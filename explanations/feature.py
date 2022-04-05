import numpy as np
import torch
import optuna
from torch.optim import Adam
from tqdm import tqdm
from captum import attr
from explanations.concept import ConceptExplainer, CAR, CAV
from torch.utils.data import DataLoader


class ConceptFeatureImportance:
    def __init__(self, attribution_name: str, concept_explainer: ConceptExplainer,
                 black_box: torch.nn.Module, device: torch.device):
        assert attribution_name in {"Gradient Shap", "Integrated Gradient"}
        if attribution_name == "Gradient Shap":
            self.attribution_method = attr.GradientShap(self.concept_importance)
        elif attribution_name == "Integrated Gradient":
            self.attribution_method = attr.IntegratedGradients(self.concept_importance)
        self.concept_explainer = concept_explainer
        self.black_box = black_box.to(device)
        self.device = device

    def attribute(self, data_loader: DataLoader, **kwargs) -> np.ndarray:
        input_shape = list(data_loader.dataset[0][0].shape)
        attr = np.empty(shape=[0]+input_shape)
        for input_features, _ in tqdm(data_loader, unit="batch", leave=False):
            input_features = input_features.to(self.device)
            attr = np.append(attr,
                             self.attribution_method.attribute(input_features, **kwargs).detach().cpu().numpy(),
                             axis=0)
        return attr


    def concept_importance(self, input_features: torch.tensor) -> torch.Tensor:
        input_features = input_features.to(self.device)
        latent_reps = self.black_box.input_to_representation(input_features)
        return self.concept_explainer.concept_importance(latent_reps)


class VanillaFeatureImportance:
    def __init__(self, attribution_name: str, black_box: torch.nn.Module, device: torch.device):
        assert attribution_name in {"Gradient Shap", "Integrated Gradient"}
        if attribution_name == "Gradient Shap":
            self.attribution_method = attr.GradientShap(black_box)
        elif attribution_name == "Integrated Gradient":
            self.attribution_method = attr.IntegratedGradients(black_box)
        self.black_box = black_box.to(device)
        self.device = device

    def attribute(self, data_loader: DataLoader, **kwargs) -> np.ndarray:
        input_shape = list(data_loader.dataset[0][0].shape)
        attr = np.empty(shape=[0]+input_shape)
        for input_features, targets in tqdm(data_loader, unit="batch", leave=False):
            targets = targets.to(self.device)
            input_features = input_features.to(self.device)
            attr = np.append(attr,
                             self.attribution_method.attribute(input_features, target=targets, **kwargs)
                             .detach().cpu().numpy(), axis=0)
        return attr


class CARCounterfactual:
    def __init__(self, concept_explainer: CAR, black_box: torch.nn.Module, device: torch.device):
        self.concept_explainer = concept_explainer
        self.black_box = black_box.to(device)
        self.device = device

    def generate(self, data_loader: DataLoader, n_epochs: int, reg_factor: float, kernel_width: float) -> tuple:
        self.concept_explainer.kernel_width = kernel_width
        input_shape = list(next(iter(data_loader))[0].shape[1:])
        batch_size = data_loader.batch_size
        counterfactuals = np.empty(shape=[0]+input_shape)
        # Generate counterfactuals
        for factual_features, _ in tqdm(data_loader, unit="batch", leave=False):
            factual_features = factual_features.to(self.device)
            factual_reps = self.black_box.input_to_representation(factual_features).detach()
            factual_importance = self.concept_explainer.concept_importance(factual_reps)
            factual_sign = torch.where(factual_importance > 0, 1, -1)
            counterfactual_features = factual_features.clone().requires_grad_(True)
            opt = Adam([counterfactual_features])
            for epoch in range(n_epochs):
                opt.zero_grad()
                counterfactual_reps = self.black_box.input_to_representation(counterfactual_features)
                counterfactual_importance = self.concept_explainer.concept_importance(counterfactual_reps)
                concept_loss = torch.sum(factual_sign*counterfactual_importance)
                l1_reg = torch.sum(torch.abs(counterfactual_features-factual_features))
                loss = concept_loss + reg_factor*l1_reg
                loss.backward()
                opt.step()
                counterfactual_features.data = torch.clamp(counterfactual_features.data, 0, 1)
            counterfactuals = np.concatenate((counterfactuals, counterfactual_features.clone().detach().cpu().numpy()))
        # Compute proportion of examples for which the concept flipped
        props_flip = []
        for batch_id, (factual_features, _) in enumerate(data_loader):
            factual_features = factual_features.to(self.device)
            factual_reps = self.black_box.input_to_representation(factual_features).detach()
            factual_concept = self.concept_explainer.predict(factual_reps.cpu().numpy())
            counterfactual_features = torch.from_numpy(
                counterfactuals[batch_size*batch_id:batch_size*batch_id+len(factual_features)]).to(self.device).float()
            counterfactual_reps = self.black_box.input_to_representation(counterfactual_features).detach()
            counterfactual_concept = self.concept_explainer.predict(counterfactual_reps.cpu().numpy())
            props_flip.append(np.count_nonzero(factual_concept != counterfactual_concept) / len(factual_concept))
        return counterfactuals, np.mean(props_flip)

    def fit_hyperparameters(self, data_loader: DataLoader, n_epochs: int) -> dict[str, float]:
        def counterfactual_efficiency(trial: optuna.Trial) -> float:
            reg_factor = trial.suggest_float("reg_factor", 1e-5, 1e-1, log=True)
            kernel_width = trial.suggest_float("kernel_width", .1, 20)
            return self.generate(data_loader, n_epochs, reg_factor, kernel_width)[1]
        study = optuna.create_study(direction="maximize")
        study.optimize(counterfactual_efficiency, n_trials=30)
        return study.best_params


class CAVCounterfactual:
    def __init__(self, concept_explainer: CAV, black_box: torch.nn.Module, device: torch.device):
        self.concept_explainer = concept_explainer
        self.black_box = black_box.to(device)
        self.device = device

    def generate(self, data_loader: DataLoader, n_epochs: int, reg_factor: float) -> tuple:
        input_shape = list(next(iter(data_loader))[0].shape[1:])
        batch_size = data_loader.batch_size
        counterfactuals = np.empty(shape=[0]+input_shape)
        cav = self.concept_explainer.get_activation_vector()
        # Generate counterfactuals
        for factual_features, _ in tqdm(data_loader, unit="batch", leave=False):
            factual_features = factual_features.to(self.device)
            factual_reps = self.black_box.input_to_representation(factual_features).detach().cpu().numpy()
            factual_concept = torch.from_numpy(self.concept_explainer.predict(factual_reps)).to(self.device)
            factual_sign = torch.where(factual_concept > 0, 1, -1)
            counterfactual_features = factual_features.clone().requires_grad_(True)
            opt = Adam([counterfactual_features])
            for epoch in range(n_epochs):
                opt.zero_grad()
                counterfactual_reps = self.black_box.input_to_representation(counterfactual_features)
                counterfactual_proj = torch.einsum('bi,bi -> b', counterfactual_reps, cav)
                concept_loss = torch.sum(factual_sign*counterfactual_proj)
                l1_reg = torch.sum(torch.abs(counterfactual_features-factual_features))
                loss = concept_loss + reg_factor*l1_reg
                loss.backward()
                opt.step()
                counterfactual_features.data = torch.clamp(counterfactual_features.data, 0, 1)
            counterfactuals = np.concatenate((counterfactuals, counterfactual_features.clone().detach().cpu().numpy()))
        # Compute proportion of examples for which the concept flipped
        props_flip = []
        for batch_id, (factual_features, _) in enumerate(data_loader):
            factual_features = factual_features.to(self.device)
            factual_reps = self.black_box.input_to_representation(factual_features).detach()
            factual_concept = self.concept_explainer.predict(factual_reps.cpu().numpy())
            counterfactual_features = torch.from_numpy(
                counterfactuals[batch_size*batch_id:batch_size*batch_id+len(factual_features)]).to(self.device).float()
            counterfactual_reps = self.black_box.input_to_representation(counterfactual_features).detach()
            counterfactual_concept = self.concept_explainer.predict(counterfactual_reps.cpu().numpy())
            props_flip.append(np.count_nonzero(factual_concept != counterfactual_concept) / len(factual_concept))
        return counterfactuals, np.mean(props_flip)

