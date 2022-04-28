import numpy as np
import torch
from torch.optim import Adam
from tqdm import tqdm
from captum import attr
from explanations.concept import ConceptExplainer, CAR, CAV
from torch.utils.data import DataLoader


class CARFeatureImportance:
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
        baselines = kwargs['baselines']
        for input_features, _ in tqdm(data_loader, unit="batch", leave=False):
            input_features = input_features.to(self.device)
            if isinstance(baselines, torch.Tensor):
                attr = np.append(attr,
                                 self.attribution_method.attribute(input_features, **kwargs).detach().cpu().numpy(),
                                 axis=0)
            elif isinstance(baselines, torch.nn.Module):
                internal_batch_size = kwargs['internal_batch_size']
                attr = np.append(attr,
                                 self.attribution_method.attribute(input_features,
                                                                   baselines=baselines(input_features),
                                                                   internal_batch_size=internal_batch_size)
                                 .detach().cpu().numpy(),
                                 axis=0)
            else:
                raise ValueError("Invalid baseline type")
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
        baselines = kwargs["baselines"]
        for input_features, targets in tqdm(data_loader, unit="batch", leave=False):
            targets = targets.to(self.device)
            input_features = input_features.to(self.device)
            if isinstance(baselines, torch.Tensor):
                attr = np.append(attr,
                                 self.attribution_method.attribute(input_features, target=targets,
                                                                   **kwargs).detach().cpu().numpy(), axis=0)
            elif isinstance(baselines, torch.nn.Module):
                internal_batch_size = kwargs['internal_batch_size']
                attr = np.append(attr,
                                 self.attribution_method.attribute(input_features, target=targets,
                                                                   baselines=baselines(input_features),
                                                                   internal_batch_size=internal_batch_size
                                                                   ).detach().cpu().numpy(), axis=0)
        return attr


class CARModulator:
    def __init__(self, concept_explainer: CAR, black_box: torch.nn.Module, device: torch.device):
        self.concept_explainer = concept_explainer
        self.black_box = black_box.to(device)
        self.device = device

    def generate(self, data_loader: DataLoader, n_epochs: int, kernel_width: float, clamp: bool = True) -> torch.Tensor:
        self.concept_explainer.kernel_width = kernel_width
        input_shape = list(next(iter(data_loader))[0].shape[1:])
        generated_inputs = torch.empty(size=[0]+input_shape)
        # Generate counterfactuals
        for factual_features, _ in tqdm(data_loader, unit="batch", leave=False):
            factual_features = factual_features.to(self.device)
            factual_reps = self.black_box.input_to_representation(factual_features).detach()
            factual_importance = self.concept_explainer.concept_importance(factual_reps)
            factual_sign = torch.where(factual_importance > 0, 1, -1)
            modulated_features = factual_features.clone().requires_grad_(True)
            opt = Adam([modulated_features])
            for epoch in range(n_epochs):
                opt.zero_grad()
                modulated_reps = self.black_box.input_to_representation(modulated_features)
                modulated_importance = self.concept_explainer.concept_importance(modulated_reps)
                concept_loss = torch.sum(factual_sign*modulated_importance)
                concept_loss.backward()
                opt.step()
                if clamp:
                    modulated_features.data = torch.clamp(modulated_features.data, 0, 1)
            generated_inputs = torch.cat((generated_inputs, modulated_features.clone().detach().cpu()))
        return generated_inputs

    def dream(self, baseline_features: torch.Tensor, n_epochs: int, kernel_width: float) -> torch.Tensor:
        self.concept_explainer.kernel_width = kernel_width
        baseline_features = baseline_features.to(self.device)
        modulated_features = baseline_features.clone().requires_grad_(True)
        opt = Adam([modulated_features])
        for epoch in range(n_epochs):
            opt.zero_grad()
            modulated_reps = self.black_box.input_to_representation(modulated_features)
            modulated_importance = self.concept_explainer.concept_importance(modulated_reps)
            (-modulated_importance).backward()
            opt.step()
            modulated_features.data = torch.clamp(modulated_features.data, 0, 1)
        return modulated_features.clone().detach().cpu()


class CAVModulator:
    def __init__(self, concept_explainer: CAV, black_box: torch.nn.Module, device: torch.device):
        self.concept_explainer = concept_explainer
        self.black_box = black_box.to(device)
        self.device = device

    def generate(self, data_loader: DataLoader, n_epochs: int, clamp: bool = True) -> torch.Tensor:
        input_shape = list(next(iter(data_loader))[0].shape[1:])
        generated_inputs = torch.empty(size=[0]+input_shape)
        cav = self.concept_explainer.get_activation_vector()
        # Generate counterfactuals
        for factual_features, _ in tqdm(data_loader, unit="batch", leave=False):
            factual_features = factual_features.to(self.device)
            factual_reps = self.black_box.input_to_representation(factual_features).detach().cpu().numpy()
            factual_concept = torch.from_numpy(self.concept_explainer.predict(factual_reps)).to(self.device)
            factual_sign = torch.where(factual_concept > 0, 1, -1)
            modulated_features = factual_features.clone().requires_grad_(True)
            opt = Adam([modulated_features])
            for epoch in range(n_epochs):
                opt.zero_grad()
                modulated_reps = self.black_box.input_to_representation(modulated_features)
                modulated_proj = torch.einsum('bi,bi -> b', modulated_reps, cav)
                concept_loss = torch.sum(factual_sign*modulated_proj)
                concept_loss.backward()
                opt.step()
                if clamp:
                    modulated_features.data = torch.clamp(modulated_features.data, 0, 1)
            generated_inputs = torch.cat((generated_inputs, modulated_features.clone().detach().cpu()))
        return generated_inputs

    def dream(self, baseline_features: torch.Tensor, n_epochs: int) -> torch.Tensor:
        baseline_features = baseline_features.to(self.device)
        modulated_features = baseline_features.clone().requires_grad_(True)
        opt = Adam([modulated_features])
        cav = self.concept_explainer.get_activation_vector()
        for epoch in range(n_epochs):
            opt.zero_grad()
            modulated_reps = self.black_box.input_to_representation(modulated_features)
            modulated_proj = torch.einsum('bi,bi -> ', modulated_reps, cav)
            (-modulated_proj).backward()
            opt.step()
            modulated_features.data = torch.clamp(modulated_features.data, 0, 1)
        return modulated_features.clone().detach().cpu()


"""
    def fit_hyperparameters(self, data_loader: DataLoader, n_epochs: int) -> dict[str, float]:
        def counterfactual_efficiency(trial: optuna.Trial) -> float:
            reg_factor = trial.suggest_float("reg_factor", 1e-5, 1e-1, log=True)
            kernel_width = trial.suggest_float("kernel_width", .1, 20)
            return self.generate(data_loader, n_epochs, reg_factor, kernel_width)[1]
        study = optuna.create_study(direction="maximize")
        study.optimize(counterfactual_efficiency, n_trials=30)
        return study.best_params

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
            nearest_counterfactual_reps = self.get_nearest_counterfactual_reps(factual_reps)
            counterfactual_features = factual_features.clone().requires_grad_(True)
            opt = Adam([counterfactual_features])
            for epoch in range(n_epochs):
                opt.zero_grad()
                counterfactual_reps = self.black_box.input_to_representation(counterfactual_features)
                concept_loss = torch.sum(torch.abs(nearest_counterfactual_reps-counterfactual_reps))
                l1_reg = torch.sum(torch.abs(counterfactual_features-factual_features))
                loss = concept_loss + reg_factor*l1_reg
                loss.backward()
                opt.step()
                counterfactual_features.data = torch.clamp(counterfactual_features.data, 0, 1)
            counterfactuals = np.concatenate((counterfactuals, counterfactual_features.clone().detach().cpu().numpy()))
        # Compute proportion of examples for which the concept flipped
        succes_rates = []
        for batch_id, (factual_features, _) in enumerate(data_loader):
            factual_features = factual_features.to(self.device)
            factual_reps = self.black_box.input_to_representation(factual_features).detach()
            factual_concept = self.concept_explainer.predict(factual_reps.cpu().numpy())
            counterfactual_features = torch.from_numpy(
                counterfactuals[batch_size*batch_id:batch_size*batch_id+len(factual_features)]).to(self.device).float()
            counterfactual_reps = self.black_box.input_to_representation(counterfactual_features).detach()
            counterfactual_concept = self.concept_explainer.predict(counterfactual_reps.cpu().numpy())
            succes_rates.append(np.count_nonzero(factual_concept != counterfactual_concept) / len(factual_concept))
        return counterfactuals, np.mean(succes_rates)

    def get_nearest_counterfactual_reps(self, factual_reps: torch.Tensor) -> torch.Tensor:
        nearest_counterfactual_reps = torch.zeros(factual_reps.shape, device=self.device)
        kernel = self.concept_explainer.get_kernel_function()
        predicted_concepts = torch.from_numpy(self.concept_explainer.predict(factual_reps.cpu().numpy())).to(self.device)
        positive_idx = (predicted_concepts == 1)
        factual_positive_reps = factual_reps[positive_idx]
        factual_negative_reps = factual_reps[~positive_idx]
        concept_positive_reps = torch.from_numpy(self.concept_explainer.get_concept_reps(True)).to(self.device)
        concept_negative_reps = torch.from_numpy(self.concept_explainer.get_concept_reps(False)).to(self.device)
        positive_gram = kernel(factual_positive_reps, concept_negative_reps)
        negative_gram = kernel(factual_negative_reps, concept_positive_reps)
        nearest_negative_idx = torch.argmax(positive_gram, -1)
        nearest_positive_idx = torch.argmax(negative_gram, -1)
        counterfactual_positive_reps = concept_positive_reps[nearest_positive_idx]
        counterfactual_negative_reps = concept_negative_reps[nearest_negative_idx]
        nearest_counterfactual_reps[positive_idx] = counterfactual_negative_reps
        nearest_counterfactual_reps[~positive_idx] = counterfactual_positive_reps
        return nearest_counterfactual_reps
"""