import numpy as np
import torch
from tqdm import tqdm
from captum import attr
from explanations.concept import ConceptExplainer
from torch.utils.data import DataLoader


class CARFeatureImportance:
    def __init__(
        self,
        attribution_name: str,
        concept_explainer: ConceptExplainer,
        black_box: torch.nn.Module,
        device: torch.device,
    ):
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
        attr = np.empty(shape=[0] + input_shape)
        baselines = kwargs["baselines"]
        for input_features, _ in tqdm(data_loader, unit="batch", leave=False):
            input_features = input_features.to(self.device)
            if isinstance(baselines, torch.Tensor):
                attr = np.append(
                    attr,
                    self.attribution_method.attribute(input_features, **kwargs)
                    .detach()
                    .cpu()
                    .numpy(),
                    axis=0,
                )
            elif isinstance(baselines, torch.nn.Module):
                internal_batch_size = kwargs["internal_batch_size"]
                attr = np.append(
                    attr,
                    self.attribution_method.attribute(
                        input_features,
                        baselines=baselines(input_features),
                        internal_batch_size=internal_batch_size,
                    )
                    .detach()
                    .cpu()
                    .numpy(),
                    axis=0,
                )
            else:
                raise ValueError("Invalid baseline type")
        return attr

    def concept_importance(self, input_features: torch.tensor) -> torch.Tensor:
        input_features = input_features.to(self.device)
        latent_reps = self.black_box.input_to_representation(input_features)
        return self.concept_explainer.concept_importance(latent_reps)


class VanillaFeatureImportance:
    def __init__(
        self, attribution_name: str, black_box: torch.nn.Module, device: torch.device
    ):
        assert attribution_name in {"Gradient Shap", "Integrated Gradient"}
        if attribution_name == "Gradient Shap":
            self.attribution_method = attr.GradientShap(black_box)
        elif attribution_name == "Integrated Gradient":
            self.attribution_method = attr.IntegratedGradients(black_box)
        self.black_box = black_box.to(device)
        self.device = device

    def attribute(self, data_loader: DataLoader, **kwargs) -> np.ndarray:
        input_shape = list(data_loader.dataset[0][0].shape)
        attr = np.empty(shape=[0] + input_shape)
        baselines = kwargs["baselines"]
        for input_features, targets in tqdm(data_loader, unit="batch", leave=False):
            targets = targets.to(self.device)
            input_features = input_features.to(self.device)
            if isinstance(baselines, torch.Tensor):
                attr = np.append(
                    attr,
                    self.attribution_method.attribute(
                        input_features, target=targets, **kwargs
                    )
                    .detach()
                    .cpu()
                    .numpy(),
                    axis=0,
                )
            elif isinstance(baselines, torch.nn.Module):
                internal_batch_size = kwargs["internal_batch_size"]
                attr = np.append(
                    attr,
                    self.attribution_method.attribute(
                        input_features,
                        target=targets,
                        baselines=baselines(input_features),
                        internal_batch_size=internal_batch_size,
                    )
                    .detach()
                    .cpu()
                    .numpy(),
                    axis=0,
                )
        return attr
