import numpy as np
from captum import attr
from explanations.concept import ConceptExplainer
import torch
from torch.utils.data import DataLoader


class FeatureImportance:
    def __init__(self, attribution_name: str, concept_explainer: ConceptExplainer,
                 black_box: torch.nn.Module, device: torch.device):
        assert attribution_name in {"Gradient Shap"}
        if attribution_name == "Gradient Shap":
            self.attribution_method = attr.GradientShap(self.concept_importance)
        self.concept_explainer = concept_explainer
        self.black_box = black_box.to(device)
        self.device = device

    def attribute(self, data_loader: DataLoader, **kwargs) -> np.ndarray:
        attr = []
        for input_features, _ in data_loader:
            input_features = input_features.to(self.device)
            attr += self.attribution_method.attribute(input_features, **kwargs).tolist()
        return np.ndarray(attr)


    def concept_importance(self, input_features: torch.tensor) -> torch.Tensor:
        input_features = input_features.to(self.device)
        latent_reps = self.black_box.input_to_representation(input_features)
        return self.concept_explainer.concept_importance(latent_reps)
