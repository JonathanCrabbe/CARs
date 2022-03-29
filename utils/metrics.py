import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from explanations.concept import ConceptExplainer


def perturbation_metric(data_loader: DataLoader, attribution: np.ndarray, device: torch.device, model: nn.Module,
                        concept_explainer: ConceptExplainer, baselines: torch.Tensor, n_perts: list[int]) -> np.ndarray:
    """
    Compute the perturbation sensitivity metric for the specified number of perturbations
    Args:
        data_loader: data loader for the examples that are perturbed
        attribution: feature importance
        model: model for which perturbations are computed
        baselines: baseline features used for perturbation
        n_perts: list containing the number of perturbed features for each perturbation

    Returns:
        a list of perturbation sensitivity for each n_pert in n_perts
    """
    pert_sensitivity = np.empty((len(n_perts), len(data_loader)))
    batch_size = data_loader.batch_size
    for pert_id, n_pert in enumerate(n_perts):
        for batch_id, (batch_features, _) in enumerate(data_loader):
            batch_features = batch_features.to(device)
            batch_shape = batch_features.shape
            flat_batch_shape = batch_features.view(len(batch_features), -1).shape
            batch_attribution = attribution[batch_id*batch_size:batch_id*batch_size+len(batch_features)]
            mask = torch.ones(flat_batch_shape, device=device)
            top_pixels = torch.topk(torch.abs(torch.from_numpy(batch_attribution)).view(flat_batch_shape), n_pert)[1]
            for k in range(n_pert):
                mask[:, top_pixels[:, k]] = 0  # Mask the n_pert most important entries
            batch_features_pert = mask*batch_features.view(flat_batch_shape) +\
                                (1-mask)*baselines.view(1, -1)
            batch_features_pert = batch_features_pert.view(batch_shape)
            # Compute the latent shift between perturbed and unperturbed inputs
            batch_reps = model.input_to_representation(batch_features).detach()
            concept_importance = concept_explainer.concept_importance(batch_reps)
            batch_reps_pert = model.input_to_representation(batch_features_pert).detach()
            concept_importance_pert = concept_explainer.concept_importance(batch_reps_pert)
            pert_sensitivity[pert_id, batch_id] = torch.mean(torch.abs(concept_importance-concept_importance_pert)).item()
    return pert_sensitivity
