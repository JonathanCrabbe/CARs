import abc
from abc import ABC
import numpy as np


class ConceptExplainer(ABC):
    """
    An abstract class that contains the interface for any post-hoc concept explainer
    """
    @abc.abstractmethod
    def __init__(self):
        self.latent_reps = None
        self.concept_labels = None

    @abc.abstractmethod
    def fit(self, latent_reps: np.ndarray, concept_labels: np.ndarray) -> None:
        """
        Fit the concept classifier to the dataset (latent_reps, concept_labels)
        Args:
            latent_reps: latent representations of the examples illustrating the concept
            concept_labels: labels indicating the presence (1) or absence (0) of the concept
        """
        self.latent_reps = latent_reps
        self.concept_labels = concept_labels
        assert(latent_reps.shape[0] == concept_labels.shape[0])

    @abc.abstractmethod
    def predict(self, latent_reps: np.ndarray) -> np.ndarray:
        """
        Predicts the presence or absence of the concept for the latent representations
        Args:
            latent_reps: representations of the test examples
        Returns:
            concepts labels indicating the presence (1) or absence (0) of the concept
        """

    @abc.abstractmethod
    def concept_importance(self, latent_reps: np.ndarray) -> np.ndarray:
        """
        Predicts the relevance of a concept for the latent representations
        Args:
            latent_reps: representations of the test examples
        Returns:
            concepts scores for each example
        """

    def get_latent_reps(self, positive_set: bool):
        """
        Get the latent representation of the positive/negative examples
        Args:
            positive_set: True returns positive examples, False returns negative examples
        Returns:
            Latent representations of the requested set
        """
        return self.latent_reps[self.concept_labels == int(positive_set)]






