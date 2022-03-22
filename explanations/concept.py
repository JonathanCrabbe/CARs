import abc
import numpy as np
import torch
import torch.nn.functional as F
from abc import ABC
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier


class ConceptExplainer(ABC):
    """
    An abstract class that contains the interface for any post-hoc concept explainer
    """
    @abc.abstractmethod
    def __init__(self, device: torch.device, batch_size: int = 50):
        self.concept_reps = None
        self.concept_labels = None
        self.classifier = None
        self.device = device
        self.batch_size = batch_size

    @abc.abstractmethod
    def fit(self, concept_reps: np.ndarray, concept_labels: np.ndarray) -> None:
        """
        Fit the concept classifier to the dataset (latent_reps, concept_labels)
        Args:
            concept_reps: latent representations of the examples illustrating the concept
            concept_labels: labels indicating the presence (1) or absence (0) of the concept
        """
        assert (concept_reps.shape[0] == concept_labels.shape[0])
        self.concept_reps = concept_reps
        self.concept_labels = concept_labels

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

    def get_concept_reps(self, positive_set: bool):
        """
        Get the latent representation of the positive/negative examples
        Args:
            positive_set: True returns positive examples, False returns negative examples
        Returns:
            Latent representations of the requested set
        """
        return self.concept_reps[self.concept_labels == int(positive_set)]


class CAR(ConceptExplainer, ABC):
    def __init__(self, device: torch.device, batch_size: int = 50, kernel: str = 'rbf'):
        super(CAR, self).__init__(device, batch_size)
        self.kernel = kernel

    def fit(self, concept_reps: np.ndarray, concept_labels: np.ndarray) -> None:
        """
        Fit the concept classifier to the dataset (latent_reps, concept_labels)
        Args:
            kernel: kernel function
            latent_reps: latent representations of the examples illustrating the concept
            concept_labels: labels indicating the presence (1) or absence (0) of the concept
        """
        super(CAR, self).fit(concept_reps, concept_labels)
        classifier = SVC(kernel=self.kernel)
        classifier.fit(concept_reps, concept_labels)
        self.classifier = classifier

    def predict(self, latent_reps: np.ndarray) -> np.ndarray:
        """
        Predicts the presence or absence of the concept for the latent representations
        Args:
            latent_reps: representations of the test examples
        Returns:
            concepts labels indicating the presence (1) or absence (0) of the concept
        """
        return self.classifier.predict(latent_reps)

    def concept_importance(self, latent_reps: np.ndarray) -> np.ndarray:
        """
        Predicts the relevance of a concept for the latent representations
        Args:
            latent_reps: representations of the test examples
        Returns:
            concepts scores for each example
        """

    def get_kernel_function(self) -> callable:
        """
        Get the kernel funtion underlying the CAR
        Returns: kernel function as a callable with arguments (h1, h2)
        """
        # The implementation should unstack one tensor to return a kernel matrix of shape len(h1) x len(h2)!
        if self.kernel == 'rbf':
            latent_reps_std = torch.from_numpy(np.std(self.latent_reps, axis=0)).to(self.device)
            latent_dim = self.latent_reps.shape[-1]
            return lambda h1, h2: torch.exp(-torch.sum(((h1 - h2)/(latent_dim*latent_reps_std))**2, dim=-1))

    def concept_density(self, latent_reps: np.ndarray, positive_set: bool) -> np.ndarray:
        kernel = self.get_kernel_function()
        ...


class CAV(ConceptExplainer, ABC):
    def __init__(self, device: torch.device, batch_size: int = 50):
        super(CAV, self).__init__(device, batch_size)

    def fit(self, concept_reps: np.ndarray, concept_labels: np.ndarray) -> None:
        """
        Fit the concept classifier to the dataset (latent_reps, concept_labels)
        Args:
            kernel: kernel function
            latent_reps: latent representations of the examples illustrating the concept
            concept_labels: labels indicating the presence (1) or absence (0) of the concept
        """
        super(CAV, self).fit(concept_reps, concept_labels)
        classifier = SGDClassifier(alpha=0.01, max_iter=1000, tol=1e-3)
        classifier.fit(concept_reps, concept_labels)
        self.classifier = classifier

    def predict(self, latent_reps: np.ndarray) -> np.ndarray:
        """
        Predicts the presence or absence of the concept for the latent representations
        Args:
            latent_reps: representations of the test examples
        Returns:
            concepts labels indicating the presence (1) or absence (0) of the concept
        """
        return self.classifier.predict(latent_reps)

    def concept_importance(self, latent_reps: np.ndarray, labels: torch.Tensor = None, num_classes: int = None,
                           rep_to_output: callable = None) -> np.ndarray:
        """
        Predicts the relevance of a concept for the latent representations
        Args:
            latent_reps: representations of the test examples
            labels: the labels associated to the representations one-hot encoded
            num_classes: the number of classes
            rep_to_output: black-box mapping the representation space to the output space
        Returns:
            concepts scores for each example
        """
        one_hot_labels = F.one_hot(labels, num_classes).to(self.device)
        latent_reps = torch.from_numpy(latent_reps).to(self.device).requires_grad_()
        outputs = rep_to_output(latent_reps)
        grads = torch.autograd.grad(outputs, latent_reps, grad_outputs=one_hot_labels)
        cav = torch.tensor(self.classifier.coef_).to(self.device).float()
        return torch.einsum("bi,bi->b", cav, grads[0]).detach().cpu().numpy()









