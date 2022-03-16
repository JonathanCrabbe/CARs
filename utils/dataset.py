import torch
import numpy as np
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from pathlib import Path


def generate_mnist_concept_dataset(concept_classes: list[int], data_dir: Path, train: bool, subset_size: int,
                                   random_seed: int) -> tuple:
    """
    Return a concept dataset with positive/negatives for MNIST
    Args:
        random_seed: random seed for reproducibility
        subset_size: size of the positive and negative subset
        concept_classes: the classes where the concept is present in MNIST
        data_dir: directory where MNIST is saved
        train: sample from the training set

    Returns:
        a concept dataset of the form X (features),y (concept labels)
    """
    dataset = MNIST(data_dir, train=train, download=True)
    data_transform = transforms.Compose([transforms.ToTensor()])
    dataset.transform = data_transform
    targets = dataset.targets
    mask = torch.zeros(len(targets))
    for idx, target in enumerate(targets):  # Scan the dataset for valid examples
        if target in concept_classes:
            mask[idx] = 1
    positive_idx = torch.nonzero(mask).flatten()
    negative_idx = torch.nonzero(1-mask).flatten()
    positive_loader = torch.utils.data.DataLoader(dataset, batch_size=subset_size,
                                                  sampler=SubsetRandomSampler(positive_idx))
    negative_loader = torch.utils.data.DataLoader(dataset, batch_size=subset_size,
                                                  sampler=SubsetRandomSampler(negative_idx))
    positive_images, positive_labels = next(iter(positive_loader))
    negative_images, negative_labels = next(iter(negative_loader))
    X = np.concatenate((positive_images.cpu().numpy(), negative_images.cpu().numpy()), 0)
    y = np.concatenate((np.ones(subset_size), np.zeros(subset_size)), 0)
    np.random.seed(random_seed)
    rand_perm = np.random.permutation(len(X))
    return X[rand_perm], y[rand_perm]
