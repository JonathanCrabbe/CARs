import pandas as pd
import torch
import kaggle
import os
import logging
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from abc import ABC
from imblearn.over_sampling import SMOTE


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


class ECGDataset(Dataset, ABC):
    def __init__(self, data_dir: Path, train: bool, balance_dataset: bool,
                 random_seed: int = 42, binarize_label: bool = True):
        """
        Generate a ECG dataset
        Args:
            data_dir: directory where the dataset should be stored
            train: True if the training set should be returned, False for the testing set
            balance_dataset: True if the classes should be balanced with SMOTE
            random_seed: random seed for reproducibility
            binarize_label: True if the label should be binarized (0: normal heartbeat, 1: abnormal heartbeat)
        """
        self.data_dir = data_dir
        if not data_dir.exists():
            os.makedirs(data_dir)
            self.download()
        # Read CSV; extract features and labels
        file_path = data_dir/"mitbih_train.csv" if train else data_dir/"mitbih_test.csv"
        df = pd.read_csv(file_path)
        X = df.iloc[:, :187].values
        y = df.iloc[:, 187].values
        if balance_dataset:
            n_normal = np.count_nonzero(y == 0)
            balancing_dic = {0: n_normal, 1: int(n_normal/4), 2: int(n_normal/4),
                             3: int(n_normal/4), 4: int(n_normal/4)}
            smote = SMOTE(random_state=random_seed, sampling_strategy=balancing_dic)
            X, y = smote.fit_resample(X, y)
        if binarize_label:
            y = np.where(y >= 1, 1, 0)
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def download(self) -> None:
        logging.info(f"Downloading ECG dataset in {self.data_dir}")
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files('shayanfazeli/heartbeat', path=self.data_dir, unzip=True)
        logging.info(f"ECG dataset downloaded in {self.data_dir}")




