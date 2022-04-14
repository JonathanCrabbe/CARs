import pandas as pd
import os
import logging
import torch
import pickle
import numpy as np
import linecache
from PIL import Image
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader, Dataset, BatchSampler
from pathlib import Path
from abc import ABC
from imblearn.over_sampling import SMOTE

"""
The code for the CUB dataset is adapted from 
https://github.com/yewsiang/ConceptBottleneck/tree/a2fd8184ad609bf0fb258c0b1c7a0cc44989f68f
"""


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
        import kaggle
        logging.info(f"Downloading ECG dataset in {self.data_dir}")
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files('shayanfazeli/heartbeat', path=self.data_dir, unzip=True)
        logging.info(f"ECG dataset downloaded in {self.data_dir}")


class CUBDataset(Dataset):
    """
    Returns a compatible Torch Dataset object customized for the CUB dataset
    """

    N_ATTRIBUTES = 312
    N_CLASSES = 200
    attribute_map = [1, 4, 6, 7, 10, 14, 15, 20, 21, 23, 25, 29, 30, 35, 36, 38, 40, 44, 45, 50, 51, 53, 54, 56, 57, 59,
                     63, 64, 69, 70, 72, 75, 80, 84, 90, 91, 93, 99, 101, 106, 110, 111, 116, 117, 119, 125, 126, 131,
                     132, 134, 145, 149, 151, 152, 153, 157, 158, 163, 164, 168, 172, 178, 179, 181, 183, 187, 188, 193,
                     194, 196, 198, 202, 203, 208, 209, 211, 212, 213, 218, 220, 221, 225, 235, 236, 238, 239, 240,
                     242, 243, 244, 249, 253, 254, 259, 260, 262, 268, 274, 277, 283, 289, 292, 293, 294, 298, 299, 304,
                     305, 308, 309, 310, 311]

    def __init__(self, pkl_file_paths, use_attr, no_img, uncertain_label, image_dir, n_class_attr, transform=None):
        """
        Arguments:
        pkl_file_paths: list of full path to all the pkl data
        use_attr: whether to load the attributes (e.g. False for simple finetune)
        no_img: whether to load the images (e.g. False for A -> Y model)
        uncertain_label: if True, use 'uncertain_attribute_label' field (i.e. label weighted by uncertainty score, e.g. 1 & 3(probably) -> 0.75)
        image_dir: default = 'images'. Will be append to the parent dir
        n_class_attr: number of classes to predict for each attribute. If 3, then make a separate class for not visible
        transform: whether to apply any special transformation. Default = None, i.e. use standard ImageNet preprocessing
        """
        self.data = []
        self.is_train = any(["train" in path for path in pkl_file_paths])
        if not self.is_train:
            assert any([("test" in path) or ("val" in path) for path in pkl_file_paths])
        for file_path in pkl_file_paths:
            self.data.extend(pickle.load(open(file_path, 'rb')))
        self.transform = transform
        self.use_attr = use_attr
        self.no_img = no_img
        self.uncertain_label = uncertain_label
        self.image_dir = image_dir
        self.n_class_attr = n_class_attr

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_data = self.data[idx]
        img_path = img_data['img_path']
        # Trim unnecessary paths
        try:
            idx = img_path.split('/').index('CUB_200_2011')
            if self.image_dir != 'images':
                img_path = '/'.join([self.image_dir] + img_path.split('/')[idx+1:])
                #img_path = img_path.replace('images/', '')
            else:
                img_path = '/'.join(img_path.split('/')[idx:])
            img = Image.open(img_path).convert('RGB')
        except:
            img_path_split = img_path.split('/')
            split = 'train' if self.is_train else 'test'
            img_path = '/'.join(img_path_split[:2] + [split] + img_path_split[2:])
            img = Image.open(img_path).convert('RGB')

        class_label = img_data['class_label']
        if self.transform:
            img = self.transform(img)

        if self.use_attr:
            if self.uncertain_label:
                attr_label = img_data['uncertain_attribute_label']
            else:
                attr_label = img_data['attribute_label']
            if self.no_img:
                if self.n_class_attr == 3:
                    one_hot_attr_label = np.zeros((self.N_ATTRIBUTES, self.n_class_attr))
                    one_hot_attr_label[np.arange(self.N_ATTRIBUTES), attr_label] = 1
                    return one_hot_attr_label, class_label
                else:
                    return attr_label, class_label
            else:
                return img, class_label, attr_label
        else:
            return img, class_label

    def concept_instance_count(self, concept_id) -> int:
        """
        Counts the number of time a concept appears in the dataset
        Args:
            concept_id: integer identifying the concept

        Returns:
            Integer counting the occurrence of the concept
        """
        count = 0
        for data_dic in self.data:
            count += data_dic["attribute_label"][concept_id]
        return count

    def concept_name(self, concept_id) -> str:
        """
        Get the name of a concept
        Args:
            concep_id: integer identifying the concept

        Returns:
            String corresponding to the concept name
        """
        attributes_path = Path(self.image_dir)/"attributes/attributes.txt"
        full_name = linecache.getline(str(attributes_path), self.attribute_map[concept_id])
        full_name = full_name.split(" ")[1]  # Remove the line number
        concept_name, concept_value = full_name.split("::")
        concept_value = concept_value[:-1] # Remove the breakline character
        concept_value = concept_value.replace("_", " ")  # Put spacing in concept values
        concept_name = concept_name[4:]  # Remove the "has_" characters
        concept_name = concept_name.replace("_", " ")  # Put spacing in concept names
        return f"{concept_value} {concept_name}".title()


class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
    """

    def __init__(self, dataset, indices=None):
        # if indices is not provided,
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) \
            if indices is None else indices

        # if num_samples is not provided,
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices)

        # distribution of classes in the dataset
        label_to_count = {}
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1

        # weight for each sample
        weights = [1.0 / label_to_count[self._get_label(dataset, idx)]
                   for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)

    def _get_label(self, dataset, idx):  # Note: for single attribute dataset
        return dataset.data[idx]['attribute_label'][0]

    def __iter__(self):
        idx = (self.indices[i] for i in torch.multinomial(
            self.weights, self.num_samples, replacement=True))
        return idx

    def __len__(self):
        return self.num_samples


def load_cub_data(pkl_paths, use_attr, no_img, batch_size, uncertain_label=False,
                  n_class_attr=2, image_dir='images', resampling=False, resol=299):
    """
    Note: Inception needs (299,299,3) images with inputs scaled between -1 and 1
    Loads data with transformations applied, and upsample the minority class if there is class imbalance and weighted loss is not used
    NOTE: resampling is customized for first attribute only, so change sampler.py if necessary
    """
    resized_resol = int(resol * 256/224)
    is_training = any(['train.pkl' in f for f in pkl_paths])
    if is_training:
        transform = transforms.Compose([
            #transforms.Resize((resized_resol, resized_resol)),
            #transforms.RandomSizedCrop(resol),
            transforms.ColorJitter(brightness=32/255, saturation=(0.5, 1.5)),
            transforms.RandomResizedCrop(resol),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), #implicitly divides by 255
            #transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [2, 2, 2])
            transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ], std = [ 0.229, 0.224, 0.225 ]),
            ])
    else:
        transform = transforms.Compose([
            #transforms.Resize((resized_resol, resized_resol)),
            transforms.CenterCrop(resol),
            transforms.ToTensor(), #implicitly divides by 255
            #transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [2, 2, 2])
            transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ], std = [ 0.229, 0.224, 0.225 ]),
            ])

    dataset = CUBDataset(pkl_paths, use_attr, no_img, uncertain_label, image_dir, n_class_attr, transform)
    if is_training:
        drop_last = True
        shuffle = True
    else:
        drop_last = False
        shuffle = False
    if resampling:
        sampler = BatchSampler(ImbalancedDatasetSampler(dataset), batch_size=batch_size, drop_last=drop_last)
        loader = DataLoader(dataset, batch_sampler=sampler)
    else:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
    return loader

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


def generate_ecg_concept_dataset(concept_class: int, data_dir: Path, train: bool, subset_size: int,
                                 random_seed: int) -> tuple:
    """
    Return a concept dataset with positive/negatives for ECG
    Args:
        random_seed: random seed for reproducibility
        subset_size: size of the positive and negative subset
        concept_class: the classes where the concept is present in ECG
        data_dir: directory where ECG is saved
        train: sample from the training set

    Returns:
        a concept dataset of the form X (features),y (concept labels)
    """
    dataset = ECGDataset(data_dir, train, balance_dataset=True, random_seed=random_seed, binarize_label=False)
    targets = dataset.y
    mask = targets == concept_class
    positive_idx = torch.nonzero(mask).flatten()
    negative_idx = torch.nonzero(~mask).flatten()
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





