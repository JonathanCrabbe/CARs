import random
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
from imblearn.over_sampling import SMOTE, RandomOverSampler
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


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
        file_path = data_dir / "mitbih_train.csv" if train else data_dir / "mitbih_test.csv"
        df = pd.read_csv(file_path)
        X = df.iloc[:, :187].values
        y = df.iloc[:, 187].values
        if balance_dataset:
            n_normal = np.count_nonzero(y == 0)
            balancing_dic = {0: n_normal, 1: int(n_normal / 4), 2: int(n_normal / 4),
                             3: int(n_normal / 4), 4: int(n_normal / 4)}
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
                img_path = '/'.join([self.image_dir] + img_path.split('/')[idx + 1:])
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

    def get_raw_image(self, idx: int,  resol: int = 299):
        img_data = self.data[idx]
        img_path = img_data['img_path']
        # Trim unnecessary paths
        try:
            idx = img_path.split('/').index('CUB_200_2011')
            if self.image_dir != 'images':
                img_path = '/'.join([self.image_dir] + img_path.split('/')[idx + 1:])
            else:
                img_path = '/'.join(img_path.split('/')[idx:])
            img = Image.open(img_path).convert('RGB')
        except:
            img_path_split = img_path.split('/')
            split = 'train' if self.is_train else 'test'
            img_path = '/'.join(img_path_split[:2] + [split] + img_path_split[2:])
            img = Image.open(img_path).convert('RGB')
        center_crop = transforms.Resize((resol, resol))
        return center_crop(img)

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

    def class_name(self, class_id) -> str:
        """
        Get the name of a class
        Args:
            class_id: integer identifying the concept

        Returns:
            String corresponding to the concept name
        """
        class_path = Path(self.image_dir) / "classes.txt"
        name = linecache.getline(str(class_path), class_id+1)
        name = name.split(".")[1]  # Remove the line number
        name = name.replace("_", " ")  # Put spacing in class names
        name = name[:-1]  # Remove breakline character
        return name.title()

    def concept_name(self, concept_id: int) -> str:
        """
        Get the name of a concept
        Args:
            concep_id: integer identifying the concept

        Returns:
            String corresponding to the concept name
        """
        attributes_path = Path(self.image_dir) / "attributes/attributes.txt"
        full_name = linecache.getline(str(attributes_path), self.attribute_map[concept_id]+1)
        full_name = full_name.split(" ")[1]  # Remove the line number
        concept_name, concept_value = full_name.split("::")
        concept_value = concept_value[:-1]  # Remove the breakline character
        concept_value = concept_value.replace("_", " ")  # Put spacing in concept values
        concept_name = concept_name[4:]  # Remove the "has_" characters
        concept_name = concept_name.replace("_", " ")  # Put spacing in concept names
        return f"{concept_name} {concept_value}".title()

    def concept_id(self, concept_name: str) -> int:
        """
        Get the integer identifying a concept
        Args:
            concept_name: the name identifying the concept

        Returns:
            Unique integer corresponding to the concept
        """
        concept_names = self.get_concept_names()
        return concept_names.index(concept_name)

    def concept_example_ids(self, concept_id: int, positive: bool = True) -> list:
        """
        Get the dataset indices of the examples that exhibit a concept
        Args:
            concept_id: integer identifying the concept
            positive: whether to return positive examples

        Returns:
            List of all the examples indices that have the concept
        """
        example_ids = []
        for idx, data_dic in enumerate(self.data):
            if data_dic["attribute_label"][concept_id] == int(positive):
                example_ids.append(idx)
        return example_ids

    def get_concept_names(self):
        """
        Get the name of all concepts
        Returns:
            List of all concept names
        """
        return [self.concept_name(i) for i in range(len(self.attribute_map))]

    def get_class_names(self):
        """
        Get the name of all concepts
        Returns:
            List of all concept names
        """
        return [self.class_name(i) for i in range(self.N_CLASSES)]

    def get_concept_categories(self) -> dict[str, list]:
        """
        Get all the groups of related concepts
        Returns:
            A dictionary with concept group names as keys and related concept indices as values
        """
        attributes_path = Path(self.image_dir) / "attributes/attributes.txt"
        groups_dic = {}
        prev_name = ""
        for concept_id in range(len(self.attribute_map)):
            line = linecache.getline(str(attributes_path), self.attribute_map[concept_id]+1)
            line = line.split(" ")[1]  # Remove the line number
            concept_name, concept_value = line.split("::")
            concept_name = concept_name[4:]  # Remove the "has_" characters
            concept_name = concept_name.replace("_", " ")  # Put spacing in concept names
            concept_name = concept_name.title()
            if concept_name == prev_name:
                groups_dic[concept_name].append(self.concept_name(concept_id))
            else:
                groups_dic[concept_name] = [self.concept_name(concept_id)]
                prev_name = concept_name
        return groups_dic

    def get_concepts_subset(self, concept_ids: list[int], instance_per_concept: int, random_seed: int) -> list[int]:
        """
        Give a list of example indices to create balance subset with several concepts
        Args:
            instance_per_concept: number of examples per concept (positive & negative)
            concept_ids: concept to consider
            random_seed: random seed for reproducibility

        Returns:
            List of example ids that can be used for subsampling
        """
        example_ids = []
        random.seed(random_seed)
        for concept_idx in concept_ids:
            positive_ids = random.sample(self.concept_example_ids(concept_idx), instance_per_concept)
            example_ids += positive_ids
            negative_ids = random.sample(self.concept_example_ids(concept_idx, False), instance_per_concept)
            example_ids += negative_ids
        return example_ids


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


class SEERDataset(Dataset):
    def __init__(self,  path_csv: str, random_seed: int, train: bool,
                 load_concept_labels: bool = False, oversample: bool = True, test_fraction: float = 0.1):
        """
        Load the SEER dataset.
        Args:
            path_csv: str, path to the dataset
            preprocess: bool, option. Perform imputation and label encoding
        Returns:
            X: the feature set
            T: days to event or censoring
            Y: Outcome or censoring
        """
        assert 0 < test_fraction < 1
        data_dir = Path(path_csv).parent
        if not (data_dir/"X_train.csv").exists(): # If the train-test split has not been performed yet
            expected_columns = [
                "Age at Diagnosis",
                "PSA Lab Value",
                "T Stage",
                "Grade",
                "AJCC Stage",
                "Primary Gleason",
                "Secondary Gleason",
                "Composite Gleason",
                "Number of Cores Positive",
                "Number of Cores Negative",
                "Number of Cores Examined",
                "Censoring",
                "Days to death or current survival status",
                "cancer related death",
                "any cause of  death",
            ]

            dataset = pd.read_csv(path_csv)
            assert set(dataset.columns) == set(expected_columns), "Invalid dataset provided."

            X = dataset.drop(
                [   "Censoring",
                    "Days to death or current survival status",
                    "cancer related death",
                    "any cause of  death",
                    "Composite Gleason",
                    "Number of Cores Negative",
                    "AJCC Stage",
                ],
                axis=1,
            )

            rename_cols = {
                "Age at Diagnosis": "Age at Diagnosis",
                "PSA Lab Value": "PSA (ng/ml)",
                "T Stage": "Clinical T stage",
                "Grade": "Histological grade group",
                "Number of Cores Positive": "Number of Cores Positive",
                "Number of Cores Examined": "Number of Cores Examined",
            }
            X = X.rename(columns=rename_cols)

            Y = dataset["cancer related death"]
            T = dataset["Days to death or current survival status"]

            # Remove empty events
            remove_empty = T > 0
            X = X[remove_empty]
            Y = Y[remove_empty]
            T = T[remove_empty]

            # One-hot encoding
            cat_columns = ["Clinical T stage", "Primary Gleason", "Secondary Gleason"]
            encoders = {}
            for col in cat_columns:
                ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
                ohe.fit(X[[col]].values)

                encoders[col] = ohe

            def encoder(df: pd.DataFrame) -> pd.DataFrame:
                output = df.copy()
                for col in encoders:
                    ohe = encoders[col]
                    encoded = pd.DataFrame(
                        ohe.transform(output[[col]].values),
                        columns=ohe.get_feature_names_out([col]),
                        index=output.index.copy(),
                    )
                    output = pd.concat([output, encoded], axis=1)
                    output.drop(columns=[col], inplace=True)

                return output

            X = encoder(X)

            # Save a training set and a test set
            test_size = int(len(X) * test_fraction)
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_seed,
                                                                stratify=Y)
            X_train.to_csv(data_dir/"X_train.csv", index=False)
            X_test.to_csv(data_dir / "X_test.csv", index=False)
            Y_train.to_csv(data_dir / "Y_train.csv", index=False)
            Y_test.to_csv(data_dir / "Y_test.csv", index=False)

        if train:
            X = pd.read_csv(data_dir/"X_train.csv")
            Y = pd.read_csv(data_dir/"Y_train.csv")

        else:
            X = pd.read_csv(data_dir / "X_test.csv")
            Y = pd.read_csv(data_dir / "Y_test.csv")


        # Imputation
        imp = IterativeImputer(missing_values=np.nan, random_state=random_seed)
        X = pd.DataFrame(imp.fit_transform(X), columns=X.columns)
        assert not X.isnull().values.any()

        # Standardize continuous features
        scaler = StandardScaler()
        num_columns = ["Age at Diagnosis", "PSA (ng/ml)", "Number of Cores Positive", "Number of Cores Examined"]
        X[num_columns] = scaler.fit_transform(X[num_columns])

        if oversample:  # Over-sample a balanced dataset
            over_sampler = RandomOverSampler(random_state=random_seed)
            X, Y = over_sampler.fit_resample(X, Y)
        G = X["Histological grade group"]
        X = X.drop(["Histological grade group"], axis=1)

        # One-hot encode concept
        G = pd.get_dummies(G)

        self.X = X
        self.Y = Y
        self.G = G
        self.load_concept_labels = load_concept_labels

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        x = self.X.iloc[[idx]].values
        x = torch.tensor(x, dtype=torch.float32).flatten()
        y = self.Y.iloc[[idx]].values[0][0]
        g = self.G.iloc[[idx]].values[0]
        if self.load_concept_labels:
            return x, y, g
        else:
            return x, y


def load_cub_data(pkl_paths, use_attr, no_img, batch_size, uncertain_label=False,
                  n_class_attr=2, image_dir='images', resampling=False, resol=299):
    """
    Note: Inception needs (299,299,3) images with inputs scaled between -1 and 1
    Loads data with transformations applied, and upsample the minority class if there is class imbalance and weighted loss is not used
    NOTE: resampling is customized for first attribute only, so change sampler.py if necessary
    """
    resized_resol = int(resol * 256 / 224)
    is_training = any(['train.pkl' in f for f in pkl_paths])
    if is_training:
        transform = transforms.Compose([
            transforms.ColorJitter(brightness=32 / 255, saturation=(0.5, 1.5)),
            transforms.RandomResizedCrop(resol),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),  # implicitly divides by 255
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((resized_resol, resized_resol)),
            transforms.ToTensor(),  # implicitly divides by 255
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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
    negative_idx = torch.nonzero(1 - mask).flatten()
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


def generate_cub_concept_dataset(concept_id: int, subset_size: int, random_seed: int,
                                 pkl_paths, use_attr, no_img, uncertain_label=False,
                                 n_class_attr=2, image_dir='images', resol=299
                                 ) -> tuple:
    """
    Return a concept dataset with positive/negatives for CUB
    Args:
        concept_id: concept integer identifier
        random_seed: random seed for reproducibility
        subset_size: size of the positive and negative subset


    Returns:
        a concept dataset of the form X (features),y (concept labels)
    """
    resized_resol = int(resol * 256 / 224)
    transform = transforms.Compose([
        transforms.Resize((resized_resol, resized_resol)),
        # transforms.CenterCrop(resol),
        transforms.ToTensor(),  # implicitly divides by 255
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = CUBDataset(pkl_paths, use_attr, no_img, uncertain_label, image_dir, n_class_attr, transform)
    positive_idx = dataset.concept_example_ids(concept_id)
    negative_idx = dataset.concept_example_ids(concept_id, False)
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


def generate_seer_concept_dataset(dataset: SEERDataset, concept_id: int, subset_size: int, random_seed: int) -> tuple:
    torch.manual_seed(random_seed)
    positive_ids = []
    negative_ids = []
    for patient_id, (patient_data, patient_label, patient_concept) in enumerate(iter(dataset)):
        if patient_concept[concept_id] == 1:
            positive_ids.append(patient_id)
        else:
            negative_ids.append(patient_id)
    random.seed(random_seed)
    random.shuffle(positive_ids)
    random.shuffle(negative_ids)
    X = torch.stack([dataset[idx][0] for idx in positive_ids[:subset_size]] + [dataset[idx][0] for idx in negative_ids[:subset_size]])
    C = torch.cat([torch.ones(subset_size), torch.zeros(subset_size)])
    rand_perm = torch.randperm(len(X))
    return X[rand_perm], C[rand_perm]
