import torch
import os
import logging
import argparse
from pathlib import Path
from torchvision import transforms
from torch.utils.data import DataLoader
from utils.dataset import ECGDataset
from models.ecg import ClassifierECG


def concept_accuracy(random_seeds: list[int], batch_size: int, latent_dim: int, train: bool,
                     save_dir: Path = Path.cwd()/"results/ecg/concept_accuracy",
                     data_dir: Path = Path.cwd()/"data/ecg", model_name: str = "model") -> None:
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    torch.manual_seed(random_seeds[0])
    model_dir = save_dir / model_name
    if not save_dir.exists():
        os.makedirs(save_dir)
    if not model_dir.exists():
        os.makedirs(model_dir)
    model = ClassifierECG(latent_dim, model_name).to(device)
    if train:
        train_set = ECGDataset(data_dir, train=True, balance_dataset=True)
        test_set = ECGDataset(data_dir, train=False, balance_dataset=False)
        train_loader = DataLoader(train_set, batch_size, shuffle=True)
        test_loader = DataLoader(test_set, batch_size, shuffle=True)
        model.fit(device, train_loader, test_loader, model_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="concept_accuracy")
    parser.add_argument('--seeds', nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--latent_dim", type=int, default=500)
    parser.add_argument("--train", action='store_true')
    args = parser.parse_args()
    if args.name == "concept_accuracy":
        concept_accuracy(args.seeds, args.batch_size, args.latent_dim, args.train)
    else:
        raise ValueError(f"{args.name} is not a valid experiment name")
