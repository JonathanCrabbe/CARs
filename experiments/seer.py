import os
import torch
import argparse
import logging
from utils.dataset import SEERDataset
from models.seer import SEERClassifier
from pathlib import Path
from torch.utils.data import random_split, DataLoader


def train_model(random_seed: int,  batch_size: int, latent_dim: int, model_name: str, test_fraction: float = 0.1,
                model_dir: Path = Path.cwd()/"results/seer", data_dir: Path = Path.cwd()/"data/seer"):
    assert 0 < test_fraction < 1
    torch.manual_seed(random_seed)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model_dir = model_dir/model_name
    if not model_dir.exists():
        os.makedirs(model_dir)

    seer_data = SEERDataset(str(data_dir/"seer.csv"), random_seed)
    train_size = int((1-test_fraction)*len(seer_data))
    test_size = len(seer_data) - train_size
    train_data, test_data = random_split(seer_data, lengths=[train_size, test_size])
    train_loader = DataLoader(train_data, batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size, shuffle=False)

    model = SEERClassifier(latent_dim, model_name)
    model.fit(device, train_loader, test_loader, model_dir, n_epoch=500, patience=50)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="concept_accuracy")
    parser.add_argument('--seeds', nargs="+", type=int, default=list(range(1, 11)))
    parser.add_argument("--batch_size", type=int, default=120)
    parser.add_argument("--latent_dim", type=int, default=50)
    parser.add_argument("--train", action='store_true')
    parser.add_argument("--plot", action='store_true')
    args = parser.parse_args()

    model_name = f"model_{args.latent_dim}"
    if args.train:
        train_model(args.seeds[0], args.batch_size, args.latent_dim, model_name)

