import torch
import os
import logging
import argparse
import matplotlib.pyplot as plt
from pathlib import Path
from utils.dataset import load_cub_data, CUBDataset, generate_cub_concept_dataset
from models.cub import CUBClassifier
from utils.hooks import register_hooks, remove_all_hooks, get_saved_representations

train_path = str(Path.cwd()/"data/cub/CUB_processed/class_attr_data_10/train.pkl")
val_path = str(Path.cwd()/"data/cub/CUB_processed/class_attr_data_10/val.pkl")
test_path = str(Path.cwd()/"data/cub/CUB_processed/class_attr_data_10/test.pkl")
img_dir = str(Path.cwd()/f"data/cub/CUB_200_2011")


def fit_model(batch_size: int, n_epochs: int, model_name: str):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model_dir = Path.cwd()/"results/cub/model"
    if not model_dir.exists():
        os.makedirs(model_dir)
    train_loader = load_cub_data([train_path, val_path], use_attr=False, batch_size=batch_size, image_dir=img_dir, no_img=False)
    test_loader = load_cub_data([test_path], use_attr=False, batch_size=batch_size, image_dir=img_dir, no_img=False)
    model = CUBClassifier(name=model_name)
    model.fit(device, train_loader, test_loader, model_dir, patience=50, n_epoch=n_epochs)


def concept_accuracy(random_seed: int):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    torch.manual_seed(random_seed)

    cub_dataset = CUBDataset([train_path, val_path], use_attr=True, no_img=False, uncertain_label=False,
                             image_dir=img_dir, n_class_attr=2)
    for i in range(len(cub_dataset.attribute_map)):
        X, y =generate_cub_concept_dataset(i, 200, random_seed, [train_path, val_path], False, False, image_dir=img_dir)
        print(cub_dataset.concept_name(i))
        print(X.shape)
        print(y.shape)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="concept_accuracy")
    parser.add_argument('--seeds', nargs="+", type=int, default=list(range(1, 11)))
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_epochs", type=int, default=1000)
    parser.add_argument("--train", action='store_true')
    parser.add_argument("--plot", action='store_true')
    args = parser.parse_args()

    model_name = f"inception_model"
    if args.train:
        fit_model(args.batch_size, args.n_epochs, model_name=model_name)

    if args.name == "concept_accuracy":
        concept_accuracy(args.seeds[0])
