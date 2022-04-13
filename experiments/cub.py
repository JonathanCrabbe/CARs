import torch
import os
import logging
from pathlib import Path
from utils.dataset import load_cub_data
from models.cub import CUBClassifier

train_path = str(Path.cwd()/"data/cub/CUB_processed/class_attr_data_10/train.pkl")
val_path = str(Path.cwd()/"data/cub/CUB_processed/class_attr_data_10/val.pkl")
test_path = str(Path.cwd()/"data/cub/CUB_processed/class_attr_data_10/test.pkl")
img_dir = str(Path.cwd()/f"data/cub/CUB_200_2011")


def fit_model():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model_dir = Path.cwd()/"results/cub/model"
    if not model_dir.exists():
        os.makedirs(model_dir)
    train_loader = load_cub_data([train_path], use_attr=False, batch_size=64, image_dir=img_dir, no_img=False)
    test_loader = load_cub_data([test_path], use_attr=False, batch_size=64, image_dir=img_dir, no_img=False)
    model = CUBClassifier()
    model.fit(device, train_loader, test_loader, model_dir)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    fit_model()