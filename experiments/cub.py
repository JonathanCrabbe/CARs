import pickle as pkl
from pathlib import Path
from utils.dataset import load_cub_data

pkl_paths = [str(Path.cwd()/f"data/cub/CUB_processed/class_attr_data_10/{name}.pkl") for name in ["train", "test", "val"]]
img_dir = str(Path.cwd()/f"data/cub/CUB_200_2011")
data_loader = load_cub_data(pkl_paths, use_attr=True, batch_size=120, image_dir=img_dir, no_img=False)
for images, concepts, labels in data_loader:
    print(images.shape)
    print(concepts.shape)
    print(len(labels))