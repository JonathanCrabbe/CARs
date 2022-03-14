import logging
import argparse
from pathlib import Path
import torch
import os
from models.mnist import ClassifierMnist
from torchvision.datasets import MNIST
from torchvision import transforms
from utils.hooks import create_forward_hook, get_saved_representations


def concept_accuracy(random_seed: int, batch_size: int, latent_dim: int,
                     save_dir: Path = Path.cwd()/"results/mnist/concept_accuracy",
                     data_dir: Path = Path.cwd()/"data/mnist"):
    torch.manual_seed(random_seed)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model_name = "model"

    # Load MNIST
    train_set = MNIST(data_dir, train=True, download=True)
    test_set = MNIST(data_dir, train=False, download=True)
    train_transform = transforms.Compose([transforms.ToTensor()])
    test_transform = transforms.Compose([transforms.ToTensor()])
    train_set.transform = train_transform
    test_set.transform = test_transform
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

    if not save_dir.exists():
        os.makedirs(save_dir)

    # Train MNIST Classifier
    model = ClassifierMnist(latent_dim, model_name)
    model.fit(device, train_loader, test_loader, save_dir, n_epoch=1)
    model.load_state_dict(torch.load(save_dir / f"{model_name}.pt"), strict=False)

    # Register hooks to extract activations
    model.maxpool1.register_forward_hook(create_forward_hook("CNN1", save_dir))
    model.maxpool2.register_forward_hook(create_forward_hook("CNN2", save_dir))
    model.fc1.register_forward_hook(create_forward_hook("Lin1", save_dir))
    model.fc2.register_forward_hook(create_forward_hook("Lin2", save_dir))
    model.test_epoch(device, test_loader)

    print(get_saved_representations("CNN1", save_dir).shape)
    print(get_saved_representations("CNN2", save_dir).shape)
    print(get_saved_representations("Lin1", save_dir).shape)
    print(get_saved_representations("Lin2", save_dir).shape)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("-name", type=str, default="concept_accuracy")
    parser.add_argument("-seed", type=int, default=42)
    parser.add_argument("-batch_size", type=int, default=120)
    parser.add_argument("-latent_dim", type=int, default=5)
    args = parser.parse_args()
    if args.name == "concept_accuracy":
        concept_accuracy(args.seed, args.batch_size, args.latent_dim)

