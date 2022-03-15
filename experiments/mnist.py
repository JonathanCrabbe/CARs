import logging
import argparse
from pathlib import Path
import torch
import os
from models.mnist import ClassifierMnist
from torchvision.datasets import MNIST
from torchvision import transforms
from utils.hooks import register_hooks, get_saved_representations
from utils.dataset import generate_mnist_concept_dataset
from explanations.concept import CAR, CAV
from sklearn.metrics import accuracy_score

concept_to_class = {"loop": [0, 6, 8, 9], "straight_lines": [1, 4, 7], "mirror_symmetry": [0, 3,  8], }


def concept_accuracy(random_seed: int, batch_size: int, latent_dim: int, train: bool,
                     save_dir: Path = Path.cwd()/"results/mnist/concept_accuracy",
                     data_dir: Path = Path.cwd()/"data/mnist"):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    torch.manual_seed(random_seed)
    model_name = "model"
    save_dir = save_dir / model_name
    if not save_dir.exists():
        os.makedirs(save_dir)

    # Load MNIST
    train_set = MNIST(data_dir, train=True, download=True)
    test_set = MNIST(data_dir, train=False, download=True)
    train_transform = transforms.Compose([transforms.ToTensor()])
    test_transform = transforms.Compose([transforms.ToTensor()])
    train_set.transform = train_transform
    test_set.transform = test_transform
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

    # Train MNIST Classifier
    model = ClassifierMnist(latent_dim, model_name)
    if train:
        model.fit(device, train_loader, test_loader, save_dir)
    model.load_state_dict(torch.load(save_dir / f"{model_name}.pt"), strict=False)

    # Register hooks to extract activations
    module_dic = register_hooks(model, save_dir)
    X_train, y_train = generate_mnist_concept_dataset(concept_to_class["loop"], data_dir, True, 200)
    model.eval()
    model(torch.from_numpy(X_train).to(device))
    car = CAR(device)
    cav = CAV(device)
    for module_name in module_dic:
        H_train = get_saved_representations(module_name, save_dir)
        car.fit(H_train, y_train)
        cav.fit(H_train, y_train)
        logging.info(accuracy_score(y_train, car.predict(H_train)))
        logging.info(accuracy_score(y_train, cav.predict(H_train)))

    """
    model.test_epoch(device, test_loader)
    for module_name in module_dic:
        logging.info(get_saved_representations(module_name, save_dir).shape)
    """


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="concept_accuracy")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=120)
    parser.add_argument("--latent_dim", type=int, default=5)
    parser.add_argument("--train", action='store_true')
    args = parser.parse_args()
    if args.name == "concept_accuracy":
        concept_accuracy(args.seed, args.batch_size, args.latent_dim, args.train)

