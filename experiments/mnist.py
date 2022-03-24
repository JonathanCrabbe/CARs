import itertools
import logging
import argparse
import torch
import os
import pandas as pd
from pathlib import Path
from models.mnist import ClassifierMnist
from torchvision.datasets import MNIST
from torchvision import transforms
from utils.hooks import register_hooks, get_saved_representations, remove_all_hooks
from utils.dataset import generate_mnist_concept_dataset
from utils.plot import plot_concept_accuracy, plot_global_explanation
from explanations.concept import CAR, CAV
from sklearn.metrics import accuracy_score
from tqdm import tqdm

concept_to_class = {"Loop": [0, 2, 6, 8, 9], "Mirror Symmetry": [0, 3,  8], "Vertical Line": [1, 4, 7],
                    "Horizontal Line": [4, 5, 7], "Curvature": [0, 2, 3, 5, 6, 8, 9]}


def train_mnist_model(latent_dim: int, batch_size: int, model_name: str = "model",
                      model_dir: Path = Path.cwd()/f"results/mnist/",
                      data_dir: Path = Path.cwd()/"data/mnist") -> None:
    logging.info("Fitting MNIST classifier")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model_dir = model_dir/model_name
    if not model_dir.exists():
        os.makedirs(model_dir)
    model = ClassifierMnist(latent_dim, model_name).to(device)
    train_set = MNIST(data_dir, train=True, download=True)
    test_set = MNIST(data_dir, train=False, download=True)
    train_transform = transforms.Compose([transforms.ToTensor()])
    test_transform = transforms.Compose([transforms.ToTensor()])
    train_set.transform = train_transform
    test_set.transform = test_transform
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
    model.fit(device, train_loader, test_loader, model_dir)


def concept_accuracy(random_seeds: list[int], latent_dim: int, plot: bool,
                     save_dir: Path = Path.cwd()/"results/mnist/concept_accuracy",
                     data_dir: Path = Path.cwd()/"data/mnist",
                     model_dir: Path = Path.cwd() / f"results/mnist/",
                     model_name: str = "model",) -> None:
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    torch.manual_seed(random_seeds[0])

    representation_dir = save_dir/f"{model_name}_representations"
    if not representation_dir.exists():
        os.makedirs(representation_dir)

    model_dir = model_dir/model_name
    model = ClassifierMnist(latent_dim, model_name)
    model.load_state_dict(torch.load(model_dir / f"{model_name}.pt"), strict=False)
    model.to(device)
    model.eval()

    # Fit a concept classifier and test accuracy for each concept
    results_data = []
    for concept_name, random_seed in itertools.product(concept_to_class, random_seeds):
        logging.info(f"Working with concept {concept_name} and seed {random_seed}")
        # Save representations for training concept examples and then remove the hooks
        module_dic, handler_train_dic = register_hooks(model, representation_dir,
                                                       f"{concept_name}_seed{random_seed}_train")
        X_train, y_train = generate_mnist_concept_dataset(concept_to_class[concept_name], data_dir,
                                                          True, 200, random_seed)
        model(torch.from_numpy(X_train).to(device))
        remove_all_hooks(handler_train_dic)
        # Save representations for testing concept examples and then remove the hooks
        module_dic, handler_test_dic = register_hooks(model, representation_dir, f"{concept_name}_seed{random_seed}_test")
        X_test, y_test = generate_mnist_concept_dataset(concept_to_class[concept_name], data_dir, False, 50, random_seed)
        model(torch.from_numpy(X_test).to(device))
        remove_all_hooks(handler_test_dic)
        # Create concept classifiers, fit them and test them for each representation space
        for module_name in module_dic:
            logging.info(f"Fitting concept classifiers for {module_name}")
            car = CAR(device)
            cav = CAV(device)
            hook_name = f"{concept_name}_seed{random_seed}_train_{module_name}"
            H_train = get_saved_representations(hook_name, representation_dir)
            car.fit(H_train, y_train)
            cav.fit(H_train, y_train)
            hook_name = f"{concept_name}_seed{random_seed}_test_{module_name}"
            H_test = get_saved_representations(hook_name, representation_dir)
            results_data.append([concept_name, module_name, random_seed, "CAR",
                                 accuracy_score(y_train, car.predict(H_train)),
                                 accuracy_score(y_test, car.predict(H_test))])
            results_data.append([concept_name, module_name, random_seed, "CAV",
                                 accuracy_score(y_train, cav.predict(H_train)),
                                 accuracy_score(y_test, cav.predict(H_test))])
    results_df = pd.DataFrame(results_data, columns=["Concept", "Layer", "Seed", "Method", "Train ACC", "Test ACC"])
    csv_path = save_dir/"metrics.csv"
    results_df.to_csv(csv_path, header=True, mode="w", index=False)
    if plot:
        plot_concept_accuracy(save_dir, None)
        for concept in concept_to_class:
            plot_concept_accuracy(save_dir, concept)


def statistical_significance(random_seed: int,  latent_dim: int,
                             save_dir: Path = Path.cwd()/"results/mnist/statistical_significance",
                             data_dir: Path = Path.cwd()/"data/mnist",
                             model_dir: Path = Path.cwd()/"results/mnist",
                             model_name: str = "model",) -> None:
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    torch.manual_seed(random_seed)
    model_dir = model_dir/model_name
    representation_dir = save_dir/f"{model_name}_representations"
    if not representation_dir.exists():
        os.makedirs(representation_dir)

    model = ClassifierMnist(latent_dim, model_name)
    model.load_state_dict(torch.load(model_dir / f"{model_name}.pt"), strict=False)
    model.to(device)
    model.eval()

    # Fit a concept classifier and test accuracy for each concept
    results_data = []
    for concept_name in concept_to_class:
        logging.info(f"Working with concept {concept_name} ")
        # Save representations for training concept examples and then remove the hooks
        module_dic, handler_train_dic = register_hooks(model, representation_dir,
                                                       f"{concept_name}_seed{random_seed}_train")
        X_train, y_train = generate_mnist_concept_dataset(concept_to_class[concept_name], data_dir,
                                                          True, 200, random_seed)
        model(torch.from_numpy(X_train).to(device))
        remove_all_hooks(handler_train_dic)

        # Create concept classifiers, fit them and test them for each representation space
        for module_name in module_dic:
            logging.info(f"Testing concept classifiers for {module_name}")
            car = CAR(device)
            cav = CAV(device)
            hook_name = f"{concept_name}_seed{random_seed}_train_{module_name}"
            H_train = get_saved_representations(hook_name, representation_dir)
            results_data.append([concept_name, module_name, "CAR", car.permutation_test(H_train, y_train)])
            results_data.append([concept_name, module_name, "CAV", cav.permutation_test(H_train, y_train)])

    results_df = pd.DataFrame(results_data, columns=["Concept", "Layer",  "Method", "p-value"])
    csv_path = save_dir / "metrics.csv"
    results_df.to_csv(csv_path, header=True, mode="w", index=False)


def global_explanations(random_seed: int, batch_size: int, latent_dim: int,  plot: bool,
                        save_dir: Path = Path.cwd()/"results/mnist/global_explanations",
                        data_dir: Path = Path.cwd()/"data/mnist",
                        model_dir: Path = Path.cwd() / f"results/mnist",
                        model_name: str = "model") -> None:
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    torch.manual_seed(random_seed)

    if not save_dir.exists():
        os.makedirs(save_dir)

    model_dir = model_dir/model_name
    model = ClassifierMnist(latent_dim, model_name)
    model.load_state_dict(torch.load(model_dir / f"{model_name}.pt"), strict=False)
    model.to(device)
    model.eval()

    # Fit a concept classifier and test accuracy for each concept
    results_data = []
    car_classifiers = [CAR(device) for _ in concept_to_class]
    cav_classifiers = [CAV(device) for _ in concept_to_class]

    for concept_name, car_classifier, cav_classifier in zip(concept_to_class, car_classifiers, cav_classifiers):
        logging.info(f"Now fitting concept classifiers for {concept_name}")
        X_train, y_train = generate_mnist_concept_dataset(concept_to_class[concept_name], data_dir,
                                                          True, 200, random_seed)
        H_train = model.input_to_representation(torch.from_numpy(X_train).to(device)).detach().cpu().numpy()
        car_classifier.fit(H_train, y_train)
        cav_classifier.fit(H_train, y_train)

    test_set = MNIST(data_dir, train=False, download=True)
    test_set.transform = transforms.Compose([transforms.ToTensor()])
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

    logging.info("Now predicting concepts on the test set")
    for X_test, y_test in tqdm(test_loader, unit="batch", leave=False):
        H_test = model.input_to_representation(X_test.to(device)).detach().cpu().numpy()
        car_preds = [car.predict(H_test) for car in car_classifiers]
        cav_preds = [cav.concept_importance(H_test, y_test, 10, model.representation_to_output)
                     for cav in cav_classifiers]
        targets = [[int(label in concept_to_class[concept]) for label in y_test] for concept in concept_to_class]

        results_data += [["TCAR", label.item()] + [int(car_pred[idx]) for car_pred in car_preds]
                         for idx, label in enumerate(y_test)]
        results_data += [["TCAV", label.item()] + [int(cav_pred[idx] > 0) for cav_pred in cav_preds]
                         for idx, label in enumerate(y_test)]
        results_data += [["Truth", label.item()] + [target[idx] for target in targets]
                         for idx, label in enumerate(y_test)]

    csv_path = save_dir / "metrics.csv"
    results_df = pd.DataFrame(results_data, columns=["Method", "Class"]+list(concept_to_class.keys()))
    results_df.to_csv(csv_path, index=False)
    if plot:
        plot_global_explanation(save_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="concept_accuracy")
    parser.add_argument('--seeds', nargs="+", type=int, default=list(range(1, 10)))
    parser.add_argument("--batch_size", type=int, default=120)
    parser.add_argument("--latent_dim", type=int, default=5)
    parser.add_argument("--train", action='store_true')
    parser.add_argument("--plot", action='store_true')
    args = parser.parse_args()

    if args.train:
        train_mnist_model(args.latent_dim,  args.batch_size, model_name="model_test")
    if args.name == "concept_accuracy":
        concept_accuracy(args.seeds, args.latent_dim, args.plot)
    elif args.name == "global_explanations":
        global_explanations(args.seeds[0], args.batch_size, args.latent_dim,  args.plot)
    elif args.name == "statistical_significance":
        statistical_significance(args.seeds[0], args.latent_dim)
    else:
        raise ValueError(f"{args.name} is not a valid experiment name")

