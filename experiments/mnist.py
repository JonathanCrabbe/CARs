import itertools
import logging
import argparse
import torch
import numpy as np
import os
import pandas as pd
from pathlib import Path
from models.mnist import ClassifierMnist, init_trainer, get_dataloader
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision import transforms
from utils.hooks import register_hooks, get_saved_representations, remove_all_hooks
from utils.dataset import generate_mnist_concept_dataset
from utils.plot import (
    plot_concept_accuracy,
    plot_global_explanation,
    plot_grayscale_saliency,
    plot_attribution_correlation,
    plot_kernel_sensitivity,
    plot_concept_size_impact,
    plot_tcar_inter_concepts,
)
from explanations.concept import CAR, CAV
from explanations.feature import CARFeatureImportance, VanillaFeatureImportance
from sklearn.metrics import accuracy_score
from sklearn.gaussian_process.kernels import Matern
from tqdm import tqdm
from utils.robustness import Attacker


concept_to_class = {
    "Loop": [0, 2, 6, 8, 9],
    "Vertical Line": [1, 4, 7],
    "Horizontal Line": [4, 5, 7],
    "Curvature": [0, 2, 3, 5, 6, 8, 9],
}


def train_mnist_model(
    latent_dim: int,
    batch_size: int,
    model_name: str = "model",
    model_dir: Path = Path.cwd() / f"results/mnist/",
    data_dir: Path = Path.cwd() / "data/mnist",
) -> None:
    logging.info("Fitting MNIST classifier")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model_dir = model_dir / model_name
    if not model_dir.exists():
        os.makedirs(model_dir)
    model = ClassifierMnist(latent_dim, model_name).to(device)
    train_set = MNIST(data_dir, train=True, download=True)
    test_set = MNIST(data_dir, train=False, download=True)
    train_transform = transforms.Compose([transforms.ToTensor()])
    test_transform = transforms.Compose([transforms.ToTensor()])
    train_set.transform = train_transform
    test_set.transform = test_transform
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False
    )
    model.fit(device, train_loader, test_loader, model_dir)


def concept_accuracy(
    random_seeds: list[int],
    latent_dim: int,
    plot: bool,
    save_dir: Path = Path.cwd() / "results/mnist/concept_accuracy",
    data_dir: Path = Path.cwd() / "data/mnist",
    model_dir: Path = Path.cwd() / f"results/mnist/",
    model_name: str = "model",
) -> None:
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    torch.manual_seed(random_seeds[0])

    representation_dir = save_dir / f"{model_name}_representations"
    if not representation_dir.exists():
        os.makedirs(representation_dir)

    model_dir = model_dir / model_name
    model = ClassifierMnist(latent_dim, model_name)
    model.load_state_dict(torch.load(model_dir / f"{model_name}.pt"), strict=False)
    model.to(device)
    model.eval()

    # Fit a concept classifier and test accuracy for each concept
    results_data = []
    for concept_name, random_seed in itertools.product(concept_to_class, random_seeds):
        logging.info(f"Working with concept {concept_name} and seed {random_seed}")
        # Save representations for training concept examples and then remove the hooks
        module_dic, handler_train_dic = register_hooks(
            model, representation_dir, f"{concept_name}_seed{random_seed}_train"
        )
        X_train, y_train = generate_mnist_concept_dataset(
            concept_to_class[concept_name], data_dir, True, 200, random_seed
        )
        model(torch.from_numpy(X_train).to(device))
        remove_all_hooks(handler_train_dic)
        # Save representations for testing concept examples and then remove the hooks
        module_dic, handler_test_dic = register_hooks(
            model, representation_dir, f"{concept_name}_seed{random_seed}_test"
        )
        X_test, y_test = generate_mnist_concept_dataset(
            concept_to_class[concept_name], data_dir, False, 50, random_seed
        )
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
            results_data.append(
                [
                    concept_name,
                    module_name,
                    random_seed,
                    "CAR",
                    accuracy_score(y_train, car.predict(H_train)),
                    accuracy_score(y_test, car.predict(H_test)),
                ]
            )
            results_data.append(
                [
                    concept_name,
                    module_name,
                    random_seed,
                    "CAV",
                    accuracy_score(y_train, cav.predict(H_train)),
                    accuracy_score(y_test, cav.predict(H_test)),
                ]
            )
    results_df = pd.DataFrame(
        results_data,
        columns=["Concept", "Layer", "Seed", "Method", "Train ACC", "Test ACC"],
    )
    csv_path = save_dir / "metrics.csv"
    results_df.to_csv(csv_path, header=True, mode="w", index=False)
    if plot:
        plot_concept_accuracy(save_dir, None, "mnist")
        for concept in concept_to_class:
            plot_concept_accuracy(save_dir, concept, "mnist")


def statistical_significance(
    random_seed: int,
    latent_dim: int,
    save_dir: Path = Path.cwd() / "results/mnist/statistical_significance",
    data_dir: Path = Path.cwd() / "data/mnist",
    model_dir: Path = Path.cwd() / "results/mnist",
    model_name: str = "model",
) -> None:
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    torch.manual_seed(random_seed)
    model_dir = model_dir / model_name
    representation_dir = save_dir / f"{model_name}_representations"
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
        module_dic, handler_train_dic = register_hooks(
            model, representation_dir, f"{concept_name}_seed{random_seed}_train"
        )
        X_train, y_train = generate_mnist_concept_dataset(
            concept_to_class[concept_name], data_dir, True, 200, random_seed
        )
        model(torch.from_numpy(X_train).to(device))
        remove_all_hooks(handler_train_dic)

        # Create concept classifiers, fit them and test them for each representation space
        for module_name in module_dic:
            logging.info(f"Testing concept classifiers for {module_name}")
            car = CAR(device)
            cav = CAV(device)
            hook_name = f"{concept_name}_seed{random_seed}_train_{module_name}"
            H_train = get_saved_representations(hook_name, representation_dir)
            results_data.append(
                [
                    concept_name,
                    module_name,
                    "CAR",
                    car.permutation_test(H_train, y_train),
                ]
            )
            results_data.append(
                [
                    concept_name,
                    module_name,
                    "CAV",
                    cav.permutation_test(H_train, y_train),
                ]
            )

    results_df = pd.DataFrame(
        results_data, columns=["Concept", "Layer", "Method", "p-value"]
    )
    csv_path = save_dir / "metrics.csv"
    results_df.to_csv(csv_path, header=True, mode="w", index=False)


def global_explanations(
    random_seed: int,
    batch_size: int,
    latent_dim: int,
    plot: bool,
    save_dir: Path = Path.cwd() / "results/mnist/global_explanations",
    data_dir: Path = Path.cwd() / "data/mnist",
    model_dir: Path = Path.cwd() / f"results/mnist",
    model_name: str = "model",
) -> None:
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    torch.manual_seed(random_seed)

    if not save_dir.exists():
        os.makedirs(save_dir)

    model_dir = model_dir / model_name
    model = ClassifierMnist(latent_dim, model_name)
    model.load_state_dict(torch.load(model_dir / f"{model_name}.pt"), strict=False)
    model.to(device)
    model.eval()

    # Fit a concept classifier and test accuracy for each concept
    results_data = []
    car_classifiers = [CAR(device) for _ in concept_to_class]
    cav_classifiers = [CAV(device) for _ in concept_to_class]

    for concept_name, car_classifier, cav_classifier in zip(
        concept_to_class, car_classifiers, cav_classifiers
    ):
        logging.info(f"Now fitting concept classifiers for {concept_name}")
        X_train, y_train = generate_mnist_concept_dataset(
            concept_to_class[concept_name], data_dir, True, 200, random_seed
        )
        H_train = (
            model.input_to_representation(torch.from_numpy(X_train).to(device))
            .detach()
            .cpu()
            .numpy()
        )
        car_classifier.fit(H_train, y_train)
        cav_classifier.fit(H_train, y_train)

    test_set = MNIST(data_dir, train=False, download=True)
    test_set.transform = transforms.Compose([transforms.ToTensor()])
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False
    )

    logging.info("Now predicting concepts on the test set")
    for X_test, y_test in tqdm(test_loader, unit="batch", leave=False):
        H_test = model.input_to_representation(X_test.to(device)).detach().cpu().numpy()
        car_preds = [car.predict(H_test) for car in car_classifiers]
        cav_preds = [
            cav.concept_importance(H_test, y_test, 10, model.representation_to_output)
            for cav in cav_classifiers
        ]
        targets = [
            [int(label in concept_to_class[concept]) for label in y_test]
            for concept in concept_to_class
        ]

        results_data += [
            ["TCAR", label.item()] + [int(car_pred[idx]) for car_pred in car_preds]
            for idx, label in enumerate(y_test)
        ]
        results_data += [
            ["TCAV", label.item()] + [int(cav_pred[idx] > 0) for cav_pred in cav_preds]
            for idx, label in enumerate(y_test)
        ]
        results_data += [
            ["True Prop.", label.item()] + [target[idx] for target in targets]
            for idx, label in enumerate(y_test)
        ]

    csv_path = save_dir / "metrics.csv"
    results_df = pd.DataFrame(
        results_data, columns=["Method", "Class"] + list(concept_to_class.keys())
    )
    results_df.to_csv(csv_path, index=False)
    if plot:
        plot_global_explanation(save_dir, "mnist")


def feature_importance(
    random_seed: int,
    batch_size: int,
    latent_dim: int,
    plot: bool,
    save_dir: Path = Path.cwd() / "results/mnist/feature_importance",
    data_dir: Path = Path.cwd() / "data/mnist",
    model_dir: Path = Path.cwd() / f"results/mnist",
    model_name: str = "model",
) -> None:
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    torch.manual_seed(random_seed)

    if not save_dir.exists():
        os.makedirs(save_dir)

    model_dir = model_dir / model_name
    model = ClassifierMnist(latent_dim, model_name)
    model.load_state_dict(torch.load(model_dir / f"{model_name}.pt"), strict=False)
    model.to(device)
    model.eval()

    # Fit a concept classifier and compute feature importance for each concept
    car_classifiers = [CAR(device) for _ in concept_to_class]
    test_set = MNIST(data_dir, train=False, download=True)
    test_set.transform = transforms.Compose([transforms.ToTensor()])
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False
    )
    attribution_dic = {}
    baselines = torch.zeros((1, 1, 28, 28)).to(device)
    for concept_name, car in zip(concept_to_class, car_classifiers):
        logging.info(f"Now fitting CAR classifier for {concept_name}")
        X_train, y_train = generate_mnist_concept_dataset(
            concept_to_class[concept_name], data_dir, True, 200, random_seed
        )
        H_train = (
            model.input_to_representation(torch.from_numpy(X_train).to(device))
            .detach()
            .cpu()
            .numpy()
        )
        car.tune_kernel_width(H_train, y_train)
        logging.info(
            f"Now computing feature importance on the test set for {concept_name}"
        )
        concept_attribution_method = CARFeatureImportance(
            "Integrated Gradient", car, model, device
        )
        attribution_dic[concept_name] = concept_attribution_method.attribute(
            test_loader, baselines=baselines
        )
        if plot:
            logging.info(f"Saving plots in {save_dir} for {concept_name}")
            X_test = test_set.data
            plot_idx = [
                torch.nonzero(test_set.targets == (n % 10))[n // 10].item()
                for n in range(100)
            ]
            for set_id in range(1, 5):
                plot_grayscale_saliency(
                    X_test,
                    attribution_dic[concept_name],
                    plot_idx[set_id * 10 : (set_id + 1) * 10],
                    save_dir,
                    f"mnist_set{set_id}",
                    concept_name.lower().replace(" ", "-"),
                )
    logging.info(f"Now computing vanilla feature importance")
    vanilla_attribution_method = VanillaFeatureImportance(
        "Integrated Gradient", model, device
    )
    attribution_dic["Vanilla"] = vanilla_attribution_method.attribute(
        test_loader, baselines=baselines
    )
    np.savez(save_dir / "attributions.npz", **attribution_dic)
    if plot:
        logging.info(f"Saving plots in {save_dir}")
        plot_attribution_correlation(save_dir, "mnist")
        X_test = test_set.data
        plot_idx = [
            torch.nonzero(test_set.targets == (n % 10))[n // 10].item()
            for n in range(100)
        ]
        for set_id in range(1, 5):
            plot_grayscale_saliency(
                X_test,
                attribution_dic["Vanilla"],
                plot_idx[set_id * 10 : (set_id + 1) * 10],
                save_dir,
                f"mnist_set{set_id}",
                "vanilla",
            )


def kernel_sensitivity(
    random_seeds: list[int],
    latent_dim: int,
    plot: bool,
    save_dir: Path = Path.cwd() / "results/mnist/kernel_sensitivity",
    data_dir: Path = Path.cwd() / "data/mnist",
    model_dir: Path = Path.cwd() / f"results/mnist/",
    model_name: str = "model",
) -> None:
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    torch.manual_seed(random_seeds[0])

    if not save_dir.exists():
        os.makedirs(save_dir)

    model_dir = model_dir / model_name
    model = ClassifierMnist(latent_dim, model_name)
    model.load_state_dict(torch.load(model_dir / f"{model_name}.pt"), strict=False)
    model.to(device)
    model.eval()

    # Fit a concept classifier and test accuracy for each concept
    kernels = {
        "Gaussian RBF": "rbf",
        "Linear": "linear",
        "Polynomial": "poly",
        "Sigmoid": "sigmoid",
        "Matern": Matern(),
    }
    cars = {
        kernel_name: CAR(device, kernel=kernels[kernel_name]) for kernel_name in kernels
    }
    results_data = []
    for concept_name, random_seed in itertools.product(concept_to_class, random_seeds):
        logging.info(f"Working with concept {concept_name} and seed {random_seed}")
        # Compute representation
        X_train, C_train = generate_mnist_concept_dataset(
            concept_to_class[concept_name], data_dir, True, 200, random_seed
        )
        H_train = (
            model.input_to_representation(torch.from_numpy(X_train).to(device))
            .detach()
            .cpu()
            .numpy()
        )
        X_val, C_val = generate_mnist_concept_dataset(
            concept_to_class[concept_name], data_dir, True, 50, random_seed + 1
        )
        H_val = (
            model.input_to_representation(torch.from_numpy(X_val).to(device))
            .detach()
            .cpu()
            .numpy()
        )
        # Create concept classifiers, fit them and test them
        for kernel_name in cars:
            car = cars[kernel_name]
            car.fit(H_train, C_train)
            results_data.append(
                [
                    "Training",
                    concept_name,
                    random_seed,
                    kernel_name,
                    accuracy_score(C_train, car.predict(H_train)),
                ]
            )
            results_data.append(
                [
                    "Validation",
                    concept_name,
                    random_seed,
                    kernel_name,
                    accuracy_score(C_val, car.predict(H_val)),
                ]
            )

    logging.info(f"Saving results in {str(save_dir)}")
    results_df = pd.DataFrame(
        results_data, columns=["Set", "Concept", "Seed", "Kernel", "Accuracy"]
    )
    csv_path = save_dir / "metrics.csv"
    results_df.to_csv(csv_path, header=True, mode="w", index=False)
    if plot:
        plot_kernel_sensitivity(save_dir, "mnist")


def concept_size_impact(
    random_seeds: list[int],
    latent_dim: int,
    concept_sizes: list[int],
    plot: bool,
    save_dir: Path = Path.cwd() / "results/mnist/concept_set_impact",
    data_dir: Path = Path.cwd() / "data/mnist",
    model_dir: Path = Path.cwd() / f"results/mnist/",
    model_name: str = "model",
) -> None:
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    torch.manual_seed(random_seeds[0])

    if not save_dir.exists():
        os.makedirs(save_dir)

    model_dir = model_dir / model_name
    model = ClassifierMnist(latent_dim, model_name)
    model.load_state_dict(torch.load(model_dir / f"{model_name}.pt"), strict=False)
    model.to(device)
    model.eval()

    # Fit a concept classifier and test accuracy for each concept
    results_data = []
    for concept_name, random_seed in itertools.product(concept_to_class, random_seeds):
        # Compute representation
        X_test, C_test = generate_mnist_concept_dataset(
            concept_to_class[concept_name], data_dir, False, 50, random_seed
        )
        H_test = (
            model.input_to_representation(torch.from_numpy(X_test).to(device))
            .detach()
            .cpu()
            .numpy()
        )
        # Create concept classifiers, fit them and test them for each representation space
        prev_size = 0
        concept_sizes.sort()
        C_train = np.empty([1], dtype=int)
        H_train = np.empty([1] + list(H_test.shape[1:]))
        for concept_size in concept_sizes:
            logging.info(
                f"Working with concept {concept_name}, seed {random_seed} and a set of size {concept_size}"
            )
            n_add = concept_size - prev_size
            prev_size = concept_size
            X_add, C_add = generate_mnist_concept_dataset(
                concept_to_class[concept_name],
                data_dir,
                True,
                n_add,
                random_seed + concept_size,
            )
            H_add = (
                model.input_to_representation(torch.from_numpy(X_add).to(device))
                .detach()
                .cpu()
                .numpy()
            )
            C_train = np.concatenate((C_train, C_add), axis=0)
            H_train = np.concatenate((H_train, H_add), axis=0)
            car = CAR(device)
            car.fit(H_train, C_train)
            results_data.append(
                [
                    concept_size,
                    random_seed,
                    concept_name,
                    accuracy_score(C_train, car.predict(H_train)),
                ]
            )

    logging.info(f"Saving results in {str(save_dir)}")
    results_df = pd.DataFrame(
        results_data,
        columns=["Concept Sets Size", "Random Seed", "Concept", "Test Accuracy"],
    )
    csv_path = save_dir / "metrics.csv"
    results_df.to_csv(csv_path, header=True, mode="w", index=False)
    if plot:
        plot_concept_size_impact(save_dir, "mnist")


def tcar_inter_concept(
    random_seed: int,
    batch_size: int,
    latent_dim: int,
    plot: bool,
    save_dir: Path = Path.cwd() / "results/mnist/tcar_inter_concept",
    data_dir: Path = Path.cwd() / "data/mnist",
    model_dir: Path = Path.cwd() / f"results/mnist",
    model_name: str = "model",
) -> None:
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    torch.manual_seed(random_seed)

    if not save_dir.exists():
        os.makedirs(save_dir)

    model_dir = model_dir / model_name
    model = ClassifierMnist(latent_dim, model_name)
    model.load_state_dict(torch.load(model_dir / f"{model_name}.pt"), strict=False)
    model.to(device)
    model.eval()

    # Fit a concept classifier and test accuracy for each concept
    results_data = []
    car_classifiers = [CAR(device) for _ in concept_to_class]

    for concept_name, car_classifier in zip(concept_to_class, car_classifiers):
        logging.info(f"Now fitting concept classifiers for {concept_name}")
        X_train, y_train = generate_mnist_concept_dataset(
            concept_to_class[concept_name], data_dir, True, 200, random_seed
        )
        H_train = (
            model.input_to_representation(torch.from_numpy(X_train).to(device))
            .detach()
            .cpu()
            .numpy()
        )
        car_classifier.fit(H_train, y_train)

    test_set = MNIST(data_dir, train=False, download=True)
    test_set.transform = transforms.Compose([transforms.ToTensor()])
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False
    )

    logging.info("Now predicting concepts on the test set")
    for X_test, y_test in tqdm(test_loader, unit="batch", leave=False):
        H_test = model.input_to_representation(X_test.to(device)).detach().cpu().numpy()
        car_preds = [car.predict(H_test) for car in car_classifiers]
        results_data += [
            [int(car_pred[idx]) for car_pred in car_preds] for idx in range(len(y_test))
        ]

    logging.info(f"Saving results in {str(save_dir)}")
    csv_path = save_dir / "metrics.csv"
    results_df = pd.DataFrame(results_data, columns=list(concept_to_class.keys()))
    results_df.to_csv(csv_path, index=False)
    if plot:
        plot_tcar_inter_concepts(save_dir, "mnist")


def adversarial_robustness(
    random_seed: int,
    batch_size: int,
    latent_dim: int,
    save_dir: Path = Path.cwd() / "results/mnist/adversarial_robustness",
    data_dir: Path = Path.cwd() / "data/mnist",
    model_dir: Path = Path.cwd() / f"results/mnist",
    model_name: str = "model",
) -> None:
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    torch.manual_seed(random_seed)

    if not save_dir.exists():
        os.makedirs(save_dir)

    model_dir = model_dir / model_name
    model = ClassifierMnist(latent_dim, model_name)
    model.load_state_dict(torch.load(model_dir / f"{model_name}.pt"), strict=False)
    model.to(device)
    model.eval()

    # Fit a concept classifier and test accuracy for each concept
    results_data = []
    car_classifiers = [CAR(device) for _ in concept_to_class]
    cav_classifiers = [CAV(device) for _ in concept_to_class]

    for concept_name, car_classifier, cav_classifier in zip(
        concept_to_class, car_classifiers, cav_classifiers
    ):
        logging.info(f"Now fitting concept classifiers for {concept_name}")
        X_train, y_train = generate_mnist_concept_dataset(
            concept_to_class[concept_name], data_dir, True, 200, random_seed
        )
        H_train = (
            model.input_to_representation(torch.from_numpy(X_train).to(device))
            .detach()
            .cpu()
            .numpy()
        )
        car_classifier.fit(H_train, y_train)
        cav_classifier.fit(H_train, y_train)

    test_set = MNIST(data_dir, train=False, download=True)
    test_set.transform = transforms.Compose([transforms.ToTensor()])
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False
    )
    attacker = Attacker(model, 100, 0.1, device)

    logging.info("Now predicting concepts on the test set")
    for attack_prop in [0, 0.05, 0.1, 0.2, 0.5, 0.7, 1]:
        logging.info(f"Working with {100*attack_prop}% of adversarial samples")
        for X_test, y_test in tqdm(test_loader, unit="batch", leave=False):
            n_attacks = int(len(X_test) * attack_prop)
            X_test = X_test.to(device)
            X_adv, X_test = torch.split(X_test, [n_attacks, len(X_test) - n_attacks])
            if n_attacks > 0:
                X_adv = attacker.make_adversarial_example(X_adv, model(X_adv))
            X_test = torch.cat([X_adv, X_test])
            H_test = (
                model.input_to_representation(X_test.to(device)).detach().cpu().numpy()
            )
            car_preds = [car.predict(H_test) for car in car_classifiers]
            cav_preds = [
                cav.concept_importance(
                    H_test, y_test, 10, model.representation_to_output
                )
                for cav in cav_classifiers
            ]
            targets = [
                [int(label in concept_to_class[concept]) for label in y_test]
                for concept in concept_to_class
            ]

            results_data += [
                [attack_prop * 100, "TCAR", label.item()]
                + [int(car_pred[idx]) for car_pred in car_preds]
                for idx, label in enumerate(y_test)
            ]
            results_data += [
                [attack_prop * 100, "TCAV", label.item()]
                + [int(cav_pred[idx] > 0) for cav_pred in cav_preds]
                for idx, label in enumerate(y_test)
            ]
            results_data += [
                [attack_prop * 100, "True Prop.", label.item()]
                + [target[idx] for target in targets]
                for idx, label in enumerate(y_test)
            ]

    csv_path = save_dir / "metrics.csv"
    results_df = pd.DataFrame(
        results_data,
        columns=["Adversarial %", "Method", "Class"] + list(concept_to_class.keys()),
    )
    results_df.to_csv(csv_path, index=False)
    scores_data = []
    classes = results_df["Class"].unique()
    methods = results_df["Method"].unique()
    adv_pcts = results_df["Adversarial %"].unique()
    concepts = concept_to_class.keys()
    for adv_pct, class_idx, concept, method in itertools.product(
        adv_pcts, classes, concepts, methods
    ):
        attr = np.array(
            results_df.loc[
                (results_df.Class == class_idx)
                & (results_df.Method == method)
                & (results_df["Adversarial %"] == adv_pct)
            ][concept]
        )
        score = np.sum(attr) / len(attr)
        scores_data.append([adv_pct, method, class_idx, concept, score])
    scores_df = pd.DataFrame(
        scores_data, columns=["Adversarial %", "Method", "Class", "Concept", "Score"]
    )
    corr_data = []
    for adv_pct in adv_pcts:
        tcar_scores = scores_df.loc[
            (scores_df.Method == "TCAR") & (scores_df["Adversarial %"] == adv_pct)
        ]["Score"]
        true_scores = scores_df.loc[
            (scores_df.Method == "True Prop.") & (scores_df["Adversarial %"] == adv_pct)
        ]["Score"]
        corr_data.append([adv_pct, "TCAR", np.corrcoef(tcar_scores, true_scores)[0, 1]])
    corr_df = pd.DataFrame(
        corr_data, columns=["Adversarial %", "Method", "Correlation"]
    )
    results_md = pd.pivot_table(
        data=corr_df,
        index="Adversarial %",
        columns="Method",
        aggfunc="mean",
        values="Correlation",
    ).to_markdown()
    logging.info(results_md)


def senn() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logging.info("Now fitting SENN model")
    senn_trainer = init_trainer(str(Path.cwd() / "configs/senn_config.json"))
    senn_trainer.run()
    senn_trainer.load_checkpoint(
        str(Path.cwd() / "results/mnist/senn/checkpoints/best_model.pt")
    )
    senn = senn_trainer.model
    senn.eval()
    senn_concept_relevance = senn.parameterizer
    senn_representation = senn.conceptizer.encode

    logging.info("Now tuning CAR concept densities")
    data_dir = Path.cwd() / "data/mnist"
    car_classifiers = {concept_name: CAR(device) for concept_name in concept_to_class}
    for concept_name in concept_to_class:
        logging.info(f"Tunning {concept_name}")
        X_train, c_train = generate_mnist_concept_dataset(
            concept_to_class[concept_name], data_dir, True, 300, 1
        )
        H_train = (
            senn_representation(
                torch.from_numpy(X_train).to(senn_trainer.config.device)
            )
            .flatten(start_dim=1)
            .detach()
            .cpu()
            .numpy()
        )
        car_classifiers[concept_name].tune_kernel_width(H_train, c_train)

    logging.info("Now computing concept relevance and densities")
    _, _, test_loader = get_dataloader(senn_trainer.config)
    human_concept_importances = []
    synthetic_concept_relevances = []
    for X_test, y_test in test_loader:
        H_test = (
            senn_representation(X_test.to(senn_trainer.config.device))
            .flatten(start_dim=1)
            .detach()
        )
        C_importance = [
            car_classifiers[concept_name]
            .concept_importance(H_test)
            .view(-1, 1)
            .cpu()
            .numpy()
            for concept_name in concept_to_class
        ]
        C_importance = np.concatenate(C_importance, axis=1)
        human_concept_importances.append(C_importance)
        C_relevance = senn_concept_relevance(
            X_test.to(senn_trainer.config.device)
        ).detach()
        class_select = (
            y_test.to(senn_trainer.config.device).view(-1, 1, 1).repeat(1, 5, 1)
        )
        C_relevance = (
            torch.gather(C_relevance, -1, index=class_select)
            .flatten(start_dim=1)
            .cpu()
            .numpy()
        )
        synthetic_concept_relevances.append(C_relevance)
    human_concept_importances = np.concatenate(human_concept_importances, axis=0)
    synthetic_concept_relevances = np.concatenate(synthetic_concept_relevances, axis=0)
    corr_data = np.corrcoef(
        human_concept_importances, synthetic_concept_relevances, rowvar=False
    )[4:, :4]
    corr_df = pd.DataFrame(
        data=corr_data,
        columns=list(concept_to_class.keys()),
        index=[f"SENN Concept {i+1}" for i in range(5)],
    )
    logging.info(corr_df.to_markdown())


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="concept_accuracy")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(1, 11)))
    parser.add_argument("--batch_size", type=int, default=120)
    parser.add_argument("--latent_dim", type=int, default=5)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument(
        "--concept_sizes", nargs="+", type=int, default=list(range(10, 310, 30))
    )
    args = parser.parse_args()

    model_name = f"model_{args.latent_dim}"
    if args.train:
        train_mnist_model(args.latent_dim, args.batch_size, model_name=model_name)
    if args.name == "concept_accuracy":
        concept_accuracy(args.seeds, args.latent_dim, args.plot, model_name=model_name)
    elif args.name == "global_explanations":
        global_explanations(
            args.seeds[0],
            args.batch_size,
            args.latent_dim,
            args.plot,
            model_name=model_name,
        )
    elif args.name == "statistical_significance":
        statistical_significance(args.seeds[0], args.latent_dim, model_name=model_name)
    elif args.name == "feature_importance":
        feature_importance(
            args.seeds[0],
            args.batch_size,
            args.latent_dim,
            args.plot,
            model_name=model_name,
        )
    elif args.name == "kernel_sensitivity":
        kernel_sensitivity(
            args.seeds, args.latent_dim, args.plot, model_name=model_name
        )
    elif args.name == "concept_size_impact":
        concept_size_impact(
            args.seeds,
            args.latent_dim,
            args.concept_sizes,
            args.plot,
            model_name=model_name,
        )
    elif args.name == "tcar_inter_concepts":
        tcar_inter_concept(
            args.seeds[0],
            args.batch_size,
            args.latent_dim,
            args.plot,
            model_name=model_name,
        )
    elif args.name == "adversarial_robustness":
        adversarial_robustness(
            args.seeds[0], args.batch_size, args.latent_dim, model_name=model_name
        )
    elif args.name == "senn":
        senn()
    else:
        raise ValueError(f"{args.name} is not a valid experiment name")
