import torch
import os
import logging
import argparse
import itertools
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader
from utils.dataset import ECGDataset, generate_ecg_concept_dataset
from models.ecg import ClassifierECG
from utils.hooks import register_hooks, get_saved_representations, remove_all_hooks
from explanations.concept import CAR, CAV
from sklearn.metrics import accuracy_score
from utils.plot import plot_concept_accuracy

concept_to_class = {"Supraventricular": 1, "Premature Ventricular": 2, "Fusion Beats": 3, "Unknown": 4}


def train_ecg_model(latent_dim: int, model_name: str, model_dir: Path,
                    data_dir: Path, device: torch.device, batch_size: int) -> None:
    model = ClassifierECG(latent_dim, model_name).to(device)
    train_set = ECGDataset(data_dir, train=True, balance_dataset=True)
    test_set = ECGDataset(data_dir, train=False, balance_dataset=False)
    train_loader = DataLoader(train_set, batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size, shuffle=True)
    model.fit(device, train_loader, test_loader, model_dir)


def concept_accuracy(random_seeds: list[int], batch_size: int, latent_dim: int, train: bool,
                     save_dir: Path = Path.cwd()/"results/ecg/concept_accuracy",
                     data_dir: Path = Path.cwd()/"data/ecg", model_name: str = "model") -> None:
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    torch.manual_seed(random_seeds[0])
    model_dir = save_dir / model_name
    representation_dir = save_dir / f"{model_name}_representations"
    if not representation_dir.exists():
        os.makedirs(representation_dir)
    if not model_dir.exists():
        os.makedirs(model_dir)

    if train:
        train_ecg_model(latent_dim, model_name, model_dir, data_dir, device, batch_size)
    model = ClassifierECG(latent_dim, model_name).to(device)
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
        X_train, y_train = generate_ecg_concept_dataset(concept_to_class[concept_name], data_dir, True, 200,
                                                        random_seed)
        model(torch.from_numpy(X_train).to(device))
        remove_all_hooks(handler_train_dic)
        # Save representations for testing concept examples and then remove the hooks
        module_dic, handler_test_dic = register_hooks(model, representation_dir,
                                                      f"{concept_name}_seed{random_seed}_test")
        X_test, y_test = generate_ecg_concept_dataset(concept_to_class[concept_name], data_dir, False, 50,
                                                      random_seed)
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
    csv_path = save_dir / "metrics.csv"
    results_df.to_csv(csv_path, header=True, mode="w", index=False)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="concept_accuracy")
    parser.add_argument('--seeds', nargs="+", type=int, default=list(range(1, 10)))
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--train", action='store_true')
    parser.add_argument("--plot", action='store_true')
    args = parser.parse_args()
    if args.name == "concept_accuracy":
        save_dir = Path.cwd() / "results/ecg/concept_accuracy"
        concept_accuracy(args.seeds, args.batch_size, args.latent_dim, args.train, save_dir=save_dir)
        if args.plot:
            plot_concept_accuracy(save_dir, None)
            for concept in concept_to_class:
                plot_concept_accuracy(save_dir, concept)
    else:
        raise ValueError(f"{args.name} is not a valid experiment name")
