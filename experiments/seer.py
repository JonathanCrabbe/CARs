import os
import torch
import argparse
import logging
import pandas as pd
import numpy as np
from utils.dataset import SEERDataset, generate_seer_concept_dataset
from models.seer import SEERClassifier
from pathlib import Path
from torch.utils.data import DataLoader
from utils.plot import plot_seer_global_explanation, plot_seer_feature_importance
from explanations.concept import CAR, CAV
from explanations.feature import CARFeatureImportance
from tqdm import tqdm
from sklearn.metrics import accuracy_score


def train_model(
    random_seed: int,
    batch_size: int,
    latent_dim: int,
    model_name: str,
    test_fraction: float = 0.1,
    model_dir: Path = Path.cwd() / "results/seer",
    data_dir: Path = Path.cwd() / "data/seer",
):
    assert 0 < test_fraction < 1
    logging.info("Now fitting a SEER classifier")
    torch.manual_seed(random_seed)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model_dir = model_dir / model_name
    if not model_dir.exists():
        os.makedirs(model_dir)

    train_data = SEERDataset(
        str(data_dir / "seer.csv"), random_seed, train=True, test_fraction=test_fraction
    )
    test_data = SEERDataset(
        str(data_dir / "seer.csv"),
        random_seed,
        train=False,
        test_fraction=test_fraction,
    )
    train_loader = DataLoader(train_data, batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size, shuffle=False)

    model = SEERClassifier(latent_dim, model_name)
    model.fit(device, train_loader, test_loader, model_dir, n_epoch=500, patience=50)


def concept_accuracy(
    random_seed: int,
    batch_size: int,
    latent_dim: int,
    model_name: str,
    test_fraction: float = 0.1,
    model_dir: Path = Path.cwd() / "results/seer",
    data_dir: Path = Path.cwd() / "data/seer",
    save_dir: Path = Path.cwd() / "results/seer/concept_accuracy",
):
    torch.manual_seed(random_seed)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    if not save_dir.exists():
        os.makedirs(save_dir)

    # Load data
    train_data = SEERDataset(
        str(data_dir / "seer.csv"),
        random_seed,
        train=True,
        test_fraction=test_fraction,
        load_concept_labels=True,
    )
    test_data = SEERDataset(
        str(data_dir / "seer.csv"),
        random_seed,
        train=False,
        test_fraction=test_fraction,
        load_concept_labels=True,
    )

    # Load model
    model_dir = model_dir / model_name
    model = SEERClassifier(latent_dim, model_name)
    model.load_state_dict(torch.load(model_dir / f"{model_name}.pt"), strict=False)
    model.to(device)

    results_data = []
    car_classifiers = [CAR(device, batch_size, kernel="linear") for _ in range(5)]
    cav_classifiers = [CAV(device, batch_size) for _ in range(5)]
    for concept_id in range(5):
        logging.info(f"Now fitting a CAR classifier for Grade {concept_id+1} patients")
        X_train, C_train = generate_seer_concept_dataset(
            train_data, concept_id, 250, random_seed
        )
        X_train = X_train.to(device)
        H_train = model.input_to_representation(X_train).detach().cpu().numpy()
        car = car_classifiers[concept_id]
        car.fit(H_train, C_train.numpy())
        cav = cav_classifiers[concept_id]
        cav.fit(H_train, C_train.numpy())
        X_test, C_test = generate_seer_concept_dataset(
            test_data, concept_id, 50, random_seed
        )
        X_test = X_test.to(device)
        H_test = model.input_to_representation(X_test).detach().cpu().numpy()
        results_data.append(
            [
                f"Grade {concept_id + 1}",
                "CAR",
                accuracy_score(C_test, car.predict(H_test)),
            ]
        )
        results_data.append(
            [
                f"Grade {concept_id + 1}",
                "CAV",
                accuracy_score(C_test, cav.predict(H_test)),
            ]
        )
    results_df = pd.DataFrame(results_data, columns=["Concept", "Method", "Test ACC"])
    logging.info(f"Saving results in {save_dir}")
    results_df.to_csv(save_dir / "metrics.csv")


def global_explanations(
    random_seed: int,
    batch_size: int,
    latent_dim: int,
    plot: bool,
    model_name: str,
    test_fraction: float = 0.1,
    model_dir: Path = Path.cwd() / "results/seer",
    data_dir: Path = Path.cwd() / "data/seer",
    save_dir: Path = Path.cwd() / "results/seer/global_explanations",
):
    torch.manual_seed(random_seed)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    if not save_dir.exists():
        os.makedirs(save_dir)

    # Load data
    train_data = SEERDataset(
        str(data_dir / "seer.csv"),
        random_seed,
        train=True,
        test_fraction=test_fraction,
        load_concept_labels=True,
    )
    test_data = SEERDataset(
        str(data_dir / "seer.csv"),
        random_seed,
        train=False,
        test_fraction=test_fraction,
        load_concept_labels=True,
    )
    test_loader = DataLoader(test_data, batch_size)

    # Load model
    model_dir = model_dir / model_name
    model = SEERClassifier(latent_dim, model_name)
    model.load_state_dict(torch.load(model_dir / f"{model_name}.pt"), strict=False)
    model.to(device)

    car_classifiers = [CAR(device, batch_size, kernel="linear") for _ in range(5)]
    for concept_id in range(5):
        logging.info(f"Now fitting a CAR classifier for Grade {concept_id+1} patients")
        X_train, C_train = generate_seer_concept_dataset(
            train_data, concept_id, 250, random_seed
        )
        X_train = X_train.to(device)
        H_train = model.input_to_representation(X_train).detach().cpu().numpy()
        car = car_classifiers[concept_id]
        car.fit(H_train, C_train.numpy())

    logging.info("Producing global explanations for the test set")
    results_data = []
    for X_test, Y_test, C_test in tqdm(test_loader, unit="batch", leave=False):
        X_test = X_test.to(device)
        H_test = model.input_to_representation(X_test).detach().cpu().numpy()
        pred_concepts = [car.predict(H_test) for car in car_classifiers]
        results_data += [
            ["TCAR", label.item()]
            + [pred_concept[example_id] for pred_concept in pred_concepts]
            for example_id, label in enumerate(Y_test)
        ]
    results_df = pd.DataFrame(
        results_data, columns=["Method", "Class"] + [f"Grade {i+1}" for i in range(5)]
    )
    logging.info(f"Saving results in {save_dir}")
    results_df.to_csv(save_dir / "metrics.csv", index=False)
    if plot:
        plot_seer_global_explanation(save_dir)


def feature_importance(
    random_seed: int,
    batch_size: int,
    latent_dim: int,
    plot: bool,
    model_name: str,
    test_fraction: float = 0.1,
    model_dir: Path = Path.cwd() / "results/seer",
    data_dir: Path = Path.cwd() / "data/seer",
    save_dir: Path = Path.cwd() / "results/seer/feature_importance",
):
    torch.manual_seed(random_seed)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    if not save_dir.exists():
        os.makedirs(save_dir)

    # Load data
    train_data = SEERDataset(
        str(data_dir / "seer.csv"),
        random_seed,
        train=True,
        test_fraction=test_fraction,
        load_concept_labels=True,
    )
    test_data = SEERDataset(
        str(data_dir / "seer.csv"),
        random_seed,
        train=False,
        test_fraction=test_fraction,
        load_concept_labels=False,
    )
    test_loader = DataLoader(test_data, batch_size)

    # Load model
    model_dir = model_dir / model_name
    model = SEERClassifier(latent_dim, model_name)
    model.load_state_dict(torch.load(model_dir / f"{model_name}.pt"), strict=False)
    model.to(device)

    results_data = []
    baselines = torch.zeros((1, 21)).to(device)
    for concept_id in range(5):
        logging.info(f"Now fitting a CAR classifier for Grade {concept_id+1} patients")
        X_train, C_train = generate_seer_concept_dataset(
            train_data, concept_id, 250, random_seed
        )
        X_train = X_train.to(device)
        H_train = model.input_to_representation(X_train).detach().cpu().numpy()
        car = CAR(device, batch_size, kernel="linear")
        car.fit(H_train, C_train.numpy())
        logging.info(
            f"Computing feature importance over the test set for Grade {concept_id+1} patients"
        )
        attribution_method = CARFeatureImportance(
            "Integrated Gradient", car, model, device
        )
        attributions = attribution_method.attribute(test_loader, baselines=baselines)
        for attribution in attributions:
            reduced_attribution = (
                np.abs(attribution[:4]).tolist()
                + [np.sum(np.abs(attribution[4:13]))]
                + [np.sum(np.abs(attribution[13:]))]
            )
            results_data.append(reduced_attribution)

    results_df = pd.DataFrame(
        results_data,
        columns=[
            "Age",
            "PSA",
            "Positive Cores",
            "Examined Cores",
            "Clinical Stage",
            "Gleason Scores",
        ],
    )
    logging.info(f"Saving results in {save_dir}")
    results_df.to_csv(save_dir / "metrics.csv", index=False)
    if plot:
        plot_seer_feature_importance(save_dir)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="concept_accuracy")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(1, 11)))
    parser.add_argument("--batch_size", type=int, default=500)
    parser.add_argument("--latent_dim", type=int, default=100)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    model_name = f"model_{args.latent_dim}"
    if args.train:
        train_model(args.seeds[0], args.batch_size, args.latent_dim, model_name)

    if args.name == "concept_accuracy":
        concept_accuracy(
            args.seeds[0], args.batch_size, args.latent_dim, model_name=model_name
        )
    elif args.name == "global_explanations":
        global_explanations(
            args.seeds[0],
            args.batch_size,
            args.latent_dim,
            args.plot,
            model_name=model_name,
        )
    elif args.name == "feature_importance":
        feature_importance(
            args.seeds[0],
            args.batch_size,
            args.latent_dim,
            args.plot,
            model_name=model_name,
        )
