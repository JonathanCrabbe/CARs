import torch
import os
import logging
import argparse
import pandas as pd
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from explanations.concept import CAR, CAV
from explanations.feature import CARFeatureImportance, VanillaFeatureImportance
from tqdm import tqdm
from utils.plot import (
    plot_concept_accuracy,
    plot_global_explanation,
    plot_attribution_correlation,
    plot_color_saliency,
)
from sklearn.metrics import accuracy_score
from pathlib import Path
from utils.dataset import load_cub_data, CUBDataset, generate_cub_concept_dataset
from models.cub import CUBClassifier, CUBResNet
from utils.hooks import register_hooks, remove_all_hooks, get_saved_representations

train_path = str(Path.cwd() / "data/cub/CUB_processed/class_attr_data_10/train.pkl")
val_path = str(Path.cwd() / "data/cub/CUB_processed/class_attr_data_10/val.pkl")
test_path = str(Path.cwd() / "data/cub/CUB_processed/class_attr_data_10/test.pkl")
img_dir = str(Path.cwd() / f"data/cub/CUB_200_2011")
img_adv_dir = str(Path.cwd() / f"data/cub/CUB_random")


def fit_model(batch_size: int, n_epochs: int, model_name: str):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model_dir = Path.cwd() / f"results/cub/{model_name}"
    if not model_dir.exists():
        os.makedirs(model_dir)
    train_loader = load_cub_data(
        [train_path, val_path],
        use_attr=False,
        batch_size=batch_size,
        image_dir=img_dir,
        no_img=False,
    )
    test_loader = load_cub_data(
        [test_path],
        use_attr=False,
        batch_size=batch_size,
        image_dir=img_dir,
        no_img=False,
    )
    if model_name == "inception_model":
        model = CUBClassifier(name=model_name)
    else:
        model = CUBResNet(name=model_name)
    model.fit(
        device, train_loader, test_loader, model_dir, patience=50, n_epoch=n_epochs
    )


def concept_accuracy(
    random_seeds: list[int],
    plot: bool,
    batch_size: int,
    save_dir: Path = Path.cwd() / "results/cub/concept_accuracy",
    model_dir: Path = Path.cwd() / f"results/cub/",
    model_name: str = "model",
):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    torch.manual_seed(random_seeds[0])
    representation_dir = save_dir / f"{model_name}_representations"
    if not representation_dir.exists():
        os.makedirs(representation_dir)

    # Load model
    model_dir = model_dir / model_name
    if model_name == "inception_model":
        model = CUBClassifier(name=model_name)
    else:
        model = CUBResNet(name=model_name)
    model.load_state_dict(torch.load(model_dir / f"{model_name}.pt"), strict=False)
    model.to(device)
    model.eval()

    # Load dataset
    train_set = CUBDataset(
        [train_path, val_path],
        use_attr=True,
        no_img=False,
        uncertain_label=False,
        image_dir=img_dir,
        n_class_attr=2,
    )
    concept_names = train_set.get_concept_names()

    # Fit a concept classifier and test accuracy for each concept
    results_data = []
    for concept_id, concept_name in enumerate(concept_names):
        for random_seed in random_seeds:
            logging.info(f"Working with concept {concept_name} and seed {random_seed}")
            # Save representations for training concept examples and then remove the hooks
            module_dic, handler_train_dic = register_hooks(
                model, representation_dir, f"{concept_name}_seed{random_seed}_train"
            )
            X_train, y_train = generate_cub_concept_dataset(
                concept_id,
                200,
                random_seed,
                [train_path, val_path],
                False,
                False,
                image_dir=img_dir,
            )
            for x_train in np.array_split(X_train, batch_size):
                model(torch.from_numpy(x_train).to(device))
            remove_all_hooks(handler_train_dic)
            # Save representations for testing concept examples and then remove the hooks
            module_dic, handler_test_dic = register_hooks(
                model, representation_dir, f"{concept_name}_seed{random_seed}_test"
            )
            X_test, y_test = generate_cub_concept_dataset(
                concept_id,
                100,
                random_seed,
                [test_path],
                False,
                False,
                image_dir=img_dir,
            )
            for x_test in np.array_split(X_test, batch_size):
                model(torch.from_numpy(x_test).to(device))
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
        plot_concept_accuracy(save_dir, None, "cub")
    results_md = pd.pivot_table(
        results_df,
        index="Layer",
        columns="Method",
        aggfunc=["mean", "sem"],
        values="Test ACC",
    ).to_markdown()
    logging.info(results_md)


def statistical_significance(
    random_seed: int,
    batch_size: int,
    save_dir: Path = Path.cwd() / "results/cub/statistical_significance",
    model_dir: Path = Path.cwd() / "results/cub",
    model_name: str = "model",
) -> None:
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    model_dir = model_dir / model_name
    representation_dir = save_dir / f"{model_name}_representations"
    if not representation_dir.exists():
        os.makedirs(representation_dir)

    # Load model
    model = CUBClassifier(model_name)
    model.load_state_dict(torch.load(model_dir / f"{model_name}.pt"), strict=False)
    model.to(device)
    model.eval()

    # Load dataset
    train_set = CUBDataset(
        [train_path, val_path],
        use_attr=True,
        no_img=False,
        uncertain_label=False,
        image_dir=img_dir,
        n_class_attr=2,
    )
    concept_names = train_set.get_concept_names()
    print(concept_names)
    print(len(concept_names))
    # Fit a concept classifier and test accuracy for each concept
    results_data = []
    for concept_id, concept_name in enumerate(concept_names):
        logging.info(f"Working with concept {concept_name} ")
        # Save representations for training concept examples and then remove the hooks
        module_dic, handler_train_dic = register_hooks(
            model, representation_dir, f"{concept_name}_seed{random_seed}_train"
        )
        X_train, y_train = generate_cub_concept_dataset(
            concept_id,
            100,
            random_seed,
            [train_path, val_path],
            False,
            False,
            image_dir=img_dir,
        )
        for x_train in np.array_split(X_train, batch_size):
            model(torch.from_numpy(x_train).to(device))
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
                    car.permutation_test(H_train, y_train, n_jobs=10, n_perm=25),
                ]
            )
            results_data.append(
                [
                    concept_name,
                    module_name,
                    "CAV",
                    cav.permutation_test(H_train, y_train, n_jobs=10, n_perm=25),
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
    plot: bool,
    car_sensitivity: bool,
    save_dir: Path = Path.cwd() / "results/cub/global_explanations",
    model_dir: Path = Path.cwd() / f"results/cub",
    model_name: str = "model",
) -> None:
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    torch.manual_seed(random_seed)
    if not save_dir.exists():
        os.makedirs(save_dir)

    # Load model
    model_dir = model_dir / model_name
    model = CUBClassifier(model_name)
    model.load_state_dict(torch.load(model_dir / f"{model_name}.pt"), strict=False)
    model.to(device)
    model.eval()

    # Load dataset
    train_set = CUBDataset(
        [train_path, val_path],
        use_attr=True,
        no_img=False,
        uncertain_label=False,
        image_dir=img_dir,
        n_class_attr=2,
    )
    class_names = train_set.get_class_names()
    concept_names = train_set.get_concept_names()
    concept_categories = train_set.get_concept_categories()

    # Fit a concept classifier and test accuracy for each concept
    results_data = []
    car_classifiers = [CAR(device) for _ in concept_names]
    cav_classifiers = [CAV(device) for _ in concept_names]

    for concept_id, (concept_name, car_classifier, cav_classifier) in enumerate(
        zip(concept_names, car_classifiers, cav_classifiers)
    ):
        logging.info(f"Now fitting concept classifiers for {concept_name}")
        X_train, y_train = generate_cub_concept_dataset(
            concept_id,
            100,
            random_seed,
            [train_path, val_path],
            False,
            False,
            image_dir=img_dir,
        )
        H_train = []
        for x_train in np.array_split(X_train, batch_size):
            H_train.append(
                model.input_to_representation(torch.from_numpy(x_train).to(device))
                .detach()
                .cpu()
                .numpy()
            )
        H_train = np.concatenate(H_train)
        car_classifier.fit(H_train, y_train)
        cav_classifier.fit(H_train, y_train)
        if car_sensitivity:
            car_classifier.tune_kernel_width(H_train, y_train)

    test_loader = load_cub_data([test_path], True, False, batch_size, image_dir=img_dir)
    logging.info("Now predicting concepts on the test set")
    for X_test, y_test, concept_labels in tqdm(test_loader, unit="batch", leave=False):
        H_test = model.input_to_representation(X_test.to(device)).detach().cpu().numpy()
        car_preds = [car.predict(H_test) for car in car_classifiers]
        cav_preds = [
            cav.concept_importance(H_test, y_test, 200, model.representation_to_output)
            for cav in cav_classifiers
        ]

        results_data += [
            ["TCAR", class_names[label]]
            + [int(car_pred[idx]) for car_pred in car_preds]
            for idx, label in enumerate(y_test)
        ]
        if car_sensitivity:
            car_sensitivity_preds = [
                car.concept_sensitivity_importance(
                    H_test, y_test, 200, model.representation_to_output
                )
                for car in car_classifiers
            ]
            results_data += [
                ["TCAR Sensitivity", class_names[label]]
                + [
                    int(car_sensitivity_pred[idx] > 0)
                    for car_sensitivity_pred in car_sensitivity_preds
                ]
                for idx, label in enumerate(y_test)
            ]
        results_data += [
            ["TCAV", class_names[label]]
            + [int(cav_pred[idx] > 0) for cav_pred in cav_preds]
            for idx, label in enumerate(y_test)
        ]
        results_data += [
            ["True Prop.", class_names[label]]
            + [
                concept_labels[concept_id][idx].item()
                for concept_id in range(len(concept_names))
            ]
            for idx, label in enumerate(y_test)
        ]

    csv_path = save_dir / "metrics.csv"
    results_df = pd.DataFrame(results_data, columns=["Method", "Class"] + concept_names)
    results_df.to_csv(csv_path, index=False)
    if plot:
        plot_global_explanation(save_dir, "cub", concept_categories)


def feature_importance(
    random_seed: int,
    batch_size: int,
    plot: bool,
    n_plots: int = 5,
    sample_per_concept: int = 50,
    save_dir: Path = Path.cwd() / "results/cub/feature_importance",
    model_dir: Path = Path.cwd() / f"results/cub",
    model_name: str = "model",
) -> None:
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    torch.manual_seed(random_seed)
    if not save_dir.exists():
        os.makedirs(save_dir)

    # Load model
    model_dir = model_dir / model_name
    model = CUBClassifier(model_name)
    model.load_state_dict(torch.load(model_dir / f"{model_name}.pt"), strict=False)
    model.to(device)
    model.eval()

    # Load test set
    test_set = CUBDataset(
        [test_path],
        use_attr=False,
        no_img=False,
        uncertain_label=False,
        image_dir=img_dir,
        n_class_attr=2,
        transform=transforms.Compose(
            [
                transforms.Resize((299, 299)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        ),
    )

    # Select concept for which feature importance is evaluated
    selected_concept_names = [
        "Belly Color Brown",
        "Belly Pattern Solid",
        "Back Pattern Striped",
        "Back Pattern Solid",
        "Breast Color Black",
        "Breast Color White",
    ]
    selected_concept_ids = [
        test_set.concept_id(concept_name) for concept_name in selected_concept_names
    ]
    subtest_ids = test_set.get_concepts_subset(
        selected_concept_ids, sample_per_concept, random_seed
    )
    subtest_set = Subset(test_set, subtest_ids)
    subtest_loader = DataLoader(subtest_set, batch_size)

    # Options for attribution methods
    attribution_dic = {}
    baselines = transforms.GaussianBlur(
        kernel_size=299, sigma=50.0
    )  # Baseline for attribution methods

    # Fit a concept classifier and compute feature importance for each concept
    car_classifiers = [CAR(device) for _ in selected_concept_names]
    for concept_number, (concept_name, car) in enumerate(
        zip(selected_concept_names, car_classifiers)
    ):
        logging.info(f"Now fitting CAR classifier for {concept_name}")
        concept_id = test_set.concept_id(concept_name)
        X_train, y_train = generate_cub_concept_dataset(
            concept_id,
            200,
            random_seed,
            [train_path, val_path],
            False,
            False,
            image_dir=img_dir,
        )
        H_train = []
        for x_train in np.array_split(X_train, batch_size):
            H_train.append(
                model.input_to_representation(torch.from_numpy(x_train).to(device))
                .detach()
                .cpu()
                .numpy()
            )
        H_train = np.concatenate(H_train)
        car.fit(H_train, y_train)
        car.tune_kernel_width(H_train, y_train)
        logging.info(
            f"Now computing feature importance on the test set for {concept_name}"
        )
        concept_attribution_method = CARFeatureImportance(
            "Integrated Gradient", car, model, device
        )
        attribution_dic[concept_name] = concept_attribution_method.attribute(
            subtest_loader, baselines=baselines, internal_batch_size=batch_size
        )
        if plot:
            logging.info(f"Saving plots in {save_dir} for {concept_name}")
            plot_slice = slice(
                2 * concept_number * sample_per_concept,
                2 * concept_number * sample_per_concept + n_plots,
            )
            positive_ids = subtest_ids[plot_slice]
            selected_images = [test_set.get_raw_image(idx) for idx in positive_ids]
            positive_saliency = attribution_dic[concept_name][plot_slice]
            plot_color_saliency(
                selected_images,
                positive_saliency,
                save_dir,
                f"cub_positive",
                concept_name.lower().replace(" ", "-"),
            )
    logging.info(f"Now computing vanilla feature importance")
    vanilla_attribution_method = VanillaFeatureImportance(
        "Integrated Gradient", model, device
    )
    attribution_dic["Vanilla"] = vanilla_attribution_method.attribute(
        subtest_loader, baselines=baselines, internal_batch_size=batch_size
    )
    np.savez(save_dir / "attributions.npz", **attribution_dic)
    if plot:
        logging.info(f"Saving plots in {save_dir}")
        plot_attribution_correlation(save_dir, "cub")


def tune_kernel(
    random_seeds: list[int],
    plot: bool,
    batch_size: int,
    save_dir: Path = Path.cwd() / "results/cub/tuning_kernel",
    model_dir: Path = Path.cwd() / f"results/cub/",
    model_name: str = "model",
):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    torch.manual_seed(random_seeds[0])
    representation_dir = save_dir / f"{model_name}_representations"
    if not representation_dir.exists():
        os.makedirs(representation_dir)

    # Load model
    model_dir = model_dir / model_name
    model = CUBClassifier(name=model_name)
    model.load_state_dict(torch.load(model_dir / f"{model_name}.pt"), strict=False)
    model.to(device)
    model.eval()

    # Load dataset
    train_set = CUBDataset(
        [train_path, val_path],
        use_attr=True,
        no_img=False,
        uncertain_label=False,
        image_dir=img_dir,
        n_class_attr=2,
    )
    concept_names = train_set.get_concept_names()

    # Fit a concept classifier and test accuracy for each concept
    results_data = []
    for concept_id, concept_name in enumerate(concept_names):
        for random_seed in random_seeds:
            logging.info(f"Working with concept {concept_name} and seed {random_seed}")
            # Save representations for training concept examples and then remove the hooks
            module_dic, handler_train_dic = register_hooks(
                model, representation_dir, f"{concept_name}_seed{random_seed}_train"
            )
            X_train, y_train = generate_cub_concept_dataset(
                concept_id,
                300,
                random_seed,
                [train_path, val_path],
                False,
                False,
                image_dir=img_dir,
            )
            for x_train in np.array_split(X_train, batch_size):
                model(torch.from_numpy(x_train).to(device))
            remove_all_hooks(handler_train_dic)
            # Save representations for testing concept examples and then remove the hooks
            module_dic, handler_test_dic = register_hooks(
                model, representation_dir, f"{concept_name}_seed{random_seed}_test"
            )
            X_test, y_test = generate_cub_concept_dataset(
                concept_id,
                300,
                random_seed,
                [test_path],
                False,
                False,
                image_dir=img_dir,
            )
            for x_test in np.array_split(X_test, batch_size):
                model(torch.from_numpy(x_test).to(device))
            remove_all_hooks(handler_test_dic)
            # Create concept classifiers, fit them and test them for each representation space
            for module_name in {"InceptionOut"}:
                logging.info(f"Fitting concept classifiers for {module_name}")
                car_cv = CAR(device)
                car = CAR(device)
                cav = CAV(device)
                hook_name = f"{concept_name}_seed{random_seed}_train_{module_name}"
                H_train = get_saved_representations(hook_name, representation_dir)
                car_cv.fit_cv(H_train, y_train)
                car.fit(H_train, y_train)
                cav.fit(H_train, y_train)
                hook_name = f"{concept_name}_seed{random_seed}_test_{module_name}"
                H_test = get_saved_representations(hook_name, representation_dir)
                results_data.append(
                    [
                        concept_name,
                        module_name,
                        random_seed,
                        "CAR-CV",
                        accuracy_score(y_train, car_cv.predict(H_train)),
                        accuracy_score(y_test, car_cv.predict(H_test)),
                    ]
                )
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
    logging.info(results_df.groupby("Method")["Test ACC"].mean())
    logging.info(results_df.groupby("Method")["Test ACC"].sem())
    if plot:
        plot_concept_accuracy(save_dir, None, "cub")


def global_explanations_mixed7b(
    random_seed: int,
    batch_size: int,
    plot: bool,
    save_dir: Path = Path.cwd() / "results/cub/global_explanations_mixed7b",
    model_dir: Path = Path.cwd() / f"results/cub",
    model_name: str = "model",
) -> None:
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    torch.manual_seed(random_seed)
    if not save_dir.exists():
        os.makedirs(save_dir)

    # Load model
    model_dir = model_dir / model_name
    model = CUBClassifier(model_name)
    model.load_state_dict(torch.load(model_dir / f"{model_name}.pt"), strict=False)
    model.to(device)
    model.eval()

    # Load dataset
    train_set = CUBDataset(
        [train_path, val_path],
        use_attr=True,
        no_img=False,
        uncertain_label=False,
        image_dir=img_dir,
        n_class_attr=2,
    )
    class_names = train_set.get_class_names()
    concept_names = train_set.get_concept_names()
    concept_categories = train_set.get_concept_categories()

    # Fit a concept classifier and test accuracy for each concept
    results_data = []
    car_classifiers = [CAR(device) for _ in concept_names]
    cav_classifiers = [CAV(device) for _ in concept_names]

    for concept_id, (concept_name, car_classifier, cav_classifier) in enumerate(
        zip(concept_names, car_classifiers, cav_classifiers)
    ):
        logging.info(f"Now fitting concept classifiers for {concept_name}")
        X_train, y_train = generate_cub_concept_dataset(
            concept_id,
            100,
            random_seed,
            [train_path, val_path],
            False,
            False,
            image_dir=img_dir,
        )
        H_train = []
        for x_train in np.array_split(X_train, batch_size):
            H_train.append(
                model.mixed_7b_rep(torch.from_numpy(x_train).to(device))
                .flatten(start_dim=1)
                .detach()
                .cpu()
                .numpy()
            )
        H_train = np.concatenate(H_train)
        car_classifier.fit(H_train, y_train)
        cav_classifier.fit(H_train, y_train)

    test_loader = load_cub_data([test_path], True, False, batch_size, image_dir=img_dir)
    logging.info("Now predicting concepts on the test set")
    for X_test, y_test, concept_labels in tqdm(test_loader, unit="batch", leave=False):
        H_test = model.mixed_7b_rep(X_test.to(device))
        H_test_flat = H_test.flatten(start_dim=1).detach().cpu().numpy()
        H_test = H_test.detach().cpu().numpy()
        car_preds = [car.predict(H_test_flat) for car in car_classifiers]
        cav_preds = [
            cav.concept_importance(H_test, y_test, 200, model.mixed_7b_rep_to_output)
            for cav in cav_classifiers
        ]

        results_data += [
            ["TCAR", class_names[label]]
            + [int(car_pred[idx]) for car_pred in car_preds]
            for idx, label in enumerate(y_test)
        ]
        results_data += [
            ["TCAV", class_names[label]]
            + [int(cav_pred[idx] > 0) for cav_pred in cav_preds]
            for idx, label in enumerate(y_test)
        ]
        results_data += [
            ["True Prop.", class_names[label]]
            + [
                concept_labels[concept_id][idx].item()
                for concept_id in range(len(concept_names))
            ]
            for idx, label in enumerate(y_test)
        ]

    csv_path = save_dir / "metrics.csv"
    results_df = pd.DataFrame(results_data, columns=["Method", "Class"] + concept_names)
    results_df.to_csv(csv_path, index=False)
    if plot:
        plot_global_explanation(save_dir, "cub", concept_categories)


def global_explanations_robustness(
    random_seed: int,
    batch_size: int,
    plot: bool,
    save_dir: Path = Path.cwd() / "results/cub/global_explanations_robustness",
    model_dir: Path = Path.cwd() / f"results/cub",
    model_name: str = "model",
) -> None:
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    torch.manual_seed(random_seed)
    if not save_dir.exists():
        os.makedirs(save_dir)

    # Load model
    model_dir = model_dir / model_name
    model = CUBClassifier(model_name)
    model.load_state_dict(torch.load(model_dir / f"{model_name}.pt"), strict=False)
    model.to(device)
    model.eval()

    # Load dataset
    train_set = CUBDataset(
        [train_path, val_path],
        use_attr=True,
        no_img=False,
        uncertain_label=False,
        image_dir=img_dir,
        n_class_attr=2,
    )
    class_names = train_set.get_class_names()
    concept_names = train_set.get_concept_names()
    concept_categories = train_set.get_concept_categories()

    # Fit a concept classifier and test accuracy for each concept
    results_data = []
    car_classifiers = [CAR(device) for _ in concept_names]
    cav_classifiers = [CAV(device) for _ in concept_names]

    for concept_id, (concept_name, car_classifier, cav_classifier) in enumerate(
        zip(concept_names, car_classifiers, cav_classifiers)
    ):
        logging.info(f"Now fitting concept classifiers for {concept_name}")
        X_train, y_train = generate_cub_concept_dataset(
            concept_id,
            100,
            random_seed,
            [train_path, val_path],
            False,
            False,
            image_dir=img_dir,
        )
        H_train = []
        for x_train in np.array_split(X_train, batch_size):
            H_train.append(
                model.input_to_representation(torch.from_numpy(x_train).to(device))
                .detach()
                .cpu()
                .numpy()
            )
        H_train = np.concatenate(H_train)
        car_classifier.fit(H_train, y_train)
        cav_classifier.fit(H_train, y_train)

    test_loader = load_cub_data(
        [test_path], True, False, batch_size, image_dir=img_adv_dir
    )
    logging.info("Now predicting concepts on the test set")
    for X_test, y_test, concept_labels in tqdm(test_loader, unit="batch", leave=False):
        H_test = model.input_to_representation(X_test.to(device)).detach().cpu().numpy()
        car_preds = [car.predict(H_test) for car in car_classifiers]
        cav_preds = [
            cav.concept_importance(H_test, y_test, 200, model.representation_to_output)
            for cav in cav_classifiers
        ]
        results_data += [
            ["TCAR", class_names[label]]
            + [int(car_pred[idx]) for car_pred in car_preds]
            for idx, label in enumerate(y_test)
        ]
        results_data += [
            ["TCAV", class_names[label]]
            + [int(cav_pred[idx] > 0) for cav_pred in cav_preds]
            for idx, label in enumerate(y_test)
        ]
        results_data += [
            ["True Prop.", class_names[label]]
            + [
                concept_labels[concept_id][idx].item()
                for concept_id in range(len(concept_names))
            ]
            for idx, label in enumerate(y_test)
        ]

    csv_path = save_dir / "metrics.csv"
    results_df = pd.DataFrame(results_data, columns=["Method", "Class"] + concept_names)
    results_df.to_csv(csv_path, index=False)
    if plot:
        plot_global_explanation(save_dir, "cub", concept_categories)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="concept_accuracy")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_epochs", type=int, default=1000)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--concept_category", type=str, default="Primary Color")
    parser.add_argument("--samples_per_concept", type=int, default=50)
    parser.add_argument("--model_type", type=str, default="inception")
    parser.add_argument("--car_sensitivity", action="store_true")
    args = parser.parse_args()

    assert args.model_type in {"resnet", "inception"}
    model_name = f"{args.model_type}_model"
    if args.train:
        fit_model(args.batch_size, args.n_epochs, model_name=model_name)
    if args.name == "concept_accuracy":
        concept_accuracy(args.seeds, args.plot, args.batch_size, model_name=model_name)
    elif args.name == "statistical_significance":
        statistical_significance(args.seeds[0], args.batch_size, model_name=model_name)
    elif args.name == "global_explanations":
        global_explanations(
            args.seeds[0],
            args.batch_size,
            args.plot,
            car_sensitivity=args.car_sensitivity,
            model_name=model_name,
        )
    elif args.name == "feature_importance":
        feature_importance(
            args.seeds[0],
            args.batch_size,
            args.plot,
            model_name=model_name,
            sample_per_concept=args.samples_per_concept,
        )
    elif args.name == "tune_kernel":
        tune_kernel(args.seeds, args.plot, args.batch_size, model_name=model_name)
    elif args.name == "global_explanations_mixed7b":
        global_explanations_mixed7b(
            args.seeds[0], args.batch_size, args.plot, model_name=model_name
        )
    elif args.name == "global_explanations_robustness":
        global_explanations_robustness(
            args.seeds[0], args.batch_size, args.plot, model_name=model_name
        )
