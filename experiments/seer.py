import os
import torch
import argparse
import logging
import pandas as pd
from utils.dataset import SEERDataset, generate_seer_concept_dataset
from utils.metrics import concept_accuracy
from models.seer import SEERClassifier
from pathlib import Path
from torch.utils.data import random_split, DataLoader
from utils.plot import plot_global_explanation
from explanations.concept import CAR, CAV


def train_model(random_seed: int,  batch_size: int, latent_dim: int, model_name: str, test_fraction: float = 0.1,
                model_dir: Path = Path.cwd()/"results/seer", data_dir: Path = Path.cwd()/"data/seer"):
    assert 0 < test_fraction < 1
    logging.info("Now fitting a SEER classifier")
    torch.manual_seed(random_seed)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model_dir = model_dir/model_name
    if not model_dir.exists():
        os.makedirs(model_dir)

    seer_data = SEERDataset(str(data_dir/"seer.csv"), random_seed)
    train_size = int((1-test_fraction)*len(seer_data))
    test_size = len(seer_data) - train_size
    train_data, test_data = random_split(seer_data, lengths=[train_size, test_size])
    train_loader = DataLoader(train_data, batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size, shuffle=False)

    model = SEERClassifier(latent_dim, model_name)
    model.fit(device, train_loader, test_loader, model_dir, n_epoch=500, patience=50)


def use_case(random_seed: int,  batch_size: int, latent_dim: int, plot: bool, model_name: str, test_fraction: float = 0.1,
             model_dir: Path = Path.cwd()/"results/seer", data_dir: Path = Path.cwd()/"data/seer",
             save_dir: Path = Path.cwd()/"results/seer/use_case"):
    torch.manual_seed(random_seed)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    if not save_dir.exists():
        os.makedirs(save_dir)

    # Load data
    seer_data = SEERDataset(str(data_dir / "seer.csv"), random_seed, load_concept_labels=True, oversample=False)
    train_size = int((1 - test_fraction) * len(seer_data))
    test_size = len(seer_data) - train_size
    train_data, test_data = random_split(seer_data, lengths=[train_size, test_size])
    train_loader = DataLoader(train_data, 1)
    test_loader = DataLoader(test_data, batch_size)

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
        X_train, C_train = generate_seer_concept_dataset(train_data, concept_id, 200, random_seed)
        X_train = X_train.to(device)
        H_train = model.input_to_representation(X_train).detach().cpu().numpy()
        car = car_classifiers[concept_id]
        car.fit(H_train, C_train.numpy())
        cav = cav_classifiers[concept_id]
        cav.fit(H_train, C_train.numpy())
        print(concept_accuracy(test_loader, car, concept_id, device, model))
        print(concept_accuracy(test_loader, cav, concept_id, device, model))



    """
    grade_data = []
    for x_train, y_train, c_train in train_loader:
        grade_data.append([y_train.item(), torch.argmax(c_train.flatten()).item()+1])
    grade_df = pd.DataFrame(grade_data, columns=["Mortality", "Grade"])
    sns.histplot(grade_df, x="Grade", hue="Mortality")
    plt.show()
    
    for X_test, Y_test, C_test in test_loader:
        X_test = X_test.to(device)
        H_test = model.input_to_representation(X_test).detach().cpu().numpy()
        car_preds = [car.predict(H_test) for car in car_classifiers]
        results_data += [["TCAR", label.item()] + [int(car_pred[idx]) for car_pred in car_preds]
                            for idx, label in enumerate(Y_test)]
        results_data += [["True Prop.", label.item()] + C_test[idx].tolist()
                         for idx, label in enumerate(Y_test)]
    
    csv_path = save_dir / "metrics.csv"
    results_df = pd.DataFrame(results_data, columns=["Method", "Class"] + [f"Grade {i}" for i in range(1, 6)])
    results_df.to_csv(csv_path, index=False)
    if plot:
        plot_global_explanation(save_dir, "seer")
    """


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="use_case")
    parser.add_argument('--seeds', nargs="+", type=int, default=list(range(1, 11)))
    parser.add_argument("--batch_size", type=int, default=500)
    parser.add_argument("--latent_dim", type=int, default=100)
    parser.add_argument("--train", action='store_true')
    parser.add_argument("--plot", action='store_true')
    args = parser.parse_args()

    model_name = f"model_{args.latent_dim}"
    if args.train:
        train_model(args.seeds[0], args.batch_size, args.latent_dim, model_name)

    if args.name == "use_case":
        use_case(args.seeds[0], args.batch_size, args.latent_dim, args.plot, model_name=model_name)

