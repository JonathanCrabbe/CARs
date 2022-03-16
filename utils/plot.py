import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import logging
import argparse
from pathlib import Path


def plot_concept_accuracy(results_dir: Path) -> None:
    metrics_df = pd.read_csv(results_dir/"metrics.csv")
    sns.boxplot(data=metrics_df, x="Layer", y="Test ACC", hue="Method")
    plt.savefig(results_dir/"overall_acc.pdf")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="concept_accuracy")
    parser.add_argument("--dataset", type=str, default="mnist")
    args = parser.parse_args()
    sns.set()
    sns.color_palette("colorblind")
    sns.set_style("white")
    if args.name == "concept_accuracy":
        plot_concept_accuracy(Path.cwd()/f"results/{args.dataset}/concept_accuracy")

