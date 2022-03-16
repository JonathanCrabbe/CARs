import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import logging
import argparse
from pathlib import Path


def plot_concept_accuracy(results_dir: Path, concept: str) -> None:
    metrics_df = pd.read_csv(results_dir/"metrics.csv")
    if concept:
        metrics_df = metrics_df[metrics_df.Concept == concept]
    sns.boxplot(data=metrics_df, x="Layer", y="Test ACC", hue="Method")
    if concept:
        plt.ylabel(f"Concept {concept} accuracy")
        plt.savefig(results_dir/f"{concept}_acc.pdf")
    else:
        plt.ylabel(f"Overall concept accuracy")
        plt.savefig(results_dir / f"overall_acc.pdf")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="concept_accuracy")
    parser.add_argument("--dataset", type=str, default="mnist")
    parser.add_argument("--concept", type=str, default=None)
    args = parser.parse_args()
    save_path = Path.cwd()/f"results/{args.dataset}/concept_accuracy"
    logging.info(f"Saving {args.name} plot for {args.dataset} in {str(save_path)}")
    sns.set()
    sns.color_palette("colorblind")
    sns.set_style("white")
    if args.name == "concept_accuracy":
        plot_concept_accuracy(save_path, args.concept)
    else:
        raise ValueError(f"{args.name} is not a valid experiment name")



