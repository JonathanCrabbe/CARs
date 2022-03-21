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
        plt.ylabel(f"Concept {concept} Accuracy")
        plt.savefig(results_dir/f"{concept}_acc.pdf")
    else:
        plt.ylabel(f"Overall Concept Accuracy")
        plt.savefig(results_dir / f"overall_acc.pdf")
    plt.close()


def plot_global_explanation(results_dir: Path) -> None:
    metrics_df = pd.read_csv(results_dir / "metrics.csv")
    sns.catplot(data=metrics_df, x="Concept", col="Class", hue="Method", kind="count", col_wrap=5)
    plt.savefig(results_dir / "global_explanations.pdf")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="concept_accuracy")
    parser.add_argument("--dataset", type=str, default="mnist")
    parser.add_argument("--concept", type=str, default=None)
    args = parser.parse_args()
    save_path = Path.cwd()/f"results/{args.dataset}/{args.name}"
    logging.info(f"Saving {args.name} plot for {args.dataset} in {str(save_path)}")
    sns.set()
    sns.color_palette("colorblind")
    sns.set_style("white")
    if args.name == "concept_accuracy":
        plot_concept_accuracy(save_path, args.concept)
    elif args.name == "global_explanations":
        plot_global_explanation(save_path)
    else:
        raise ValueError(f"{args.name} is not a valid experiment name")



