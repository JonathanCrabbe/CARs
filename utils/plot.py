import itertools
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import logging
import argparse
import textwrap
import numpy as np
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
    concepts = list(metrics_df.columns[2:])
    classes = metrics_df["Class"].unique()
    plot_data = []
    for class_idx, concept in itertools.product(classes, concepts):
        tcar_attr = np.array(metrics_df.loc[(metrics_df.Class == class_idx) & (metrics_df.Method == "TCAR")][concept])
        tcav_attr = np.array(metrics_df.loc[(metrics_df.Class == class_idx) & (metrics_df.Method == "TCAV")][concept])
        tcar_score = np.sum(tcar_attr)/len(tcar_attr)
        tcav_score = np.sum(tcav_attr)/len(tcav_attr)
        plot_data.append(["TCAR", class_idx, concept, tcar_score])
        plot_data.append(["TCAV", class_idx, concept, tcav_score])
    plot_df = pd.DataFrame(plot_data, columns=["Method", "Class", "Concept", "Score"])
    for class_idx in classes:
        ax = sns.barplot(data=plot_df.loc[plot_df.Class == class_idx], x="Concept", y="Score", hue="Method")
        wrap_labels(ax, 10)
        plt.title(f"Class: {class_idx}")
        plt.tight_layout()
        plt.savefig(results_dir / f"global_explanations_class{class_idx}.pdf")
        plt.close()


def wrap_labels(ax, width, break_long_words=False):
    labels = []
    for label in ax.get_xticklabels():
        text = label.get_text()
        labels.append(textwrap.fill(text, width=width,
                      break_long_words=break_long_words))
    ax.set_xticklabels(labels, rotation=0)


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



