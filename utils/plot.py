import itertools
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import logging
import argparse
import textwrap
import numpy as np
from pathlib import Path


def plot_concept_accuracy(results_dir: Path, concept: str, dataset_name: str) -> None:
    sns.set(font_scale=1.5)
    sns.color_palette("colorblind")
    sns.set_style("white")
    metrics_df = pd.read_csv(results_dir/"metrics.csv")
    if concept:
        metrics_df = metrics_df[metrics_df.Concept == concept]
    sns.boxplot(data=metrics_df, x="Layer", y="Test ACC", hue="Method")
    if concept:
        plt.ylabel(f"Concept {concept} Accuracy")
        plt.savefig(results_dir/f"{dataset_name}_{concept}_acc.pdf")
    else:
        plt.ylabel(f"Overall Concept Accuracy")
        plt.savefig(results_dir / f"{dataset_name}_concept_acc.pdf")
    plt.close()


def plot_global_explanation(results_dir: Path, dataset_name: str) -> None:
    sns.set(font_scale=1.2)
    sns.color_palette("colorblind")
    sns.set_style("white")
    metrics_df = pd.read_csv(results_dir / "metrics.csv")
    concepts = list(metrics_df.columns[2:])
    classes = metrics_df["Class"].unique()
    methods = metrics_df["Method"].unique()
    plot_data = []
    for class_idx, concept, method in itertools.product(classes, concepts, methods):
        attr = np.array(metrics_df.loc[(metrics_df.Class == class_idx) & (metrics_df.Method == method)][concept])
        score = np.sum(attr)/len(attr)
        plot_data.append([method, class_idx, concept, score])
    plot_df = pd.DataFrame(plot_data, columns=["Method", "Class", "Concept", "Score"])
    for class_idx in classes:
        ax = sns.barplot(data=plot_df.loc[plot_df.Class == class_idx], x="Concept", y="Score", hue="Method")
        wrap_labels(ax, 10)
        plt.title(f"Class: {class_idx}")
        plt.ylim(bottom=0, top=1.1)
        plt.tight_layout()
        plt.savefig(results_dir / f"{dataset_name}_global_class{class_idx}.pdf")
        plt.close()
    tcar_scores = plot_df.loc[plot_df.Method == "TCAR"]["Score"]
    tcav_scores = plot_df.loc[plot_df.Method == "TCAV"]["Score"]
    true_scores = plot_df.loc[plot_df.Method == "True Prop."]["Score"]
    logging.info(f"TCAR-True Prop. Correlation: {np.corrcoef(tcar_scores, true_scores)[0, 1]:.2g}")
    logging.info(f"TCAV-True Prop. Correlation: {np.corrcoef(tcav_scores, true_scores)[0, 1]:.2g}")


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
    if args.name == "concept_accuracy":
        plot_concept_accuracy(save_path, args.concept, args.dataset)
    elif args.name == "global_explanations":
        plot_global_explanation(save_path, args.dataset)
    else:
        raise ValueError(f"{args.name} is not a valid experiment name")



