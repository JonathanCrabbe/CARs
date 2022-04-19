import itertools
import os

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import logging
import argparse
import torch
import textwrap
import numpy as np
from pathlib import Path
from utils.metrics import correlation_matrix
from utils.dataset import CUBDataset


def plot_concept_accuracy(results_dir: Path, concept: str, dataset_name: str) -> None:
    sns.set(font_scale=1)
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


def plot_global_explanation(results_dir: Path, dataset_name: str, concept_categories: dict = None) -> None:
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
    if concept_categories is not None:
        for class_idx, concept_category in itertools.product(classes, concept_categories):
            save_dir = results_dir/concept_category.lower().replace(" ", "-")
            if not save_dir.exists():
                os.makedirs(save_dir)
            filtered_concepts = concept_categories[concept_category]  # Use only the concept in the given category
            ax = sns.barplot(data=plot_df[(plot_df.Class == class_idx) & (plot_df.Concept.isin(filtered_concepts))],
                             x="Concept", y="Score", hue="Method")
            remove_text_from_labels(ax, f"{concept_category} ")
            wrap_labels(ax, 10)
            plt.title(f"Class: {class_idx}")
            plt.xlabel(f"Concept: {concept_category}")
            plt.ylim(bottom=0, top=1.1)
            plt.tight_layout()
            concept_category = concept_category.lower().replace(" ", "-")
            class_idx = class_idx.lower().replace(" ", "-")
            plt.savefig(save_dir / f"{dataset_name}_global_category_{concept_category}_class_{class_idx}.pdf")
            plt.close()
    else:
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


def plot_perturbation_sensitivity(results_dir: Path, concept: str, dataset_name: str) -> None:
    sns.set(font_scale=1)
    sns.color_palette("colorblind")
    sns.set_style("white")
    metrics_df = pd.read_csv(results_dir/"metrics.csv")
    if concept:
        metrics_df = metrics_df[metrics_df.Concept == concept]
    sns.lineplot(data=metrics_df, x="Perturbed Features", y="Concept Shift", hue="Method")
    if concept:
        plt.ylabel(f"Concept {concept} Perturbation Sensitivity")
        plt.savefig(results_dir/f"{dataset_name}_{concept}_pert_sensitivity.pdf")
    else:
        plt.ylabel(f"Overall Perturbation Sensitivity")
        plt.savefig(results_dir / f"{dataset_name}_pert_sensitivity.pdf")
    plt.close()


def plot_attribution_correlation(results_dir: Path, dataset_name: str) \
        -> None:
    sns.set(font_scale=.8)
    sns.color_palette("colorblind")
    sns.set_style("white")
    attribution_dic = np.load(results_dir/"attributions.npz")
    corr_matrix = correlation_matrix(attribution_dic)
    ax = sns.heatmap(corr_matrix, vmin=-1, vmax=1, cmap=sns.diverging_palette(10, 133, as_cmap=True), cbar=True,
                     xticklabels=attribution_dic.keys(), yticklabels=attribution_dic.keys(),
                     cbar_kws={'label': 'Correlation'}, annot=True)
    wrap_labels(ax, 12, True, True)
    plt.tight_layout()
    plt.savefig(results_dir/f"{dataset_name}_attr_corr.pdf")
    plt.close()


def plot_saliency_map(images: torch.Tensor, saliency: np.ndarray, plot_indices: list[int],
                      results_dir: Path, dataset_name: str, concept_name: str) -> None:
    sns.set(font_scale=1.2)
    sns.color_palette("colorblind")
    sns.set_style("white")
    if len(images.shape) == 4:
        images = images.permute((0, 2, 3, 1))
    n_plots = len(plot_indices)
    fig, axs = plt.subplots(ncols=1, nrows=n_plots, figsize=(1.5, 1.5*n_plots))
    for ax_id, example_id in enumerate(plot_indices):
        sub_saliency = saliency[example_id]
        max_value = np.max(np.abs(sub_saliency))
        ax = axs[ax_id]

        ax.imshow(images[example_id].cpu().numpy(), cmap='gray', zorder=1)
        ax.axis('off')
        sns.heatmap(np.sum(sub_saliency, axis=0), linewidth=0, xticklabels=False, yticklabels=False,
                    ax=ax, cmap=sns.diverging_palette(10, 133, as_cmap=True), cbar=False,
                    alpha=.8, zorder=2, vmin=-max_value, vmax=max_value)
    plt.tight_layout()
    plt.savefig(results_dir/f"{dataset_name}_{concept_name}_saliency.pdf")
    plt.close()


def plot_time_series_saliency(tseries: torch.Tensor, saliency: np.ndarray, plot_indices: list[int],
                              results_dir: Path, dataset_name: str, concept_name: str) -> None:
    sns.set(font_scale=1)
    sns.color_palette("colorblind")
    sns.set_style("white")
    T = tseries.shape[1]
    n_plots = len(plot_indices)
    fig, axs = plt.subplots(ncols=1, nrows=n_plots, figsize=(20, 3*n_plots))
    for ax_id, example_id in enumerate(plot_indices):
        sub_saliency = saliency[example_id]
        max_value = np.max(np.abs(sub_saliency))
        ax = axs[ax_id]
        sns.lineplot(x=list(range(T)), y=tseries[example_id].flatten(), ax=ax)
        scatter = ax.scatter(x=list(range(T)), y=tseries[example_id].flatten(),
                             cmap=sns.diverging_palette(10, 133, as_cmap=True),
                             c=sub_saliency.flatten(), vmin=-max_value, vmax=max_value)
        ax.set_xlabel("Time")
        ax.set_ylabel("Voltage")
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label("Importance")
    plt.tight_layout()
    plt.savefig(results_dir/f"{dataset_name}_{concept_name}_saliency.pdf")
    plt.close()


def plot_modulation_impact(results_dir: Path, dataset_name: str) -> None:
    sns.set(font_scale=1.2)
    sns.color_palette("colorblind")
    sns.set_style("white")
    metrics_df = pd.read_csv(results_dir/"metrics.csv")
    ax = sns.displot(metrics_df, x="Concept Shift", hue="Method", fill=True, kde=True)
    plt.savefig(results_dir/f"{dataset_name}_concept_shift.pdf")
    plt.close()
    ax = sns.displot(metrics_df, x="Modulation Norm", hue="Method", fill=True, kde=True)
    plt.savefig(results_dir / f"{dataset_name}_modulation_norm.pdf")
    plt.close()


def plot_counterfactual_images(factuals: torch.Tensor, counterfactuals: list[np.ndarray], plot_indices: list[int],
                               results_dir: Path, dataset_name: str, concept_name: str) -> None:
    sns.set(font_scale=1.0)
    sns.color_palette("colorblind")
    sns.set_style("white")
    n_plots = len(plot_indices)
    n_cols = len(counterfactuals) + 1
    fig, axs = plt.subplots(ncols=n_cols, nrows=n_plots, figsize=(2*n_cols, 2*n_plots))
    for ax_id, example_id in enumerate(plot_indices):
        factual_image = factuals[example_id].cpu().numpy().astype(float)
        ax = axs[ax_id, 0]
        ax.imshow(factual_image, cmap='gray')
        ax.axis('off')
        ax.set_title('Original')
        ax = axs[ax_id, 1]
        counterfactual_image = counterfactuals[0][example_id].squeeze()
        ax.imshow(counterfactual_image, cmap='gray')
        ax.axis('off')
        ax.set_title('CAR Modulated')
        ax = axs[ax_id, 2]
        counterfactual_image = counterfactuals[1][example_id].squeeze()
        ax.imshow(counterfactual_image, cmap='gray')
        ax.axis('off')
        ax.set_title('CAV Modulated')
    plt.tight_layout()
    plt.savefig(results_dir/f"{dataset_name}_{concept_name}_counterfactual.pdf")
    plt.close()


def plot_counterfactual_series(factuals: torch.Tensor, counterfactuals: list[np.ndarray], plot_indices: list[int],
                               results_dir: Path, dataset_name: str, concept_name: str) -> None:
    sns.set(font_scale=1.2)
    sns.color_palette("colorblind")
    sns.set_style("white")
    n_plots = len(plot_indices)
    n_cols = len(counterfactuals) + 1
    fig, axs = plt.subplots(ncols=n_cols, nrows=n_plots, figsize=(4*n_cols, 2*n_plots))
    for ax_id, example_id in enumerate(plot_indices):
        factual_series = factuals[example_id].cpu().numpy().astype(float).squeeze()
        ax = axs[ax_id, 0]
        sns.lineplot(x=list(range(len(factual_series))), y=factual_series, ax=ax)
        ax.set_xlabel("Time")
        ax.set_ylabel("Voltage")
        ax.set_title('Original')
        ax = axs[ax_id, 1]
        counterfactual_series = counterfactuals[0][example_id].squeeze()
        sns.lineplot(x=list(range(len(counterfactual_series))), y=counterfactual_series, ax=ax)
        ax.set_xlabel("Time")
        ax.set_ylabel("Voltage")
        ax.set_title('CAR Modulated')
        ax = axs[ax_id, 2]
        counterfactual_series = counterfactuals[1][example_id].squeeze()
        sns.lineplot(x=list(range(len(counterfactual_series))), y=counterfactual_series, ax=ax)
        ax.set_xlabel("Time")
        ax.set_ylabel("Voltage")
        ax.set_title('CAV Modulated')
    plt.tight_layout()
    plt.savefig(results_dir/f"{dataset_name}_{concept_name}_counterfactual.pdf")
    plt.close()


def wrap_labels(ax, width, break_long_words=False, do_y: bool = False) -> None:
    """
    Break labels in several lines in a figure
    Args:
        ax: figure axes
        width: maximal number of characters per line
        break_long_words: if True, allow breaks in the middle of a word
        do_y: if True, apply the function to the y axis as well

    Returns:

    """
    labels = []
    for label in ax.get_xticklabels():
        text = label.get_text()
        labels.append(textwrap.fill(text, width=width,
                      break_long_words=break_long_words))
    ax.set_xticklabels(labels, rotation=0)
    if do_y:
        labels = []
        for label in ax.get_yticklabels():
            text = label.get_text()
            labels.append(textwrap.fill(text, width=width,
                                        break_long_words=break_long_words))
        ax.set_yticklabels(labels, rotation=0)


def remove_text_from_labels(ax, removed_text: str) -> None:
    """
    Remove some redundant text in the labels of a figure
    Args:
        ax: figure axes
        removed_text: string to be removed

    Returns:

    """
    labels = []
    for label in ax.get_xticklabels():
        text = label.get_text()
        labels.append(text.replace(removed_text, ""))
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
    if args.dataset == "cub":
        train_set = CUBDataset([str(Path.cwd()/"data/cub/CUB_processed/class_attr_data_10/train.pkl")],
                               use_attr=True, no_img=False, uncertain_label=False, n_class_attr=2,
                               image_dir=str(Path.cwd()/f"data/cub/CUB_200_2011"))
        concept_categories = train_set.get_concept_categories()

    else:
        concept_categories = None
    if args.name == "concept_accuracy":
        plot_concept_accuracy(save_path, args.concept, args.dataset)
    elif args.name == "global_explanations":
        plot_global_explanation(save_path, args.dataset, concept_categories=concept_categories)
    elif args.name == "feature_importance":
        plot_attribution_correlation(save_path, args.dataset)
    elif args.name == "concept_modulation":
        plot_modulation_impact(save_path, args.dataset)
    else:
        raise ValueError(f"{args.name} is not a valid experiment name")



