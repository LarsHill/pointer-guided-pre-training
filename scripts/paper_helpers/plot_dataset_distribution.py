import json
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import rcParams

from llm import project_path


def plot_class_distribution(datasets):
    sns.set(style="whitegrid")

    # Ensure that the Computer Modern font is used
    rcParams["font.family"] = "serif"
    rcParams["font.serif"] = "Computer Modern Roman"
    rcParams["text.usetex"] = True
    rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"

    tick_label_size = 18
    font_size = 20
    dataset_names = [r"GRI$_{\text{DE}}$", r"IFRS$_{\text{EN}}$", "CSAbstruct", "PubMed-RCT", "Nicta"]

    # Create a figure
    fig_width = 12  # Fixed width, you can adjust this as needed
    fig_height = 3.5 * (1 + (len(datasets) - 2) // 3)  # Calculate height based on the number of datasets
    plt.figure(figsize=(fig_width, fig_height))

    # Create a custom grid layout
    axes = [plt.subplot2grid((2, 6), (0, 0), colspan=3), plt.subplot2grid((2, 6), (0, 3), colspan=3)]
    # First row, each plot spans all 3 columns
    # Second row, each plot spans 1 column
    for i in range(2, len(datasets)):
        row = 1  # Second row
        col = (i - 2) % 3
        axes.append(plt.subplot2grid((2, 6), (row, col * 2), colspan=2))

    for i, (dataset_name, class_counts) in enumerate(datasets.items()):
        # Sort the classes and limit to the first 100
        sorted_classes = sorted(class_counts.items(), key=lambda item: item[1], reverse=True)[:100]
        classes, counts = zip(*sorted_classes)
        total_count = sum(counts)
        percentages = [count / total_count * 100 for count in counts]

        ax = axes[i]
        sns.barplot(
            x=np.arange(len(classes)),
            y=percentages,
            ax=ax,
            palette="plasma",
            legend=False,
        )

        ax.set_title(f"{dataset_names[i]}", fontsize=font_size)

        # Explicitly set the tick label font size
        for label in ax.get_xticklabels():
            label.set_fontsize(tick_label_size)
        for label in ax.get_yticklabels():
            label.set_fontsize(tick_label_size)

        # Show class names only if there are 10 or fewer classes, otherwise skip
        if len(classes) <= 10:
            ax.set_xticklabels(
                [cls.lower().capitalize() for cls in classes], rotation=35, ha="right", fontsize=font_size
            )
        else:
            ax.set_xticks([])
            ax.set_xlabel(
                (
                    f"Classes (Top 100 of {len(class_counts)})"
                    if "ifrs" in dataset_name
                    else f"Classes ({len(class_counts)})"
                ),
                fontsize=font_size,
            )

        ax.set_ylabel(r"Class Distribution (\%)", fontsize=font_size)

    plt.tight_layout()

    # Save the figure as a PDF
    plot_dir = os.path.join(project_path, "data", "plots")
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(os.path.join(plot_dir, "fine_tuning_class_distribution.pdf"), bbox_inches="tight")

    plt.show()


def main():
    stats_dir = os.path.join(project_path, "data", "finetuning_stats")

    stats = json.load(open(os.path.join(stats_dir, "all_stats.json"), "r"))

    dataset_class_stats = {}
    for name, data_stats in stats.items():
        dataset_class_stats[name] = data_stats["all"]["label_to_support"]

    plot_class_distribution(dataset_class_stats)


if __name__ == "__main__":
    main()
