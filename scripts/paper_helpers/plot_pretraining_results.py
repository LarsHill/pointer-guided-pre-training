import glob
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib import rcParams

from llm import project_path

# rc("font", **{"family": "serif", "serif": ["Computer Modern"]})
# rc("text", usetex=True)
# plt.style.use("ggplot")

run_to_model_name = {
    "pristine-arm-000": r"PointerBERT$_\text{wiki-en}$",
    "aged-computer-000": r"BERT$_\text{wiki-en}$",
    "aged-computer-002": r"RoBERTa$_\text{wiki-en}$",
    "steep-collar-000": r"PointerBERT$_\text{all-de}$",
    "shabby-sink-000": r"PointerSciBERT$_\text{wiki-en}$",
    "muffled-newspaper-000": r"PointerBERT$_\text{all}$",
    "muffled-newspaper-001": r"RoBERTa$_\text{all}$",
}


def plot_pretraining(pretraining_data):
    sns.set(style="whitegrid")

    # Ensure that the Computer Modern font is used
    rcParams["font.family"] = "serif"
    rcParams["font.serif"] = "Computer Modern Roman"
    rcParams["font.size"] = 18
    rcParams["text.usetex"] = True
    rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"

    tick_label_size = 18
    font_size = 20

    # Assuming pretraining_data is already defined as in your example and each <pd.Series> has an index that represents the training steps

    # Set up the 2x2 grid of subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex="col")

    # Flatten the axes array for easy indexing
    axes = axes.flatten()

    # Define the metrics and their positions in the grid
    metrics = [("mlm_loss", 0), ("so_loss", 1), ("mlm_acc", 2), ("so_acc", 3)]

    # Define a color palette for the models
    palette = sns.color_palette("muted", len(pretraining_data["mlm_loss"]))
    model_to_color = {model_name: color for color, model_name in zip(palette, pretraining_data["mlm_loss"])}

    # Plot each metric in its respective subplot
    for metric, idx in metrics:
        for model_idx, (model_name, series) in enumerate(pretraining_data[metric].items()):
            sns.lineplot(
                x=series.index,
                y=series.values,
                ax=axes[idx],
                label=run_to_model_name[model_name],
                color=model_to_color[model_name],
                linewidth=2,
            )

    # # Set the titles for each subplot
    axes[0].set_title("Masked Language Modeling (MLM)", fontsize=font_size)
    axes[1].set_title("Segment Ordering (SO)", fontsize=font_size)
    # axes[2].set_title("SO Loss", fontsize=font_size)
    # axes[3].set_title("SO Accuracy", fontsize=font_size)

    # Set the x-axis label only for the bottom subplots
    axes[2].set_xlabel("Training Steps (K)", fontsize=font_size)
    axes[3].set_xlabel("Training Steps (K)", fontsize=font_size)

    # Set the y-axis label for the left subplots
    axes[0].set_ylabel("Loss", fontsize=font_size)
    axes[2].set_ylabel("Accuracy", fontsize=font_size)

    # Set the y-axis label for the right subplots
    axes[1].set_ylabel("Loss", fontsize=font_size)
    axes[3].set_ylabel("Accuracy", fontsize=font_size)

    # # Hide x-axis labels for the top subplots
    # axes[0].get_xaxis().set_visible(False)
    # axes[1].get_xaxis().set_visible(False)

    # Enable the grid on all subplots for the x-axis and set the tick parameters
    for ax in axes:
        ax.grid(True, axis="x")
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))  # Ensure integer ticks

    # Hide the x-axis labels on the top row subplots
    for ax in axes[:2]:
        ax.tick_params(labelbottom=False)

    # Remove the legends from each subplot
    for ax in axes:
        ax.get_legend().remove()

    for ax in axes:
        # Explicitly set the tick label font size
        for label in ax.get_xticklabels():
            label.set_fontsize(tick_label_size)
        for label in ax.get_yticklabels():
            label.set_fontsize(tick_label_size)

    # Adjust the layout to make space for the legend below the subplots
    fig.subplots_adjust(bottom=0.4, hspace=0.3, wspace=0.3)

    # Create a legend below the subplots
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0), fontsize=18)

    # Adjust the layout to prevent overlapping
    plt.tight_layout(rect=[0, 0.17, 1, 1])  # rect=[0, 0.05, 1, 1]

    # Save the figure as a PDF
    # plot_dir = "/scratch/data/language-model-training/plots"
    plot_dir = os.path.join(project_path, "data", "plots")
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(os.path.join(plot_dir, "pretraining_results.pdf"), bbox_inches="tight")

    # Show the plot
    plt.show()


def main():
    csv_dir = os.path.join(project_path, "data", "plots")
    paths = glob.glob(os.path.join(csv_dir, "valid*.csv"))

    index_multiplier = 1250

    pretraining_data = {"mlm_loss": {}, "so_loss": {}, "nsp_loss": {}, "mlm_acc": {}, "so_acc": {}, "nsp_acc": {}}
    for path in paths:
        df = pd.read_csv(path)
        metric_name = df.columns[1].split(" - ")[1]
        try:
            metric, task = metric_name.removesuffix("-epoch").removeprefix("valid-").split("--")
        except ValueError:
            task, metric = metric_name.removesuffix("-epoch").removeprefix("valid-").split("-")
        task_metric = f"{task}_{metric}"

        for model_name in df.columns[1:]:
            if "__" not in model_name:
                model = model_name.split(" - ")[0]
                data = df[model_name]
                pretraining_data[task_metric][model] = data.dropna().reset_index(drop=True)
                pretraining_data[task_metric][model].index *= 1.250

    plot_pretraining(pretraining_data)

    # print()
    #
    # pretraining_data = {
    #     "mlm_loss": {"model_a": <pd.Series>, "model_b": <pd.Series>},
    #     "so_loss": {...},
    #     "mlm_acc": {...},
    #     "so_acc": {...},
    # }


if __name__ == "__main__":
    main()
