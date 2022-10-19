import os 
import math 
import json
from typing import Dict, List
import matplotlib

import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt 
import matplotlib.patches as  mpatches
import matplotlib.lines as mlines

from pathlib import Path



def visualize_model_size(data: pd.DataFrame, save_path: str) -> None:
    """Plot results of PLMs at different scales.

    Args:
        data: 
                model_name, setting, model_size, score, task_name
            0   bert        FT       110M        78.00, task1 
            1   bert        LP       110M        64.00, task1
            .
            .
            .
        save_path: The save path of the figure.
    
    Returns:
        None
    """
    task_name_to_formal = {
        "task1": "Concept Similarity Judgment",
        "task2": "Concept Property Judgment",
        "task3": "Conceptualization in Contexts"
    }
    # sns.set_style("ticks")
    palette = sns.color_palette()
    colors = [palette[1], palette[3], palette[0]]
    markers = ["^", "o", "X"]
    # colors = palette[1:]
    fig, axes = plt.subplots(1, 3, sharex=True, figsize=(25, 8))
    for i, task_name in enumerate(data.keys()):
        sns.lineplot(data=data[task_name], x="model_size", y="score", hue="model_name", style="setting",
                    dashes=False, markers=markers, palette=colors, legend=False, ax=axes[i], seed=42,
                    **{"markersize": 14, "linewidth": 3})
        sns.despine()
        axes[i].set_xlabel("Millions of parameters", fontsize=24)
        axes[i].set_ylabel("Accuracy (%)", fontsize=24)
        model_sizes = [20, 50, 100, 300, 1000, 3000, 11000]
        axes[i].set_xticks(ticks=[math.log10(item) for item in model_sizes])
        model_size_str = ["20", "50", "100", "300", "1,000", "3,000", "11,000"]
        axes[i].set_xticklabels(model_size_str, fontsize=22)
        # axes[i].set_yticklabels(labels=axes[i].get_yticks(), fontsize=16)
        axes[i].tick_params(axis="y", labelsize=24)
        axes[i].set_title(task_name_to_formal[task_name], {"fontsize": 24, "fontweight": "bold"})
    handles = [
        mlines.Line2D([], [], marker=markers[0], color="0", linestyle='None', markersize=20, label='Zero-shot Probing'),
        mlines.Line2D([], [], marker=markers[2], color="0", linestyle='None', markersize=20, label='Linear Probing'),
        mlines.Line2D([], [], marker=markers[1], color="0", linestyle='None', markersize=20, label='Fine-tuning'),
        mpatches.Patch(facecolor=colors[0], label="BERT"),
        mpatches.Patch(facecolor=colors[1], label="GPT-2"),
        mpatches.Patch(facecolor=colors[2], label="T5")
    ]
    fig.legend(handles=handles, loc="lower center", frameon=False, ncol=6, prop={'size': 24}, bbox_to_anchor=(0.5, -0.1))
    plt.savefig(save_path, dpi=300, format="pdf", bbox_inches='tight', pad_inches=0.0)
    plt.close()


def prepare_data_for_model_size(data_path: str) -> pd.DataFrame:
    """Prepare results for plotting.

    Args:
        data_path: The data path of overall results. 
    
    Returns:
        A Dataframe.
    """
    data = json.load(open(data_path))
    model_name_to_params = {
        "bert-small": 29,
        "bert-medium": 42,
        "bert-base-uncased": 110,
        "bert-large-uncased": 330,
        "gpt2": 125,
        "gpt2-medium": 355,
        "gpt2-large": 774,
        "gpt2-xl": 1500,
        "t5-small": 60,
        "t5-base": 220,
        "t5-large": 770,
        "t5-3b": 3000,
        "t5-11b": 11000
    }
    plot_model_names = [
        "bert-small", "bert-medium", "bert-base-uncased", "bert-large-uncased",
        "gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl",
        "t5-small", "t5-base", "t5-large", "t5-3b", "t5-11b"
    ]
    plot_data = dict()
    for task_name in data.keys():
        task_data = {
            "model_name": [],
            "setting": [],
            "model_size": [],
            "score": [],
        }
        for model_name in plot_model_names:
            for setting in [["PB", "PB"], ["ood-FT-mean", "FT"], ["ood-LP-mean", "LP"]]:
                # if setting[0] == "ood-LP-mean" and int(data[task_name][model_name][setting[0]]) == -1:
                #     continue
                # if setting[0] == "PB":
                #     if (task_name == "task1" and int(data[task_name][model_name][setting[0]]["hard"]["topk"]["1"]) == -1) \
                #         or (task_name == "task2" and int(data[task_name][model_name][setting[0]]["accuracy"]) == -1) \
                #             or (task_name == "task3" and int(data[task_name][model_name][setting[0]]["hard_accuracy"]) == -1):
                #             continue
                task_data["model_name"].append(model_name.split("-")[0].upper())
                task_data["setting"].append(setting[1])
                task_data["model_size"].append(math.log10(model_name_to_params[model_name]))  
                if setting[0] == "PB":
                    if task_name == "task1":
                        task_data["score"].append(data[task_name][model_name][setting[0]]["hard"]["topk"]["1"])
                    elif task_name == "task2":
                        task_data["score"].append(data[task_name][model_name][setting[0]]["accuracy"])
                    else:
                        task_data["score"].append(data[task_name][model_name][setting[0]]["hard_accuracy"])
                else:
                    task_data["score"].append(data[task_name][model_name][setting[0]])
        plot_data[task_name] = task_data
    return plot_data


def visualize_iid(data: Dict[str, List], save_path: str) -> None:
    """Plot results of data with conceptual overlap.

    Args:
        data: A Python dict with the following format:
            {
                "Task": [],
                "Accuracy": [],
                "Data_Type": []
            }
        save_path: The save path.
    
    Returns:
        None 
    """
    sns.set_style("ticks")
    palette = sns.color_palette()
    palette = [palette[0], palette[1]]
    ax = sns.barplot(data=data, x="Task", y="Accuracy", hue="Data_Type", palette=palette)
    ax.set_ylim(ymin=25)
    sns.despine()
    handles = [mpatches.Patch(facecolor=palette[0], label="w/o concept overlap"),
           mpatches.Patch(facecolor=palette[1], label="w/ concept overlap")]
    plt.legend(handles=handles, frameon=False, ncol=1, loc="upper right", prop={'size': 16}, bbox_to_anchor=(1, 1.14))
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    ax.set_xlabel("", fontsize=20)
    ax.set_ylabel("Accuracy (%)", fontsize=20)
    plt.savefig(save_path, dpi=300, format="pdf", bbox_inches='tight', pad_inches=0.0)
    plt.close()


def plot_iid() -> None:
    """Plot results on conceptual overlap data of all three tasks.
    """
    model_name = "bert-base-uncased"
    task_name_to_formal = {
        "task1": "CSJ",
        "task2": "CPJ",
        "task3": "CiC"
    }
    plot_data = dict(Task=[], Accuracy=[], Data_Type=[])
    data = json.load(open("results/result.json"))
    for task in ["task1", "task2", "task3"]:
        plot_data["Task"].append(task_name_to_formal[task])
        plot_data["Accuracy"].append(data[task][model_name]["ood-FT-mean"])
        plot_data["Data_Type"].append("Benchmark")
        plot_data["Task"].append(task_name_to_formal[task])
        plot_data["Accuracy"].append(data[task][model_name]["iid-FT-mean"])
        plot_data["Data_Type"].append("Concept Overlap")
    visualize_iid(plot_data, "figures/iid.pdf")


if __name__ == "__main__":
    output_dir = Path("figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = os.path.join(output_dir, "model_scale.pdf")
    data = prepare_data_for_model_size("results/result.json")
    visualize_model_size(data, save_path)
    # plot_iid()



