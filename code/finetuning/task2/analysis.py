import os
import json 
import scipy

import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 

from typing import Dict


def parse_schema(file_path: str) -> Dict[str, str]:
    """Return wikidata schema.
    Args:
        file_path: schema filepath
    Returns:
        schema: a python dict whose key is a concept and the value is 
                the subClassOf the key . 
                e.g., {"soccerPlayer": "Player"}
    """
    schema = {}
    with open(file_path) as f:
        for line in f.readlines():
            child, parent = line.strip().split("\t\t")
            assert child not in schema
            schema[child] = parent
    return schema


def flatten_schema(schema: Dict[str, str]) -> Dict[str, str]:
    flat_schema = {}
    for child in schema.keys():
        parent = schema[child]
        if parent != "Root":
            while True:
                if schema[parent] == "Root":
                    break
                parent = schema[parent]
        else:
            parent = child
        flat_schema[child] = parent
    return flat_schema


def analyze_fasle_negative():
    model_names = ["bert-base-uncased", "roberta-base",
        "gpt2", "gpt-neo-125M",
        "bart-base", "t5-base"
    ]
    for model_name in model_names:
        rate = 0
        for seed in ["42", "43", "44"]:
            input_path = os.path.join(f"output/FT/{seed}", f"{model_name}.json")
            data = json.load(open(input_path))
            total_error, false_pos = 0, 0
            for item in data:
                if item["label"] != item["pred"]:
                    total_error += 1
                if item["label"] == 0 and item["pred"] == 1:
                    false_pos += 1
            rate += false_pos/total_error
        rate /= 3
        print(model_name, round(rate*100, 1))


def number_to_key(number, bucket_size=2):
    if number > 40:
        return ">40"
    else:
        base = int((number // bucket_size) * bucket_size)
        return f"{base}-{base+bucket_size}"


def _analyze_bias(data, searched_data):
    bucket_size = 2
    bins = []
    for i in range(0, 20):
        bins.append(f"{i*bucket_size}-{(i+1)*bucket_size}")
    bins.append(">40")
    bm25_to_acc = dict()
    for bin in bins:
        bm25_to_acc[bin] = dict(total=0, error=0, scores=[])
    neg_data = []
    for item in data:
        if item["label"] == 0:
            neg_data.append(item)
    for i in range(len(neg_data)):
        assert neg_data[i]["text"] == searched_data[i]["text"]
        assert neg_data[i]["concept"]["id"] == searched_data[i]["concept"]["id"]
        assert neg_data[i]["label"] == searched_data[i]["label"]
        if len(searched_data[i]["score"]) == 0:
            bm25_to_acc[bins[0]]["total"] += 1
            if neg_data[i]["pred"] == 1:
                bm25_to_acc[bins[0]]["error"] += 1
            continue
        bm25_to_acc[number_to_key(searched_data[i]["score"][0])]["scores"].append(searched_data[i]["score"][0])
        bm25_to_acc[number_to_key(searched_data[i]["score"][0])]["total"] += 1
        if neg_data[i]["pred"] == 1:
            bm25_to_acc[number_to_key(searched_data[i]["score"][0])]["error"] += 1
    for key in bm25_to_acc.keys():
        bm25_to_acc[key]["error-rate"] = bm25_to_acc[key]["error"] / bm25_to_acc[key]["total"]
    return bm25_to_acc


def rsquared(x, y):
    """ Return R^2 where x and y are array-like."""

    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
    return r_value**2, p_value


def visualize_error_rate_on_buckets(data, save_path: str):
    """
    Args:
        data: A python dict with the following format:
            {   
                "x": [],
                "y": []
            }
    """
    sns.set_style("ticks")
    # sns.set_context("paper")
    # palette = sns.color_palette("pastel")
    palette = sns.color_palette()
    palette = [palette[0], palette[1]]
    ax = sns.regplot(data=data, x="x", y="y", color=palette[0],
                    line_kws={"linewidth":3.5})
    sns.despine()
    r_square, p_value = rsquared(data["x"], data["y"])
    ax.text(5, 0.80*100, "$R^2$=%.2f, p=%.2f $\\times$ $10^{%d}$" % (round(r_square, 2), convert_format(p_value)[0], convert_format(p_value)[1]), fontsize=16)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    ax.set_ylim(ymin=10)
    ax.set_xlabel("BM25 score", fontsize=20)
    ax.set_ylabel("False Positive Rate (%)", fontsize=20)
    plt.savefig(save_path, dpi=300, format="pdf", bbox_inches='tight', pad_inches=0.0)
    plt.close()


def average(data_list):
    total = 0
    for item in data_list:
        total += item 
    return total / len(data_list)


def plot_error_rate():
    model_name = "bert-base-uncased"
    data = json.load(open(f"output/FT/42/{model_name}.json"))
    searched_data = json.load(open("../../../data/task2/data/search/searched_data.json"))
    bm25_to_acc = _analyze_bias(data, searched_data)
    print(bm25_to_acc)
    data = {
        "x": [],
        "y": []
    }
    for key in bm25_to_acc:
        data["x"].append(average(bm25_to_acc[key]["scores"]))
        data["y"].append(bm25_to_acc[key]["error-rate"] * 100)
    data["x"] = np.array(data["x"])
    data["y"] = np.array(data["y"])
    visualize_error_rate_on_buckets(data, "error-rate.pdf")


def convert_format(num):
    _pow = 0
    while num < 1:
        _pow -= 1
        num = num * 10
    return round(num, 2), _pow


def visualize_all_error_rate_on_buckets(data, save_path: str):
    """
    Args:
        data: A python dict with the following format:
            {   
                model_name: {
                    "x": [],
                    "y": []
                }
            }
    """
    model_name_to_formal = {
        "bert-base-uncased": "$\mathregular{BERT_{BASE}}$",
        "roberta-base": "$\mathregular{RoBERTa_{BASE}}$",
        "gpt2": "$\mathregular{{GPT-2}_{BASE}}$",
        "gpt-neo-125M": "$\mathregular{{GPT-Neo}_{BASE}}$",
        "bart-base": "$\mathregular{BART_{BASE}}$",
        "t5-base": "$\mathregular{T5_{BASE}}$",
    }
    sns.set_style("ticks")
    # sns.set_context("paper")
    # palette = sns.color_palette("pastel")
    palette = sns.color_palette()
    palette = [palette[0], palette[1]]
    fig, axes = plt.subplots(2, 3, sharex=True, figsize=(25, 13))
    for i, model_name in enumerate(data.keys()):
        row = int(i % 2)
        col = int(i // 2)
        ax = sns.regplot(data=data[model_name], x="x", y="y", color=palette[0], ax=axes[row][col],
                        line_kws={"linewidth":3.5})
        sns.despine()
        ax.set_xlabel("BM25 score", fontsize=20)
        ax.set_ylabel("False Positive Rate (%)", fontsize=20)
        r_square, p_value = rsquared(data[model_name]["x"], data[model_name]["y"])
        y_pos = 0.61
        if model_name in ["gpt-neo-125M", "gpt2"]:
            y_pos = 0.5
        elif model_name in ["bert-base-uncased", "bart-base"]:
            y_pos = 0.8
        ax.text(5, y_pos*100, "$R^2$=%.2f, p=%.2f $\\times$ $10^{%d}$" % (round(r_square, 2), convert_format(p_value)[0], convert_format(p_value)[1]), fontsize=18)
        ax.tick_params(axis="x", labelsize=18) 
        ax.tick_params(axis="y", labelsize=18) 
        ax.set_ylim(ymin=5)
        ax.set_title(model_name_to_formal[model_name], {"fontsize": 18, "fontweight": "bold"})

    # plt.xticks(fontsize=18)
    # plt.yticks(fontsize=18)
    plt.savefig(save_path, dpi=300, format="pdf", bbox_inches='tight', pad_inches=0.0)
    plt.close()

def plot_all_error_rate():
    model_names = [
        "bert-base-uncased",
        "roberta-base",
        "gpt2",
        "gpt-neo-125M",
        "bart-base",
        "t5-base"
    ]
    all_plot_data = {}
    for model_name in model_names:
        data = json.load(open(f"output/FT/42/{model_name}.json"))
        searched_data = json.load(open("../../../data/task2/data/search/searched_data.json"))
        bm25_to_acc = _analyze_bias(data, searched_data)
        print(bm25_to_acc)
        data = {
            "x": [],
            "y": []
        }
        for key in bm25_to_acc:
            data["x"].append(average(bm25_to_acc[key]["scores"]))
            data["y"].append(bm25_to_acc[key]["error-rate"]*100)
        data["x"] = np.array(data["x"])
        data["y"] = np.array(data["y"])
        all_plot_data[model_name] = data 
    visualize_all_error_rate_on_buckets(all_plot_data, "all-error-rate.pdf")


if __name__ == "__main__":
    # analyze_fasle_negative()
    plot_error_rate()
    plot_all_error_rate()


