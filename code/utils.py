from collections import defaultdict
import enum
import os 
import pdb 
import csv 
import json
from pathlib import Path
from pyexpat import model 
import numpy as np 


model_name_to_formal = {
    "bert-small": "\\BERTsmall",
    "bert-medium": "\\BERTmedium",
    "bert-base-uncased": "\\BERTbase",
    "bert-large-uncased": "\\BERTlarge",
    "roberta-base": "\\Rbase",
    "gpt2": "\\GPTbase",
    "gpt2-medium": "\\GPTmedium",
    "gpt2-large": "\\GPTlarge",
    "gpt2-xl": "\\GPTxl",
    "gpt-neo-125M": "\\GPTNeobase",
    "bart-base": "\\BARTbase",
    "t5-small": "\\Tsmall",
    "t5-base": "\\Tbase",
    "t5-large": "\\Tlarge",
    "t5-3b": "\\Txl",
    "t5-11b": "\\Txxl"
}

main_names = [
    "bert-base-uncased", "roberta-base", "gpt2", "gpt-neo-125M", "bart-base", "t5-base"
]


def merge_result():
    """
    Returns:
        {
            task_name: {
                model_name:{
                    "PB": xx,
                    "ood-FT": [],
                    "ood-LP": [],
                    "ood-FT-mean": xx,
                    "ood-FT-std": xx,
                    "ood-LP-mean": xx,
                    "ood-LP-std: xx
                }
            }
        }
    """
    model_names = [
        "bert-small",  "bert-medium", "bert-base-uncased", "bert-large-uncased",
        "roberta-base", 
        "gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl",
        "gpt-neo-125M",
        "bart-base",
        "t5-small", "t5-base", "t5-large", "t5-3b", "t5-11b"
    ]
    model_results = dict()
    for task_name in ["task1", "task2", "task3"]:
        model_results[task_name] = defaultdict(dict)
        probing_dir = f"probing/{task_name}/output/results"
        for model_name in model_names:
            model_probing_results = json.load(open(os.path.join(probing_dir, model_name+".json")))
            probing_result = None 
            if task_name == "task1":
                if model_name in ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl", "gpt-neo-125M"]:
                    probing_result = model_probing_results["all"]
                else:
                    probing_result = model_probing_results["e1"]
            elif task_name == "task2":
                if model_name in ["gpt2", "bert-small", "t5-3b", "t5-11b"]:
                    probing_result = model_probing_results["all"]
                else:
                    probing_result = model_probing_results["concept"]
                if model_name in main_names:
                    if model_name in ["bart-base"]:
                        consistency = model_probing_results["all"]["consistency"]["overall"]["consistency"]
                    elif model_name in ["bert-small"]:
                        consistency = model_probing_results["concept"]["consistency"]["overall"]["consistency"]
                    else:
                        consistency = model_probing_results["answer"]["consistency"]["overall"]["consistency"]
                    model_results[task_name][model_name]["PB-chain"] = consistency * 100
            elif task_name == "task3":
                if model_name in ["bert-small", "bert-medium", "bert-base-uncased", "bert-large-uncased", "gpt2-medium", "gpt2-xl", "t5-small", "t5-base", "t5-large", "t5-3b", "t5-11b"]:
                    probing_result = model_probing_results["all"]
                else:
                    probing_result = model_probing_results["concept"]
            model_results[task_name][model_name]["PB"] = probing_result
            for seed_idx, seed in enumerate([42, 43, 44]):
                tuning_dir = f"finetuning/{task_name}/results/{seed}"
                tuning_results = json.load(open(os.path.join(tuning_dir, model_name+".json")))
                if seed_idx == 0:
                    model_results[task_name][model_name]["ood-FT"] = [float(tuning_results["ood-FT"])]
                    model_results[task_name][model_name]["ood-LP"] = [float(tuning_results["ood-LP"])]
                    if "iid-FT" in tuning_results:
                        model_results[task_name][model_name]["iid-FT"] = [float(tuning_results["iid-FT"])]
                        model_results[task_name][model_name]["iid-LP"] = [float(tuning_results["iid-LP"])]
                    if task_name == "task2" and model_name in main_names:
                        model_results[task_name][model_name]["ood-FT-chain"] = [tuning_results["ood-FT_consistency"]["overall"]["consistency"]*100]
                        model_results[task_name][model_name]["ood-LP-chain"] = [tuning_results["ood-LP_consistency"]["overall"]["consistency"]*100]
                else:
                    model_results[task_name][model_name]["ood-FT"].append(float(tuning_results["ood-FT"]))
                    model_results[task_name][model_name]["ood-LP"].append(float(tuning_results["ood-LP"]))
                    if "iid-FT" in tuning_results:
                        model_results[task_name][model_name]["iid-FT"].append(float(tuning_results["iid-FT"]))
                        model_results[task_name][model_name]["iid-LP"].append(float(tuning_results["iid-LP"]))
                    if task_name == "task2" and model_name in main_names:
                        model_results[task_name][model_name]["ood-FT-chain"].append(tuning_results["ood-FT_consistency"]["overall"]["consistency"]*100)
                        model_results[task_name][model_name]["ood-LP-chain"].append(tuning_results["ood-LP_consistency"]["overall"]["consistency"]*100)
                if seed_idx == 2:
                    model_results[task_name][model_name]["ood-FT-mean"] = np.mean(np.array(model_results[task_name][model_name]["ood-FT"]))
                    model_results[task_name][model_name]["ood-FT-std"] = np.std(np.array(model_results[task_name][model_name]["ood-FT"]))
                    model_results[task_name][model_name]["ood-LP-mean"] = np.mean(np.array(model_results[task_name][model_name]["ood-LP"]))
                    model_results[task_name][model_name]["ood-LP-std"] = np.std(np.array(model_results[task_name][model_name]["ood-LP"]))
                    if "iid-FT" in tuning_results:
                        model_results[task_name][model_name]["iid-FT-mean"] = np.mean(np.array(model_results[task_name][model_name]["iid-FT"]))
                        model_results[task_name][model_name]["iid-FT-std"] = np.std(np.array(model_results[task_name][model_name]["iid-FT"]))
                        model_results[task_name][model_name]["iid-LP-mean"] = np.mean(np.array(model_results[task_name][model_name]["iid-LP"]))
                        model_results[task_name][model_name]["iid-LP-std"] = np.std(np.array(model_results[task_name][model_name]["iid-LP"]))
                    if task_name == "task2" and model_name in main_names:
                        model_results[task_name][model_name]["ood-FT-chain-mean"] = np.mean(np.array(model_results[task_name][model_name]["ood-FT-chain"]))
                        model_results[task_name][model_name]["ood-FT-chain-std"] = np.std(np.array(model_results[task_name][model_name]["ood-FT-chain"]))
                        model_results[task_name][model_name]["ood-LP-chain-mean"] = np.mean(np.array(model_results[task_name][model_name]["ood-LP-chain"]))
                        model_results[task_name][model_name]["ood-LP-chain-std"] = np.std(np.array(model_results[task_name][model_name]["ood-LP-chain"]))
    json.dump(model_results, open("results/result.json", "w"), indent=4)


def merge_main():
    model_names = [
        "bert-base-uncased",
        "roberta-base",
        "gpt2",
        "gpt-neo-125M",
        "bart-base",
        "t5-base",
    ]
    model_name_to_latex = {
        "bert-base-uncased": ["BERT", "110M"],
        "roberta-base": ["RoBERTa", "125M"],
        "gpt2": ["GPT-2", "117M"],
        "gpt-neo-125M": ["GPT-Neo", "125M"],
        "bart-base": ["BART", "140M"],
        "t5-base": ["T5", "220M"],
    }
    round_number = 1
    save_name = "results/main.latex"
    all_results = json.load(open("results/result.json"))
    with open(save_name, "w") as f:
        for model_name in model_names:
            model_result = dict()
            for task_name in ["task1", "task2", "task2-chain", "task3"]:
                model_result[task_name] = {
                    "probing": -1,
                    "FT-mean": -1,
                    "FT-std": -1,
                    "LP-mean": -1,
                    "LP-std": -1,
                }
                if task_name == "task2-chain":
                    model_result[task_name]["probing"] = round(all_results["task2"][model_name]["PB-chain"], round_number)
                    model_result[task_name]["LP-mean"] = round(all_results["task2"][model_name]["ood-LP-chain-mean"], round_number)
                    model_result[task_name]["LP-std"] = round(all_results["task2"][model_name]["ood-LP-chain-std"], 2)
                    model_result[task_name]["FT-mean"] = round(all_results["task2"][model_name]["ood-FT-chain-mean"], round_number)
                    model_result[task_name]["FT-std"] = round(all_results["task2"][model_name]["ood-FT-chain-std"], 2)
                    continue
                if task_name == "task1":
                    model_result[task_name]["probing"] = round(all_results[task_name][model_name]["PB"]["hard"]["topk"]["1"], round_number)
                elif task_name == "task2":
                    model_result[task_name]["probing"] = round(all_results[task_name][model_name]["PB"]["accuracy"], round_number)
                elif task_name == "task3":
                    model_result[task_name]["probing"] = round(all_results[task_name][model_name]["PB"]["hard_accuracy"], round_number)
                model_result[task_name]["LP-mean"] = round(all_results[task_name][model_name]["ood-LP-mean"], round_number)
                model_result[task_name]["LP-std"] = round(all_results[task_name][model_name]["ood-LP-std"], 2)
                model_result[task_name]["FT-mean"] = round(all_results[task_name][model_name]["ood-FT-mean"], round_number)
                model_result[task_name]["FT-std"] = round(all_results[task_name][model_name]["ood-FT-std"], 2)
            f.write(model_name_to_formal[model_name]+"&")
            latex_result = []
            for key in model_result.keys():
                # latex_result.append("")
                latex_result.append("$" + str(model_result[key]["probing"]) + "$")
                latex_result.append("$" + str(model_result[key]["LP-mean"]) + "_{\\textsc{%.2f}}" % model_result[key]["LP-std"] + "$")
                latex_result.append("$" + str(model_result[key]["FT-mean"]) + "_{\\textsc{%.2f}}" % model_result[key]["FT-std"] + "$")
            f.write("&".join(latex_result))
            f.write("\\\\")
            f.write("\n")


def merge_probing():
    model_names = [
        "bert-small",
        "bert-medium",
        "bert-base-uncased",
        "bert-large-uncased",
        "roberta-base",
        "gpt2",
        "gpt2-medium",
        "gpt2-large",
        "gpt2-xl",
        "gpt-neo-125M",
        "bart-base",
        "t5-small",
        "t5-base",
        "t5-large",
        "t5-3b",
        "t5-11b"
    ]
    round_number = 1
    save_name = "results/all_probing.latex"
    with open(save_name, "w") as f:
        for model_name in model_names:
            model_result = []
            for task_name in ["task1", "task2", "task3"]:
                all_results = json.load(open(f"probing/{task_name}/output/results/{model_name}.json"))
                # parse 
                if task_name == "task1":
                    model_result.append(round(all_results["e1"]["hard"]["topk"]["1"], round_number))
                    model_result.append(round(all_results["e2"]["hard"]["topk"]["1"], round_number))
                    model_result.append(round(all_results["all"]["hard"]["topk"]["1"], round_number))
                elif task_name == "task2":
                    model_result.append(round(all_results["concept"]["accuracy"], round_number))
                    model_result.append(round(all_results["answer"]["accuracy"], round_number))
                    model_result.append(round(all_results["all"]["accuracy"], round_number))
                elif task_name == "task3":
                    model_result.append(round(all_results["concept"]["hard_accuracy"], round_number))
                    model_result.append(round(all_results["all"]["hard_accuracy"], round_number))
            f.write(model_name_to_formal[model_name]+"&")
            for i in range(len(model_result)):
                model_result[i] = "$" + str(model_result[i]) + "$"
            f.write("&".join(model_result))
            f.write("\\\\")
            f.write("\n")


def merge_finetuning():
    model_names = [
        "bert-small",
        "bert-medium",
        "bert-base-uncased",
        "bert-large-uncased",
        "roberta-base",
        "gpt2",
        "gpt2-medium",
        "gpt2-large",
        "gpt2-xl",
        "gpt-neo-125M",
        "bart-base",
        "t5-small",
        "t5-base",
        "t5-large",
        "t5-3b",
        "t5-11b"
    ]
    round_number = 1
    all_results = json.load(open("results/result.json"))
    for task_name in ["task1", "task2", "task3"]:
        save_name = f"results/finetuning_{task_name}.latex"
        with open(save_name, "w") as f:
            for model_name in model_names:
                if model_name not in all_results[task_name]:
                    continue
                model_result = []
                model_result.extend([round(acc, round_number) for acc in all_results[task_name][model_name]["ood-LP"]])
                model_result.append(round(all_results[task_name][model_name]["ood-LP-mean"], round_number))
                model_result.append(round(all_results[task_name][model_name]["ood-LP-std"], 2))
                model_result.extend([round(acc, round_number) for acc in all_results[task_name][model_name]["ood-FT"]])
                model_result.append(round(all_results[task_name][model_name]["ood-FT-mean"], round_number))
                model_result.append(round(all_results[task_name][model_name]["ood-FT-std"], 2))
                f.write(model_name_to_formal[model_name]+"&")
                for i in range(len(model_result)):
                    model_result[i] = "$" + str(model_result[i]) + "$"
                f.write("&".join(model_result))
                f.write("\\\\")
                f.write("\n")


def merge_iid():
    model_names = [
        "bert-base-uncased",
        "roberta-base",
        "gpt2",
        "gpt-neo-125M",
        "bart-base",
        "t5-base"
    ]
    all_results = json.load(open("results/result.json"))
    for setting in ["FT", "LP"]:
        save_name = f"results/iid_{setting}.latex"
        with open(save_name, "w") as f:
            for model_name in model_names:
                model_result = []
                for task_name in ["task1", "task2", "task3"]:
                    model_result.append(round(all_results[task_name][model_name][f"iid-{setting}-mean"], 1))
                    delta = all_results[task_name][model_name][f"ood-{setting}-mean"]
                    model_result.append(round(delta, 1))
                f.write(model_name_to_formal[model_name]+"&")
                for i in range(len(model_result)):
                    # if i % 2 == 0:
                    model_result[i] = "$" + str(model_result[i]) + "$"
                    # else:
                    #     if model_result[i] > 0:
                    #         model_result[i] = "$" + "{" + "\color{red}" + "+" + str(model_result[i]) + "}" + "$"
                    #     else:
                    #         model_result[i] = "$" + "{" + "\color{blue}" + str(model_result[i]) + "}" + "$"
                f.write("&".join(model_result))
                f.write("\\\\")
                f.write("\n")


def merge_lexical_overlap():
    model_names = [
        "bert-base-uncased",
        "roberta-base",
        "gpt2",
        "gpt-neo-125M",
        "bart-base",
        "t5-base"
    ]
    all_results = json.load(open("result.json"))
    save_name = f"results/lexical_overlap.latex"
    with open(save_name, "w") as f:
        for model_name in model_names:
            model_result = []
            for task_name in ["task1", "task3"]:
                easy_data = json.load(open(f"probing/{task_name}/all_output/results/{model_name}.json"))
                if task_name == "task1":
                    if model_name in ["gpt2", "gpt-neo-125M"]:
                        model_result.append(round(easy_data["all"]["easy"]["topk"]["1"], 1))
                    else:
                        model_result.append(round(easy_data["e1"]["easy"]["topk"]["1"], 1))
                    delta = all_results[task_name][model_name]["PB"]["hard"]["topk"]["1"]
                    model_result.append(round(delta, 1))
                else:
                    if model_name in ["bert-base-uncased", "t5-base"]:
                        model_result.append(round(easy_data["all"]["easy_accuracy"], 1))
                    else:
                        model_result.append(round(easy_data["concept"]["easy_accuracy"], 1))
                    delta = all_results[task_name][model_name]["PB"]["hard_accuracy"]
                    model_result.append(round(delta, 1))
            f.write(model_name_to_formal[model_name]+"&")
            for i in range(len(model_result)):
                # if i % 2 == 0:
                model_result[i] = "$" + str(model_result[i]) + "$"
                # else:
                #     if model_result[i] > 0:
                #         model_result[i] = "$" + "{" + "\color{red}" + "+" + str(model_result[i]) + "}" + "$"
                #     else:
                #         model_result[i] = "$" + "{" + "\color{blue}" + str(model_result[i]) + "}" + "$"
            f.write("&".join(model_result))
            f.write("\\\\")
            f.write("\n")





if __name__ == "__main__":
    if not os.path.exists("results"):
        os.mkdir("results")
    merge_result()
    merge_probing()
    # merge_finetuning()
    merge_main()
    # merge_iid()
    # merge_lexical_overlap()




            
