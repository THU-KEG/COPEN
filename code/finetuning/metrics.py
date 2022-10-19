import pdb
from typing import Tuple 
import torch 
import numpy as np

from collections import defaultdict
from sklearn.metrics import f1_score


def compute_F1(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    pos_labels = list(range(1, 42))
    micro_f1 = f1_score(labels, predictions, labels=pos_labels, average="micro") * 100.0
    return {"micro_f1": micro_f1}


def softmax(logits, dim=-1):
    logits = torch.tensor(logits)
    return torch.softmax(logits, dim=dim).numpy()


def compute_accuracy(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(softmax(logits), axis=-1)
    accuracy = (predictions == labels).sum() / labels.shape[0]
    return {"accuracy": accuracy}


def compute_accuracy_for_qa(eval_pred):
    logits, labels = eval_pred
    begin_logits, end_logits = logits[0], logits[1]
    begin_labels, end_labels = labels[0], labels[1]
    begin_predictions = np.argmax(softmax(begin_logits), axis=-1)
    end_predictions = np.argmax(softmax(end_logits), axis=-1)
    correct = (begin_predictions == begin_labels) * (end_predictions == end_labels)
    accuracy = correct.sum() / correct.shape[0]
    return {"accuracy": accuracy}


def compute_consistency(data):
    chains = defaultdict(list)
    for item in data:
        if "chain_id" in item:
            chains[item["chain_id"]].append(item)
    metrics = {}
    for chain_id, chain in chains.items():
        if len(chain) == 1:
            continue

        chain_length = len(chain)
        if chain_length not in metrics:
            metrics[chain_length] = {"num_chains": 0, "consist_correct": 0, "consistency": 0}

        metrics[chain_length]["num_chains"] += 1
        consistent = True
        for item in chain:
            if item["label"] != item["pred"]:
                consistent = False
                break 
        if consistent:
            metrics[chain_length]["consist_correct"] += 1
    total_chains, total_correct = 0, 0
    for key in metrics:
        metrics[key]["consistency"] = metrics[key]["consist_correct"] / metrics[key]["num_chains"]
        total_chains += metrics[key]["num_chains"]
        total_correct += metrics[key]["consist_correct"]
    metrics["overall"] = {}
    metrics["overall"]["num_chains"] = total_chains
    metrics["overall"]["consist_correct"] = total_correct
    metrics["overall"]["consistency"] = total_correct / total_chains
    return metrics




