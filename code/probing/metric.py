import os 
import pdb 
import sys

sys.path.append("..")
import json 
from typing import Counter, List, Dict, Tuple 
from pathlib import Path
import logging

from cp_arguments import CPArgs


MAXIMUM = 10**20


def computeSpanLoss(results: List[Dict], special_tokens: List[str]) -> List[Dict]:
    results_with_span_loss = []
    for i, item in enumerate(results):
        special_token_loss = {k:0 for k in special_tokens}
        special_token_counter = {k:0 for k in special_tokens}
        for i, token in enumerate(item["tokens"]):
            if token["marker"] in special_token_counter:
                special_token_counter[token["marker"]] += 1
                special_token_loss[token["marker"]] += token["loss"]
        item_with_span_loss = {
            "tokens": item["tokens"],
        }
        for k, v in special_token_loss.items():
            item_with_span_loss[k] = -v / (special_token_counter[k]+1e-10)
        results_with_span_loss.append(item_with_span_loss)
    return results_with_span_loss

# -------------------- Metric for Conceptual Similarity Judgment (CSJ)  --------------------#

def computePerplexityForCSJ(mask_position: str, 
                            results: List[Dict], 
                            special_tokens: List[str],
                            data: List[Dict]) -> List[Dict]:
    if mask_position != "all": 
        results = computeSpanLoss(results, special_tokens)
    index = 0
    for i in range(len(data)):
        pred = None
        flag = -MAXIMUM
        for j in range(len(data[i]["candidates"])):
            prob = results[index][mask_position]
            index += 1
            data[i]["candidates"][j]["prob"] = prob
            if prob >= flag:
                flag = prob
                pred = data[i]["candidates"][j]["name"]
        data[i]["pred"] = pred
    assert index == len(results)
    return data


def computeTopk(preds, label, k=3):
    preds = [ent["name"] for ent in sorted(preds, key=lambda item: item["prob"], reverse=True)]
    if label in preds[:k]:
        return 1 
    return 0 


def computeMR(preds, label):
    preds = [ent["name"] for ent in sorted(preds, key=lambda item: item["prob"], reverse=True)]
    pos = -1
    for i, pred in enumerate(preds):
        if pred == label:
            pos = 1 / (i+1)
            break 
    # assert pos != -1
    return pos 


def computeAccuracyForCSJ(all_data, results: List[Dict], args: CPArgs, special_tokens: List[str], logger: logging.Logger) -> None:
    all_scored_data = computePerplexityForCSJ(args.mask_position, results, special_tokens, all_data)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = f"{args.mask_position}-{args.output_file}"
    json.dump(all_scored_data, open(os.path.join(output_dir, output_file), "w"), indent=4)
    # compute accuracy
    results = dict(total=1e-10, topk=Counter(), mrr=0)
    for item in all_scored_data:
        label = -1
        for candidate in item["candidates"]:
            if candidate["id"] == item["label"]:
                label = candidate["name"]
        results["total"] += 1
        preds = item["candidates"]
        for i in range(1, 11):
            results["topk"][f"{i}"] += computeTopk(preds, label, i)
        results["mrr"] += computeMR(preds, label) 
    results["mrr"] /= results["total"]
    for k in results["topk"].keys():
        results["topk"][k] /= results["total"]
        results["topk"][k] *= 100
    logger.info(results)
    return results

# -------------------- Metric for Conceptual Property Judgment (CPJ)  --------------------#

def computeProbForCPJ(mask_position: CPArgs,
                        results: List[Dict], 
                        special_tokens: List[str],
                        data: List[Dict]) -> List[Dict]:
    if mask_position != "all": 
        results = computeSpanLoss(results, special_tokens)
    index = 0
    for i in range(len(data)):
        positive_pred = results[index][mask_position]
        index += 1
        negative_pred = results[index][mask_position]
        index += 1
        data[i]["pred"] = int(positive_pred > negative_pred)
    assert index == len(results)
    return data


def computeAccuracyForCPJ(all_data, results: List[Dict], args: CPArgs, special_tokens: List[str], logger: logging.Logger) -> None:
    all_scored_data = computeProbForCPJ(args.mask_position, results, special_tokens, all_data)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = f"{args.mask_position}-{args.output_file}"
    json.dump(all_scored_data, open(os.path.join(output_dir, output_file), "w"), indent=4)
    # compute accuracy
    crr = 0
    for item in all_scored_data:
        if item["label"] == item["pred"]:
            crr += 1
    logger.info("Accuracy: %.4f, Correct: %d, Total: %d" % (crr/len(all_scored_data), crr, len(all_scored_data)))
    return dict(
        accuracy=crr/len(all_scored_data)*100
    )

# -------------------- Metric for Conceptualization in Contexts (CiC)  --------------------#

def computeProbForCiC(mask_position: str, 
                            results: List[Dict], 
                            special_tokens: List[str],
                            data: List[Dict]) -> List[Dict]:
    if mask_position != "all": 
        results = computeSpanLoss(results, special_tokens)
    index = 0
    for i in range(len(data)):
        pplOfLabels = []
        pred = None
        flag = -MAXIMUM
        for path in data[i]["concept_chains"]:
            pplOfPath = []
            for label in path:
                prob = results[index][mask_position]
                index += 1
                pplOfPath.append({"con": label, "prob": prob})
                if prob > flag:
                    flag = prob
                    pred = label
            pplOfLabels.append(pplOfPath)
        data[i]["concept_chains"] = pplOfLabels
        data[i]["pred"] = pred 
    assert index == len(results)
    return data


def computeAcc(all_scored_data: List[Dict], mode="all") -> Tuple[float, int, int]:
    crr, tot = 0, 1e-10
    for item in all_scored_data:
        if item["label"] != -1:
            if item["label"].split("_")[-1] == item["pred"].split("_")[-1]:
                crr += 1
        else:
            pass
        tot += 1
    return crr / tot, crr, tot


def computeAccuracyForCiC(all_data, results: List[Dict], args: CPArgs, special_tokens: List[str], logger: logging.Logger) -> None:
    all_scored_data = computeProbForCiC(args.mask_position, results, special_tokens, all_data)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = f"{args.mask_position}-{args.output_file}"
    json.dump(all_scored_data, open(os.path.join(output_dir, output_file), "w"), indent=4)
    logger.info("All Accuracy: %.4f, Correct: %d, Total: %d" % (computeAcc(all_scored_data, "all")))
    return dict(
        all_accuracy=computeAcc(all_scored_data, "all")[0]*100
    )