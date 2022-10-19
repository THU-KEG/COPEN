import os 
import pdb 
import sys

sys.path.append("..")
import json 
from typing import Counter, List, Dict, Tuple 
from pathlib import Path
import logging

from cp_arguments import CPArgumentParser, CPArgs
from utils import compute_consistency

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
        for path in data[i]["label"]:
            pplOfPath = []
            for label in path:
                prob = results[index][mask_position]
                index += 1
                pplOfPath.append({"con": label, "prob": prob})
                if prob > flag:
                    flag = prob
                    pred = label
            pplOfLabels.append(pplOfPath)
        data[i]["label"] = pplOfLabels
        data[i]["pred"] = pred 
    assert index == len(results)
    return data


def computeAcc(all_scored_data: List[Dict], mode="all") -> Tuple[float, int, int]:
    # compute accuracy
    crr, tot = 0, 1e-10
    for item in all_scored_data:
        if mode != "all" and item["id"] != mode:
            continue
        if item["my_label"].split("_")[-1] == item["pred"].split("_")[-1]:
            crr += 1
        tot += 1
    return crr / tot, crr, tot


def computeAccuracyForCiC(all_data, results: List[Dict], args: CPArgs, special_tokens: List[str], logger: logging.Logger) -> None:
    all_scored_data = computeProbForCiC(args.mask_position, results, special_tokens, all_data)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = f"{args.mask_position}-{args.output_file}"
    json.dump(all_scored_data, open(os.path.join(output_dir, output_file), "w"), indent=4)
    logger.info("Easy Accuracy: %.4f, Correct: %d, Total: %d" % (computeAcc(all_scored_data, "easy")))
    logger.info("Hard Accuracy: %.4f, Correct: %d, Total: %d" % (computeAcc(all_scored_data, "hard")))
    logger.info("All Accuracy: %.4f, Correct: %d, Total: %d" % (computeAcc(all_scored_data, "all")))
    return dict(
        easy_accuracy=computeAcc(all_scored_data, "easy")[0]*100,
        hard_accuracy=computeAcc(all_scored_data, "hard")[0]*100,
        all_accuracy=computeAcc(all_scored_data, "all")[0]*100
    )


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
        prob = results[index][mask_position]
        index += 1
        data[i]["y"]["prob"] = prob
        if prob >= flag:
            flag = prob
            pred = data[i]["y"]["name"]

        for j in range(len(data[i]["n"])):
            prob = results[index][mask_position]
            index += 1
            data[i]["n"][j]["prob"] = prob
            if prob >= flag:
                flag = prob
                pred = data[i]["n"][j]["name"]
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
    assert pos != -1
    return pos 


def computeAccuracyForCSJ(all_data, results: List[Dict], args: CPArgs, special_tokens: List[str], logger: logging.Logger) -> None:
    all_scored_data = computePerplexityForCSJ(args.mask_position, results, special_tokens, all_data)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = f"{args.mask_position}-{args.output_file}"
    json.dump(all_scored_data, open(os.path.join(output_dir, output_file), "w"), indent=4)
    # compute accuracy
    results = dict(
        hard=dict(total=1e-10, topk=Counter(), mrr=0),
        easy=dict(total=1e-10, topk=Counter(), mrr=0)
    )
    for item in all_scored_data:
        item_type = item["id"][:4]
        results[item_type]["total"] += 1
        label = item["y"]["name"]
        preds = []
        for pred in item["n"]:
            preds.append(pred)
        preds.append(item["y"])
        for i in range(1, 11):
            results[item_type]["topk"][f"{i}"] += computeTopk(preds, label, i)
        results[item_type]["mrr"] += computeMR(preds, label) 
    # tot = len(all_scored_data)
    for type in results.keys():
        results[type]["mrr"] /= results[type]["total"]
        for k in results[type]["topk"].keys():
            results[type]["topk"][k] /= results[type]["total"]
            results[type]["topk"][k] *= 100
    logger.info(results)
    return results


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
        accuracy=crr/len(all_scored_data)*100,
        consistency=compute_consistency(all_scored_data)
    )