import os 
import json
import pdb 
import random
import numpy as np 
from tqdm import tqdm 

from pathlib import Path
from typing import Dict, List

import spacy
from scipy import stats


def convert_format(item):
    """
    Returns:
    {
        "query": {},
        "y": {},
        "candidates": [] # ranked list
    }
    """
    candidates = []
    for candidate in item["n"]:
        candidates.append(candidate)
    candidates.append(item["y"])
    candidates = sorted(candidates, key=lambda item: item["prob"], reverse=True)
    rank = -1 
    for i, candidate in enumerate(candidates):
        if item["y"]["id"] == candidate["id"]:
            rank = i
    assert rank != -1
    new_item = {
        "query": item["query"],
        "y": item["y"],
        "rank": rank,
        "candidates": candidates
    }
    return new_item


def _sample_predictions(input_path, save_path, num_samples=10):
    data = json.load(open(input_path))
    format_data = []
    for item in data:
        format_item = convert_format(item)
        if format_item["rank"] == 0 or format_item["rank"] > 5:
            continue
        format_data.append(format_item)
    print(len(data), len(format_data))
    sampled_data = random.sample(format_data, k=10)
    json.dump(sampled_data, open(save_path, "w"), indent=4)


def sample_predictions():
    random.seed(42)
    output_dir = Path("output/case-study")
    output_dir.mkdir(parents=True, exist_ok=True)
    input_path = "output/e1-bert-base-uncased-probs.json"
    save_path = os.path.join(output_dir, "bert.json")
    _sample_predictions(input_path, save_path)


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


def _compute_fine_coarse_grained_rank(data: List, flat_schema: Dict[str, str], sort_key: str) -> None:
    """Compute MRR of candidate entities of different types.

    Compute MRR of hard distractors and easy distractors.

    Args:
        data: The results.
        flat_schema: The flat schema.
        sort_key: key for sorting.
    """
    # data = json.load(open(input_path))
    grain_rank = []
    filtered_easy = 0
    for item in data:
        if item["id"].split("-")[0] == "easy":
            filtered_easy += 1
            continue
        candidates = []
        for candidate in item["n"]:
            candidates.append(candidate)
        candidates = sorted(candidates, key=lambda item: item[sort_key], reverse=True)
        new_item = {
            "query": item["query"],
            "y": item["y"],
            "fine": {
                "item": [],
                "rank": 0,
            },
            "coarse": {
                "item": [],
                "rank": 0
            }   
        }
        for i, candidate in enumerate(candidates):
            if flat_schema[item["query"]["concept"]] == flat_schema[candidate["concept"]]:
                new_item["fine"]["item"].append(candidate)
                new_item["fine"]["rank"] += 1 / (i+1)
            else:
                new_item["coarse"]["item"].append(candidate)
                new_item["coarse"]["rank"] += 1 / (i+1)
        if len(new_item["fine"]["item"]) == 0:
            new_item["fine"]["rank"] = -1
        else:
            new_item["fine"]["rank"] /= len(new_item["fine"]["item"])
        new_item["coarse"]["rank"] /= len(new_item["coarse"]["item"])
        grain_rank.append(new_item)
    mean_rank = dict(fine=0, coarse=0)
    fine_tot = 0
    for item in grain_rank:
        mean_rank["coarse"] += (item["coarse"]["rank"] / len(grain_rank))
        if item["fine"]["rank"] == -1:
            continue
        fine_tot += 1
        mean_rank["fine"] += item["fine"]["rank"]
    mean_rank["fine"] /= fine_tot
    print("Filterd easy data: %d" % filtered_easy)
    return mean_rank


def compute_fine_coarse_grained_rank() -> None:
    schema = parse_schema("../../../data/schema/father.txt")
    flat_schema = flatten_schema(schema)
    model_names = [
        "bert-base-uncased",
        "roberta-base",
        "gpt2",
        "gpt-neo-125M",
        "bart-base",
        "t5-base"
    ]
    for model_name in model_names:
        probing_mode = "e1"
        if model_name in ["gpt2", "gpt-neo-125M"]:
            probing_mode = "all"
        input_path = os.path.join("output", f"{probing_mode}-{model_name}-probs.json")
        mean_rank = _compute_fine_coarse_grained_rank(json.load(open(input_path)), flat_schema, "prob")
        for key in mean_rank.keys():
            mean_rank[key] = round(mean_rank[key]*100, 1) 
        print(model_name, mean_rank)


def compute_prop(results: List[Dict], special_tokens: List[str], mask_position: str) -> List[Dict]:
    results_with_span_prob = []
    for i, item in enumerate(results):
        special_token_prob = {k:0 for k in special_tokens}
        special_token_counter = {k:0 for k in special_tokens}
        for i, token in enumerate(item["tokens"]):
            if mask_position == "all":
                special_token_counter["all"] += 1
                special_token_prob["all"] += token["loss"]
            elif token["marker"] in special_token_counter:
                special_token_counter[token["marker"]] += 1
                special_token_prob[token["marker"]] += token["loss"]
        item_with_span_prob = {
            "tokens": item["tokens"],
        }
        for k, v in special_token_prob.items():
            item_with_span_prob[k] = -v / (special_token_counter[k]+1e-10)
        results_with_span_prob.append(item_with_span_prob)
    return results_with_span_prob


def compute_similarity_PLMs(mask_position: str, 
                            results: List[Dict], 
                            special_tokens: List[str],
                            data: List[Dict]) -> List[Dict]:
    MAXIMUM = 10000000000
    results = compute_prop(results, special_tokens, mask_position)
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


def compute_pearsonr():
    model_name = "/tmp/glove"
    nlp = spacy.load(model_name)
    model_names = [
        "bert-base-uncased",
        "roberta-base",
        "gpt2",
        "gpt-neo-125M",
        "bart-base",
        "t5-base"
    ]
    for model_name in model_names:
        probing_mode = "e1"
        if model_name in ["gpt2", "gpt-neo-125M"]:
            probing_mode = "all"
        input_path = os.path.join("output", f"{model_name}-probs.json")
        original_data = json.load(open("../../../data/task1/data/ood/test.json"))
        data = compute_similarity_PLMs(probing_mode, json.load(open(input_path)), ["e1", "all"], original_data)
        PLMs_sim = []
        word_sim = []
        for item in tqdm(data):
            PLMs_sim.append(item["y"]["prob"])
            doc_query = nlp(item["query"]["name"])
            doc_candidate = nlp(item["y"]["name"])
            word_sim.append(doc_query.similarity(doc_candidate))
            for neg in item["n"]:
                PLMs_sim.append(neg["prob"])
                doc_candidate = nlp(neg["name"])
                word_sim.append(doc_query.similarity(doc_candidate))
        pearsonr, _ = stats.pearsonr(PLMs_sim, word_sim)
        print(model_name, pearsonr)


def compute_ws_rank():
    schema = parse_schema("../../../data/schema/father.txt")
    flat_schema = flatten_schema(schema)
    model_name = "/tmp/glove"
    nlp = spacy.load(model_name)
    model_names = [
        "bert-base-uncased",
        "roberta-base",
        "gpt2",
        "gpt-neo-125M",
        "bart-base",
        "t5-base"
    ]
    for model_name in model_names:
        data = json.load(open("../../../data/task1/data/ood/test.json"))
        for item in tqdm(data):
            doc_query = nlp(item["query"]["name"])
            doc_candidate = nlp(item["y"]["name"])
            item["y"]["ws"] =  doc_query.similarity(doc_candidate)
            for neg in item["n"]:
                doc_candidate = nlp(neg["name"])
                neg["ws"] = doc_query.similarity(doc_candidate)
        mean_rank = _compute_fine_coarse_grained_rank(data, flat_schema, "ws")
        for key in mean_rank.keys():
            mean_rank[key] = round(mean_rank[key]*100, 1) 
        print(model_name, mean_rank)


if __name__ == "__main__":
    compute_fine_coarse_grained_rank()
    # compute_pearsonr()
    # compute_ws_rank()


