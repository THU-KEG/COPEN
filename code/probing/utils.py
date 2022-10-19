import os 
import sys
import math 

sys.path.append("..")
import pdb 
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

from transformers import PreTrainedTokenizer
from collections import defaultdict

from cp_arguments import CPArgs

LARGE_NUMBER = 1000000000000


def dumpResult(args: CPArgs,
                tokenizer: PreTrainedTokenizer,
                all_tokens: List[str],
                all_token_markers: List[str],
                all_masked_lm_loss: List[float]) -> List[Dict]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_predict_file = os.path.join(args.output_dir, args.output_file)
    if not args.post_process_logits:
        return parseResult(args, all_masked_lm_loss, all_tokens, all_token_markers, "begin", "end", output_predict_file)
    else:
        return parseResultLabelWords(args, tokenizer, all_masked_lm_loss, all_tokens, all_token_markers, "begin", "end", output_predict_file)


def parseResult(args: CPArgs,
                result: List[float], 
                all_tokens: List[str], 
                all_token_markers: List[str], 
                bos_token: str,
                eos_token: str, 
                output_file: Union[Path, str, None]=None) -> List[Dict]:
    assert len(all_tokens) == len(all_token_markers)
    with open(output_file, "w") as writer:
        print("***** Predict results *****")
        idx = 0
        sentences = []
        sentence = {}
        tokens = []
        for i, token in enumerate(all_tokens):
            # # start of a sentence
            # if all_token_markers[i] == bos_token:
            #     sentence = {}
            #     tokens = []

            if all_token_markers[i] == args.mask_position \
                or (args.mask_position == "all" and all_token_markers[i] not in [bos_token, eos_token]):
                word_loss = float(result[idx])
                prob = float(np.exp(-word_loss))
                idx += 1
            else:
                word_loss = -1
                prob = -1

            # add token
            tokens.append({"token": all_tokens[i],
                            "marker": all_token_markers[i],
                            "loss": word_loss,
                            "prob": prob})

            # end of a sentence
            if all_token_markers[i] == eos_token:
                sentence["tokens"] = tokens
                if args.mask_position == "all":
                    tot_loss = 0
                    for token in tokens:
                        tot_loss += token["loss"]
                    sentence["all"] = -float(np.exp(tot_loss/len(tokens)))
                sentences.append(sentence)
                sentence = {}
                tokens = []
        # pdb.set_trace()
        assert idx == len(result)
        if output_file is not None:
            print("Saving results to %s" % output_file)
            writer.write(json.dumps(sentences, indent=2, ensure_ascii=False))
        
        return sentences


def parseResultLabelWords(args: CPArgs,
                tokenizer: PreTrainedTokenizer,
                result: List[float], 
                all_tokens: List[str], 
                all_token_markers: List[str], 
                bos_token: str,
                eos_token: str, 
                output_file: Union[Path, str, None]=None) -> List[Dict]:
    assert len(all_tokens) == len(all_token_markers)
    with open(output_file, "w") as writer:
        print("***** Predict results *****")
        idx = 0
        sentences = []
        sentence = {}
        tokens = []
        for i, token in enumerate(all_tokens):
            if all_token_markers[i] == "answer":
                _prob = result[idx]
                prob = {}
                for j, id in enumerate(args.label_words):
                    word = tokenizer.decode([id])
                    prob[word] = _prob[j]
                sentence["preds"] = prob
                idx += 1
            else:
                prob = -1

            # add token
            tokens.append({"token": all_tokens[i],
                            "marker": all_token_markers[i],
                            "prob": prob})

            # end of a sentence
            if all_token_markers[i] == eos_token:
                sentence["tokens"] = tokens
                sentences.append(sentence)
                sentence = {}
                tokens = []
        assert idx == len(result)
        if output_file is not None:
            print("Saving results to %s" % output_file)
            writer.write(json.dumps(sentences, indent=2, ensure_ascii=False))
        
        return sentences


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
