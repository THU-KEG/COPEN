import os
import sys
sys.path.append("..")
import pdb
import copy 
import json
import argparse
from argparse import Namespace

from pathlib import Path 
from typing import List, Tuple, Dict
from transformers import BertTokenizer, RobertaTokenizer, GPT2Tokenizer, \
                            BartTokenizer, T5Tokenizer, AutoTokenizer

from processor_utils import convertExamplesToFeatures


class InputExample(object):
    def __init__(self, unique_id, tokens, con_pos, answer_pos):
        self.unique_id = unique_id
        self.tokens = tokens 
        self.con_pos = con_pos
        self.answer_pos = answer_pos 


def readExamplesForSC(input_file):
    """Read a list of `InputExample`s from an input file."""
    examples = []
    unique_id = 0
    data = json.load(open(input_file))
    template = "The statement is".split()
    for item in data:
        tokens = item["text"].split()
        con_pos = item["concept"]["pos"]
        for answer in ["true", "false"]:
            all_tokens = copy.deepcopy(tokens) + template 
            answer_pos = [len(all_tokens), len(all_tokens)+1]
            all_tokens.append(answer)
            all_tokens.append(".")
            examples.append(
                InputExample(unique_id, all_tokens, con_pos, answer_pos)
            )
            unique_id += 1
    return examples


def encodeTextWithTemplateForSC(example, args, tokenizer):
    """Returns entity_positions and tokens."""
    tokens, con_pos, answer_pos = example.tokens, example.con_pos, example.answer_pos
    tokenized_token_markers = ["begin"]
    if args.model_type in ["bert"]:
        tokenized_tokens = [tokenizer.cls_token]
    elif args.model_type in ["roberta", "gpt2", "gpt_neo"]:
        tokenized_tokens = [tokenizer.bos_token] 
    elif args.model_type in ["bart", "t5"]:
        tokenized_tokens = []
        tokenized_token_markers = []
    mask_left, mask_right = -1, -1
    for i, token in enumerate(tokens):
        _tokens = tokenizer.tokenize(token)
        tokenized_tokens.extend(_tokens)
        if i >= con_pos[0] and i < con_pos[1]:
            tokenized_token_markers.extend(['concept']*len(_tokens))
        elif i >= answer_pos[0] and i< answer_pos[1]:
            tokenized_token_markers.extend(['answer']*len(_tokens))
        else:
            tokenized_token_markers.extend(['#']*len(_tokens))
    # end 
    if args.model_type in ["bert"]:
        tokenized_tokens.append(tokenizer.sep_token)
    elif args.model_type in ["roberta", "gpt2", "gpt_neo", "bart", "t5"]:
        tokenized_tokens.append(tokenizer.eos_token)
    tokenized_token_markers.append("end")
    assert len(tokenized_tokens) == len(tokenized_token_markers)
    # assert mask_left != -1 and mask_right != -1

    return tokenized_tokens, tokenized_token_markers, mask_left, mask_right


def process_data(args, tokenizer):
    examples = readExamplesForSC(args.input_file)
    all_features, all_tokens, all_token_markers = convertExamplesToFeatures(examples, args, tokenizer, args.max_seq_length, encodeTextWithTemplateForSC)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(os.path.join(output_dir, f"data-{args.mask_position}.jsonl"), "w") as f:
        for features in all_features:
            item = dict()
            item["input"] = tokenizer.convert_ids_to_tokens(features.input_ids)
            item["output"] = tokenizer.convert_ids_to_tokens(features.masked_lm_ids)
            item["output_mask"] = features.masked_lm_positions
            f.write(json.dumps(item)+"\n")
    with open(os.path.join(output_dir, f"all_tokens.json"), "w") as f:
        json.dump(all_tokens, f)
    with open(os.path.join(output_dir, f"all_token_markers.json"), "w") as f:
        json.dump(all_token_markers, f)
            

if __name__ == "__main__":
    model_types = [
        ["bert", "bert-base-uncased"],
        ["roberta", "roberta-base"],
        ["gpt2", "gpt2"],
        ["gpt_neo", "EleutherAI/gpt-neo-125M"],
        ["bart", "facebook/bart-base"],
        ["t5", "t5-small"]
    ]
    for model_type in model_types:
        if model_type[0] == "bert":
            tokenizer = BertTokenizer.from_pretrained(model_type[1])
        elif model_type[0] == "roberta":
            tokenizer = RobertaTokenizer.from_pretrained(model_type[1])
        elif model_type[0] == "gpt2":
            tokenizer = GPT2Tokenizer.from_pretrained(model_type[1])
        elif model_type[0] == "gpt_neo":
            tokenizer = GPT2Tokenizer.from_pretrained(model_type[1])
        elif model_type[0] == "bart":
            tokenizer = BartTokenizer.from_pretrained(model_type[1])
        elif model_type[0] == "t5":
            tokenizer = T5Tokenizer.from_pretrained(model_type[1])
        else:
            raise ValueError
        for template in ["template1"]:
            for mask_position in ["all"]:
                params = {
                    "model_type": model_type[0],
                    "model_name_or_path": model_type[1],
                    "input_file": os.path.join(os.path.join("data", "ood"), "test.json"),
                    "output_dir": os.path.join(os.path.join(os.path.join(os.path.join("data", "probing"), model_type[0]), template)),
                    "mask_position": mask_position,
                    "max_seq_length": 120
                }
                args = Namespace(**params)
                print(args)
                process_data(args, tokenizer)

