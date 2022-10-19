import os
import sys
sys.path.append("..")
import pdb
import json
import argparse
from argparse import Namespace

from pathlib import Path 
from typing import List, Tuple, Dict
from transformers import BertTokenizer, RobertaTokenizer, GPT2Tokenizer, \
                            BartTokenizer, T5Tokenizer

from processor_utils import convertExamplesToFeatures


# template
templates = {
    "T1": "Is E1 similar with E2 ?",
}


class InputExample(object):
    def __init__(self, unique_id, tokens):
        self.unique_id = unique_id
        self.tokens = tokens 


def readExamplesForES(input_file):
    """Read a list of `InputExample`s from an input file."""
    examples = []
    unique_id = 0
    data = json.load(open(input_file))
    for item in data:
        query = item["query"]["name"]
        pos = item["y"]["name"]
        text = f"{query}\t{pos}"
        examples.append(
          InputExample(unique_id, text)
        )
        unique_id += 1
        for neg in item["n"]:
            neg = neg["name"]
            text = f"{query}\t{neg}"
            examples.append(
              InputExample(unique_id, text)
            )
            unique_id += 1
    return examples


def encodeTextWithTemplateForES(example, args, tokenizer):
    """Returns entity_positions and tokens."""
    query_entity, candi_entity = example.tokens.split('\t')
    text_before_E1 = ""
    text_between = " is conceptually similar with"
    text_after_E2 = "."
    mask_left, mask_right = -1, -1
    def _tokenize(text, marker, tokenized_tokens, tokenized_token_markers):
        _tokens = tokenizer.tokenize(text)
        mask_left = len(tokenized_tokens)
        tokenized_tokens.extend(_tokens)
        tokenized_token_markers.extend([marker for _ in range(len(_tokens))])
        mask_right = len(tokenized_tokens)
        return mask_left, mask_right
        
    # begin
    tokenized_token_markers = ['begin']
    if args.model_type in ["bert"]:
        tokenized_tokens = [tokenizer.cls_token]
    elif args.model_type in ["roberta", "gpt2", "gpt_neo"]:
        tokenized_tokens = [tokenizer.bos_token] 
    elif args.model_type in ["bart", "t5"]:
        tokenized_tokens = []
        tokenized_token_markers = []
    # before
    _tokenize(text_before_E1, "#", tokenized_tokens, tokenized_token_markers)
    # E1
    e1_mask_left, e1_mask_right = _tokenize(query_entity, "e1", tokenized_tokens, tokenized_token_markers)
    # between
    _tokenize(text_between, "#", tokenized_tokens, tokenized_token_markers)
    # E2
    e2_mask_left, e2_mask_right = _tokenize(candi_entity, "e2", tokenized_tokens, tokenized_token_markers)
    # after
    _tokenize(text_after_E2, "#", tokenized_tokens, tokenized_token_markers)
    # # append position for [mask]
    # mask_left = len(tokenized_tokens)
    # _tokenize("Yes", "answer", tokenized_tokens, tokenized_token_markers)
    # mask_right = len(tokenized_tokens)
    # _tokenize(".", "#", tokenized_tokens, tokenized_token_markers)
    # end
    if args.mask_position == "e1":
        mask_left = e1_mask_left 
        mask_right = e1_mask_right
    elif args.mask_position == "e2":
        mask_left = e2_mask_left
        mask_right = e2_mask_right
    else:
        pass 
    if args.model_type in ["bert"]:
        tokenized_tokens.append(tokenizer.sep_token)
    elif args.model_type in ["roberta", "gpt2", "gpt_neo", "bart", "t5"]:
        tokenized_tokens.append(tokenizer.eos_token)
    tokenized_token_markers.append("end")
    assert len(tokenized_tokens) == len(tokenized_token_markers)
    return tokenized_tokens, tokenized_token_markers, mask_left, mask_right


def process_data(args, tokenizer):
    examples = readExamplesForES(args.input_file)
    all_features, all_tokens, all_token_markers = convertExamplesToFeatures(examples, args, tokenizer, args.max_seq_length, encodeTextWithTemplateForES)
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
        elif model_type[0] in ["gpt2", "gpt_neo"]:
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





