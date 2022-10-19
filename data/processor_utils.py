import os
import pdb
import sys 
import json
import logging


from typing import List, Tuple, Dict
from tqdm import tqdm
from argparse import Namespace
from pathlib import Path

from transformers import BertTokenizer, RobertaTokenizer, GPT2Tokenizer, \
                            BartTokenizer, T5Tokenizer, AutoTokenizer


class InputFeatures(object):
    def __init__(self, input_ids, masked_lm_positions=None, masked_lm_ids=None):
        self.input_ids = input_ids
        self.masked_lm_positions = masked_lm_positions
        self.masked_lm_ids = masked_lm_ids


def convertExamplesToFeatures(examples, args, tokenizer, max_seq_length, encode_fn):
    """Convert a set of `InputExample`s to a list of `InputFeatures`."""
    all_features = []
    all_tokens = []
    all_token_markers = []

    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            print("Writing example %d of %d" % (ex_index, len(examples)))

        features, tokens, token_markers = convertSingleExample(example, 
                                            max_seq_length, args, tokenizer, encode_fn)
        all_features.extend(features)
        all_tokens.extend(tokens)
        all_token_markers.extend(token_markers)

    return all_features, all_tokens, all_token_markers


def convertSingleExample(example, max_seq_length, args, tokenizer, encode_fn):
    """Converts a single `InputExample` into a single `InputFeatures`."""
    tokens, token_markers, mask_left, mask_right = encode_fn(example, args, tokenizer)

    if len(tokens) > max_seq_length:
        tokens = tokens[:max_seq_length-1]
        token_markers = token_markers[:max_seq_length-1]
        if args.model_type in ["bert"]:
            tokens.append(tokenizer.sep_token)
        elif args.model_type in ["roberta", "gpt2", "gpt_neo", "bart", "t5"]:
            tokens.append(tokenizer.eos_token)
        token_markers.append('end')
        print(
            "An input exceeds max_seq_length limit: ",
            example.tokens
        )

    input_tokens = tokens
    input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
    token_count = len(input_tokens)

    if args.model_type in ["bert", "roberta"]:
        features = createMaskForMLM(mask_left, mask_right, input_ids, 
                                    tokenizer) if args.mask_position != "all" else \
                    createMaskForMLMPerToken(token_count, input_ids, 
                                    tokenizer)
    elif args.model_type in ["gpt2", "gpt_neo", "bart", "t5"]:
        features = createMaskForLM(args, mask_left, mask_right, input_ids, 
                                    tokenizer) if args.mask_position != "all" else \
                    createMaskForLMPerToken(args,token_count, input_ids, 
                                    tokenizer)
    return features, input_tokens, token_markers


def createMaskForMLM(mask_left, mask_right, input_ids, tokenizer):
    """Mask each token/word sequentially. Only used for bert/roberta"""
    features = []
    num_mask = mask_right - mask_left
    for i in range(mask_left, mask_right):
        assert num_mask != 0
        input_ids_new, masked_lm_positions, masked_lm_labels = createMaskedLMPredictionForMLM(input_ids, i, tokenizer, num_mask)
        feature = InputFeatures(
            input_ids=input_ids_new,
            masked_lm_positions=masked_lm_positions,
            masked_lm_ids=masked_lm_labels)
        features.append(feature)
        num_mask -= 1
    assert num_mask == 0
    return features


def createMaskForMLMPerToken(token_count, input_ids, tokenizer):
    """Mask each token/word sequentially. Only used for bert/roberta"""
    features = []
    for i in range(1, token_count-1):
        input_ids_new, masked_lm_positions, masked_lm_labels = createMaskedLMPredictionForMLM(input_ids, i, tokenizer)
        feature = InputFeatures(
            input_ids=input_ids_new,
            masked_lm_positions=masked_lm_positions,
            masked_lm_ids=masked_lm_labels)
        features.append(feature)
    return features


def createMaskedLMPredictionForMLM(input_ids, mask_position, tokenizer, mask_count=1):
    new_input_ids = list(input_ids)
    masked_lm_labels = [0] * len(input_ids)
    masked_lm_labels[mask_position] = input_ids[mask_position]
    masked_lm_positions = [0] * len(input_ids)
    masked_lm_positions[mask_position] = 1
    for i in range(mask_position, mask_position+mask_count):
        new_input_ids[i] = tokenizer.convert_tokens_to_ids([tokenizer.mask_token])[0]
    return new_input_ids, masked_lm_positions, masked_lm_labels


def createMaskForLM(args, mask_left, mask_right, input_ids, tokenizer):
    masked_lm_labels = [0] * len(input_ids)
    masked_lm_positions = [0] * len(input_ids)
    if args.model_type not in ["t5"]:
        for pos in range(mask_left, mask_right):
            masked_lm_labels[pos] = input_ids[pos]
            masked_lm_positions[pos] = 1
            # input_ids[pos] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
        if args.model_type in ["bart"]:
            masked_lm_labels = input_ids
            input_ids = input_ids[:mask_left] + \
                        tokenizer.convert_tokens_to_ids([tokenizer.mask_token]) + \
                        input_ids[mask_right:]
    else:
        idx = 0
        for pos in range(mask_left, mask_right):
            masked_lm_labels[idx] = input_ids[pos]
            masked_lm_positions[idx] = 1
            idx += 1
        mask_id = tokenizer.convert_tokens_to_ids(["<extra_id_0>"])[0]
        masked_lm_labels = [mask_id] + masked_lm_labels[:-1]
        masked_lm_positions = [0] + masked_lm_positions[:-1]
        input_ids = input_ids[:mask_left] + [mask_id] + input_ids[mask_right:]
    feature = InputFeatures(
        input_ids=input_ids,
        masked_lm_positions=masked_lm_positions,
        masked_lm_ids=masked_lm_labels)
    return [feature]


def createMaskForLMPerToken(args, token_count, input_ids, tokenizer):
    features = []
    begin = 1 if args.model_type in ["gpt2", "gpt_neo"] else 0
    for i in range(begin, token_count-1):
        feature = createMaskForLM(args, i, i+1, input_ids, tokenizer)
        features.extend(feature)
    return features



def convert_examples_to_features_for_finetuning(args):
    if args.model_type == "bert":
        tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)
        processor = BertProcessor(tokenizer)
    elif args.model_type == "roberta":
        tokenizer = RobertaTokenizer.from_pretrained(args.model_name_or_path)
        processor = RobertaProcessor(tokenizer)
    elif args.model_type in ["gpt2", "gpt_neo"]:
        tokenizer = GPT2Tokenizer.from_pretrained(args.model_name_or_path)
        processor = GPT2Processor(tokenizer)
    elif args.model_type == "bart":
        tokenizer = BartTokenizer.from_pretrained(args.model_name_or_path)
        processor = BartProcessor(tokenizer)
    elif args.model_type == "t5":
        tokenizer = T5Tokenizer.from_pretrained(args.model_name_or_path)
        if task_name == "task1":
            processor = T5Processor(tokenizer, prefix="choose the most similar entity to")
        elif task_name == "task2":
            processor = T5Processor(tokenizer, prefix="verify:")
        elif task_name == "task3":
            processor = T5Processor(tokenizer, prefix="select concept:")
    else:
        raise ValueError

    input_data = json.load(open(args.input_file))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    with open(args.output_file, "w") as f:
        for example in tqdm(input_data):
            features = processor.convert_example_to_features(example)
            f.write(json.dumps(features)+"\n")


if __name__ == "__main__":
    task_name, task_type = sys.argv[1], sys.argv[2]
    if task_name == "task1":
        if task_type == "mc":
            from task1.finetuning_data_processor_for_multiple_choice import (
                BertProcessor, RobertaProcessor, GPT2Processor, BartProcessor
            )
            from task1.finetuning_data_processor_for_question_answering import T5Processor
        elif task_type == "qa":
            from task1.finetuning_data_processor_for_question_answering import (
                BertProcessor, RobertaProcessor, GPT2Processor, BartProcessor, T5Processor
            )
    elif task_name == "task2":
        if task_type == "sc":
            from task2.finetuning_data_processor import (
                    BertProcessor, RobertaProcessor, GPT2Processor, BartProcessor
                )
        from task2.finetuning_data_processor import T5Processor
    elif task_name == "task3":
        if task_type == "mc":
            from task3.finetuning_data_processor_for_multiple_choice import (
                BertProcessor, RobertaProcessor, GPT2Processor, BartProcessor
            )
            from task3.finetuning_data_processor_for_question_answering import T5Processor
        elif task_type == "qa":
            from task3.finetuning_data_processor_for_question_answering import (
                BertProcessor, RobertaProcessor, GPT2Processor, BartProcessor, T5Processor
            )
    else:
        raise ValueError("No such task.")

    model_types = [
        ["bert", "bert-base-uncased"],
        ["roberta", "roberta-base"],
        ["gpt2", "gpt2"],
        ["gpt_neo", "EleutherAI/gpt-neo-125M"],
        ["bart", "facebook/bart-base"],
        ["t5", "t5-small"]
    ]
    for model_type in model_types:
        # for type in ["iid", "ood"]:
        for type in ["iid"]:
            for split in ["train", "dev", "test"]:
                params = {
                    "task_name": task_name,
                    "model_type": model_type[0],
                    "model_name_or_path": model_type[1],
                    "input_file": os.path.join(os.path.join(os.path.join(task_name, "data"), type), split+".json"),
                    "output_dir": os.path.join(os.path.join(os.path.join(os.path.join(task_name, "data"), type), task_type), model_type[0]),
                }
                if model_type[0] in ["t5"]:
                    params["output_dir"] = \
                        os.path.join(os.path.join(os.path.join(os.path.join(task_name, "data"), type), "qa"), model_type[0])
                params["output_file"] = os.path.join(params["output_dir"], split+".jsonl")
                args = Namespace(**params)
                print(args)
                convert_examples_to_features_for_finetuning(args)


