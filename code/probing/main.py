"""BERT language model predict."""

import os
import pdb
import sys 
sys.path.append("..")
import json
import pickle
import logging
import argparse
from typing import List, Tuple, Dict 
from tqdm import tqdm
from pathlib import Path

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset, dataloader

from cp_arguments import CPArgumentParser, CPArgs
from transformers import PreTrainedTokenizer, BertTokenizer, RobertaTokenizer, \
                            GPT2Tokenizer, BartTokenizer, T5Tokenizer

# argument parser
parser = CPArgumentParser(CPArgs, description="Concept Probing")
args = parser.parse_args_into_dataclasses()
args: CPArgs
args = parser.parse_file_config(args)[0]
model_name_or_path = args.model_name_or_path.split("/")[-1]
args.output_file = "{}-probs.json".format(model_name_or_path)
args.logging_file = "{}.log".format(model_name_or_path)

# set logging name
logging_dir = Path(os.path.join(args.output_dir, "logs"))
logging_dir.mkdir(parents=True, exist_ok=True)
logging_file = os.path.join(logging_dir, args.logging_file) \
                if args.logging_file is not None else None 
# set logger
logging.basicConfig(
    filename=logging_file,
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.DEBUG,
)
# define a Handler which writes INFO messages or higher to the sys.stderr
console = logging.StreamHandler()
console.setLevel(logging.INFO)
# set a format which is simpler for console use
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s -   %(message)s")
# tell the handler to use this format
console.setFormatter(formatter)
# add the handler to the root logger
logging.getLogger('').addHandler(console)
logger = logging.getLogger(__name__)
# logging arguments
logger.debug("\n"+"-"*50)
args_to_logging = ""
for k in list(vars(args).keys()):
    args_to_logging += "%s:\t%s\n" % (k, vars(args)[k])
logger.info(args_to_logging)


from utils import dumpResult
from model import BertForConceptProbing, RobertaForConceptProbing, GPT2ForConceptProbing, \
                    GPTNeoForConceptProbing, BartForConceptProbing, T5ForConceptProbing
from data_processor import ProbingDataset
from metric import computeAccuracyForCSJ, computeAccuracyForCPJ, computeAccuracyForCiC


def constructDataloader(args: CPArgs,
                        tokenizer: PreTrainedTokenizer) -> Tuple[DataLoader, List[str], List[str]]:
    dataset = ProbingDataset(args, tokenizer)
    dataloader = DataLoader(dataset=dataset, batch_size=args.per_device_eval_batch_size, shuffle=False, sampler=None, collate_fn=dataset.collate_fn)
    all_tokens = json.load(open(os.path.join(args.input_dir, "all_tokens.json")))
    all_token_markers = json.load(open(os.path.join(args.input_dir, "all_token_markers.json")))
    return dataloader, all_tokens, all_token_markers


def extend_lm_loss(all_lm_loss, lm_loss):
    all_lm_loss.extend(lm_loss)


def main(args) -> List[Dict]:
    # tokenizer 
    if args.model_type == "bert":
        tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)
        model = BertForConceptProbing(args)
    elif args.model_type == "roberta":
        tokenizer = RobertaTokenizer.from_pretrained(args.model_name_or_path)
        model = RobertaForConceptProbing(args)
    elif args.model_type == "gpt2":
        tokenizer = GPT2Tokenizer.from_pretrained(args.model_name_or_path)
        model = GPT2ForConceptProbing(args)
    elif args.model_type == "gpt_neo":
        tokenizer = GPT2Tokenizer.from_pretrained(args.model_name_or_path)
        model = GPTNeoForConceptProbing(args)
    elif args.model_type == "bart":
        tokenizer = BartTokenizer.from_pretrained(args.model_name_or_path)
        model = BartForConceptProbing(args)
    elif args.model_type == "t5":
        tokenizer = T5Tokenizer.from_pretrained(args.model_name_or_path)
        model = T5ForConceptProbing(args)
    else:
        raise ValueError
    if args.post_process_logits:
        # set label word 
        label_words = ["true", "false"]
        label_word_ids = []
        for word in label_words:
            word_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(word))
            assert len(word_ids) == 1
            label_word_ids.append(word_ids[0])
        args.label_words = label_word_ids
        logger.info("Label words: %s, %s", str(label_words), str(label_word_ids))

    model.cuda()
    model.eval()
    # dataloader
    dataloader, all_tokens, all_token_markers = constructDataloader(args, tokenizer)
    logger.info("***** Running prediction*****")
    logger.info("  Steps = %d", len(dataloader))
    logger.info("  Batch size = %d", args.per_device_eval_batch_size)
    all_lm_loss = []
    for batch in tqdm(dataloader):
        lm_loss = model(**batch).detach().cpu().tolist()
        # pdb.set_trace()
        extend_lm_loss(all_lm_loss, lm_loss)

    results = dumpResult(args, tokenizer, all_tokens, all_token_markers, all_lm_loss)

    return results


def save_result(results):
    checkpoint_dir = Path(os.path.join(args.output_dir, "results"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    output_file = os.path.join(checkpoint_dir, model_name_or_path+".json")
    # save to file 
    if os.path.exists(output_file):
        data = json.load(open(output_file, "r"))
    else:
        data = dict()
    if args.mask_position in data:
        print("---Warning! %s has existed in %s" % (args.mask_position, output_file))
    data[args.mask_position] = results
    json.dump(data, open(output_file, "w"), indent=4)


if __name__ == "__main__":
    if args.recompute:
        results = main(args)
    else:
        results = json.load(open(os.path.join(args.output_dir, args.output_file)))
    # load data 
    all_data = json.load(open(args.test_file))
    special_tokens = [args.mask_position]
    if args.task_name == "CiC":
        results = computeAccuracyForCiC(all_data, results, args, special_tokens, logger)
    elif args.task_name == "CSJ":
        results = computeAccuracyForCSJ(all_data, results, args, special_tokens, logger)
    elif args.task_name == "CPJ":
        results = computeAccuracyForCPJ(all_data, results, args, special_tokens, logger)
    save_result(results)


