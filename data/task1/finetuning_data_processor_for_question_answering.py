
import enum
import os
import pdb 
import json
import random 
import string

from pathlib import Path
from tqdm import tqdm 
from argparse import Namespace

from transformers import BertTokenizer, RobertaTokenizer, \
                    GPT2Tokenizer, BartTokenizer, T5Tokenizer

class DataProcessor:
    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer

    def add_template(self, entity1, entity2):
        return entity1 + " is conceptually similar with " + entity2 + "."

    def convert_example_to_features(self, example):
        all_candidates = [example["y"]] + example["n"]
        random.shuffle(all_candidates)
        features = dict()
        features["input"] = self.tokenizer.tokenize("Among")
        features["start_position"] = -1 
        features["end_position"] = -1
        for i, candidate in enumerate(all_candidates):
            if candidate["id"] == example["y"]["id"]:
                features["start_position"] = len(features["input"])
            features["input"] += self.tokenizer.tokenize(" "+candidate["name"]+",")
            if candidate["id"] == example["y"]["id"]:
                features["end_position"] = len(features["input"]) - 1
        features["input"] += self.tokenizer.tokenize(" which one is most conceptually similar to " 
                                                     + example["query"]["name"] + " ?")
        assert features["start_position"] != -1 and features["end_position"] != -1
        return features

    
class BertProcessor(DataProcessor):
    def __init__(self, tokenizer) -> None:
        super().__init__(tokenizer)
    
    def convert_example_to_features(self, example):
        features = super().convert_example_to_features(example)
        features["input"] = [self.tokenizer.cls_token] + features["input"] + [self.tokenizer.sep_token]
        return features

class RobertaProcessor(DataProcessor):
    def __init__(self, tokenizer) -> None:
        super().__init__(tokenizer)
    
    def convert_example_to_features(self, example):
        features = super().convert_example_to_features(example)
        features["input"] = [self.tokenizer.bos_token] + features["input"] + [self.tokenizer.eos_token]
        return features


class GPT2Processor(DataProcessor):
    def __init__(self, tokenizer) -> None:
        super().__init__(tokenizer)
    
    def convert_example_to_features(self, example):
        features = super().convert_example_to_features(example)
        features["input"] = [self.tokenizer.bos_token] + features["input"] + [self.tokenizer.eos_token]
        return features


class BartProcessor(DataProcessor):
    def __init__(self, tokenizer) -> None:
        super().__init__(tokenizer)
    
    def convert_example_to_features(self, example):
        features = super().convert_example_to_features(example)
        features["input"] = features["input"] + [self.tokenizer.eos_token]
        return features


class T5Processor(DataProcessor):
    def __init__(self, tokenizer, prefix="choose the most similar entity to") -> None:
        super().__init__(tokenizer)
        self.tokenizer = tokenizer
        self.prefix = prefix
    
    def convert_example_to_features(self, example):
        all_candidates = [example["y"]] + example["n"]
        random.shuffle(all_candidates)
        features = dict()
        features["input"] = self.prefix + " " + example["query"]["name"] + ":"
        features["label"] = ""
        all_labels = string.ascii_uppercase
        for i, candidate in enumerate(all_candidates):
            if candidate["id"] == example["y"]["id"]:
                # features["label"] = self.tokenizer.tokenize(candidate["name"])
                features["label"] = [all_labels[i], self.tokenizer.eos_token]
            features["input"] += f"({all_labels[i]})" + " " + candidate["name"] + ","
        features["input"] = self.tokenizer.tokenize(features["input"][:-1]) + [self.tokenizer.eos_token]
        assert features["label"] != ""
        return features




