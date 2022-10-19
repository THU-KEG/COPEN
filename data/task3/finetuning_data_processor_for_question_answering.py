
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

    def convert_example_to_features(self, example):
        features = dict()
        features["input"] = self.tokenizer.tokenize(example["sentence"])
        features["start_position"] = -1 
        features["end_position"] = -1
        golden_label = example["my_label"].split("_")[-1]
        all_concepts = set()
        for path in example["label"]:
            for label in path:
                concept = label.split("_")[-1]
                all_concepts.add(concept)
        all_concepts = list(all_concepts)
        for i, concept in enumerate(all_concepts):
            if concept == golden_label:
                features["start_position"] = len(features["input"])
            features["input"] += self.tokenizer.tokenize(" "+concept+",")
            if concept == golden_label:
                features["end_position"] = len(features["input"]) - 1
        features["input"][-1:] = self.tokenizer.tokenize(" .")
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
    def __init__(self, tokenizer, prefix="select concept:") -> None:
        super().__init__(tokenizer)
        self.tokenizer = tokenizer
        self.prefix = prefix 
    
    def convert_example_to_features(self, example):
        features = dict()
        features["input"] = self.tokenizer.tokenize(self.prefix)
        features["label"] = ""
        # tokenize context
        for i, token in enumerate(example["sentence"].split()):
            if i == example["entity"]["pos"][0]:
                features["input"].append("<entity>")
            features["input"] += self.tokenizer.tokenize(" "+token)
            if i == example["entity"]["pos"][1]-1:
                features["input"].append("</entity>")
        # add candidate concepts 
        features["input"] += self.tokenizer.tokenize("Select a contextually related concept for") \
                        + self.tokenizer.tokenize(example["entity"]["name"]) \
                        + self.tokenizer.tokenize("from")
        golden_label = example["my_label"].split("_")[-1]
        all_concepts = set()
        for path in example["label"]:
            for label in path:
                concept = label.split("_")[-1]
                all_concepts.add(concept)
        all_concepts = list(all_concepts)
        all_labels = string.ascii_uppercase
        for i, concept in enumerate(all_concepts):
            if concept == golden_label:
                features["label"] = [all_labels[i], self.tokenizer.eos_token]
            features["input"] += self.tokenizer.tokenize(f" ({all_labels[i]}) {concept},")
        features["input"][-1:] = self.tokenizer.tokenize(" .")
        features["input"].append(self.tokenizer.eos_token)
        return features




