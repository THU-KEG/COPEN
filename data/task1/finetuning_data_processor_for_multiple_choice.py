
import enum
import os
import pdb 
import json
import random 

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
        features["input"] = []
        features["label"] = -1 
        for i, candidate in enumerate(all_candidates):
            if candidate["id"] == example["y"]["id"]:
                features["label"] = i 
            instance = self.add_template(example["query"]["name"], candidate["name"])
            features["input"].append(self.tokenizer.tokenize(instance))
        assert features["label"] in range(0, len(all_candidates))
        return features

    
class BertProcessor(DataProcessor):
    def __init__(self, tokenizer) -> None:
        super().__init__(tokenizer)
    
    def convert_example_to_features(self, example):
        features = super().convert_example_to_features(example)
        final_inputs = []
        for instance in features["input"]:
            instance = [self.tokenizer.cls_token] + instance + [self.tokenizer.sep_token]
            final_inputs.append(instance)
        features["input"] = final_inputs
        return features


class RobertaProcessor(DataProcessor):
    def __init__(self, tokenizer) -> None:
        super().__init__(tokenizer)
    
    def convert_example_to_features(self, example):
        features = super().convert_example_to_features(example)
        final_inputs = []
        for instance in features["input"]:
            instance = [self.tokenizer.bos_token] + instance + [self.tokenizer.eos_token]
            final_inputs.append(instance)
        features["input"] = final_inputs
        return features


class GPT2Processor(DataProcessor):
    def __init__(self, tokenizer) -> None:
        super().__init__(tokenizer)
    
    def convert_example_to_features(self, example):
        features = super().convert_example_to_features(example)
        final_inputs = []
        for instance in features["input"]:
            instance = [self.tokenizer.bos_token] + instance + [self.tokenizer.eos_token]
            final_inputs.append(instance)
        features["input"] = final_inputs
        return features


class BartProcessor(DataProcessor):
    def __init__(self, tokenizer) -> None:
        super().__init__(tokenizer)
    
    def convert_example_to_features(self, example):
        features = super().convert_example_to_features(example)
        final_inputs = []
        for instance in features["input"]:
            instance = instance + [self.tokenizer.eos_token]
            final_inputs.append(instance)
        features["input"] = final_inputs
        return features


class T5Processor(DataProcessor):
    def __init__(self, tokenizer) -> None:
        super().__init__(tokenizer)
        self.tokenizer = tokenizer
    
    def convert_example_to_features(self, example):
        pass



