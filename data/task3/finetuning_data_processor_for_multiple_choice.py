
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


def isVowel(token):
    if token[0].lower() in ["a", "e", "i", "o"]:
        return True 
    else:
        return False

class DataProcessor:
    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer

    def add_template(self, context, entity, concept):
        return context + " " + entity + (" is an " if isVowel(concept) else " is a ") + concept + "."

    def convert_example_to_features(self, example):
        features = dict()
        features["input"] = []
        features["label"] = -1 
        golden_label = example["my_label"].split("_")[-1]
        all_concepts = set()
        for path in example["label"]:
            for label in path:
                concept = label.split("_")[-1]
                all_concepts.add(concept)
        all_concepts = list(all_concepts)
        for i, concept in enumerate(all_concepts):
            instance = self.add_template(example["sentence"], example["entity"]["name"], concept)
            features["input"].append(self.tokenizer.tokenize(instance))
            if concept == golden_label:
                features["label"] = i
        assert features["label"] in list(range(0, len(all_concepts)))
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
    
    def convert_example_to_features(self, example, special_token="<CV>"):
        pass 



