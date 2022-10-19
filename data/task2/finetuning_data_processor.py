
import os
import pdb 
import json

from pathlib import Path 
from tqdm import tqdm 

from transformers import BertTokenizer, RobertaTokenizer, \
                    GPT2Tokenizer, BartTokenizer, T5Tokenizer

class DataProcessor:
    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer
    
    def convert_example_to_features(self, example):
        """
        Returns:
            A python dict: {
                `input`: [`the`, `sun`, `has`, `no`, `eyes`, `.`],
                `label`: 1
            }
        """
        features = dict()
        features["input"] = self.tokenizer.tokenize(example["text"])
        features["label"] = example["label"] 
        if features["label"] not in [0, 1]:
            features["label"] = 0
            print(example)
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
    def __init__(self, tokenizer, prefix="") -> None:
        super().__init__(tokenizer)
        self.tokenizer = tokenizer
        self.prefix = prefix
    
    def convert_example_to_features(self, example):
        """
        Original Input: The sun has no eyes. 
        Processed Input: <CV> The sun has no eyes.
        Original Target: 1 
        Processed Target: True
        """
        features = dict()
        input_text = self.prefix + example["text"]
        features["input"] = self.tokenizer.tokenize(input_text) + [self.tokenizer.eos_token]
        features["label"] = "true" if example["label"] else "false"
        features["label"] = self.tokenizer.tokenize(features["label"]) + [self.tokenizer.eos_token]
        return features




