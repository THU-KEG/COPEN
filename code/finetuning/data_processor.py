import os
import pdb
import json
import torch 
import logging


from typing import List, Tuple, Dict
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, TensorDataset

from transformers import BertTokenizer, RobertaTokenizer, GPT2Tokenizer, \
                            BartTokenizer, T5Tokenizer


class DatasetForSequenceClassification(Dataset):
    def __init__(self, args, tokenizer, config, input_file) -> None:
        self.args = args 
        self.tokenizer = tokenizer
        self.config = config
        self.data = []
        with open(input_file) as f:
            for line in f.readlines():
                self.data.append(json.loads(line.strip()))
    
    def __len__(self):
        return len(self.data)

    
    def get_features(self, item):
        tokens = self.tokenizer.convert_tokens_to_ids(item["input"])
        input_ids = torch.zeros(self.args.max_seq_length, dtype=torch.long) + self.config.pad_token_id
        input_mask = torch.zeros(self.args.max_seq_length, dtype=torch.float32)
        length = min(len(tokens), self.args.max_seq_length)
        input_ids[:length] = torch.tensor(tokens, dtype=torch.long)[:length]
        input_mask[:length] = 1

        feature_dict = dict()
        feature_dict["input_ids"] = input_ids
        feature_dict["attention_mask"] = input_mask
        labels = torch.tensor(item["label"], dtype=torch.long)
        feature_dict["labels"] = labels
        return feature_dict


    def __getitem__(self, index):
        return self.get_features(self.data[index])
    
    
    def collate_fn(self, batch):
        output_batch = dict()
        for key in batch[0].keys():
            output_batch[key] = torch.stack([x[key] for x in batch], dim=0)
        input_length = int(output_batch["attention_mask"].sum(-1).max())
        for key in ["input_ids", "attention_mask"]:
            output_batch[key] = output_batch[key][:, :input_length]
        return output_batch


class DatasetForMultipleChoice(Dataset):
    def __init__(self, args, tokenizer, config, input_file) -> None:
        self.args = args 
        self.tokenizer = tokenizer
        self.config = config
        self.data = []
        with open(input_file) as f:
            for line in f.readlines():
                self.data.append(json.loads(line.strip()))
        self.num_choices = args.num_choices
    
    def __len__(self):
        return len(self.data)

    
    def get_features(self, item):
        input_ids = torch.zeros(self.num_choices, self.args.max_seq_length, dtype=torch.long) + self.config.pad_token_id
        input_mask = torch.zeros(self.num_choices, self.args.max_seq_length, dtype=torch.float32)
        choice_mask = torch.zeros(self.num_choices, dtype=torch.float32)
        mc_token_ids = torch.zeros(self.num_choices, dtype=torch.long)
        if self.args.model_type == "bart":
            for i in range(self.num_choices):
                input_ids[i, 0] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.eos_token)
        for i, instance in enumerate(item["input"]):
            _input_ids = self.tokenizer.convert_tokens_to_ids(instance)
            length = min(len(_input_ids), self.args.max_seq_length)
            input_ids[i, :length] = torch.tensor(_input_ids, dtype=torch.long)[:length]
            input_mask[i, :length] = 1
            choice_mask[i] = 1
            if len(_input_ids) > self.args.max_seq_length:
                if self.args.model_type in ["roberta", "gpt2", "gpt_neo", "bart", "t5"]:
                    input_ids[i, -1] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.eos_token)
                elif self.args.model_type in ["bert"]:
                    input_ids[i, -1] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.sep_token)
            if self.args.model_type == "gpt2":
                cls_position = min(len(_input_ids), self.args.max_seq_length-1)
                input_ids[i, cls_position] = self.tokenizer.convert_tokens_to_ids("[CLS]")
                input_mask[i, cls_position] = 1
                mc_token_ids[i] = cls_position

        feature_dict = dict()
        feature_dict["input_ids"] = input_ids
        feature_dict["attention_mask"] = input_mask
        feature_dict["choice_mask"] = choice_mask
        if self.args.model_type == "gpt2":
            feature_dict["mc_token_ids"] = mc_token_ids

        if self.args.model_type == "t5":
            _labels = self.tokenizer.convert_tokens_to_ids([item["label"]])
            token_labels = torch.zeros(self.args.max_output_length, dtype=torch.long)
            token_labels[:len(_labels)] = torch.tensor(_labels, dtype=torch.long)
            feature_dict["token_labels"] = token_labels
            labels = torch.tensor(-1, dtype=torch.long)
        else:
            labels = torch.tensor(item["label"], dtype=torch.long)
        feature_dict["labels"] = labels
        return feature_dict


    def __getitem__(self, index):
        return self.get_features(self.data[index])
    
    
    def collate_fn(self, batch):
        output_batch = dict()
        for key in batch[0].keys():
            output_batch[key] = torch.stack([x[key] for x in batch], dim=0)
        input_length = int(output_batch["attention_mask"].sum(-1).max())
        for key in ["input_ids", "attention_mask"]:
            output_batch[key] = output_batch[key][:, :, :input_length]
        return output_batch
            

class DatasetforQuestionAnswering(Dataset):
    def __init__(self, args, tokenizer, config, input_file) -> None:
        self.args = args 
        self.tokenizer = tokenizer
        self.config = config
        self.data = []
        with open(input_file) as f:
            for line in f.readlines():
                self.data.append(json.loads(line.strip()))
    
    def __len__(self):
        return len(self.data)


    def get_features(self, item):
        tokens = self.tokenizer.convert_tokens_to_ids(item["input"])
        input_ids = torch.zeros(self.args.max_seq_length, dtype=torch.long)
        if self.args.model_type == "gpt2":
            input_ids += self.config.pad_token_id
        input_mask = torch.zeros(self.args.max_seq_length, dtype=torch.float32)
        length = min(len(tokens), self.args.max_seq_length)
        input_ids[:length] = torch.tensor(tokens, dtype=torch.long)[:length]
        input_mask[:length] = 1

        feature_dict = dict()
        feature_dict["input_ids"] = input_ids
        feature_dict["attention_mask"] = input_mask
        # pdb.set_trace()

        if self.args.model_type == "t5":
            _labels = self.tokenizer.convert_tokens_to_ids(item["label"] if isinstance(item["label"], list) else [item["label"]])
            token_labels = torch.zeros(self.args.max_output_length, dtype=torch.long) - 100
            token_labels[:len(_labels)] = torch.tensor(_labels, dtype=torch.long)
            feature_dict["labels"] = token_labels
            decoder_attention_mask = torch.zeros(self.args.max_output_length, dtype=torch.float32)
            decoder_attention_mask[:len(_labels)] = 1
            feature_dict["decoder_attention_mask"] = decoder_attention_mask
        else:
            start_position = torch.tensor(item["start_position"], dtype=torch.long)
            end_position = torch.tensor(item["end_position"], dtype=torch.long) 
            feature_dict["start_positions"] = start_position
            feature_dict["end_positions"] = end_position
        return feature_dict


    def __getitem__(self, index):
        return self.get_features(self.data[index])
    
    
    def collate_fn(self, batch):
        output_batch = dict()
        for key in batch[0].keys():
            output_batch[key] = torch.stack([x[key] for x in batch], dim=0)
        input_length = int(output_batch["attention_mask"].sum(-1).max())
        for key in ["input_ids", "attention_mask"]:
            output_batch[key] = output_batch[key][:, :input_length]
        if "labels" in output_batch:
            output_length = int(output_batch["decoder_attention_mask"].sum(-1).max())
            for key in ["labels", "decoder_attention_mask"]:
                output_batch[key] = output_batch[key][:, :output_length]
        return output_batch
            
