import os
import pdb
import json
import torch 
import logging


from typing import List, Tuple, Dict
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, TensorDataset


class ProbingDataset(Dataset):
    def __init__(self, args, tokenizer) -> None:
        self.args = args 
        self.tokenizer = tokenizer
        self.data = []
        with open(os.path.join(args.input_dir,f"data-{args.mask_position}.jsonl")) as f:
            for line in f.readlines():
                self.data.append(json.loads(line.strip()))
    
    def __len__(self):
        return len(self.data)

    
    def get_features(self, item):
        tokens = self.tokenizer.convert_tokens_to_ids(item["input"])
        input_ids = torch.zeros(self.args.max_seq_length, dtype=torch.long)
        input_mask = torch.zeros(self.args.max_seq_length, dtype=torch.float32)
        input_ids[:len(tokens)] = torch.tensor(tokens, dtype=torch.long)
        input_mask[:len(tokens)] = 1

        labels = self.tokenizer.convert_tokens_to_ids(item["output"])
        masked_labels = torch.zeros(self.args.max_output_length, dtype=torch.long)
        masked_lm_positions = torch.zeros(self.args.max_output_length, dtype=torch.long)
        masked_labels[:len(labels)] = torch.tensor(labels, dtype=torch.long)
        masked_lm_positions[:len(labels)] = torch.tensor(item["output_mask"], dtype=torch.long)
        
        return {
            "input_ids": input_ids,
            "input_mask": input_mask,
            "masked_labels": masked_labels,
            "masked_lm_positions": masked_lm_positions
        }


    def __getitem__(self, index):
        return self.get_features(self.data[index])
    
    
    def collate_fn(self, batch):
        output_batch = dict()
        for key in batch[0].keys():
            output_batch[key] = torch.stack([x[key] for x in batch], dim=0).cuda()
        input_length = int(output_batch["input_mask"].sum(-1).max())
        for key in ["input_ids", "input_mask"]:
            output_batch[key] = output_batch[key][:, :input_length]
        output_length = input_length
        for key in ["masked_labels", "masked_lm_positions"]:
            output_batch[key] = output_batch[key][:, :output_length]
        return output_batch
            
