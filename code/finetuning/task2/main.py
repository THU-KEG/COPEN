import os 
import sys
sys.path.append("..")
sys.path.append("../..")
import pdb 
import json 

import numpy as np 
from sklearn.metrics import precision_score, recall_score, f1_score

import torch 
from torch.utils.tensorboard import SummaryWriter
from transformers.integrations import TensorBoardCallback
from transformers import Trainer
from pathlib import Path
from transformers import EarlyStoppingCallback

from cp_arguments import CPArgumentParser, CPArgs
from model import get_model
from data_processor import DatasetForSequenceClassification
from metrics import compute_accuracy
from utils import dump_result, set_seed, get_submissions

# argument parser
parser = CPArgumentParser(CPArgs, description="Concept Probing")
args = parser.parse_args_into_dataclasses()
args: CPArgs
args = parser.parse_file_config(args)[0]

# to exclude parent folder 
model_name_or_path = args.model_name_or_path.split("/")[-1]
args.output_dir = args.output_dir.replace(args.model_name_or_path, model_name_or_path)
args.logging_dir = args.output_dir

# set seed
set_seed(args.seed)

# writter 
writer = SummaryWriter(args.logging_dir)
tensorboardCallBack = TensorBoardCallback(writer)
earlystoppingCallBack = EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience, early_stopping_threshold=args.early_stopping_threshold)

# model 
model, tokenizer, config = get_model(args, args.model_type, args.model_name_or_path, 
                                    args.model_name_or_path, new_tokens=[])
model.cuda()

if args.pipeline:
    num_layers = config.num_hidden_layers
    gpu_numbers = torch.cuda.device_count()
    num_per_gpus = int(num_layers / gpu_numbers)
    device_map = {i : [j for j in range(num_per_gpus * i, num_per_gpus * (i + 1))] for i in range(gpu_numbers)}
    model.parallelize(device_map)

# dataset 
train_dataset = DatasetForSequenceClassification(args, tokenizer, config, args.train_file)
eval_dataset = DatasetForSequenceClassification(args, tokenizer, config, args.validation_file)

# Trainer 
trainer = Trainer(
    args = args,
    model = model,
    train_dataset = train_dataset,
    eval_dataset = eval_dataset, 
    compute_metrics = compute_accuracy,
    data_collator = train_dataset.collate_fn,
    tokenizer = tokenizer,
    callbacks = [tensorboardCallBack, earlystoppingCallBack]
)
trainer.train()

if args.do_predict:
    test_dataset = DatasetForSequenceClassification(args, tokenizer, config, args.test_file)
    preds, labels, metrics = trainer.predict(
        test_dataset = test_dataset
    )
    # writer.add_scalar(tag="test_accuracy", scalar_value=metrics["test_accuracy"])
    print(metrics)

    # write to result file 
    dump_result(args, model_name_or_path, metrics)

    # get submissions to codalab
    predictions = []
    for pred in preds:
        predictions.append(int(np.argmax(pred)))
    all_data = json.load(open("../../../data/task2/data/test.json"))
    save_path = os.path.join(args.output_dir, "cpj_submissions.json")
    get_submissions(all_data, predictions, save_path)

