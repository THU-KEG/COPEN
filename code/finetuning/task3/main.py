import os  
import sys
sys.path.append("..")
sys.path.append("../..")

import torch 
from torch.utils.tensorboard import SummaryWriter
from transformers.integrations import TensorBoardCallback
from transformers import Trainer
from pathlib import Path
from transformers import EarlyStoppingCallback

from cp_arguments import CPArgumentParser, CPArgs
from model import get_model
from data_processor import DatasetForMultipleChoice, DatasetforQuestionAnswering
from metrics import compute_accuracy, compute_accuracy_for_qa
from utils import dump_result, set_seed

# argument parser
parser = CPArgumentParser(CPArgs, description="Concept Probing")
args = parser.parse_args_into_dataclasses()
args: CPArgs
args = parser.parse_file_config(args)[0]

# to exclude parent folder 
model_name_or_path = args.model_name_or_path.split("/")[-1]
args.output_dir = args.output_dir.replace(args.model_name_or_path, model_name_or_path)
args.logging_dir = args.output_dir

if args.task_type == "qa":
    args.label_names = ["start_positions", "end_positions"]
else:
    args.label_names = ["labels"]

# set seed
set_seed(args.seed)

# writter 
writer = SummaryWriter(args.logging_dir)
tensorboardCallBack = TensorBoardCallback(writer)
earlystoppingCallBack = EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience, early_stopping_threshold=args.early_stopping_threshold)

# model 
model, tokenizer, config = get_model(args, args.model_type, args.model_name_or_path, 
                                    args.model_name_or_path, new_tokens=["[CLS]"])
model.cuda()

if args.pipeline:
    num_layers = config.num_hidden_layers
    gpu_numbers = torch.cuda.device_count()
    num_per_gpus = int(num_layers / gpu_numbers)
    device_map = {i : [j for j in range(num_per_gpus * i, num_per_gpus * (i + 1))] for i in range(gpu_numbers)}
    model.parallelize(device_map)

# dataset 
train_dataset = DatasetForMultipleChoice(args, tokenizer, config, args.train_file) if args.task_type != "qa" else \
                DatasetforQuestionAnswering(args, tokenizer, config, args.train_file)
eval_dataset = DatasetForMultipleChoice(args, tokenizer, config, args.validation_file) if args.task_type != "qa" else \
                DatasetforQuestionAnswering(args, tokenizer, config, args.validation_file)

# Trainer 
trainer = Trainer(
    args = args,
    model = model,
    train_dataset = train_dataset,
    eval_dataset = eval_dataset, 
    compute_metrics = compute_accuracy if args.task_type != "qa" else compute_accuracy_for_qa,
    data_collator = train_dataset.collate_fn,
    tokenizer = tokenizer,
    callbacks = [tensorboardCallBack, earlystoppingCallBack]
)
trainer.train()

if args.do_predict:
    test_dataset = DatasetForMultipleChoice(args, tokenizer, config, args.test_file)if args.task_type != "qa" else \
                DatasetforQuestionAnswering(args, tokenizer, config, args.test_file)
    preds, labels, metrics = trainer.predict(
        test_dataset = test_dataset
    )
    writer.add_scalar(tag="test_accuracy", scalar_value=metrics["test_accuracy"])
    print(metrics)

# write to result file 
dump_result(args, model_name_or_path, metrics)
