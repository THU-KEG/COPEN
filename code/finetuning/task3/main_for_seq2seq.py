import os 
import pdb 
import sys
sys.path.append("..")
sys.path.append("../..")
import json

from torch.utils.tensorboard import SummaryWriter
from transformers.integrations import TensorBoardCallback
from transformers import Seq2SeqTrainer
from pathlib import Path
from transformers import EarlyStoppingCallback

from cp_arguments import CPArgumentParser, CPArgs
from model import get_model
from data_processor import DatasetforQuestionAnswering
from utils import dump_result, set_seed, get_submissions

import numpy as np 


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
                                    args.model_name_or_path, new_tokens=["<entity>", "</entity>"])
model.cuda()

# dataset 
train_dataset = DatasetforQuestionAnswering(args, tokenizer, config, args.train_file)
eval_dataset = DatasetforQuestionAnswering(args, tokenizer, config, args.validation_file)



def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    if args.ignore_pad_token_for_loss:
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    # decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    # result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    # return result
    assert len(decoded_preds) == len(decoded_labels)
    correct = 0
    for i in range(len(decoded_preds)):
        if decoded_preds[i] == decoded_labels[i]:
            correct += 1
    # pdb.set_trace()
    return dict(
        accuracy=correct/len(decoded_labels)
    )
    
# Trainer 
trainer = Seq2SeqTrainer(
    args = args,
    model = model,
    train_dataset = train_dataset,
    eval_dataset = eval_dataset, 
    compute_metrics = compute_metrics,
    data_collator = train_dataset.collate_fn,
    tokenizer = tokenizer,
    callbacks = [tensorboardCallBack, earlystoppingCallBack]
)
trainer.train()

if args.do_predict:
    test_dataset = DatasetforQuestionAnswering(args, tokenizer, config, args.test_file)
    preds, labels, metrics = trainer.predict(
        test_dataset = test_dataset
    )
    # writer.add_scalar(tag="test_accuracy", scalar_value=metrics["test_accuracy"])
    print(metrics)

    # write to result file 
    dump_result(args, model_name_or_path, metrics)

    # get submissions to codalab
    all_data = json.load(open("../../../data/task3/data/ood/test.json"))
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    import string; id2char = string.ascii_uppercase
    char2id = {char: id for id, char in enumerate(id2char)}
    predictions = []
    for pred, item in zip(decoded_preds, all_data):
        predictions.append(item["candidates"][char2id[pred]])
    save_path = os.path.join(args.output_dir, "cic_submissions.json")
    get_submissions(all_data, decoded_preds, save_path)