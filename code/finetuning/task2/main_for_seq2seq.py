import os 
import pdb 
import sys
sys.path.append("..")
sys.path.append("../..")
import json 

import numpy as np 
from sklearn.metrics import precision_score, recall_score, f1_score

from torch.utils.tensorboard import SummaryWriter
from transformers.integrations import TensorBoardCallback
from transformers import Seq2SeqTrainer
from pathlib import Path
from transformers import EarlyStoppingCallback

from cp_arguments import CPArgumentParser, CPArgs
from model import get_model
from data_processor import DatasetforQuestionAnswering
from utils import dump_result, set_seed
from metrics import compute_consistency

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
                                    args.model_name_or_path, new_tokens=[])
model.cuda()

# dataset 
train_dataset = DatasetforQuestionAnswering(args, tokenizer, config, args.train_file)
eval_dataset = DatasetforQuestionAnswering(args, tokenizer, config, args.validation_file)



def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    # pdb.set_trace()
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
    writer.add_scalar(tag="test_accuracy", scalar_value=metrics["test_accuracy"])
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    predictions = []
    for pred in decoded_preds:
        assert pred in ["true", "false"]
        if pred == "true":
            predictions.append(1)
        else:
            predictions.append(0)
    if args.ignore_pad_token_for_loss:
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    labels = []
    for label in decoded_labels:
        assert label in ["true", "false"]
        if label == "true":
            labels.append(1)
        else:
            labels.append(0)
    metrics["Precision"]  = precision_score(labels, predictions)
    metrics["recall"]  = recall_score(labels, predictions)
    metrics["f1"]  = f1_score(labels, predictions)
    print(metrics)
    # accuracy 
    assert len(predictions) == len(labels)
    correct = 0
    for pred, label in zip(predictions, labels):
        if pred == label:
            correct += 1
    assert correct / len(labels) == metrics["test_accuracy"]
    # dump predictions
    # test_data = json.load(open("../../../data/task2/data/ood/test.json"))
    # mode = "FT"
    # if args.freeze_backbone_parameters:
    #     mode = "LP"
    # output_dir = Path(os.path.join(os.path.join("output", mode), str(args.seed)))
    # output_dir.mkdir(parents=True, exist_ok=True)
    # for idx in range(len(labels)):
    #     assert test_data[idx]["label"] == labels[idx]
    #     test_data[idx]["pred"] = predictions[idx]
    # json.dump(test_data, open(os.path.join(output_dir, f"{model_name_or_path}.json"), "w"), indent=4)
    # consist_metric = compute_consistency(test_data)
    # metrics["consistency"] = {}
    # for key, value in consist_metric.items():
    #     metrics["consistency"][key] = value 

# write to result file 
dump_result(args, model_name_or_path, metrics)
