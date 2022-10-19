from ast import boolop
from asyncio import FastChildWatcher
from collections import defaultdict
from curses import meta
from email.policy import default
import pdb 
import json
import dataclasses

from argparse import Namespace
from dataclasses import dataclass, field
from pathlib import Path
from re import I
from typing import Optional, List

import yaml
from transformers import HfArgumentParser, TrainingArguments


@dataclass
class CPArgs(TrainingArguments):
    # global task config #
    task_type: str = field(
        default="",
        metadata={
            "help": "which task type to perform"
        }
    )
    task_name: str = field(
        default="",
        metadata={
            "help": "which task to perform"
        }
    )

    # i/o config #
    train_file: Optional[str] = field(
        default=None, 
        metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, 
        metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_file: Optional[str] = field(
        default=None, 
        metadata={"help": "A csv or a json file containing the test data."}
    )
    input_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "The input dir for probing."
        }
    )
    input_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The input path for probing."
        }
    )
    conf_path: str = field(
        default=None,
        metadata={
            "help": "config file"
        }        
    )

    # probing config # 
    mask_position: str = field(
        default="",
        metadata={
            "help": "which probing mode to use"
        }
    )
    post_process_logits: bool = field(
        default=False,
        metadata={
            "help": "whether post process logits"
        }
    )
    recompute: bool = field(
        default=False,
        metadata={
            "help": "whether recompute logits"
        }
    )

    # training config # 
    num_choices: int = field(
        default=21,
        metadata={
            "help": "Number of choices"
        }
    )
    grid_search: bool = field(
        default=False,
        metadata={
            "help": "whether grid search"
        }
    )
    pipeline: bool = field(
        default=False,
        metadata={
            "help": "whether to use model parallel"
        }
    )
    freeze_backbone_parameters: bool = field(
        default=False,
        metadata={
            "help": "Whether to freeze language model params."
        }
    )
    early_stopping_patience: int = field(
        default=5,
        metadata={
            "help": "early stopping patience"   
        }
    )
    early_stopping_threshold: float = field(
        default=0.005,
        metadata={
            "help": "threshold within which patience add 1"
        }
    )

    # model config # 
    model_type: str = field(
        default="bert",
        metadata={
            "help": "transformer model",
            "choices": {"bert", "roberta", "gpt2", "gpt_neo", "bart", "t5"}
        }
    )
    model_name_or_path: str = field(
        default='bert-base-uncased',
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )

    # data config # 
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_output_length: int = field(
        default=128, 
        metadata={
            "help": "The maximum output length for encoder-decoder architecture (BART, T5)."
        }
    )
    generation_max_length: int = field(
        default=128, 
        metadata={
            "help": "The maximum generation length for the decoder of T5."
        }
    )
    generation_num_beams: int = field(
        default=3, 
        metadata={
            "help": "The beam size for the decoder of T5."
        }
    )
    ignore_pad_token_for_loss: bool = field(
        default=False, 
        metadata={
            "help": ""
        }
    )
    predict_with_generate: bool = field(
        default=False, 
        metadata={
            "help": ""
        }
    )


class CPArgumentParser(HfArgumentParser):
    def parse_cli_args(self, args=None, namespace=None):
        # make sure that args are mutable
        args = list(args)

        # default Namespace built from parser defaults
        if namespace is None:
            namespace = Namespace()

        # parse the arguments and exit if there are any errors
        _UNRECOGNIZED_ARGS_ATTR = '_unrecognized_args'
        namespace, args = self._parse_known_args(args, namespace)
        if hasattr(namespace, _UNRECOGNIZED_ARGS_ATTR):
            args.extend(getattr(namespace, _UNRECOGNIZED_ARGS_ATTR))
            delattr(namespace, _UNRECOGNIZED_ARGS_ATTR)
        return namespace, args


    def parse_args_into_dataclasses(self, **kwargs):
        from sys import argv 
        args = argv[1:]
        namespace, _ = self.parse_cli_args(args=args)
        return namespace 


    def parse_file_config(self, cli_args):
        conf_path = Path(cli_args.conf_path)
        if conf_path.suffix in ['.yml', '.yaml']:
            with open(conf_path) as f:
                conf = yaml.safe_load(f)
        elif conf_path.suffix == '.json':
            with open(conf_path) as f:
                conf = json.load(f)
        else:
            raise ValueError(f'Not Supported Format: {conf_path.suffix}')
        namespace, _ = self.parse_known_args(args=None, namespace=Namespace(**conf))
        outputs = []
        for dtype in self.dataclass_types:
            keys = {f.name for f in dataclasses.fields(dtype) if f.init}
            inputs = {k: v for k, v in vars(namespace).items() if k in keys}
            # cli args first
            for k, v in vars(cli_args).items():
                inputs[k] = v
            for k in keys:
                delattr(namespace, k)
            obj = dtype(**inputs)
            outputs.append(obj)
        if len(namespace.__dict__) > 0:
            # additional namespace.
            outputs.append(namespace)
        
        return (*outputs,)


