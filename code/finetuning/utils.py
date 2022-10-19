import os 
import json 
import torch 
import random 

from pathlib import Path 
import numpy as np 


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def dump_result(args, model_name_or_path, metrics):
    # write to result file 
    checkpoint_dir = Path(os.path.join("results", str(args.seed)))
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    output_file = os.path.join(checkpoint_dir, model_name_or_path+".json")
    # split type and FT/LP
    split_type = "ood" if "ood" in args.train_file else "iid"
    assert split_type in args.train_file
    tuning_type = "FT" if not args.freeze_backbone_parameters else "LP"
    result_name = f"{split_type}-{tuning_type}"
    # grid search 
    if args.grid_search:
        result_name += f"-{args.num_train_epochs}-{args.learning_rate}-{args.per_device_train_batch_size}-{args.gradient_accumulation_steps}"
    # save to file 
    if os.path.exists(output_file):
        data = json.load(open(output_file, "r"))
        if result_name in data:
            print("---Warning! %s has existed in %s---" % (result_name, output_file))
    else:
        data = dict()
    for key in metrics.keys():
        if key == "test_accuracy":
            data[result_name] = metrics["test_accuracy"]*100
        else:
            data[result_name+"_"+key] = metrics[key]
    json.dump(data, open(output_file, "w"), indent=4)
    # clear checkpoints 
    all_ckpts = os.listdir(args.output_dir)
    for ckpt in all_ckpts:
        if "checkpoint" in ckpt:
            ckpt_path = os.path.join(args.output_dir, ckpt)
            print(f"Removing {ckpt_path}...")
            os.system(f"rm -r {ckpt_path}")



