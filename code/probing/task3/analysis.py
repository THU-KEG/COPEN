import os 
import json

from pathlib import Path


def _analyze_error_type(item):
    all_paths = []
    for path in item["label"]:
        _path = []
        for label in path:
            _path.append(label["con"].split("_")[-1])
        all_paths.append(_path)
    label = item["my_label"].split("_")[-1]
    pred = item["pred"].split("_")[-1]
    disambiguation = True 
    # pdb.set_trace()
    for path in all_paths:
        if label in path and pred in path:
            disambiguation = False
            break
    return disambiguation


def analyze_error_type():
    model_names = [
        "bert-base-uncased",
        "roberta-base",
        "gpt2",
        "gpt-neo-125M",
        "bart-base",
        "t5-base"
    ]
    for model_name in model_names:
        print("-"*5, model_name, "-"*5)
        if model_name in ["bert-base-uncased", "t5-base"]:
            input_path = f"output/all-{model_name}-probs.json"
        else:
            input_path = f"output/concept-{model_name}-probs.json"
        data = json.load(open(input_path))
        result = {
            "disambiguation": 0,
            "in-path": 0
        }
        total_errors = 0
        for item in data:
            if item["my_label"].split("_")[-1] == item["pred"].split("_")[-1]:
                continue
            if len(item["label"]) < 2:
                continue
            if item["id"] == "easy":
                continue
            disambiguation = _analyze_error_type(item)
            if disambiguation:
                result["disambiguation"] += 1
            else:
                result["in-path"] += 1
            total_errors += 1
        print(result)
        result["disambiguation"] /= total_errors
        result["in-path"] /= total_errors
        result["disambiguation"] =  round(result["disambiguation"]*100, 1)
        result["in-path"] = round(result["in-path"]*100, 1)
        print(result)


def _get_context_free_pred(context_free, context, output_path):
    entity2concepts = dict()
    assert len(context_free) == len(context)
    for i in range(len(context_free)):
        item_free = context_free[i]
        item = context[i]
        assert item["sentence"] == item_free["sentence"]
        assert item["my_label"] == item_free["my_label"]
        entity = item["entity"]["name"]
        if item["entity"]["name"] in entity2concepts:
            entity2concepts[entity]["context-free"].append(item_free["pred"].split("_")[-1])
            entity2concepts[entity]["contextual"].append(item["pred"].split("_")[-1])
            entity2concepts[entity]["label"].append(item["my_label"].split("_")[-1])
        else:
            entity2concepts[entity] = {
                "context-free": [item_free["pred"].split("_")[-1],],
                "contextual": [item["pred"].split("_")[-1], ],
                "label": [item["my_label"].split("_")[-1],]
            }
    json.dump(entity2concepts, open(output_path, "w"), indent=4)


def get_context_free_pred():
    model_names = [
        "bert-base-uncased",
        "roberta-base",
        "gpt2",
        "gpt-neo-125M",
        "bart-base",
        "t5-base"
    ]
    output_dir = Path("output/out-of-context/comparison")
    output_dir.mkdir(exist_ok=True)
    for model_name in model_names:
        print(f"---{model_name}---")
        if model_name in ["bert-base-uncased", "t5-base"]:
            context_free = json.load(open(f"output/out-of-context/all-{model_name}-probs.json"))
            context = json.load(open(f"output/all-{model_name}-probs.json"))
        else:
            context_free = json.load(open(f"output/out-of-context/concept-{model_name}-probs.json"))
            context = json.load(open(f"output/concept-{model_name}-probs.json"))
        output_path = os.path.join(output_dir, f"{model_name}.json")
        _get_context_free_pred(context_free, context, output_path)


def analyze_context_errors():
    model_names = [
        "bert-base-uncased",
        "roberta-base",
        "gpt2",
        "gpt-neo-125M",
        "bart-base",
        "t5-base"
    ]
    for model_name in model_names:
        result = json.load(open(os.path.join("output/out-of-context/comparison", f"{model_name}.json")))
        tot_err, same_err = 0, 0 
        for key in result.keys():
            assert len(result[key]["context-free"]) == len(result[key]["contextual"])
            assert len(result[key]["context-free"]) == len(result[key]["label"])
            tot_ins = len(result[key]["context-free"])
            for i in range(tot_ins):
                if result[key]["contextual"][i] != result[key]["label"][i]:
                    tot_err += 1
                    if result[key]["contextual"][i] == result[key]["context-free"][i]:
                        same_err += 1
        print(model_name, round(same_err/tot_err*100, 1))


if __name__ == "__main__":
    # get_context_free_pred()
    # analyze_context_errors()
    analyze_error_type()


