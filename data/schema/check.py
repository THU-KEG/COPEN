import json 

from typing import Dict



def parse_schema(file_path: str) -> Dict[str, str]:
    """Return wikidata schema.
    Args:
        file_path: schema filepath
    Returns:
        schema: a python dict whose key is a concept and the value is 
                the subClassOf the key . 
                e.g., {"soccerPlayer": "Player"}
    """
    schema = {}
    with open(file_path) as f:
        for line in f.readlines():
            child, parent = line.strip().split("\t\t")
            assert child not in schema
            schema[child] = parent
    return schema


def flatten_schema(schema: Dict[str, str]) -> Dict[str, str]:
    flat_schema = {}
    for child in schema.keys():
        parent = schema[child]
        if parent != "Root":
            while True:
                if schema[parent] == "Root":
                    break
                parent = schema[parent]
        else:
            parent = child
        flat_schema[child] = parent
    return flat_schema


def _check_task1(data, concepts, flat_schema):
    in_data = set()
    for item in data:
        top = flat_schema[item["query"]["concept"]]
        assert top in concepts
        in_data.add(top)
    print(concepts)
    print(list(in_data))

def _check_task2(data, concepts, flat_schema):
    id2Name = {}
    with open("mapping.txt") as f:
        for line in f.readlines():
            id, name = line.strip().split("\t\t")
            id2Name[id] = name
    in_data = set()
    for item in data:
        top = flat_schema[id2Name[item["id"]]]
        assert top in concepts
        in_data.add(top)
    print(concepts)
    print(list(in_data))


def _check_task3(data, concepts, flat_schema):
    in_data = set()
    for item in data:
        top = flat_schema[item["label"][0][0].split("_")[-1]]
        assert top in concepts
        in_data.add(top)
    print(concepts)
    print(list(in_data))


def check_task(task_name):
    task_name_to_fn = {
        "task1": _check_task1,
        "task2": _check_task2,
        "task3": _check_task3
    }
    check_fn = task_name_to_fn[task_name]
    schema = parse_schema("father.txt")
    flat_schema = flatten_schema(schema)
    
    train_concepts = json.load(open("train_top_concepts.json"))
    train = json.load(open(f"../{task_name}/data/ood/train.json"))    
    dev = json.load(open(f"../{task_name}/data/ood/dev.json"))
    print("--------Train--------")
    check_fn(train, train_concepts, flat_schema)
    check_fn(dev, train_concepts, flat_schema)
    
    print("--------Test-------")
    test_concepts = json.load(open("test_top_concepts.json"))
    test = json.load(open(f"../{task_name}/data/ood/test.json"))
    check_fn(test, test_concepts, flat_schema)


def count_concept():
    schema = parse_schema("father.txt")
    flat_schema = flatten_schema(schema)
    
    train_concepts = json.load(open("train_top_concepts.json"))
    test_concepts = json.load(open("test_top_concepts.json"))
    print(len(train_concepts))
    print(len(test_concepts))
    all_train = []
    all_test = []
    for key in schema.keys():
        if flat_schema[key] in train_concepts:
            all_train.append(key)
        elif flat_schema[key] in test_concepts:
            all_test.append(key)
        else:
            print(key)
    print(len(all_train))
    print(len(all_test))


if __name__ == "__main__":
    # print("-"*20, "task1", "-"*20)
    # check_task("task1")
    # print("\n\n", "-"*20, "task2", "-"*20)
    # check_task("task2")
    # print("\n\n", "-"*20, "task3", "-"*20)
    # check_task("task3")
    count_concept()