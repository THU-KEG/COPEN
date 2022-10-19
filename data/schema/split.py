import os 
import json 
import random 

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


def split_top_concepts():
    # split schema to train/test
    schema = parse_schema("/data/ph/Concept-Prompt/utils/wikidata-process-toolkits/data/ontology/father.txt")
    top_concepts = []
    for con in schema.keys():
        if schema[con] == "Root":
            top_concepts.append(con)
    random.shuffle(top_concepts)
    print("%s top concepts" % str(len(top_concepts)))
    flag = len(top_concepts) // 2
    train_tcs = top_concepts[:flag]
    test_tcs = top_concepts[flag:]
    print("Train top concepts: %s" % json.dumps(train_tcs))
    print("Test top concepts: %s" % json.dumps(test_tcs))
    json.dump(train_tcs, open("train_top_concepts.json", "w"))
    json.dump(test_tcs, open("test_top_concepts.json", "w"))


if __name__ == "__main__":
    random.seed(42)    
    split_top_concepts()