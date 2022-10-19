import os
import pdb 
import spacy 
import json 

from tqdm import tqdm 
from pathlib import Path 


def compute_hit(candidates, golden_id):
    sorted_candidates = sorted(candidates, key=lambda item: item["similarity"], reverse=True)
    if sorted_candidates[0]["id"] == golden_id:
        return 1 
    else:
        return 0


def compute_per_instance(query, candidates, nlp):
    doc_query = nlp(query["name"])
    assert doc_query.has_vector
    for candidate in candidates:
        doc_candidate = nlp(candidate["name"])
        assert doc_candidate.has_vector
        candidate["similarity"] = doc_query.similarity(doc_candidate)
    return candidates


def eval_embedding_similarity(data, model_name, output_dir, prefix="test_"):
    nlp = spacy.load(model_name)
    metric = dict(total=0, correct=0)
    for item in tqdm(data):
        candidates = [item["y"]] + item["n"]
        candidates_with_sim = compute_per_instance(item["query"], candidates, nlp)
        item["y"] = candidates_with_sim[0]
        item["n"] = candidates_with_sim[1:]
        metric["total"] += 1
        metric["correct"] += compute_hit(candidates_with_sim, item["y"]["id"])
    metric["accuracy"] = metric["correct"] / metric["total"]
    print(metric)
    json.dump(data, open(os.path.join(output_dir, prefix+"embedding_similarity.json"), "w"), indent=4)
    

if __name__ == "__main__":
    model_name = "/tmp/glove"
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    data = json.load(open("../../../data/task1/data/ood/version3/test.json"))
    eval_embedding_similarity(data, model_name, output_dir)