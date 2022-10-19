import sys 
import json 
import random

def _random_guess(file_path):
    data = json.load(open(file_path))
    correct = 0
    for example in data:
        golden_label = example["my_label"].split("_")[-1]
        all_concepts = set()
        for path in example["label"]:
            for label in path:
                concept = label.split("_")[-1]
                all_concepts.add(concept)
        all_concepts = list(all_concepts)
        pred = random.sample(all_concepts, k=1)[0]
        correct += pred == golden_label
    print("Accuracy: %.4f, Correct: %d, Total: %d" % (correct/len(data), correct, len(data)))
    return correct / len(data)


def random_guess():
    avg_acc = 0
    all_seeds = [42, 43, 44]
    for seed in all_seeds:
        random.seed(seed)
        avg_acc += _random_guess("data/ood/test.json")
    print("Average accuracy: %.4f" % (avg_acc / len(all_seeds)))


if __name__ == "__main__":
    random_guess()
