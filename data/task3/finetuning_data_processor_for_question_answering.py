
import string 

class DataProcessor:
    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer

    def convert_example_to_features(self, example):
        raise NotImplementedError()


class T5Processor(DataProcessor):
    def __init__(self, tokenizer, prefix="select concept:") -> None:
        super().__init__(tokenizer)
        self.tokenizer = tokenizer
        self.prefix = prefix 
    
    def convert_example_to_features(self, example):
        features = dict()
        features["input"] = self.tokenizer.tokenize(self.prefix)
        features["label"] = ""
        # tokenize context
        for i, token in enumerate(example["sentence"].split()):
            if i == example["entity"]["pos"][0]:
                features["input"].append("<entity>")
            features["input"] += self.tokenizer.tokenize(" "+token)
            if i == example["entity"]["pos"][1]-1:
                features["input"].append("</entity>")
        # add candidate concepts 
        features["input"] += self.tokenizer.tokenize("Select a contextually related concept for") \
                        + self.tokenizer.tokenize(example["entity"]["name"]) \
                        + self.tokenizer.tokenize("from")
        if example["label"] != -1:
            golden_label = example["label"].split("_")[-1]
        else:
            golden_label = -1
        all_labels = string.ascii_uppercase
        all_concepts = example["candidates"]
        for i, concept in enumerate(all_concepts):
            if concept == golden_label:
                features["label"] = [all_labels[i], self.tokenizer.eos_token]
            features["input"] += self.tokenizer.tokenize(f" ({all_labels[i]}) {concept},")
        features["input"][-1:] = self.tokenizer.tokenize(" .")
        features["input"].append(self.tokenizer.eos_token)
        return features




