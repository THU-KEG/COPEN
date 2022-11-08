import string


class DataProcessor:
    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer

    def add_template(self, entity1, entity2):
        return entity1 + " is conceptually similar with " + entity2 + "."

    def convert_example_to_features(self, example):
        raise NotImplementedError()


class T5Processor(DataProcessor):
    def __init__(self, tokenizer, prefix="choose the most similar entity to") -> None:
        super().__init__(tokenizer)
        self.tokenizer = tokenizer
        self.prefix = prefix
    
    def convert_example_to_features(self, example):
        features = dict()
        features["input"] = self.prefix + " " + example["query"]["name"] + ":"
        features["label"] = ""
        all_labels = string.ascii_uppercase
        for i, candidate in enumerate(example["candidates"]):
            if candidate["id"] == example["label"]:
                features["label"] = [all_labels[i], self.tokenizer.eos_token]
            features["input"] += f"({all_labels[i]})" + " " + candidate["name"] + ","
        features["input"] = self.tokenizer.tokenize(features["input"][:-1]) + [self.tokenizer.eos_token]
        return features




