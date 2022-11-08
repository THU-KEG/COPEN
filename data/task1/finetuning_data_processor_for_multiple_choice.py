

class DataProcessor:
    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer

    def add_template(self, entity1, entity2):
        return entity1 + " is conceptually similar with " + entity2 + "."

    def convert_example_to_features(self, example):
        features = dict()
        features["input"] = []
        features["label"] = -1 
        for i, candidate in enumerate(example["candidates"]):
            if candidate["id"] == example["label"]:
                features["label"] = i 
            instance = self.add_template(example["query"]["name"], candidate["name"])
            features["input"].append(self.tokenizer.tokenize(instance))
        return features

    
class BertProcessor(DataProcessor):
    def __init__(self, tokenizer) -> None:
        super().__init__(tokenizer)
    
    def convert_example_to_features(self, example):
        features = super().convert_example_to_features(example)
        final_inputs = []
        for instance in features["input"]:
            instance = [self.tokenizer.cls_token] + instance + [self.tokenizer.sep_token]
            final_inputs.append(instance)
        features["input"] = final_inputs
        return features


class RobertaProcessor(DataProcessor):
    def __init__(self, tokenizer) -> None:
        super().__init__(tokenizer)
    
    def convert_example_to_features(self, example):
        features = super().convert_example_to_features(example)
        final_inputs = []
        for instance in features["input"]:
            instance = [self.tokenizer.bos_token] + instance + [self.tokenizer.eos_token]
            final_inputs.append(instance)
        features["input"] = final_inputs
        return features


class GPT2Processor(DataProcessor):
    def __init__(self, tokenizer) -> None:
        super().__init__(tokenizer)
    
    def convert_example_to_features(self, example):
        features = super().convert_example_to_features(example)
        final_inputs = []
        for instance in features["input"]:
            instance = [self.tokenizer.bos_token] + instance + [self.tokenizer.eos_token]
            final_inputs.append(instance)
        features["input"] = final_inputs
        return features


class BartProcessor(DataProcessor):
    def __init__(self, tokenizer) -> None:
        super().__init__(tokenizer)
    
    def convert_example_to_features(self, example):
        features = super().convert_example_to_features(example)
        final_inputs = []
        for instance in features["input"]:
            instance = instance + [self.tokenizer.eos_token]
            final_inputs.append(instance)
        features["input"] = final_inputs
        return features


