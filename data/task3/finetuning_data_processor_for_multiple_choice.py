

def isVowel(token):
    if token[0].lower() in ["a", "e", "i", "o"]:
        return True 
    else:
        return False

class DataProcessor:
    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer

    def add_template(self, context, entity, concept):
        return context + " " + entity + (" is an " if isVowel(concept) else " is a ") + concept + "."

    def convert_example_to_features(self, example):
        features = dict()
        features["input"] = []
        features["label"] = -1
        if example["label"] != -1:
            golden_label = example["label"].split("_")[-1]
        else:
            golden_label = -1
        all_concepts = example["candidates"]
        for i, concept in enumerate(all_concepts):
            instance = self.add_template(example["sentence"], example["entity"]["name"], concept)
            features["input"].append(self.tokenizer.tokenize(instance))
            if concept == golden_label:
                features["label"] = i
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



