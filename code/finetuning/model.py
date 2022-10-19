import pdb
import torch
import torch.nn as nn 
import torch.nn.functional as F

from transformers import (
    BertConfig, 
    BertTokenizer, 
    BertForSequenceClassification,
    BertForMultipleChoice,
    BertForQuestionAnswering
)
from transformers import (
    RobertaConfig, 
    RobertaTokenizer, 
    RobertaForSequenceClassification,
    RobertaForMultipleChoice,
    RobertaForQuestionAnswering
)
from transformers import (
    GPT2Config, 
    GPT2Tokenizer, 
    GPT2ForSequenceClassification,
    GPT2DoubleHeadsModel
)
from transformers import (
    GPTNeoConfig,
    GPT2Tokenizer,
    GPTNeoForSequenceClassification
)
from transformers import (
    BartConfig, 
    BartTokenizer, 
    BartForSequenceClassification
)
from transformers import (
    T5Config, 
    T5Tokenizer, 
    T5ForConditionalGeneration
)
from transformers.models.bart.modeling_bart import BartForQuestionAnswering

from transformers.utils.model_parallel_utils import assert_device_map, get_device_map
from transformers.models.bart.modeling_bart import BartForQuestionAnswering

from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.modeling_outputs import  SequenceClassifierOutputWithPast
LARGE_NUM = 100000000



class GPT2ForSequenceClassification(GPT2ForSequenceClassification):
    # @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def parallelize(self, device_map=None):
        self.device_map = (
            get_device_map(len(self.transformer.h), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.transformer.h))
        self.transformer.parallelize(self.device_map)
        self.score = self.score.to(self.transformer.first_device)
        self.model_parallel = True

    def deparallelize(self):
        self.transformer.deparallelize()
        self.transformer = self.transformer.to("cpu")
        self.score = self.score.to("cpu")
        self.model_parallel = False
        torch.cuda.empty_cache()
    
    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        hidden_states = hidden_states.to(self.score.weight.device)
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size, sequence_length = input_ids.shape[:2]
        else:
            batch_size, sequence_length = inputs_embeds.shape[:2]

        assert (
            self.config.pad_token_id is not None or batch_size == 1
        ), "Cannot handle batch sizes > 1 if no padding token is defined."
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = torch.ne(input_ids, self.config.pad_token_id).sum(-1) - 1
            else:
                sequence_lengths = -1
                logger.warning(
                    f"{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be "
                    f"unexpected if using padding tokens in conjunction with `inputs_embeds.`"
                )

        pooled_logits = logits[range(batch_size), sequence_lengths]

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


class BertForMultipleChoice(BertForMultipleChoice):
    def __init__(self, config, args):
        super().__init__(config)
        self.args = args

    def forward(
        self,
        input_ids,
        attention_mask,
        choice_mask,
        labels
    ):
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        logits = outputs.logits + (1 - choice_mask) * -LARGE_NUM
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits, labels)
        return dict(
            loss=loss,
            logits=logits
        )


class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.out_proj(x)
        return x


class RobertaForMultipleChoice(RobertaForMultipleChoice):
    def __init__(self, config, args):
        super().__init__(config)
        self.args = args

    def forward(
        self,
        input_ids,
        attention_mask,
        choice_mask,
        labels
    ):
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        logits = outputs.logits + (1 - choice_mask) * -LARGE_NUM
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits, labels)
        return dict(
            loss=loss,
            logits=logits
        )


class GPT2ForMultipleChoice(GPT2DoubleHeadsModel):
    def __init__(self, config, args):
        super().__init__(config)
        self.args = args

    def forward(
        self,
        input_ids,
        attention_mask,
        choice_mask,
        mc_token_ids,
        labels
    ):
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            mc_token_ids=mc_token_ids,
        )
        mc_logits = outputs.mc_logits + (1 - choice_mask) * -LARGE_NUM
        loss_fct = nn.CrossEntropyLoss()
        mc_loss = loss_fct(mc_logits.view(-1, mc_logits.size(-1)), labels.view(-1))
        return dict(
            loss=mc_loss,
            logits=mc_logits
        )
    

class GPTNeoForMultipleChoice(GPTNeoForSequenceClassification):
    def __init__(self, config, args):
        super().__init__(config)
        self.args = args 
    
    def forward(
        self,
        input_ids, 
        attention_mask,
        choice_mask,
        labels
    ):
        num_choices = input_ids.shape[1]
        input_ids = input_ids.view(-1, input_ids.size(-1))
        attention_mask = attention_mask.view(-1, attention_mask.size(-1))
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        reshaped_logits = outputs.logits.view(-1, num_choices)
        reshaped_logits = reshaped_logits + (1 - choice_mask) * -LARGE_NUM
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(reshaped_logits, labels)
        return dict(
            loss=loss,
            logits=reshaped_logits
        )


class BartForMultipleChoice(BartForSequenceClassification):
    def __init__(self, config, args):
        super().__init__(config)
        self.args = args

    def forward(
        self,
        input_ids,
        attention_mask,
        choice_mask,
        labels
    ):
        num_choices = input_ids.shape[1]
        input_ids = input_ids.view(-1, input_ids.size(-1))
        attention_mask = attention_mask.view(-1, attention_mask.size(-1))
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        reshaped_logits = outputs.logits.view(-1, num_choices)
        reshaped_logits = reshaped_logits + (1 - choice_mask) * -LARGE_NUM
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(reshaped_logits, labels)
        return dict(
            loss=loss,
            logits=reshaped_logits
        )


def get_model_for_sc(args, model_type, model_name_or_path, tokenizer_name, new_tokens:list = []):
    if model_type == "bert":
        config = BertConfig.from_pretrained(model_name_or_path)
        config.num_labels = 2
        config.output_hidden_states = False 
        config.output_attentions = False 
        config.keys_to_ignore_at_inference = ["loss", "hidden_states", "attentions"]
        model = BertForSequenceClassification.from_pretrained(model_name_or_path, config=config)
        tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
    elif model_type == "roberta":
        config = RobertaConfig.from_pretrained(model_name_or_path)
        config.num_labels = 2
        config.output_hidden_states = False 
        config.output_attentions = False 
        config.keys_to_ignore_at_inference = ["loss", "hidden_states", "attentions"]
        model = RobertaForSequenceClassification.from_pretrained(model_name_or_path, config=config)
        tokenizer = RobertaTokenizer.from_pretrained(tokenizer_name)
    elif model_type == "gpt2":
        config = GPT2Config.from_pretrained(model_name_or_path)
        config.num_labels = 2
        config.use_cache = False 
        config.output_hidden_states = False 
        config.output_attentions = False 
        config.keys_to_ignore_at_inference = ["loss", "past_key_values", "hidden_states", "attentions"]
        model = GPT2ForSequenceClassification.from_pretrained(model_name_or_path, config=config)
        tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_name)
        model.config.pad_token_id = model.config.eos_token_id
    elif model_type == "gpt_neo":
        config = GPTNeoConfig.from_pretrained(model_name_or_path)
        config.num_labels = 2
        config.use_cache = False
        config.output_hidden_states = False 
        config.output_attentions = False
        config.keys_to_ignore_at_inference = ["loss", "past_key_values", "hidden_states", "attentions"]
        model = GPTNeoForSequenceClassification.from_pretrained(model_name_or_path, config=config)
        tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_name)
        model.config.pad_token_id = model.config.eos_token_id
    elif model_type == "bart":
        config = BartConfig.from_pretrained(model_name_or_path)
        config.num_labels = 2
        config.use_cache = False 
        config.output_hidden_states = False 
        config.output_attentions = False 
        config.keys_to_ignore_at_inference = ["loss", "past_key_values", "decoder_hidden_states", "decoder_attentions", 
        "cross_attentions", "encoder_last_hidden_state", "encoder_hidden_states", "encoder_attentions"]
        model = BartForSequenceClassification.from_pretrained(model_name_or_path, config=config)
        tokenizer = BartTokenizer.from_pretrained(tokenizer_name)
    elif model_type == "t5":
        model = T5ForConditionalGeneration.from_pretrained(model_name_or_path)
        tokenizer = T5Tokenizer.from_pretrained(tokenizer_name)
    else:
        raise ValueError("Unsupported Model")
    
    for token in new_tokens:
        tokenizer.add_tokens(token, special_tokens = True)
    if len(new_tokens) > 0:
        model.resize_token_embeddings(len(tokenizer))

    config = model.config
    return model, tokenizer, config


def get_model_for_mc(args, model_type, model_name_or_path, tokenizer_name, new_tokens:list = []):
    if model_type == "bert":
        config = BertConfig.from_pretrained(model_name_or_path)
        config.output_hidden_states = False 
        config.output_attentions = False 
        config.keys_to_ignore_at_inference = ["loss", "hidden_states", "attentions"]
        model = BertForMultipleChoice.from_pretrained(model_name_or_path, config=config, args=args)
        tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
    elif model_type == "roberta":
        config = RobertaConfig.from_pretrained(model_name_or_path)
        config.output_hidden_states = False 
        config.output_attentions = False 
        config.keys_to_ignore_at_inference = ["loss", "hidden_states", "attentions"]
        model = RobertaForMultipleChoice.from_pretrained(model_name_or_path, config=config, args=args)
        tokenizer = RobertaTokenizer.from_pretrained(tokenizer_name)
    elif model_type == "gpt2":
        config = GPT2Config.from_pretrained(model_name_or_path)
        config.use_cache = False 
        config.output_hidden_states = False 
        config.output_attentions = False 
        model = GPT2ForMultipleChoice.from_pretrained(model_name_or_path, config=config, args=args)
        tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_name)
        model.config.pad_token_id = model.config.eos_token_id
    elif model_type == "gpt_neo":
        config = GPTNeoConfig.from_pretrained(model_name_or_path)
        config.num_labels = 1
        config.use_cache = False 
        config.output_hidden_states = False 
        config.output_attentions = False
        model = GPTNeoForMultipleChoice.from_pretrained(model_name_or_path, config=config, args=args)
        tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_name)
        model.config.pad_token_id = model.config.eos_token_id
    elif model_type == "bart":
        config = BartConfig.from_pretrained(model_name_or_path)
        config.num_labels = 1
        config.use_cache = False 
        config.output_hidden_states = False 
        config.output_attentions = False 
        model = BartForMultipleChoice.from_pretrained(model_name_or_path, config=config, args=args)
        tokenizer = BartTokenizer.from_pretrained(tokenizer_name)
    else:
        raise ValueError("Unsupported Model")
    
    for token in new_tokens:
        tokenizer.add_tokens(token, special_tokens = True)
    if len(new_tokens) > 0:
        model.resize_token_embeddings(len(tokenizer))

    config = model.config
    return model, tokenizer, config


def get_model_for_qa(args, model_type, model_name_or_path, tokenizer_name, new_tokens:list = []):
    if model_type == "bert":
        config = BertConfig.from_pretrained(model_name_or_path)
        config.num_labels = 2
        model = BertForQuestionAnswering.from_pretrained(model_name_or_path, config=config)
        tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
    elif model_type == "roberta":
        config = RobertaConfig.from_pretrained(model_name_or_path)
        config.num_labels = 2
        model = RobertaForQuestionAnswering.from_pretrained(model_name_or_path, config=config)
        tokenizer = RobertaTokenizer.from_pretrained(tokenizer_name)
    elif model_type == "bart":
        config = BartConfig.from_pretrained(model_name_or_path)
        config.num_labels = 1
        model = BartForQuestionAnswering.from_pretrained(model_name_or_path, config=config, args=args)
        tokenizer = BartTokenizer.from_pretrained(tokenizer_name)
    elif model_type == "t5":
        config = T5Config.from_pretrained(model_name_or_path)
        if args.freeze_backbone_parameters:
            config.tie_word_embeddings = False 
        model = T5ForConditionalGeneration.from_pretrained(model_name_or_path, config=config)
        tokenizer = T5Tokenizer.from_pretrained(tokenizer_name)
    else:
        raise ValueError("Unsupported Model")
    
    for token in new_tokens:
        tokenizer.add_tokens(token, special_tokens = True)
    if len(new_tokens) > 0:
        model.resize_token_embeddings(len(tokenizer))

    config = model.config
    return model, tokenizer, config


def freeze_backbone_parameters(args, model, config):
    if args.model_type == "bert":
        for name, param in model.named_parameters():
            if "classifier" in name:
                print("---%s is not frezon---" % name)
                continue
            param.requires_grad = False
    elif args.model_type == "roberta":
        if args.task_type == "sc":
            model.classifier = RobertaClassificationHead(config)
        for name, param in model.named_parameters():
            if "classifier" in name:
                print("---%s is not frezon---" % name)
                continue
            param.requires_grad = False
    elif args.model_type == "gpt2":
        for name, param in model.named_parameters():
            if "score" in name or "multiple_choice_head" in name:
                print("---%s is not frezon---" % name)
                continue
            param.requires_grad = False
    elif args.model_type == "gpt_neo":
        for name, param in model.named_parameters():
            if "score" in name:
                print("---%s is not frezon---" % name)
                continue
            param.requires_grad = False 
    elif args.model_type == "bart":
        model.classification_head = nn.Linear(config.d_model, config.num_labels)
        for name, param in model.named_parameters():
            if "classification_head" in name:
                print("---%s is not frezon---" % name)
                continue
            param.requires_grad = False
    elif args.model_type == "t5":
        for name, param in model.named_parameters():
            if "lm_head" in name:
                print("---%s is not frezon---" % name)
                continue
            param.requires_grad = False
    else:
        raise ValueError("No such model to freeze parameters.")


def get_model(args, model_type, model_name_or_path, tokenizer_name, new_tokens:list = []):
    if args.task_type == "sc":
        model, tokenizer, config = get_model_for_sc(args, model_type, model_name_or_path, tokenizer_name, new_tokens)
    elif args.task_type == "mc":
        model, tokenizer, config = get_model_for_mc(args, model_type, model_name_or_path, tokenizer_name, new_tokens)
    elif args.task_type == "qa":
        model, tokenizer, config = get_model_for_qa(args, model_type, model_name_or_path, tokenizer_name, new_tokens)
    else:
        raise ValueError("No such task.")
    if args.freeze_backbone_parameters:
        freeze_backbone_parameters(args, model, config)
    return model, tokenizer, config 





