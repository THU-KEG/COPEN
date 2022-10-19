import pdb 
import torch
import torch.nn as nn 
import torch.nn.functional as F


from transformers import BertModel, BertConfig, BertTokenizer, BertForMaskedLM
from transformers import RobertaModel, RobertaConfig, RobertaTokenizer, RobertaForMaskedLM
from transformers import GPT2Model, GPT2Config, GPT2Tokenizer, GPT2LMHeadModel
from transformers import GPTNeoForCausalLM
from transformers import BartModel, BartConfig, BartTokenizer, BartForConditionalGeneration
from transformers import T5Model, T5Config, T5Tokenizer, T5ForConditionalGeneration


class BertForConceptProbing(nn.Module):
    def __init__(self, args):
        super(BertForConceptProbing, self).__init__()
        self.bert = BertForMaskedLM.from_pretrained(args.model_name_or_path)
        self.args = args
  
    def forward(self, 
                input_ids, 
                input_mask, 
                masked_lm_positions,
                masked_labels):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=input_mask)
        logits = outputs.logits
        batch_size, seq_length = input_ids.shape[0], input_ids.shape[1]
        # compute loss 
        if not self.args.post_process_logits: 
            loss_fct = nn.CrossEntropyLoss(reduction="none")
            masked_lm_loss = loss_fct(logits.reshape(-1, logits.size(-1)), masked_labels.reshape(-1)) # (batch_size, seq_length)
            masked_lm_loss = masked_lm_loss.reshape(-1, seq_length)
            masked_lm_loss = torch.masked_select(masked_lm_loss, masked_lm_positions.to(torch.bool))
            return masked_lm_loss
        else: # return logits 
            logits = logits * (masked_lm_positions.to(torch.float32).unsqueeze(-1))
            logits = torch.softmax(logits, dim=-1) # (batch_size, seq_length, vocab_size)
            logits = logits[:, :, self.args.label_words] # (batch_size, seq_length, label_word_cnt)
            logits = torch.masked_select(logits, masked_lm_positions.unsqueeze(-1).to(torch.bool))
            logits = logits.reshape(-1, len(self.args.label_words))
            return logits

        
class RobertaForConceptProbing(nn.Module):
    def __init__(self, args):
        super(RobertaForConceptProbing, self).__init__()
        self.roberta = RobertaForMaskedLM.from_pretrained(args.model_name_or_path)
        self.args = args
  
    def forward(self, 
                input_ids, 
                input_mask, 
                masked_lm_positions,
                masked_labels):
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=input_mask)
        # pdb.set_trace()
        logits = outputs.logits
        batch_size, seq_length = input_ids.shape[0], input_ids.shape[1]
        # compute loss 
        if not self.args.post_process_logits:
            loss_fct = nn.CrossEntropyLoss(reduction="none")
            masked_lm_loss = loss_fct(logits.reshape(-1, logits.size(-1)), masked_labels.reshape(-1)) # (batch_size, seq_length)
            masked_lm_loss = masked_lm_loss.reshape(-1, seq_length)
            masked_lm_loss = torch.masked_select(masked_lm_loss, masked_lm_positions.to(torch.bool))
            return masked_lm_loss
        else: # return logits 
            logits = logits * (masked_lm_positions.to(torch.float32).unsqueeze(-1))
            logits = torch.softmax(logits, dim=-1) # (batch_size, seq_length, vocab_size)
            logits = logits[:, :, self.args.label_words] # (batch_size, seq_length, label_word_cnt)
            logits = torch.masked_select(logits, masked_lm_positions.unsqueeze(-1).to(torch.bool))
            logits = logits.reshape(-1, len(self.args.label_words))
            return logits        


class GPT2ForConceptProbing(nn.Module):
    def __init__(self, args):
        super(GPT2ForConceptProbing, self).__init__()
        self.gpt2 = GPT2LMHeadModel.from_pretrained(args.model_name_or_path)
        self.args = args
    
    def forward(self, 
                input_ids, 
                input_mask, 
                masked_lm_positions,
                masked_labels):
        labels = masked_labels
        outputs = self.gpt2(
            input_ids=input_ids,
            attention_mask=input_mask
        )
        lm_logits = outputs.logits
        # Shift so that tokens < n predict n
        shift_logits = lm_logits[..., :-1, :].contiguous() # (batch_size, seq_length-1, vocab_size)
        shift_labels = labels[..., 1:].contiguous()  # (batch_size, seq_length-1)
        if not self.args.post_process_logits:
            # flatten the tokens
            loss_fct = nn.CrossEntropyLoss(reduction="none")
            flat_loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1)) # (batch_size * (seq_length-1))
            # reshape loss 
            loss = flat_loss.reshape(-1, shift_labels.shape[1]) # (batch_size, seq_length-1)
            masked_lm_loss = torch.masked_select(loss, masked_lm_positions[:, 1:].to(torch.bool))
            return masked_lm_loss 
        else:
            logits = shift_logits * (masked_lm_positions[:, 1:].to(torch.float32).unsqueeze(-1))
            logits = torch.softmax(logits, dim=-1) # (batch_size, seq_length-1, vocab_size)
            logits = logits[:, :, self.args.label_words] # (batch_size, seq_length-1, label_word_cnt)
            logits = torch.masked_select(logits, masked_lm_positions[:, 1:].unsqueeze(-1).to(torch.bool))
            logits = logits.reshape(-1, len(self.args.label_words)) # (cnt, label_word_cnt)
            return logits    


class GPTNeoForConceptProbing(nn.Module):
    def __init__(self, args):
        super(GPTNeoForConceptProbing, self).__init__()
        self.gpt_neo = GPTNeoForCausalLM.from_pretrained(args.model_name_or_path)
        self.args = args
    
    def forward(self, 
                input_ids, 
                input_mask, 
                masked_lm_positions,
                masked_labels):
        labels = masked_labels
        outputs = self.gpt_neo(
            input_ids=input_ids,
            attention_mask=input_mask
        )
        lm_logits = outputs.logits
        # Shift so that tokens < n predict n
        shift_logits = lm_logits[..., :-1, :].contiguous() # (batch_size, seq_length-1, vocab_size)
        shift_labels = labels[..., 1:].contiguous()  # (batch_size, seq_length-1)
        if not self.args.post_process_logits:
            # flatten the tokens
            loss_fct = nn.CrossEntropyLoss(reduction="none")
            flat_loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1)) # (batch_size * (seq_length-1))
            # reshape loss 
            loss = flat_loss.reshape(-1, shift_labels.shape[1]) # (batch_size, seq_length-1)
            masked_lm_loss = torch.masked_select(loss, masked_lm_positions[:, 1:].to(torch.bool))
            return masked_lm_loss 
        else:
            logits = shift_logits * (masked_lm_positions[:, 1:].to(torch.float32).unsqueeze(-1))
            logits = torch.softmax(logits, dim=-1) # (batch_size, seq_length-1, vocab_size)
            logits = logits[:, :, self.args.label_words] # (batch_size, seq_length-1, label_word_cnt)
            logits = torch.masked_select(logits, masked_lm_positions[:, 1:].unsqueeze(-1).to(torch.bool))
            logits = logits.reshape(-1, len(self.args.label_words)) # (cnt, label_word_cnt)
            return logits    


class BartForConceptProbing(nn.Module):
    def __init__(self, args):
        super(BartForConceptProbing, self).__init__()
        self.bart = BartForConditionalGeneration.from_pretrained(args.model_name_or_path)
        self.args = args
    
    def forward(self, 
                input_ids, 
                input_mask, 
                masked_lm_positions,
                masked_labels):
        labels = masked_labels
        outputs = self.bart(
            input_ids=input_ids,
            attention_mask=input_mask,
            labels = labels
        )
        lm_logits = outputs.logits
        if not self.args.post_process_logits:
            # flatten the tokens
            loss_fct = nn.CrossEntropyLoss(reduction="none")
            flat_loss = loss_fct(lm_logits.reshape(-1, lm_logits.size(-1)), labels.reshape(-1)) # (batch_size * seq_length)
            # reshape loss 
            loss = flat_loss.reshape(-1, labels.shape[1])  # (batch_size, seq_length)
            masked_lm_loss = torch.masked_select(loss, masked_lm_positions.to(torch.bool))
            return masked_lm_loss 
        else:
            logits = lm_logits * (masked_lm_positions.to(torch.float32).unsqueeze(-1))
            logits = torch.softmax(logits, dim=-1) # (batch_size, seq_length, vocab_size)
            logits = logits[:, :, self.args.label_words] # (batch_size, seq_length, label_word_cnt)
            logits = torch.masked_select(logits, masked_lm_positions.unsqueeze(-1).to(torch.bool))
            logits = logits.reshape(-1, len(self.args.label_words))
            return logits    


class T5ForConceptProbing(nn.Module):
    def __init__(self, args):
        super(T5ForConceptProbing, self).__init__()
        self.t5 = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path)
        self.args = args
    
    def forward(self, 
                input_ids, 
                input_mask, 
                masked_lm_positions,
                masked_labels):
        labels = masked_labels
        outputs = self.t5(
            input_ids=input_ids,
            attention_mask=input_mask,
            labels = labels
        )
        lm_logits = outputs.logits
        if not self.args.post_process_logits:
            # flatten the tokens
            loss_fct = nn.CrossEntropyLoss(reduction="none")
            flat_loss = loss_fct(lm_logits.reshape(-1, lm_logits.size(-1)), labels.reshape(-1)) # (batch_size * seq_length)
            # reshape loss 
            loss = flat_loss.reshape(-1, labels.shape[1])   # (batch_size, seq_length)
            masked_lm_loss = torch.masked_select(loss, masked_lm_positions.to(torch.bool))
            return masked_lm_loss 
        else:
            logits = lm_logits * (masked_lm_positions.to(torch.float32).unsqueeze(-1))
            logits = torch.softmax(logits, dim=-1) # (batch_size, seq_length, vocab_size)
            logits = logits[:, :, self.args.label_words] # (batch_size, seq_length, label_word_cnt)
            logits = torch.masked_select(logits, masked_lm_positions.unsqueeze(-1).to(torch.bool))
            logits = logits.reshape(-1, len(self.args.label_words))
            return logits  

