import seqeval
import seqeval.metrics
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForTokenClassification
from typing import List
import argparse
import gzip
import numpy as np
import numpy as np
import os
import torch
import transformers

## Label converter functions
def split_label(label, num):
    if num == 0:
        return None
    if label[0] == "B":
        return [label] + ["I" + label[1:]] * (num - 1)
    else:
        return [label] * num

def create_tokenized_labels(labels, original_ranges, token_ranges) -> List[int]:
    new_labels = []
    tok_id = 0
    label_id = 0
    cur_tok = token_ranges[tok_id]
    tok_ranges = token_ranges[1:-1] # Remove start and end tokens
    
    for start, end in original_ranges:
        current_label = labels[label_id]
        label_id += 1
        counter = 0
        while tok_id < len(tok_ranges) and cur_tok[1] <= end:  # Iterate over token_ranges
            cur_tok = tok_ranges[tok_id]
            counter += 1
            tok_id += 1
            if cur_tok[1] == end:
                break

        new_token_labels = split_label(current_label, counter) # Create label for every token_range
        if new_token_labels:
            new_labels.extend(new_token_labels)
    return new_labels

def recombine_to_original_labels(tok_labels, original_ranges, token_ranges):
    org_labels = []
    tok_id = 0
    label_id = 0

    # Remove start and end tokens
    tok_ranges = token_ranges[1:-1] 
    # tok_labels = tok_labels[1:-1]
    
    cur_tok = token_ranges[tok_id]
    
    for i, (start, end) in enumerate(original_ranges):
        current_label = tok_labels[label_id]
        inner_labels = []
        while cur_tok[1] <= end:
            inner_labels.append(current_label)
            
            if tok_id >= len(tok_ranges) - 1:
                break
            
            tok_id += 1 
            label_id += 1
            cur_tok = tok_ranges[tok_id]
            
        if len(inner_labels) > 0:
            org_labels.append(inner_labels[0])
    
    return org_labels


class XTREMEDataset(torch.utils.data.Dataset):
    def __init__(self, path, tokenizer_path="bert-base-multilingual-cased"):

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, cache_dir="./cache")
        
        
        self.sentences = []
        self.sent_labels = []
        self.ranges = []
        
        self.tokens = []
        self.token_labels = []
        self.token_ranges = []
        
        self.label_to_num = {"[PAD]": 0,
                             "[CLS]": 1,
                             "[SEP]": 2,
                             "B-ORG": 3,
                             "I-ORG": 4,
                             "B-LOC": 5,
                             "I-LOC": 6,
                             "B-PER": 7,
                             "I-PER": 8,
                             "O": 9,
                            }
        self.num_to_label = {num: label for label, num in self.label_to_num.items()}
        
        if type(path) == str:
            self.import_data(path)
        else:
            for p in path:
                self.import_data(p)
                
        self.tokenize_data()
        
    def tokenize_data(self, max_len=512):
        for i, (sent, labels, ranges) in enumerate(zip(self.sentences, 
                                                       self.sent_labels, 
                                                       self.ranges)):
            sent_string = " ".join(sent)
            tokens = self.tokenizer(sent_string, 
                                    return_offsets_mapping=True, 
                                    max_length=512, 
                                    truncation=True)
            
            tok_ranges = tokens["offset_mapping"]
            tok_labels = create_tokenized_labels(labels, ranges, tok_ranges)

            # if len(tok_labels) != len(tok_ranges) - 2:
            #     error_sent = " ".join(tokenizer.convert_ids_to_tokens(tokens['input_ids']))
            #     print(f"ERROR IN LEN IN INDEX {i} ({len(tok_labels)}) : ({len(tok_ranges) - 2})")
            
            self.tokens.append(tokens)
            self.token_ranges.append(tok_ranges)
            self.token_labels.append(tok_labels)
            
        
    def import_data(self, path):
        counter = 0
        
        with gzip.open(path, 'r') as file:
            cur_sent = []
            cur_labels = []
            cur_range = []
            prev_idx = 0
            
            for line in file:
                # New sentence if file contains an empty line
                if not line.split():
                    self.sentences.append(cur_sent)
                    self.sent_labels.append(cur_labels)
                    self.ranges.append(cur_range)
                    cur_sent = []
                    cur_labels = []
                    cur_range = []
                    prev_idx = 0
                    continue
                    
                    
                word, label = line.decode().split()
                
                # Create unique numbered labels
                if label not in self.label_to_num.keys():
                    new_num = len(self.label_to_num)
                    self.label_to_num[label] = new_num
                    self.num_to_label[new_num] = label
                    
                # num = self.label_to_num[label]
                num = label
                
                
                # Build the word ranges for the sentence
                word_len = len(word)
                cur_range.append((prev_idx, prev_idx + word_len))
                prev_idx += word_len + 1 # add 1 for the space from later concatenation
                
                # Add the word and label to the current sentence
                cur_sent.append(word)
                cur_labels.append(num)
                
    
                
    def tokens_to_input_format(tokens):
        return NotImplemented

    def __getitem__(self, idx):
        data = {**self.tokens[idx]}
        data["sentence"] = self.sentences[idx]
        data["token_labels"] = self.token_labels[idx]
        data["original_labels"] = self.sent_labels[idx]
        data["label_to_num"] = self.label_to_num
        data["num_to_label"] = self.num_to_label
        
        return data

    def __len__(self):
        return len(self.sentences)


class CollateFunctor:
    def __init__(self, tokenizer, max_len, device):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.device = device
        
    def batch_to_device(self, batch):
        new_batch = {}
        for key, value in batch.items():
            if torch.is_tensor(value):
                new_batch[key] = value.to(self.device)
            else:
                new_batch[key] = value
        return new_batch

    def __call__(self, batch):
        input_ids = []
        token_type_ids = []
        attention_mask = []
        labels = []
        max_batch_len = max(len(sample["input_ids"]) for sample in batch)
        max_len = max(self.max_len, max_batch_len)
        
        # Iterate over each sample in the batch
        for i, sample in enumerate(batch):
            # Pad or truncate input_ids, token_type_ids, attention_mask
            toks_labels_numerical = [sample["label_to_num"][label] for label in sample["token_labels"]]
            toks_labels_numerical = [1, *toks_labels_numerical, 2] # Add start and end token

            cur_ids = sample['input_ids']
            
            input_ids.append(torch.tensor(cur_ids))
            token_type_ids.append(torch.tensor(sample["token_type_ids"]))
            attention_mask.append(torch.tensor(sample["attention_mask"]))
            label_pad_size = (len(cur_ids) - len(toks_labels_numerical))
            labels.append(torch.tensor(toks_labels_numerical + label_pad_size * [0]))  
            # print(f" * lab:{labels[0].shape}")
            # print(f" * ids{input_ids[0].shape}\n")
            
        # Pad sequences to ensure uniform length
        input_ids = pad_sequence(input_ids,
                                 batch_first=True, 
                                 padding_value=self.tokenizer.pad_token_id)
        
        token_type_ids = pad_sequence(token_type_ids, batch_first=True, padding_value=0)
        attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
        # print(f"LEN BEFORE PAD: {len(labels)}")
        labels = pad_sequence(labels, batch_first=True, padding_value=0)
        # print(f"LEN AFTER PAD: {len(labels)}")
        labels = torch.stack([term for term in labels])
        # print(f"SHAPE AFTER STACK: {labels.shape}")
        inputs = {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

        inputs['labels'] = labels.clone().detach()
        
        shape = inputs['labels'].shape
        
        # for key, value in inputs.items():
        #     if torch.is_tensor(value):
        #         print(f"\nSHAPE: * {key}: {value.shape}")

        # print(inputs['labels'])
         
        return self.batch_to_device(inputs)