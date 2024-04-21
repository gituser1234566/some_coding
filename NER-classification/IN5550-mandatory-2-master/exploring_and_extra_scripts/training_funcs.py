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

# tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased", cache_dir="./cache")

os.environ["TOKENIZERS_PARALLELISM"] = "false"

## TRAINING
def train_epoch(model, train_loader, optimizer, lr_scheduler, device=None, print_every=1000):
    model.train()
    
    for i, batch in enumerate(train_loader):
        optimizer.zero_grad()
        inputs = {key: value.to(device) for key, value in batch.items()}
        loss = model(**inputs).loss

        # backward pass
        loss.backward()
        
        # update weights
        optimizer.step()
        lr_scheduler.step()
    return loss.item()    

@torch.no_grad()
def evaluate(model, val_loader, device):
    model.eval()
    total_correct, total_samples = 0, 0
    for batch in val_loader:
        #batch to divice
        inputs = {key: value.to(device) for key, value in batch.items()}
        #batch to model on device
        outputs = model(**inputs)
        
        
        attention_mask = inputs['attention_mask']

        # Mask the [CLS] and [SEP] token as well
        for mask in attention_mask:
            unmasked = torch.where(mask == 1)[0]
            mask[unmasked[0]] = 0
            mask[unmasked[-1]] = 0
        
        active_logits = outputs.logits.view(-1, outputs.logits.shape[-1])[attention_mask.view(-1) == 1]  
        active_labels = inputs['labels'].view(-1)[attention_mask.view(-1) == 1]

        total_correct += (active_logits.argmax(dim=1) == active_labels).sum().item()  
        total_samples += active_labels.shape[0]

    accuracy = total_correct / total_samples
    return accuracy


def compute_metrics(model, data_loader,device):
    model.eval()

    # Remove ignored index (special tokens)
    true_labels = []
    true_predictions = []
    total_correct, total_samples = 0, 0
    
    #label to numeric dict
    label_to_num =  {"B-ORG": 0,
                             "I-ORG": 1,
                             "B-LOC": 2,
                             "I-LOC": 3,
                             "B-PER": 4,
                             "I-PER": 5,
                             "O": 6,}
    #numeric to label dict
    num_to_label = {v: k for k, v in label_to_num.items()}
    
    int_to_label = num_to_label
    for batch in data_loader:
        
        #batch to divice
        inputs = {key: value.to(device) for key, value in batch.items()}
        
        #batch to model on device
        outputs = model(**inputs)
        attention_mask = inputs['attention_mask']

        # Mask the [CLS] and [SEP] token as well
        for mask in attention_mask:
            unmasked = torch.where(mask == 1)[0]
            mask[unmasked[0]] = 0
            mask[unmasked[-1]] = 0
        
        active_logits = outputs.logits.view(-1, outputs.logits.shape[-1])[attention_mask.view(-1) == 1]  
        active_labels = inputs['labels'].view(-1)[attention_mask.view(-1) == 1]
        
        total_correct += (active_logits.argmax(dim=1) == active_labels).sum().item()  
        total_samples += active_labels.shape[0]
        
        active_labels = [int_to_label[int(label)] for label in active_labels]
        active_logits = [int_to_label[int(label)] for label in active_logits.argmax(dim=1)]
        true_labels.append(active_labels)
        true_predictions.append(active_logits)
        

    results = seqeval.metrics.classification_report(true_labels, true_predictions, digits=4, output_dict=True)
    results["accuracy"] = total_correct / total_samples
    return results


