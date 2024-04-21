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
from xtreme_dataset_smart import XTREMEDataset, CollateFunctor

# tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased", cache_dir="./cache")

os.environ["TOKENIZERS_PARALLELISM"] = "false"

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



## TRAINING
def train_epoch(model, train_loader, optimizer, lr_scheduler, device=None, print_every=1000):
    model.train()
    loss_value = 0
    for i, batch in enumerate(train_loader):
        optimizer.zero_grad()

        loss = model(**batch).loss

        # backward pass
        loss.backward()

        # update weights
        optimizer.step()
        lr_scheduler.step()
        
        loss_value += loss.item()
        
    return loss_value / len(train_loader)

    # data["sentence"] = self.sentences[idx]
    # data["token_labels"] = self.token_labels[idx]
    # data["original_labels"] = self.sent_labels[idx]
    # data["label_to_num"] = self.label_to_num
    # data["num_to_label"] = self.num_to_label

@torch.no_grad()
def predict(model, data_loader, device):
    model.eval()
    predictions = []
    true_labels = []
    num_to_label = data_loader.dataset.num_to_label

    for batch in data_loader:
        # Send the batch to the same device as the model
        outputs = model(**batch)
        attention_mask = batch['attention_mask']
        
        # Mask out the [CLS] and [SEP] tokens
        for mask in attention_mask:
            unmasked_positions = torch.nonzero(mask).squeeze()
            mask[unmasked_positions[0]] = 0  # [CLS]
            mask[unmasked_positions[-1]] = 0  # [SEP]

        # Get logits and labels for active parts of the loss calculation
        active_logits = outputs.logits.view(-1, outputs.logits.shape[-1])[attention_mask.view(-1) == 1]
        active_labels = batch['labels'].view(-1)[attention_mask.view(-1) == 1]

        # Predictions are the argmax of logits per token
        batch_predictions = active_logits.argmax(dim=1)

        predictions.extend(batch_predictions.tolist())
        true_labels.extend(active_labels.tolist())
    
    predictions = [num_to_label[int(label)] for label in predictions]
    true_labels = [num_to_label[int(label)] for label in true_labels]

    return predictions, true_labels

@torch.no_grad()
def evaluate(model, val_loader, device):
    model.eval()
    total_correct, total_samples = 0, 0
    for batch in val_loader:
        outputs = model(**batch)
        attention_mask = batch['attention_mask']

        # Mask the [CLS] and [SEP] token as well
        for mask in attention_mask:
            unmasked = torch.where(mask == 1)[0]
            mask[unmasked[0]] = 0
            mask[unmasked[-1]] = 0
        
        active_logits = outputs.logits.view(-1, outputs.logits.shape[-1])[attention_mask.view(-1) == 1]  
        active_labels = batch['labels'].view(-1)[attention_mask.view(-1) == 1]

        total_correct += (active_logits.argmax(dim=1) == active_labels).sum().item()  
        total_samples += active_labels.shape[0]

    accuracy = total_correct / total_samples
    return accuracy


def compute_metrics(model, data_loader):
    model.eval()

    # Remove ignored index (special tokens)
    true_labels = []
    true_predictions = []
    total_correct, total_samples = 0, 0
    int_to_label = data_loader.dataset.num_to_label
    
    for batch in data_loader:
        outputs = model(**batch)
        attention_mask = batch['attention_mask']

        # Mask the [CLS] and [SEP] token as well
        for mask in attention_mask:
            unmasked = torch.where(mask == 1)[0]
            mask[unmasked[0]] = 0
            mask[unmasked[-1]] = 0
        
        active_logits = outputs.logits.view(-1, outputs.logits.shape[-1])[attention_mask.view(-1) == 1]  
        active_labels = batch['labels'].view(-1)[attention_mask.view(-1) == 1]
        
        total_correct += (active_logits.argmax(dim=1) == active_labels).sum().item()  
        total_samples += active_labels.shape[0]
        
        active_labels = [int_to_label[int(label)] for label in active_labels]
        active_logits = [int_to_label[int(label)] for label in active_logits.argmax(dim=1)]
        true_labels.append(active_labels)
        true_predictions.append(active_logits)
        

    results = seqeval.metrics.classification_report(true_labels,
                                                    true_predictions,
                                                    digits=4,
                                                    output_dict=True)
    
    results["accuracy"] = total_correct / total_samples
    return results


# # Hyperparameters
if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description='Train a model on the SNLI dataset')
    parser.add_argument('--model', type=str, default='bert-base-multilingual-cased', help='bert-base-multilingual-cased or xlm-roberta-base')
    parser.add_argument('--batch_size', type=int, default=32, help='The batch size')
    parser.add_argument('--epochs', type=int, default=3, help='The number of epochs to train')
    parser.add_argument('--lr', type=float, default=2e-5, help='The learning rate')
    parser.add_argument('--freeze', type=bool, default=True, help='If to freeze the earlier BERT weights')
    parser.add_argument('--seed', type=int, default=42, help='The random seed')
    parser.add_argument('--warmup_steps', type=int, default=50, help='The number of warmup steps')
    parser.add_argument('--gradient_clipping', type=float, default=10.0, help='The gradient clipping value')
    parser.add_argument('--print_every', type=int, default=1000, help='Print the loss every n:th batch')

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_name = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n\n{'='*100}\n"
            f"Training model: {args.model}\n * {args.epochs} epochs\n * learning rate is {args.lr}\n"  +
            f" * batch size is {args.batch_size}\n * weights are frozen\n * device is on {device}" +
            f" * warmup steps: {args.warmup_steps}\n * printing loss on" +
            f" every 100th batch\n{'='*100}\n\n")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir="./cache")
    model = AutoModelForTokenClassification.from_pretrained(args.model, 
                                                            cache_dir="./cache", 
                                                            num_labels=10).to(device)

    # Freeze all layers of the pre-trained BERT model
    if args.freeze and args.model != "google/electra-base-discriminator":
        params = model.bert.named_parameters()

        if args.model == "xlm-roberta-base":
            model.roberta.named_parameters()
            
        for name, param in params:  
            param.requires_grad = False 


    collate = CollateFunctor(tokenizer, 512, device)

    print(f"\nData preprocessing...")
    
    train_set = XTREMEDataset("./data_split/train-en-split.tsv.gz")
    val_set = XTREMEDataset("./data_split/val-en-split.tsv.gz")
    
    train_loader = torch.utils.data.DataLoader(train_set, 
                                            batch_size=args.batch_size, 
                                            shuffle=True,
                                            drop_last=True,
                                            collate_fn=collate)
    
    val_loader = torch.utils.data.DataLoader(val_set,
                                            batch_size=args.batch_size,
                                            shuffle=False,
                                            drop_last=True,
                                            collate_fn=collate)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr
    )
    
    lr_scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=len(train_loader) * args.epochs
    )

    print(f"\nTraining...\n")
    for epoch in range(args.epochs):
        train_epoch(model, train_loader, optimizer, lr_scheduler, device, args.print_every)
        report = compute_metrics(model, val_loader)
        
        micro_avg = report["micro avg"]
        micro_f1 = micro_avg["f1"]
        micro_acc = micro_avg["accuracy"]
        micro_prec = micro_avg["precision"]
        micro_recall = micro_avg["recall"]
        
        macro_avg = report["macro avg"]
        macro_f1 = macro_avg["f1"]
        macro_acc = macro_avg["accuracy"]
        macro_prec = macro_avg["precision"]
        macro_recall = macro_avg["recall"]
        
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f" * Micro Average:")
        print(f" * * f1: {micro_f1:.4f}, accuracy: {micro_acc:.4f}, precision: {micro_prec:.4f}, recall: {micro_recall:.4f}\n")
        print(f" * Macro Average:")
        print(f" * * f1: {macro_f1:.4f}, accuracy: {macro_acc:.4f}, precision: {macro_prec:.4f}, recall: {macro_recall:.4f}\n\n")
    
    ## Evaluation
    report = compute_metrics(model, val_loader, device)
    report_lines = report.split('\n')

    print("Classification Report on val set:\n")
    for line in report_lines:
        print(line.strip())
