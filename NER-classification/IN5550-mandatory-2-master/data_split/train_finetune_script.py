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


## DATASET
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
    
    train_set = XTREMEDataset("./train-en-split.tsv.gz")
    val_set = XTREMEDataset("./val-en-split.tsv.gz")
    
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
