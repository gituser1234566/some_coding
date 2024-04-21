import transformers
from transformers import AutoTokenizer, AutoModelForTokenClassification
from typing import List
import argparse
import gzip
import os
import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

## Keyboard arguments

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train a model on the SNLI dataset')
    parser.add_argument('--model', type=str, default='bert-base-multilingual-cased', help='The model to use')
    parser.add_argument('--batch_size', type=int, default=16, help='The batch size')
    parser.add_argument('--epochs', type=int, default=8, help='The number of epochs to train')
    parser.add_argument('--lr', type=float, default=2e-5, help='The learning rate')
    parser.add_argument('--freeze', type=bool, default=True, help='If to freeze the earlier BERT weights')
    parser.add_argument('--seed', type=int, default=42, help='The random seed')
    parser.add_argument('--warmup_steps', type=int, default=50, help='The number of warmup steps')
    parser.add_argument('--gradient_clipping', type=float, default=10.0, help='The gradient clipping value')
    return parser.parse_args([])

args = parse_arguments()

os.environ["TOKENIZERS_PARALLELISM"] = "false"


## Helper functions

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
    tok_ranges = token_ranges[1:-1]
    
    for start, end in original_ranges:
        current_label = labels[label_id]
        label_id += 1
        counter = 0
        while tok_id < len(tok_ranges) and cur_tok[1] <= end:
            cur_tok = tok_ranges[tok_id]
            counter += 1
            tok_id += 1
            if cur_tok[1] == end:
                break

        new_token_labels = split_label(current_label, counter)
        if new_token_labels:
            new_labels.extend(new_token_labels)
    return new_labels

def recombine_to_original_labels(tok_labels, original_ranges, token_ranges):
    org_labels = []
    tok_id = 0
    label_id = 0

    cur_tok = token_ranges[tok_id]
    tok_ranges = token_ranges[1:-1]
    
    
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
        org_labels.append(inner_labels[0])



## Dataset class

class XTREMEDataset(torch.utils.data.Dataset):
    def __init__(self, path):
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased", cache_dir="./cache")
        
        
        self.sentences = []
        self.sent_labels = []
        self.ranges = []
        
        self.tokens = []
        self.token_labels = []
        self.token_ranges = []

        # Convert between numerical representation
        self.label_to_num = {"[PAD}": 0, "[CLS]": 1, "[SEP]": 2}
        self.num_to_label = {}
        
        self.import_data(path)
        self.tokenize_data()
        
    def tokenize_data(self, max_len=512):
        for i, (sent, labels, ranges) in enumerate(zip(self.sentences, self.sent_labels, self.ranges)):
            sent_string = " ".join(sent)
            tokens = self.tokenizer(sent_string, return_offsets_mapping=True, max_length=512, truncation=True)
            tok_ranges = tokens["offset_mapping"]
            # print(f"len labs {len(labels)}, ranges: {len(ranges)}, toks: {len(tok_ranges)}")
            tok_labels = create_tokenized_labels(labels, ranges, tok_ranges)

            if len(tok_labels) != len(tok_ranges) - 2:
                error_sent = " ".join(tokenizer.convert_ids_to_tokens(tokens['input_ids']))
                print(f"ERROR IN LEN IN INDEX {i} ({len(tok_labels)}) : ({len(tok_ranges) - 2})")
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
                
                if label not in self.label_to_num.keys():
                    new_num = len(self.label_to_num)
                    self.label_to_num[label] = new_num
                    self.num_to_label[new_num] = label
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
        for sample in batch:
            # Pad or truncate input_ids, token_type_ids, attention_mask
            toks_labels_numerical = [sample["label_to_num"][label] for label in sample["token_labels"]]
            toks_labels_numerical = [1, *toks_labels_numerical, 2] # Add start and end token

            cur_ids = sample['input_ids']
            
            input_ids.append(torch.tensor(cur_ids))
            token_type_ids.append(torch.tensor(sample["token_type_ids"]))
            attention_mask.append(torch.tensor(sample["attention_mask"]))
            labels.append(torch.tensor(toks_labels_numerical + (len( input_ids)-len(cur_ids))*[0]))  
            
        # Pad sequences to ensure uniform length
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        token_type_ids = pad_sequence(token_type_ids, batch_first=True, padding_value=0,)
        attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
        labels = pad_sequence(labels, batch_first=True, padding_value=0)
        labels = torch.stack([term.squeeze(0) for term in labels])
        inputs = {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

        inputs['labels'] = labels.clone().detach()
         
        return self.batch_to_device(inputs)


## Training and Evaluation Loop


def train_epoch(model, train_loader, optimizer, lr_scheduler, device, val_loader=None):
    model.train()
    for i, batch in enumerate(train_loader):
        optimizer.zero_grad()

        print(f"Batch shape: {batch['input_ids'].shape}")
        print(f"Token shape: {batch['token_type_ids'].shape}")
        print(f"Attention shape: {batch['attention_mask'].shape}")
        print(f"Labels shape: {batch['labels'].shape}")
        print(f"Labels: {batch['labels'][0]}")
        print(f"Inputs: {batch['input_ids'][0]}\n")
        
        loss = model(**batch).loss
        epoch_loss = loss.item()
        # backward pass
        loss.backward()

        # update weights
        optimizer.step()
        lr_scheduler.step()
    
        if i % 100 == 0:
            print(f" * * epoch loss: {epoch_loss:.2f}")

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


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_name = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased", cache_dir="./cache")
    model = AutoModelForTokenClassification.from_pretrained("bert-base-multilingual-cased", 
                                                            cache_dir="./cache", 
                                                            num_labels=10).to(device)

    # Freeze all layers of the pre-trained BERT model  
    if args.freeze:
        for name, param in model.bert.named_parameters():  
            param.requires_grad = False 

    
        

    print(f"\n\n{'='*100}\n"
        f"Training model: {args.model} for {args.epochs} epochs\n * learning rate is {args.lr}\n"  +
        f" * batch size is {args.batch_size}\n * BERT weights are frozen\n * device is on {device}" +
        f" * warmup steps: {args.warmup_steps}\n * printing loss on" +
        f" every 100th batch\n{'='*100}\n\n")

    collate = CollateFunctor(tokenizer, 512, device)

    train_data = XTREMEDataset("./data/train-en.tsv.gz")
    train_loader = torch.utils.data.DataLoader(train_data, 
                                            batch_size=args.batch_size, 
                                            shuffle=True,
                                            drop_last=True,
                                            collate_fn=collate)
    
    val_data = XTREMEDataset("./data/dev-en.tsv.gz", train=False)
    val_loader = torch.utils.data.DataLoader(val_data,
                                            batch_size=args.batch_size,
                                            shuffle=True,
                                            drop_last=True,
                                            collate_fn=collate)
    
    test_loader = val_loader

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr
    )
    lr_scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=len(train_loader) * args.epochs
    )

    for epoch in range(args.epochs):
        train_epoch(model, train_loader, optimizer, lr_scheduler, device)
        accuracy = evaluate(model, val_loader, device)
        print(f"Epoch {epoch + 1}/{args.epochs}: accuracy (NB: CURRENTLY ON TRAIN) = {accuracy:.2%}\n")

    test_accuracy = evaluate(model, test_loader, device)
    print(f"Test accuracy (NB: ON TRAINING SET!!!) {test_accuracy:.2%}")

