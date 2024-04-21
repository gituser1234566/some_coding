import argparse
import random
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForTokenClassification
import transformers
from train_finetune_script import XTREMEDataset, compute_metrics, evaluate, train_epoch
from train_finetune_script import CollateFunctor



def test_params(model, 
                tokenizer,
                train_loader, 
                val_loader,
                test_loader,
                optimizer,
                batch_size=32, 
                epochs=15, 
                lr=0.0002, 
                freeze=True, 
                dropout=0.1, 
                warmup_steps=50, 
                activation='GELU',
                model_name="bert-base-multilingual-cased",
                languages=["en"]):
    
    print(f"\n\n{'='*100}\n"
        f"Training model: {model_name}\n * {epochs} epochs\n * learning rate is {lr}\n"  +
        f" * batch size is {batch_size}\n * frozen weights: {freeze}\n * device is on {device}" +
        f" * warmup steps: {warmup_steps}\n * activation: {activation}\n" +
        f" * languages: {', '.join(languages)}\n{'-'*100}\n")
    
    # Freeze all layers of the pre-trained BERT model
    if freeze:
        params = None
        if model_name == "xlm-roberta-base":
            params = model.roberta.named_parameters()
        elif model_name == "bert-base-multilingual-cased":
            params = model.bert.named_parameters()
        elif model_name == "google/electra-base-discriminator":
            params = model.electra.named_parameters()
        elif model_name == "flaubert/flaubert_base_cased":
            params = model.transformer.named_parameters()
        else:
            params = model.distilbert.named_parameters()

        for _, param in params:  
            param.requires_grad = False 
            
            
    # Set the dropout
    
    for m in model.modules():
        if m.__class__.__name__.startswith("Dropout"):
            m.p = dropout

    lr_scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=len(train_loader) * epochs
    )

    for epoch in range(epochs):
        train_epoch(model, train_loader, optimizer, lr_scheduler, device)
        report = compute_metrics(model, val_loader)
        
        accuracy = report["accuracy"]
        
        micro_avg = report["micro avg"]
        micro_f1 = micro_avg["f1-score"]
        micro_prec = micro_avg["precision"]
        micro_recall = micro_avg["recall"]
        
        macro_avg = report["macro avg"]
        macro_f1 = macro_avg["f1-score"]
        macro_prec = macro_avg["precision"]
        macro_recall = macro_avg["recall"]
        
        
        print(f"Epoch {epoch+1}/{epochs}, accuracy: {accuracy:.4f}")
        print(f" * Micro Average: f1: {micro_f1:.4f}, precision: {micro_prec:.4f}, recall: {micro_recall:.4f}")
        print(f" * Macro Average: f1: {macro_f1:.4f}, precision: {macro_prec:.4f}, recall: {macro_recall:.4f}\n")
    
    ## Evaluation
    report = compute_metrics(model, test_loader)

    print("Classification Report on test set:\n")
    print(f"{'_'*40}")
    
    for key, value in report.items():
        print(f" {key}:")
        if type(value) == float:
            print(f"  * {float(value):.4f}")
            continue
        
        for metric, score in value.items():
            print(f"  * {metric}: {float(score):.4f}")
            
    print(f"{'_'*40}\n\n")
        
def split_range(string, num_type=int):
    return [num_type(i) for i in string.split(",")]


# Add arguments

parser = argparse.ArgumentParser()
parser.add_argument('--model', 
                    type=str, 
                    default='bert-base-multilingual-cased', 
                    help='bert-base-multilingual-cased or xlm-roberta-base')

parser.add_argument('--train', type=str, default="./data/train-en.tsv.gz")
parser.add_argument('--val', type=str, default="./data/val-en.tsv.gz")
parser.add_argument('--test', type=str, default="./data/dev-en.tsv.gz")
parser.add_argument('--batch_size', type=str, default="16,32,64")
parser.add_argument('--dropout', type=str, default="0.1,0.3,0.5")
parser.add_argument('--activation', type=str, default="GELU", help='Comma separated GELU,RELU')
parser.add_argument('--epochs', type=int, default=10, help='The number of epochs to train')
parser.add_argument('--lr', type=float, default=2e-5, help='The learning rate')
parser.add_argument('--freeze', type=bool, default=True, help='If to freeze the earlier BERT weights')
parser.add_argument('--seed', type=int, default=42, help='The random seed')
parser.add_argument('--warmup_steps', type=int, default=50, help='The number of warmup steps')
parser.add_argument('--gradient_clipping', type=float, default=10.0, help='The gradient clipping value')
parser.add_argument('--languages', type=str, default="en", help='Comma separated languages')
parser.add_argument('--print_every', type=int, default=1000, help='Print the loss every n:th batch')

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_name = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir="./cache")
model = AutoModelForTokenClassification.from_pretrained(args.model, 
                                                        cache_dir="./cache", 
                                                        num_labels=10).to(device)



print(f"\nData preprocessing...")

# Load the data
collate = CollateFunctor(tokenizer, 512, device)
train_set = XTREMEDataset("./train-en-split.tsv.gz",args.model)
val_set = XTREMEDataset("./val-en-split.tsv.gz",args.model)
test_set = XTREMEDataset("./data/dev-en.tsv.gz",args.model)

optimizer = torch.optim.AdamW(
    model.parameters(), lr=args.lr
)

batch_sizes = split_range(args.batch_size)
dropouts = split_range(args.dropout, float)
activation_functions = args.activation.split(",")
languages = args.languages.split(",")

print(f"TESTING FOR MUTLIPLE PARAMETERS:\n\n")
for batch_size in batch_sizes:
    train_loader = torch.utils.data.DataLoader(train_set, 
                                        batch_size=batch_size, 
                                        shuffle=True,
                                        drop_last=True,
                                        collate_fn=collate)

    val_loader = torch.utils.data.DataLoader(val_set,
                                            batch_size=batch_size,
                                            shuffle=False,
                                            drop_last=True,
                                            collate_fn=collate)
    
    test_loader = torch.utils.data.DataLoader(test_set,
                                            batch_size=batch_size,
                                            shuffle=False,
                                            drop_last=True,
                                            collate_fn=collate)
    
    for dropout in dropouts:
        for activation in activation_functions:
            # Test with and without Freeze
            test_params(model=model,
                        model_name=args.model,
                        tokenizer=tokenizer, 
                        train_loader=train_loader, 
                        val_loader=val_loader,
                        test_loader=test_loader,
                        optimizer=optimizer,
                        batch_size=batch_size, 
                        epochs=args.epochs, 
                        lr=args.lr, 
                        dropout=dropout, 
                        warmup_steps=args.warmup_steps, 
                        activation=activation,
                        freeze=False,
                        languages=languages)
            
            test_params(model=model,
                        model_name=args.model,
                        tokenizer=tokenizer, 
                        train_loader=train_loader, 
                        val_loader=val_loader,
                        test_loader=test_loader,
                        optimizer=optimizer,
                        batch_size=batch_size, 
                        epochs=args.epochs, 
                        lr=args.lr, 
                        dropout=dropout, 
                        warmup_steps=args.warmup_steps, 
                        activation=activation,
                        freeze=True,
                        languages=languages)
    
