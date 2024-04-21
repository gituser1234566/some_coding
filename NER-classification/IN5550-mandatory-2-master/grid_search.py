import argparse
import random
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForTokenClassification
import transformers
from train_finetune_script import compute_metrics, evaluate, train_epoch
from train_finetune_script import CollateFunctor
import os
from xtreme_dataset_smart import XTREMEDataset

# torch.cuda.empty_cache()
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:10192"

# Add arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', 
                    type=str, 
                    default='bert-base-multilingual-cased', 
                    help='bert-base-multilingual-cased, xlm-roberta-base or google/electra-base-discriminator')

parser.add_argument('--train', type=str, default="./data/train-en.tsv.gz")
parser.add_argument('--val', type=str, default="./data/val-en.tsv.gz")
parser.add_argument('--test', type=str, default="./data/dev-en.tsv.gz")
parser.add_argument('--batch_size', type=str, default="64")
parser.add_argument('--dropout', type=str, default="0.3")
parser.add_argument('--activation', type=str, default="GELU", help='Comma separated GELU,RELU')
parser.add_argument('--epochs', type=int, default=10, help='The number of epochs to train')
parser.add_argument('--lr', type=float, default=2e-5, help='The learning rate')
parser.add_argument('--freeze', type=bool, default=True, help='If to freeze the earlier BERT weights')
parser.add_argument('--seed', type=int, default=42, help='The random seed')
parser.add_argument('--warmup_steps', type=int, default=50, help='The number of warmup steps')
parser.add_argument('--gradient_clipping', type=float, default=10.0, help='The gradient clipping value')
parser.add_argument('--print_every', type=int, default=1000, help='Print the loss every n:th batch')
parser.add_argument('--train_language', type=str, default="en")
parser.add_argument('--test_language', type=str, default="en")

args = parser.parse_args()



highest_f1 = 0
best_model = None
best_model_info = None

def test_params(model, 
                tokenizer,
                train_loader, 
                val_loader,
                test_loader,
                optimizer,
                batch_size=16, 
                epochs=15, 
                lr=0.0002, 
                freeze=True, 
                dropout=0.1, 
                warmup_steps=50, 
                activation='GELU',
                model_name="bert-base-multilingual-cased",
                train_language="en",
                test_language="en"):
    
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    
    model_info = f"Training model: {model_name}\n * {epochs} epochs\n * learning rate is {lr}\n"+ \
                    f" * train language: {train_language}\n * test language: {test_language}\n" + \
                    f" * dropout: {dropout}\n * batch size is {batch_size}\n" + \
                    f" * frozen weights: {freeze}\n * activation: {activation}\n" + \
                    f" * device is on {device_name}\n * warmup steps: {warmup_steps}"
    
    print(f"\n\n{'='*100}\n{model_info}\n{'_'*100}\n")
    
    # Freeze all layers of the pre-trained BERT model
    if model_name != "google/electra-base-discriminator":
        params = None
        if model_name == "xlm-roberta-base":
            params = model.roberta.named_parameters()
        else:
            params = model.bert.named_parameters()
            
        num_of_params = len(list(params))
        
        grad_margin = num_of_params/2

        
        for i, param in params:  
            if i < num_of_params-grad_margin:
                param.requires_grad = freeze 
            
            accuracy = report["accuracy"]
            
    # Set the dropout
    
    for m in model.modules():
        if m.__class__.__name__.startswith("Dropout"):
            m.p = dropout

    lr_scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=len(train_loader) * epochs
    )

    for epoch in range(epochs):
        loss = train_epoch(model, train_loader, optimizer, lr_scheduler, device)
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
        
        
        print(f"Epoch {epoch+1}/{epochs}, accuracy: {accuracy:.4f}, loss: {loss:.4f}")
        print(f" * Micro Average: f1: {micro_f1:.4f}, precision: {micro_prec:.4f}, recall: {micro_recall:.4f}")
        print(f" * Macro Average: f1: {macro_f1:.4f}, precision: {macro_prec:.4f}, recall: {macro_recall:.4f}\n")

    
    ## Evaluation
    report = compute_metrics(model, test_loader)

    print("Classification Report on test set:\n")
    print(f"{'_'*40}")
    
    avg_f1 = (report["micro avg"]["f1-score"] + report["macro avg"]["f1-score"]) / 2
    
    global highest_f1, best_model, best_model_info
    if avg_f1 > highest_f1:
        highest_f1 = report["micro avg"]["f1-score"]
        best_model = model
        best_model_info = model_info
        
    for key, value in report.items():
        print(f" {key}:")
        if type(value) == float:
            print(f"  * {float(value):.4f}")
            continue
        
        for metric, score in value.items():
            print(f"  * {metric}: {float(score):.4f}")
            
    print(f"{'_'*40}\n\n")
    
    return avg_f1
        
def split_range(string, num_type=int):
    return [num_type(i) for i in string.split(",")]



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_name = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir="./cache")



print(f"\nData preprocessing...")




batch_sizes = split_range(args.batch_size)
dropouts = split_range(args.dropout, float)
activation_functions = args.activation.split(",")
train_language = args.train_language

train_split = args.train_language.split(",")
if len(train_split) > 2:
    train_language = train_split

print(f"TESTING FOR MUTLIPLE PARAMETERS:\n\n")
for test_language in ["en"]:
    # Load the data
    collate = CollateFunctor(tokenizer, 512, device)
    train_set = XTREMEDataset(f"./data_split/train-{train_language}-split.tsv.gz", args.model)
    val_set = XTREMEDataset(f"./data_split/val-{test_language}-split.tsv.gz", args.model)
    test_set = XTREMEDataset(f"./data/dev-{test_language}.tsv.gz", args.model)
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
                model = AutoModelForTokenClassification.from_pretrained(args.model, 
                                                                cache_dir="./cache", 
                                                                num_labels=10).to(device)
                
                optimizer = torch.optim.AdamW(
                    model.parameters(), lr=args.lr
                )
        
                f1 = test_params(model=model,
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
                            train_language=args.train_language,
                            test_language=test_language)
                
                model = AutoModelForTokenClassification.from_pretrained(args.model, 
                                                                cache_dir="./cache", 
                                                                num_labels=10).to(device)
                
                # path = "."
                # name = f"{args.model}_({args.train_language}-{test_language})_f1_{float(f1):.4f}.pth"
                # print(f"Saving model to {path}")
                # torch.save(best_model.state_dict(), f"{path}/{name}")
                # print(f"Model saved!\n\n{'-'*100}\n")
                
                
                optimizer = torch.optim.AdamW(
                    model.parameters(), lr=args.lr
                )

                
                
            
best_model = best_model
f1 = highest_f1
print(f"\n\n{'='*100}\nBEST MODEL (f1: {f1:.4}):\n{best_model_info}\n")

path = "."
name = f"{args.model}_({args.train_language}-{args.test_language}|{args.test_language})_f1_{float(f1):.4f}.pth"
print(f"Saving model to {path}")

torch.save(best_model.state_dict(), f"{path}/{name}")

print(f"Model saved!\n\n{'-'*100}\n")
