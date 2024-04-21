import argparse
import random
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForTokenClassification
import transformers
from dataset_processing import datacollector
from data_loading_xlm import util_Dataset,CollateFunctor
from training_funcs import compute_metrics, train_epoch
import time

# In this experiments we freeze all weights exept the classification head and no freeze.


#Function for testing hyperparamters
def test_params(model, 
                tokenizer,
                train_loader, 
                val_loader,
                test_loader,
                optimizer,
                batch_size=32, 
                epochs=4, 
                lr=0.0002, 
                freeze=True, 
                dropout=0.1, 
                warmup_steps=50, 
                activation='GELU',
                model_name="xlm-roberta-base",
                lang=None):
    #print out hyperparameters for specific run
    print(f"\n\n{'='*100}\n"
        f"Training model: {model_name}\n * {epochs} epochs\n * learning rate is {lr}\n"  +
        f" * batch size is {batch_size}\n * frozen weights: {freeze}\n * device is on {device}" +
        f" * warmup steps: {warmup_steps}\n * activation: {activation}\n" +
        f" * languages: {lang}\n{'-'*100}\n"+ f" * dropout: {dropout}\n{'-'*100}\n")
    
  
    #freeze or do not freeze        
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

#set schedueler
    lr_scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=len(train_loader) * epochs
    )
    
    #train and validate on validation set and compute NER statistics
    start_time = time.time()
    for epoch in range(epochs):
        train_epoch(model, train_loader, optimizer, lr_scheduler, device)
        report = compute_metrics(model, val_loader,device)
        
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
    end_time = time.time()
    duration = end_time - start_time
    print(f" time for training and evaluating:{duration} for train language combination:{args.train} and test language: {lang}")
    ## Evaluation
    report = compute_metrics(model, test_loader,device)

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
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', 
    type=str, default='/fp/projects01/ec30/models/xlm-roberta-base',
    help='The model to use /fp/projects01/ec30/models/bert-base-multilingual-cased,/fp/projects01/ec30/models/distiluse-base-multilingual-cased-v1 or /fp/projects01/ec30/models/xlm-roberta-base/ ')
    
    parser.add_argument('--train', type=str, default="train-en.tsv.gz")
    parser.add_argument('--test', type=str, default="dev-en.tsv.gz,dev-de.tsv.gz,dev-it.tsv.gz,dev-sw.tsv.gz,dev-af.tsv.gz")
    parser.add_argument('--batch_size', type=str, default="64")
    parser.add_argument('--dropout', type=str, default="0.1,0.3")
    parser.add_argument('--activation', type=str, default="GELU", help='Comma separated GELU,RELU')
    parser.add_argument('--epochs', type=int, default=4, help='The number of epochs to train')
    parser.add_argument('--lr', type=float, default=2e-5, help='The learning rate')
    parser.add_argument('--freeze', type=bool, default=True, help='If to freeze the earlier BERT weights')
    parser.add_argument('--seed', type=int, default=42, help='The random seed')
    parser.add_argument('--warmup_steps', type=int, default=50, help='The number of warmup steps')
    parser.add_argument('--gradient_clipping', type=float, default=10.0, help='The gradient clipping value')
    return parser.parse_args([])


args= parse_arguments()
torch.manual_seed(args.seed)

#set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_name = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir="./cache")


print(f"\nData preprocessing...")
# Load the english training data compute input to model and return plots on the statistics
max_len_sent=10
data_collector_train_val = datacollector([args.train],max_len_sent,tokenizer)
train_set,val_set=data_collector_train_val(train=True,test=False)

data_collector_train_val.data_statistic()
data_collector_train_val.plot_NER_dist("train_en")

train_set = util_Dataset(train_set)
val_set = util_Dataset(val_set)




#language keys
langkey= ['dev-en','dev-de','dev-it','dev-sw','dev-af']

#extract hyperparameters from arguments
language=args.test.split(",")
batch_sizes = split_range(args.batch_size)
dropouts = split_range(args.dropout, float)
activation_functions = args.activation.split(",")


print(f"TESTING FOR MUTLIPLE PARAMETERS:\n\n")

#for test languages
for i,lang in enumerate(language):
    
    #for testset compute input to model
    data_collector_test = datacollector([lang],max_len_sent,tokenizer)
    test_set=data_collector_test(train=False,test=True)
    test_set = util_Dataset(test_set)
    data_collector_train_val.data_statistic()
    data_collector_train_val.plot_NER_dist(langkey[i])
    
    #vary batch size hyperparameter and put into dataloader
    for batch_size in batch_sizes:
        
        train_loader = torch.utils.data.DataLoader(
                train_set, batch_size=batch_size, shuffle=True, drop_last=True,
                collate_fn=CollateFunctor(tokenizer, 40)
                )
                
        val_loader = torch.utils.data.DataLoader(
                val_set, batch_size=batch_size, shuffle=True, drop_last=True,
                collate_fn=CollateFunctor(tokenizer, 40)
                )
                
        test_loader = torch.utils.data.DataLoader(
                test_set, batch_size=batch_size, shuffle=True, drop_last=True,
                collate_fn=CollateFunctor(tokenizer, 40)
                )
        #vary dropout and activation
        for dropout in dropouts:
            for activation in activation_functions:
                # Test with and without Freeze
                
                #set new model for each hyperparamterer config
                model = AutoModelForTokenClassification.from_pretrained(args.model, 
                                                                cache_dir="./cache", 
                                                                num_labels=7).to(device)
                #set optmizizer
                optimizer = torch.optim.AdamW(
                    model.parameters(), lr=args.lr
                )
        
                
                
                
                #run model on hyperparameters without freezing the weights
                test_params(model=model, 
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
                            lang=lang)
    
                #reset model 
                model = AutoModelForTokenClassification.from_pretrained(args.model, 
                                                                cache_dir="./cache", 
                                                                num_labels=7).to(device)
                #set optimizer
                optimizer = torch.optim.AdamW(
                    model.parameters(), lr=args.lr
                )
                
                #run model on hyperparamters with freezed weights
                test_params(model=model,
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
                            lang=lang)