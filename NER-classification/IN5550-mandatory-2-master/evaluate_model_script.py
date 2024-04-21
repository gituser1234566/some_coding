import argparse
import random
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoConfig
import transformers
from train_finetune_script import XTREMEDataset, compute_metrics, evaluate, predict, train_epoch
from train_finetune_script import CollateFunctor
import os
import matplotlib as plt

torch.cuda.empty_cache()

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:10192"


# Add arguments

parser = argparse.ArgumentParser()
parser.add_argument('--model', 
                    type=str, 
                    default='bert-base-multilingual-cased_(en-en)_f1_0.8222.pth', 
                    help='path')

parser.add_argument('--pretrained_model', type=str, default="bert-base-multilingual-cased")
parser.add_argument('--languages', type=str, default="en,it,de,af,sw")

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_name = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize the model configuration using the same number of labels that you trained with.
config = AutoConfig.from_pretrained(
    args.pretrained_model,
    num_labels=10,
    cache_dir="./cache"
)

# Create a new model with the configuration.
model = AutoModelForTokenClassification.from_config(config)

# Load your previously saved state dictionary into this model.
# Ensure `args.model` is the path to your `.pth` file that you saved with `torch.save`.
model.load_state_dict(torch.load(args.model, map_location=device))
model.to(device)

tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)

def split_range(string, num_type=int):
    return [num_type(i) for i in string.split(",")]

languages = split_range(args.languages, str)

loss = []
f1 = []
average_f1 = 0

for language in languages:
    print(f"Testing model {args.model} on language {language}")
    test_dataset = XTREMEDataset(f"./data/dev-{language}.tsv.gz", args.pretrained_model,)
    test_loader = DataLoader(test_dataset,
                             batch_size=16,
                             shuffle=False,
                             collate_fn=CollateFunctor(tokenizer, 512, device))
    
    
    predictions, labels = predict(model, test_loader, device)
    report = compute_metrics(model, test_loader)
    average_f1 += report["macro avg"]["f1-score"]
    
    print(f"Classification Report on {language} test set:\n")
    print(f"{'_'*40}")
    for key, value in report.items():
        print(f" {key}:")
        if type(value) == float:
            print(f"  * {float(value):.4f}")
            continue
        
        for metric, score in value.items():
            print(f"  * {metric}: {float(score):.4f}")
            
        
        print(f"pred: {predictions[0][:20]}")
        print(f"lab: {labels[0][:20]}")
    
    print(f"{'_'*40}\n\n")

print(f"\n * Average language f1: {average_f1/len(languages):.4f}\n\n{'='*40}\n\n")    
    