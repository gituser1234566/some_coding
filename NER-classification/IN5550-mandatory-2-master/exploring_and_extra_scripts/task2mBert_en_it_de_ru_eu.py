import os
import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import AutoModelForTokenClassification, AutoTokenizer
from seqeval.metrics import classification_report,accuracy_score
import logging
from dataset_processing import datacollector
from data_loading import util_Dataset,CollateFunctor
import matplotlib.pyplot as plt
from transformers import  TrainingArguments, Trainer
import numpy as np
from datasets import load_metric
from seqeval.metrics import classification_report
import json
import os
from itertools import combinations
import time
import training_funcs
import numpy as np

torch.cuda.empty_cache()


#For commented code look to task2xlm_roberta.py and task2Bert.py.
# The commenting for this will be exactly the same.

def test_train_Experimental_combinations2(key_train,key_test):
  
    train_files=["train-en.tsv.gz","train-it.tsv.gz","train-de.tsv.gz","train-ru.tsv.gz","train-eu.tsv.gz"]
    
   
    combinations_5 = list(combinations(train_files, 5))

   
    
    all_combinations_train = combinations_5

    # Create a dictionary with training key
    combination_dict_train = {}
    for comb in all_combinations_train:
        print(comb)
        key = '-'.join([os.path.basename(path).split('.')[0] for path in comb])
        combination_dict_train[key] = list(comb)
    
    test_files=[["dev-de.tsv.gz"],["dev-it.tsv.gz"],["dev-en.tsv.gz"],["dev-sw.tsv.gz"],["dev-af.tsv.gz"]]
    
    combination_dict_test = {}
    for comb in test_files:
        key = '-'.join([os.path.basename(path).split('.')[0] for path in comb])
        combination_dict_test[key] = comb
    
    return combination_dict_train[key_train] ,combination_dict_test[key_test]


def train_epoch(model, train_loader, optimizer, lr_scheduler, device):
    model.train()
    

  
    for batch in train_loader:
        
        inputs = {key: value.to(device) for key, value in batch.items()}
       
        optimizer.zero_grad()
       
        # forward pass
        
        model_out = model(**inputs)
        
        
        
        loss = model_out.loss
        
        # backward pass
        loss.backward()

        # update weights
        lr_scheduler.step()
        optimizer.step()

    return loss.item()
        
 

@torch.no_grad()
def compute_metrics(model,data,all_labels_flatt_unique,device,epoch):
    
    all_predictions = []
    all_labels = []
    model.eval()
    total_correct, total_samples = 0, 0
    for batch in data:
        inputs = {key: value.to(device) for key, value in batch.items()}
        
        outputs = model(**inputs)
        predictions=outputs.logits.to("cpu")
        

        labels=batch["labels"].to("cpu")
        
        predictions = np.argmax(predictions, axis=2)
        
     
        filtered_predictions = []
        filtered_labels = []
        for pred_seq, label_seq in zip(predictions, labels):
            pred_seq_filtered = []
            label_seq_filtered = []
            for pred, label in zip(pred_seq, label_seq):
                if label != -100:
                    pred_seq_filtered.append(pred)
                    label_seq_filtered.append(label)
            filtered_predictions.append(pred_seq_filtered)
            filtered_labels.append(label_seq_filtered)

           
        for pred_seq, label_seq in zip(filtered_predictions, filtered_labels):
            pred_labels = [all_labels_flatt_unique[pred] for pred in pred_seq]
            true_labels = [all_labels_flatt_unique[label] for label in label_seq]
            all_predictions.append(pred_labels)
            all_labels.append(true_labels)

    report = classification_report(all_labels, all_predictions,output_dict=True)
    report["accuracy"]=  accuracy_score(y_true=all_labels, y_pred=all_predictions)
    accuracy = report["accuracy"]
    
    micro_avg = report["micro avg"]
    micro_f1 = micro_avg["f1-score"]
    micro_prec = micro_avg["precision"]
    micro_recall = micro_avg["recall"]
    
    macro_avg = report["macro avg"]
    macro_f1 = macro_avg["f1-score"]
    macro_prec = macro_avg["precision"]
    macro_recall = macro_avg["recall"]
    
    
    print(f"Epoch {epoch+1}/{4}, accuracy: {accuracy:.4f}")
    print(f" * Micro Average: f1: {micro_f1:.4f}, precision: {micro_prec:.4f}, recall: {micro_recall:.4f}")
    print(f" * Macro Average: f1: {macro_f1:.4f}, precision: {macro_prec:.4f}, recall: {macro_recall:.4f}\n")

    return report

def plot_f1_and_loss_scores(f1_micro_avg, f1_macro_avg,loss,text2, f1_weighted_avg, text, save_path,save_path2):
    epochs = range(1, len(f1_micro_avg) + 1)

    # Plot F1 scores
    fig, ax = plt.subplots() 
    ax.plot(epochs, f1_micro_avg, label='F1 Micro Avg')
    ax.plot(epochs, f1_macro_avg, label='F1 Macro Avg')
    ax.plot(epochs, f1_weighted_avg, label='F1 Weighted Avg')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('F1 Score')
    ax.set_title(text)
    ax.legend()

    if save_path:
        plt.savefig(save_path)
    plt.close()
    
    
    fig, ax = plt.subplots() 
    ax.plot(range(len(loss)),loss, label='Loss')
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Cross Entropy')
    ax.set_title(text2)
    ax.legend()
    if save_path2:
        plt.savefig(save_path2)
    plt.close()
  

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train a model on the SNLI dataset')
    parser.add_argument('--model', type=str, default='/fp/projects01/ec30/models/bert-base-multilingual-cased', help='The model to use /fp/projects01/ec30/models/bert-base-multilingual-cased,  /fp/projects01/ec30/models/distiluse-base-multilingual-cased-v1 or /fp/projects01/ec30/models/xlm-roberta-base/ ')
    parser.add_argument('--batch_size', type=int, default=32, help='The batch size')
    parser.add_argument('--epochs', type=int, default=4, help='The number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-4, help='The learning rate')
    parser.add_argument('--seed', type=int, default=42, help='The random seed')
    parser.add_argument('--warmup_steps', type=int, default=50, help='The number of warmup steps')
    parser.add_argument('--gradient_clipping', type=float, default=10.0, help='The gradient clipping value')
    return parser.parse_args([])



def main():

    logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)
    logger = logging.getLogger(__name__)

    args = parse_arguments()
    logger.info(f"Setting random seed to {args.seed}")
    torch.manual_seed(args.seed)

    # set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
         args.model,
        cache_dir="./cache"
    )
    max_len_sent=10
    
    


    train_key=["train-en-train-it-train-de-train-ru-train-eu"]
    test_key = ['dev-de','dev-it','dev-en','dev-sw','dev-af']
    
    f1_avg=0
    for i in range(len(train_key)):
        average_f1=0
        start_time = time.time()
        for j in range(len(test_key)):
            
            
            train_path,test_path=test_train_Experimental_combinations2(train_key[i],test_key[j])
            
            max_len_sent=10
            logger.info("Loading datasets")
            
            data_collector_train_val = datacollector(train_path,max_len_sent,tokenizer)
            train_set,val_set=data_collector_train_val(train=True,test=False)
            data_collector_train_val.data_statistic()
            data_collector_train_val.plot_NER_dist(train_key[i]+test_key[j])
            train_set = util_Dataset(train_set)
            val_set = util_Dataset(val_set)
        
            data_collector_test = datacollector(test_path,max_len_sent,tokenizer)
            test_set=data_collector_test(train=False,test=True)
            data_collector_test.data_statistic()
            data_collector_train_val.plot_NER_dist(train_key[i]+test_key[j])
            test_set = util_Dataset(test_set)
        
            train_loader = torch.utils.data.DataLoader(
            train_set, batch_size= args.batch_size, shuffle=True, drop_last=True,
            collate_fn=CollateFunctor(tokenizer, 40)
            )
        
            val_loader = torch.utils.data.DataLoader(
            val_set, batch_size= args.batch_size, shuffle=True, drop_last=True,
            collate_fn=CollateFunctor(tokenizer, 40)
            )
        
            test_loader = torch.utils.data.DataLoader(
            test_set, batch_size= args.batch_size, shuffle=True, drop_last=True,
            collate_fn=CollateFunctor(tokenizer, 40)
            )
        
            model = AutoModelForTokenClassification.from_pretrained(
             args.model,
            cache_dir="./cache",
            trust_remote_code=True,
            num_labels=7
            ).to(device)
        
        
        
            optimizer = torch.optim.AdamW(
            model.parameters(), lr= args.lr
            )
            lr_scheduler = transformers.get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps= args.warmup_steps, num_training_steps=len(train_loader) *  args.epochs
            )
        
            loss_list=[]
            f1_micro_avg=[]
            f1_macro_avg=[]
            f1_weighted_avg=[]
            label_unique_val=["B-ORG",
                         "I-ORG",
                         "B-LOC",
                         "I-LOC",
                         "B-PER",
                         "I-PER",
                         "O"]
            logger.info("Training model")
            loss_list=[]
            for epoch in range(args.epochs):
                loss=train_epoch(model, train_loader, optimizer, lr_scheduler, device)
                loss_list.append(loss)
                eval_metric_val=compute_metrics(model,val_loader,label_unique_val,device,epoch)
                f1_micro_avg.append(eval_metric_val["micro avg"]['f1-score'])
                f1_macro_avg.append(eval_metric_val["macro avg"]['f1-score'])
                f1_weighted_avg.append(eval_metric_val["weighted avg"]['f1-score'])
                
            average_f1+=f1_macro_avg[-1] 
            
            print("Classification Report on test set:\n")
            print(f"{'_'*40}")

            print(f"trainset:{train_key[i]} and testset:{test_key[j]}:\n")
            print(f"{'_'*40}")
            
            report = training_funcs.compute_metrics(model, test_loader,device)    
            for key, value in report.items():
                print(f" {key}:")
                if type(value) == float:
                    print(f"  * {float(value):.4f}")
                    continue
                
                for metric, score in value.items():
                    print(f"  * {metric}: {float(score):.4f}")
                    
            print(f"{'_'*40}\n\n")

            text= f"F1 score Validation-set for simulation: {train_key[i],test_key[j]}"
            text2=f"Loss Training-set for simulation: {train_key[i],test_key[j]}"
            path= f"plots/additional_ru_eu/hindi_replaced_Russian/f1_bert/f1{train_key[i],test_key[j]}"
            path2= f"plots/additional_ru_eu/hindi_replaced_Russian/loss_bert/Loss{train_key[i],test_key[j]}"
            
            plot_f1_and_loss_scores(f1_micro_avg, f1_macro_avg,loss_list,text2, f1_weighted_avg, text,path,path2)

        
           
        if  f1_avg< average_f1:
            
            model_name=f" training config :{train_key[i]}"
            f1_avg=average_f1
            
            model_save=model.to("cpu")
            save_model_named="ner_modelBert_hi-eu-en-it-de"
            model_save.save_pretrained(save_model_named)
            tokenizer.save_pretrained("tokenizer")

            label_list=[
                        "B-ORG",
                         "I-ORG",
                         "B-LOC",
                         "I-LOC",
                         "B-PER",
                         "I-PER",
                         "O"]
            label2id= {label:str(id) for id,label in enumerate(label_list)}
            id2label={str(id):label for id,label in enumerate(label_list) }



            config=json.load(open(f"{save_model_named}/config.json"))

            config["id2label"]=id2label
            config["label2id"]=label2id


            json.dump(config,open(f"{save_model_named}/config.json","w"))
            model_save=model.to(device)
            report_best = training_funcs.compute_metrics(model_save, test_loader,device)    
            if i*j==(len(train_key)-1)*(len(train_key)-1):
               print(f"Best model for simulation had the following {model_name} ") 


               for key, value in report_best.items():
                 print(f" {key}:")
                 if type(value) == float:
                    print(f"  * {float(value):.4f}")
                    continue
                
                 for metric, score in value.items():
                    print(f"  * {metric}: {float(score):.4f}")
                    
               print(f"{'_'*40}\n\n")

            
        
        

        end_time = time.time()
        duration = end_time - start_time
        print(f"Iteration  {i+1}, Inner iteration  {j+1}: {duration:.4f} seconds for train language combination:{train_key[i]} ")
            
                
    

if __name__ == "__main__":
    main()
