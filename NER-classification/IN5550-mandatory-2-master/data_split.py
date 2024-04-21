

# Sklearn datasplit

import argparse
import gzip
import os
import pandas as pd
from sklearn.model_selection import train_test_split


train_split = 0.8

parser = argparse.ArgumentParser()
parser.add_argument('--language', type=str, default='en', help='Language code')
args = parser.parse_args()

language = args.language

file_name = f"./data/train-{language}.tsv.gz"

with gzip.open(file_name, "rb") as f:
    cur_sent = []
    cur_labels = []
    cur_range = []
    prev_idx = 0
    
    sentences = []
    sent_labels = []
    
    for line in f:
        # New sentence if file contains an empty line
        if not line.split():
            sentences.append(cur_sent)
            sent_labels.append(cur_labels)
            cur_sent = []
            cur_labels = []
            cur_range = []
            prev_idx = 0
            continue
            
            
        word, label = line.decode().split()
        num = label
        
        # Build the word ranges for the sentence
        word_len = len(word)
        cur_range.append((prev_idx, prev_idx + word_len))
        prev_idx += word_len + 1 # add 1 for the space from later concatenation
        
        # Add the word and label to the current sentence
        cur_sent.append(word)
        cur_labels.append(num)
    
    df = pd.DataFrame({'sentence': sentences, 'label': sent_labels})
    
    print(df.head())
    
    train, val = train_test_split(df, test_size=1-train_split, random_state=42, stratify=df['label'])
    
    
    train_words = pd.DataFrame()
    train_labels = pd.DataFrame()
    
    
    for i in range(len(train)):
        train_words = train_words.append(pd.DataFrame(train.iloc[i]['sentence'], columns=['sentence']))
        train_labels = train_labels.append(pd.DataFrame(train.iloc[i]['label'], columns=['label']))
        print(train_words)
        break
    
    

    
    # Store the data as .gz
    train_file_name = f'train-split-{language}.tsv'
    val_file_name = f'val-split-{language}.tsv'
    
    # Save as tab separated .gz file
    train = pd.concat([train_words, train_labels], axis=1)
    train.to_csv(train_file_name, sep='\t', index=False, compression='gzip')
    
    val = pd.concat([val['sentence'], val['label']], axis=1)
    val.to_csv(val_file_name, sep='\t', index=False, compression='gzip')
    
    
    

