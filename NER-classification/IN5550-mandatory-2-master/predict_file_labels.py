from train_finetune_script import create_tokenized_labels, recombine_to_original_labels
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoConfig
import seqeval
import seqeval.metrics
import pandas as pd
import numpy as np
from smart_open import open
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', 
                    type=str, 
                    default='/fp/projects01/ec30/eirikeg_torkilef_rasyed/assignment2/bert-base-multilingual-cased_en-it-de-ru-th-en_f1_0.8358.pth')
parser.add_argument('--file_path', type=str, default="/fp/projects01/ec30/IN5550/obligatories/2/surprise/surprise_test_set.tsv")
parser.add_argument('--out_path', type=str, default="surprise_data_predictions.tsv")



args = parser.parse_args()

# Configuration variables
file_path = args.file_path
output_file = args.out_path
base_model= "bert-base-multilingual-cased"
fine_tuned_model_path = args.model
MAX_TOKEN_LENGTH = 500 # BERT's maximum token length
    

label_to_num = {"[PAD]": 0,
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

num_to_label = {v: k for k, v in label_to_num.items()}

sentences = []
labels = []


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_name = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize the model configuration using the same number of labels that you trained with.
config = AutoConfig.from_pretrained(
    base_model,
    num_labels=10,
    cache_dir="./cache"
)

# Create a new model with the configuration.
model = AutoModelForTokenClassification.from_config(config)

# Load your previously saved state dictionary into this model.
# Ensure `args.model` is the path to your `.pth` file that you saved with `torch.save`.
model.load_state_dict(torch.load(fine_tuned_model_path, map_location=device))
model.to(device)

tokenizer = AutoTokenizer.from_pretrained(base_model, cache_dir="./cache")



def tokenize_and_predict(cur_sent, cur_ranges):
    # Tokenize the sentence
                    sent_string = " ".join(cur_sent)
                    tokens = tokenizer(sent_string, return_offsets_mapping=True, truncation=False)
                    token_ranges = tokens["offset_mapping"]
                    tokens["input_ids"] = tokens["input_ids"][1:-1] # Remove [CLS] and [SEP] tokens
                    
                    num_tokens_left = len(tokens["input_ids"])
                    
                    token_labels = []
                    token_index = 0 # Start index for slicing the tokens
                    
                    # Handle cases where the sentence is too long
                    while num_tokens_left > 0:
                        
                        # Ensure we don't try to index out of bounds.
                        # The -2 is to account for the [CLS] and [SEP] tokens
                        chunk_size = min(num_tokens_left, MAX_TOKEN_LENGTH - 2)
                        chunk_tokens = {key: tokens[key][token_index:token_index + chunk_size] for key in tokens}
    
                        # Manually add [CLS] and [SEP] token IDs to the input IDs for this chunk
                        chunk_tokens["input_ids"] = [101] + chunk_tokens["input_ids"] + [102]
                        
                        # Manually add attention mask bits for [CLS] and [SEP] (these should be 1s)
                        chunk_tokens["attention_mask"] = [1] + chunk_tokens["attention_mask"] + [1]
    
                        chunk_ranges = chunk_tokens["offset_mapping"]
                        
                        with torch.no_grad():
                            
                            # Convert the tokens to tensors
                            chunk_input_ids = torch.tensor(chunk_tokens["input_ids"]).unsqueeze(0).to(device)
                            chunk_attention_mask = torch.tensor(chunk_tokens["attention_mask"]).unsqueeze(0).to(device)
                            
                            if chunk_input_ids.shape[1] > MAX_TOKEN_LENGTH:
                                print(f"chunk_size: {chunk_size}, num_tokens_left: {num_tokens_left}, len(input_ids): {len(chunk_input_ids.squeeze())}")
                            
                            # Get the model's prediction
                            outputs = model(chunk_input_ids, attention_mask=chunk_attention_mask)
                            predictions = torch.argmax(outputs.logits, dim=2).squeeze(0).cpu().numpy()
                            predictions = [num_to_label[pred] for pred in predictions]
                            
                            # Remove the [CLS] and [SEP] tokens
                            predictions = predictions[1:-1]
                                
                            token_labels.extend(predictions)
                            
                        # Prepare index for next slice
                        token_index += chunk_size
                        num_tokens_left -= chunk_size
                    


                    
                    # Recombine labels to original word ranges
                    pred_labels = recombine_to_original_labels(token_labels, cur_ranges, token_ranges)
                    
                    sentences.append(cur_sent)
                    labels.append(pred_labels)
                
                    if len(pred_labels) != len(cur_sent):
                        print(f"tok_labels: {predictions}")
                        print(f"cur_sent{cur_sent}")
                        print(f"pred_labels{pred_labels}\n\n")
                        print(f"Length mismatch {len(pred_labels)}, {len(cur_sent)}, {len(token_labels)}, {len(cur_ranges)}")
                        print(cur_sent)
                        print(pred_labels)
                        print(cur_ranges)
                        print(chunk_ranges)
                        print(tokens)
                    


def ner_tag_from_file(path):
    with open(path, 'r') as file:
                cur_sent = []
                cur_ranges = []
                prev_idx = 0
                
                for i, line in enumerate(file):
                    line_string = line.split()
                    # If line is a word, add to current sentence
                    if line_string:
                        word = line_string[0]
                        
                        
                        # Build the word ranges for the sentence
                        word_len = len(word)
                        cur_ranges.append((prev_idx, prev_idx + word_len))
                        prev_idx += word_len + 1 # add 1 for the space from later concatenation
                        
                        # Add the word and label to the current sentence
                        cur_sent.append(word)
    
                    # At every new line: tokenize the sentence and get the model's predicted labels
                    else:
                        tokenize_and_predict(cur_sent, cur_ranges)
                        cur_sent = []
                        cur_labels = []
                        cur_ranges = []
                        prev_idx = 0  
                        continue

                # If file does not end on empty line
                if len(cur_sent):
                    tokenize_and_predict(cur_sent, cur_ranges)



# # Convert sentence data to word-level output file
def convert_to_words(df):
    words = []
    labels = []
    for i, row in df.iterrows():
        sentence = row["sentence"]
        word_labels = row["labels"]
        words.extend(sentence)
        labels.extend(word_labels)

        # Add NaN row after each sentence
        words.append(np.nan)
        labels.append(np.nan)

    return pd.DataFrame({'word': words, 'label': labels})

def save_file(df_sent_to_word, name):
    # Use "\n" to directly represent newlines for empty rows
    df_sent_to_word.to_csv(name, sep='\t', index=False, header=False)



if __name__ == "__main__":
    print(f"Running script for predicting labels for {args.file_path}")
    ner_tag_from_file(file_path)

    sent_df = pd.DataFrame({"sentence": sentences, "labels": labels})

    word_df = convert_to_words(sent_df)
    
    print(f"Predictions stored at {output_file}")
    save_file(word_df, output_file)
