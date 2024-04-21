import os
import pandas as pd
from nltk.tokenize import sent_tokenize
from operator import itemgetter
from sklearn.preprocessing import LabelEncoder
import itertools
from itertools import chain
from datasets import Dataset
from transformers import AutoTokenizer
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter
from matplotlib import pyplot as plt

#from datasets.utils.logging import disable_progress_bar



#Class takes as input nam of file max example len and tokenizer and returns the list 
# of dicts where each dict contains relevant input for language model.The main usecase 
# is calling its call function,If the train it set to true for call function it
# splits the data into test and train and returns separate train and val set if 
# test is set to true tranformation is done on test 

class datacollector:
    def __init__(self, name_of_file,max_len_sent,tokenizer):
        self.name_of_file = name_of_file
        self.max_len_sent=max_len_sent
        self.tokenizer=tokenizer
        
    def __call__(self,train:bool,test:bool):
        if not isinstance(train, bool):
            raise TypeError("train parameter must be a boolean")
        
        self.train=train
        self.test=test
        
        #get path
        current_directory = os.getcwd()
        paths_to_print = self.import_func(self.name_of_file, current_directory)
        
        #get files as dataframe
        doc = self.read_file_to_df(paths_to_print)
        
        #words in first col    
        train_text_col = doc.iloc[:, 0].to_string(index=False)
 
        #labels in second
        train_lab_col = doc.iloc[:, 1].to_string(index=False)
        
        #transform to string of list
        train_text_list = train_text_col.split()
        train_label_list = train_lab_col.split()
        
        #devide test and labels into nested list of examples
        text_sent,label_sent=self.tokenize_text_and_labels(train_text_list, train_label_list,self.max_len_sent)

        #flat labellist
        self.all_labels = list(itertools.chain.from_iterable(label_sent))
        
        #tranform labels to numeric
        label_encoder = LabelEncoder()
        label_list=[
                        "B-ORG",
                         "I-ORG",
                         "B-LOC",
                         "I-LOC",
                         "B-PER",
                         "I-PER",
                         "O"]
        label_encoder.fit(label_list)
        
        numerical_NER = [list(label_encoder.transform(sublist_list)) for sublist_list in label_sent]
       

        
        all_labels_flatt_unique=list(np.unique(np.array(self.all_labels)))

        #model dict to be passed into dict form 
        data = {"tokens": text_sent, "ner_tags": numerical_NER}
        # Create a dataset from the Pandas DataFrame
        data = Dataset.from_dict(data)
        #disable_progress_bar()
        tokenized_data = data.map(self.tokenize_and_adjust_labels, batched=True)

        if test==True:
           self.nr_examples_test=len(text_sent)
           self.all_label_test=self.all_labels
        
        if train==True:
            # if train reapete above steps
            text_sent_train,text_sent_val,train_label,val_label=train_test_split(text_sent, label_sent,test_size=0.2)

            train_numerical_NER=self.numeric_target(train_label)
            val_numerical_NER=self.numeric_target(val_label)
           
            data_train=data = {"tokens": text_sent_train, "ner_tags": train_numerical_NER[0]}
            data_train = Dataset.from_dict(data_train)
            data_val = {"tokens": text_sent_val, "ner_tags": val_numerical_NER[0]}
            data_val = Dataset.from_dict(data_val)

            self.nr_examples_val=len(text_sent_val)
            self.nr_example_train=len(text_sent_train)
            self.all_label_train=list(chain.from_iterable(train_label))
            self.all_label_val=list(chain.from_iterable(val_label))
            
            #disable_progress_bar()
            tokenized_train_data=data_train.map(self.tokenize_and_adjust_labels, batched=True)
            tokenized_eval_data=data_val.map(self.tokenize_and_adjust_labels, batched=True)
            return tokenized_train_data,tokenized_eval_data
        return tokenized_data
    
    #function for converting labels to numerics
    def numeric_target(self,labels):
        all_labels = list(itertools.chain.from_iterable(labels))
        label_encoder = LabelEncoder()
        label_list=[
                        "B-ORG",
                         "I-ORG",
                         "B-LOC",
                         "I-LOC",
                         "B-PER",
                         "I-PER",
                         "O"]
        label_encoder.fit(label_list)
        
        numerical_NER = [list(label_encoder.transform(sublist_list)) for sublist_list in labels]
       

        
        all_labels_flatt_unique=list(np.unique(np.array(all_labels)))
        return numerical_NER,all_labels_flatt_unique
    
    #function for linking file name to file path 
    def find_substring_index(self, substring_list, string_list):
        indexes = []
        for substring in substring_list:
            try:
                index = next(i for i, string in enumerate(string_list) if substring in string)
                indexes.append(index)
            except StopIteration:
                indexes.append(-1)
        return indexes
    
    # import_func gets the path using find_substring_index 
    def import_func(self, sub_string_list, current_dir):
        dir_path = current_dir

        # Construct the base path for the data directory
        dir_path = os.path.join(dir_path, "data")

        files = os.listdir(dir_path)

        list_paths = [os.path.join(dir_path, files[i]) for i in range(len(files))]
        match_on = self.find_substring_index(sub_string_list, list_paths)

        path_get = itemgetter(*match_on)
        get_paths = path_get(list_paths)
        return get_paths

    #Extract file as DataFrame
    def read_file_to_df(self, path_in_list):
        if not path_in_list:
            return []
        
        df_list = []
        if len(self.name_of_file)>1:
            path_in_list=list(path_in_list)
        
        if len(self.name_of_file)==1:
           path_in_list=[path_in_list]
        
        for path_in in path_in_list:
            
            with open(path_in, "rb") as f:
                df = pd.read_csv(f.name, sep="\t",names=["text","label"])
                df_list.append(df)
        merged_df = pd.concat(df_list, ignore_index=True)  
        #print(merged_df)
        return merged_df

    # this function devide the words into example of len in max_len_sent
    def tokenize_text_and_labels(self,text_list, labels_list,max_len_sen):
 
    
    
        # Initialize lists to store tokenized text and labels
        tokenize_nested_text=[]
        tokenize_nested_label=[]
        tokenized_texts = []
        tokenized_labels = [] 
        # Keep track of the current index in the tokenized text
        current_index = 0
        len_text_list=len(text_list)
        count_for=0
        # Iterate through sentences
        for i,word in enumerate(text_list):
            count_for+=1
            
                
            
            if current_index<=self.max_len_sent:
                
                tokenized_texts.append(word)
                
                if ("," in word or "." in word) and len(word)==1:
                   
                # Append the corresponding label to the tokenized labels
                   tokenized_labels.append("O")
                   current_index=current_index 
                else:
                    tokenized_labels.append(labels_list[i])
                    current_index+=1
                
            if current_index>self.max_len_sent:
                current_index=0
                tokenized_texts = []
                tokenized_labels = [] 
            
            if current_index==self.max_len_sent:
                tokenize_nested_text.append(tokenized_texts)
                tokenize_nested_label.append(tokenized_labels)
            
            if count_for==len_text_list:
               break
           
            
        return tokenize_nested_text, tokenize_nested_label

    #tokenise function thate expands labels the correct way so if , the word Find has label O,
    # get tokenized as [fi,##nd] then label get extended to [O,O]
    def tokenize_and_adjust_labels(self,samples_per_split):
      tokenized_samples = self.tokenizer.batch_encode_plus(samples_per_split["tokens"],return_tensors='pt', padding=True, truncation=True ,is_split_into_words=True)
     
      adjusted_labels = []
      #print(len(tokenized_samples["input_ids"]))
      for k in range(0, len(tokenized_samples["input_ids"])):
        prev_wordid = -1
        word_ids_list = tokenized_samples.word_ids(batch_index=k)
        existing_label_ids = samples_per_split["ner_tags"][k]
      
          
        j = -1
        adjusted_label_ids = []
       
        for wordid in word_ids_list:
          if(wordid is None):
            adjusted_label_ids.append(-100)
          elif(wordid!=prev_wordid):
            j = j + 1
            adjusted_label_ids.append(existing_label_ids[j])
            prev_wordid = wordid
          else:
            label_name = self.all_labels[existing_label_ids[j]]
            adjusted_label_ids.append(existing_label_ids[j])
            
        adjusted_labels.append(adjusted_label_ids)
      tokenized_samples["labels"] = adjusted_labels
      return tokenized_samples
     
     #Function for finding data statistics 
    def data_statistic(self):
          
        

         if self.train==True:
            label_counts = Counter(self.all_label_train)
            
            total_labels = len(self.all_label_train) 
           
            self.label_train_distribution = {label: count / total_labels for label, count in label_counts.items()} 
           
            label_counts = Counter(self.all_label_val)
            #print(label_counts) 
            total_labels = len(self.all_label_val) 
           # print(total_labels) 
            self.label_val_distribution = {label: count / total_labels for label, count in label_counts.items()}   
            #print(self.label_val_distribution)
         if self.test==True:
           label_counts = Counter(self.all_label_test)
           total_labels = len(self.all_label_test) 
           self.label_test_distribution = {label: count / total_labels for label, count in label_counts.items()}  
           
           return self.nr_examples_test
            
         return self.nr_examples_val,self.nr_example_train
    
    #Function for plotting NER dist for data 
    def plot_NER_dist(self,name_key):
         current_directory = os.getcwd()
         dir_path = os.path.join(current_directory, "plots/")
         os.makedirs(dir_path, exist_ok=True)
         if self.train==True:
         
            labels, frequencies = zip(*self.label_train_distribution.items())
            fig, ax = plt.subplots() 
            ax.bar(labels, frequencies)
            ax.set_xlabel('Labels')  
            ax.set_ylabel('Frequency')  
            ax.set_title("Label Distribution")
             
            fig.subplots_adjust(bottom=0.3) 
            ax.text(0.5, -0.15,  f' Plot for {self.name_of_file},train-dev split of 80/20 where nr training example is:{self.nr_example_train} \n each example len is:{self.max_len_sent}',fontsize=10,horizontalalignment='center',verticalalignment='top', transform=ax.transAxes)
            
            
            file_name = os.path.join(dir_path, "Train_NER_dist"+name_key)
            # Save the plot
            plt.savefig(file_name)
            plt.close() 

            fig, ax = plt.subplots() 
            labels, frequencies = zip(*self.label_val_distribution.items())
            ax.bar(labels, frequencies)
            ax.set_xlabel('Labels')  
            ax.set_ylabel('Frequency')  
            ax.set_title("Label Distribution")
            fig.subplots_adjust(bottom=0.3)
            ax.text(0.5, -0.15, f' Plot for {self.name_of_file},train-dev split 80/20 where nr training example is:{self.nr_examples_val} \n each example len is:{self.max_len_sent}', fontsize=10,horizontalalignment='center',verticalalignment='top', transform=ax.transAxes)
           
            
            file_name = os.path.join(dir_path, "Val_NER_dist"+name_key)
            # Save the plot
            plt.savefig(file_name)
            plt.close() 
            
         if self.test==True:

            labels, frequencies = zip(*self.label_test_distribution.items())
            fig, ax = plt.subplots() 
            labels, frequencies = zip(*self.label_val_distribution.items())
            ax.bar(labels, frequencies)
            ax.set_xlabel('Labels')  
            ax.set_ylabel('Frequency')  
            ax.set_title("Label Distribution")
            fig.subplots_adjust(bottom=0.2)
            ax.text(0.5, -0.15, f' Plot for {self.name_of_file}, nr of test example is:{self.nr_examples_test} \n where each example len is:{self.max_len_sent}',fontsize=10, horizontalalignment='center',verticalalignment='top', transform=ax.transAxes)

            
            file_name = os.path.join(dir_path, "Test_NER_dist"+name_key)
            # Save the plot
            plt.savefig(file_name)
            plt.close()  




