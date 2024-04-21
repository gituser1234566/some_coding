import torch
from torch.nn.utils.rnn import pad_sequence


#Purpose of this class is to change datacollector class results form dataset_processing.py 
# into torch.utils.data.Dataset
class util_Dataset(torch.utils.data.Dataset):
    def __init__(self, data, label_vocab=None):


        self.data=data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

#CollateFunctor for padding
class CollateFunctor:
    def __init__(self, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __call__(self, batch):
        input_ids = []
        token_type_ids = []
        attention_mask = []
        labels = []
        
        # Iterate over each sample in the batch
        for sample in batch:
            # Pad or truncate input_ids, token_type_ids, attention_mask
            input_ids.append(torch.tensor(sample["input_ids"]))
            token_type_ids.append(torch.tensor(sample["token_type_ids"]))
            attention_mask.append(torch.tensor(sample["attention_mask"]))
            # Add padding to labels
  
            labels.append(torch.tensor(sample["labels"]  ))
        
        # Pad sequences to ensure uniform length
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        token_type_ids = pad_sequence(token_type_ids, batch_first=True, padding_value=0,)
        attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)
        
        inputs = {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
       
         
        return inputs