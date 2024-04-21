from torchvision import datasets, models, transforms
import numpy as np
#from __future__ import print_function, division
from sklearn.metrics import average_precision_score 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
# import matplotlib.pyplot as plt
import time
import os

# import skimage.io
import PIL.Image

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from tqdm import tqdm


class MulticlassAccuracyAndAP(torch.nn.Module):
    def __init__(self, num_classes):
        super(MulticlassAccuracyAndAP, self).__init__()
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.predictions = []
        self.targets = []
        self.prob=[]
        
    def update_state(self, y_true, y_pred,pred_prob):
        # Convert one-hot encoded predictions to class labels
        

        # Accumulate predictions and targets
        self.predictions.append(y_pred)
        self.targets.append(y_true)
        self.prob.append(pred_prob)
        
    def compute_metrics(self):
        # Concatenate accumulated predictions and targets
        all_predictions = torch.cat(self.predictions,dim=0).view(-1).long().numpy()
        
        all_targets = torch.cat(self.targets,dim=0).view(-1).long().numpy()
        prob=torch.cat(self.prob,dim=0).view(-1).numpy()
        
        print("target",all_targets)
        print("pred",all_predictions)
        print("prob",prob)
        # Initialize dictionaries to store accuracy and AP per class
        
        class_ap = {}

        # Compute accuracy and AP per class
        for class_idx in range(6):
            print(class_idx)
            class_targets = (all_targets == class_idx)
           
            class_predictions = all_predictions[class_targets]
            
            class_probs=prob[class_targets]
            
            class_targets1 = all_targets[class_targets]
            
            binary_vec= (class_targets1==class_predictions)
            
            binary_vec=binary_vec
            
            class_ap[class_idx] = average_precision_score(binary_vec,class_probs)
           
    # Compute mean accuracy and mean AP over all classes
    
        mean_ap = sum(class_ap.values()) / self.num_classes

        return  class_ap, mean_ap
    


def evaluate(model, dataloader, losscriterion, device,num_classes,test:bool,save_name):
    model.eval()

    softmax_list=[]
    all_image_ind_list = []
    all_labels_list = []
    losses = []
    predictions=[]
    curcount = 0
    accuracy = 0
    metrics_calculator = MulticlassAccuracyAndAP(num_classes)
    with torch.no_grad():
        for ctr, data in enumerate(dataloader):
            inputs = data['image'].to(device)
            outputs = model(inputs)

            labels = data['label']

            cpuout = outputs.to('cpu')
            image_idx=data["idx"]
            loss = losscriterion(cpuout, labels)
            losses.append(loss.item())
            
            cpuout=torch.softmax(cpuout, dim=1)
            
            

            
            
            preds=torch.argmax(cpuout, dim=1)
            preds_prob=cpuout[torch.arange(len(preds)), preds]
            
            softmax_list.extend(preds_prob)
            all_image_ind_list.extend(image_idx)
            all_labels_list.extend(labels)
            predictions.extend(preds)
            labels = labels.float()
            
            corrects = torch.sum(preds == labels.data) / float(labels.shape[0])
            accuracy = accuracy * (curcount / float(curcount + labels.shape[0])) + corrects.float() * (
                        labels.shape[0] / float(curcount + labels.shape[0]))
            curcount += labels.shape[0]
            
            metrics_calculator.update_state(labels,preds, preds_prob)
            
            
            del inputs, labels, outputs
            torch.cuda.empty_cache()
            
    class_ap, mean_ap = metrics_calculator.compute_metrics() 
    
    if test==True:
       concatenated_tensor = torch.cat((torch.tensor(softmax_list).unsqueeze(1),
                                 torch.tensor(all_image_ind_list).unsqueeze(1),
                                 torch.tensor(all_labels_list).unsqueeze(1),torch.tensor(predictions).unsqueeze(1)), dim=1)
        
       torch.save(concatenated_tensor, f'{save_name} softmax_scores.pt')   
    
    return  np.mean(losses), class_ap, mean_ap,accuracy.item()