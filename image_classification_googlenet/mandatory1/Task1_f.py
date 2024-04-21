import torch
import torchvision.models as models
from torchvision import datasets, models, transforms
import numpy as np
import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets, models, transforms
from torchvision import transforms
import evaluate_funcs
from dataloader import Data_utils
import os
from matplotlib import pyplot as plt


     

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


saved_model_path = 'saved_model.pt'
saved_model_state_dict = torch.load(saved_model_path)


model = models.googlenet(pretrained=True)


num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 6)  # make the change


model.load_state_dict(saved_model_state_dict)

batchsize_tr = 5
batchsize_test = 5
maxnumepochs = 2


device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")

numcl = 6
# transforms
data_transforms = {}
data_transforms['train'] = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
data_transforms['val'] = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

data_transforms['test'] = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


datasets = {}

datasets['test'] = Data_utils(testing=False,Test=True,num=False,transform=data_transforms['test'])

dataloaders = {}
dataloaders['test'] = torch.utils.data.DataLoader(datasets['test'], batch_size=batchsize_test, shuffle=False)


criterion = torch.nn.CrossEntropyLoss(weight=None, size_average=None, ignore_index=-100, reduce=None,
                                        reduction='mean')

model=model.to(device=device)
num_classes=6
meanlosses, class_ap, mean_ap_test,accuracy  = evaluate_funcs.evaluate(model=model, dataloader=dataloaders['test'], losscriterion=criterion,
                                    device=device,num_classes=num_classes,test=True,save_name="save")





loaded_softmax = torch.load("save softmax_scores.pt")

sorted_indices = torch.argsort(loaded_softmax[:, 0],descending=True)


sorted_tensor =loaded_softmax[sorted_indices]


desired_labels = [1, 3, 5]  
num_to_label_dict={ "0":'buildings', "1":'forest', "2":'glacier', "3":'mountain', "4":'sea', "5":'street'}

for label in desired_labels:
    
    filtered_tensor = sorted_tensor[sorted_tensor[:, 2] == label]

    
    top_indices = filtered_tensor[:10, 1].long()
    bottom_indices = filtered_tensor[-10:, 1].long()
    print(top_indices)

    for i, index in enumerate(top_indices.tolist()):
        
        
        sample = datasets['test'][index]
        
        
        image = sample['image']
        
        label=sample["label"]
        image = image.numpy().transpose((1, 2, 0))

            
        plt.subplot(len(desired_labels), 4,   i + 1)
        plt.imshow(image)
        plt.axis('off')
        plt.title(f'Top {i+1} {num_to_label_dict[str(label)]}', fontsize=9, fontweight='light')
    plt.savefig(f"top10 {label}")
    plt.close()         
    for i, index in enumerate(bottom_indices.tolist()):
        
        sample = datasets['test'][index]

        
        image = sample['image']
        image = image.numpy().transpose((1, 2, 0))
        label=sample["label"]
        plt.subplot(len(desired_labels), 4,   i + 1)
        plt.imshow(image)
        plt.axis('off')
        plt.title(f'Bottom {i+1} {num_to_label_dict[str(label)]}', fontsize=7, fontweight='light')
    plt.savefig(f"bottom10{label}")
    plt.close()    