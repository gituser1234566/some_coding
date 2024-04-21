import numpy as np 
from sklearn.metrics import average_precision_score 
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import  models, transforms
from torchvision import transforms
from tqdm import tqdm
from dataloader import Data_utils

#The following code is highly inspired by the weekly exersize 5

def train_epoch(model, trainloader, losscriterion, device, optimizer):
    model.train()

    losses = list()
    with tqdm(total=len(trainloader)) as pbar:
        for batch_idx, data in enumerate(trainloader):

            inputs = data['image'].to(device)
            labels = data['label'].to(device)

            optimizer.zero_grad()

            output = model(inputs)
            loss = losscriterion(output, labels)

            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            if batch_idx % 100 == 0:
                print('current mean of losses ', np.mean(losses))

            pbar.update(1)
            
            del inputs, labels, output
            torch.cuda.empty_cache()
    return np.mean(losses)





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
        
        

        # Accumulate predictions and targets
        self.predictions.append(y_pred)
        self.targets.append(y_true)
        self.prob.append(pred_prob)
        
    def compute_metrics(self):
        
        all_predictions = torch.cat(self.predictions,dim=0).view(-1).long().numpy()
        
        all_targets = torch.cat(self.targets,dim=0).view(-1).long().numpy()
        prob=torch.cat(self.prob,dim=0).view(-1).numpy()
        
      
       
        
        class_ap = {}

        # Compute accuracy and AP per class
        for class_idx in range(6):
            #print(class_idx)
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
    accuracy_list=[]
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
            
            corrects = torch.sum(preds == labels) / float(labels.shape[0])
            accuracy = accuracy * (curcount / float(curcount + labels.shape[0])) + corrects.float() * (
                        labels.shape[0] / float(curcount + labels.shape[0]))
            curcount += labels.shape[0]
            accuracy_list.append(accuracy)
            metrics_calculator.update_state(labels,preds, preds_prob)
            
            
            del inputs, labels, outputs
            torch.cuda.empty_cache()
            
    class_ap, mean_ap = metrics_calculator.compute_metrics() 
    
    if test==True:
       concatenated_tensor = torch.cat((torch.tensor(softmax_list).unsqueeze(1),
                                 torch.tensor(all_image_ind_list).unsqueeze(1),
                                 torch.tensor(all_labels_list).unsqueeze(1),torch.tensor(predictions).unsqueeze(1)), dim=1)
        
       torch.save(concatenated_tensor, f'{save_name} softmax_scores.pt')   
    print("meanloss:",np.mean(losses), "ap_class:",class_ap,"mean_accuracy:", mean_ap,accuracy.item())
    return  np.mean(losses), class_ap, mean_ap,np.mean(accuracy_list)

def train_model(dataloader_train, dataloader_val, model, losscriterion, optimizer, scheduler, num_epochs,
                           device,num_classes):
    best_measure = 0
    best_epoch = -1
    loss_train_list,loss_val_list=[],[]
    class_ap_list, mean_ap_list=[],[]
    accuracy_list=[]
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        model.train(True)
        loss_train = train_epoch(model, dataloader_train, losscriterion, device, optimizer)

        if scheduler is not None:
            scheduler.step()

        model.train(False)
        losses_val, class_ap, mean_ap,accuracy = evaluate(model, dataloader_val, losscriterion, device,num_classes,test=False,save_name="no_save")
        
        class_ap_list.append(class_ap)
        accuracy_list.append(accuracy)
        mean_ap_list.append(mean_ap)
        loss_val_list.append(losses_val)
        loss_train_list.append(loss_train)
        
        #meanlosses, class_ap, mean_ap = evaluate(model, dataloader_train, losscriterion, device,num_classes)
        print(' Loss:', loss_train, "class AP:",class_ap,"mAP:", mean_ap, "Validation loss:",losses_val,"val_accuracy:",accuracy)
        measure=mean_ap
        if measure > best_measure:  # higher is better or lower is better?
            bestweights = model.state_dict()
            best_measure=measure
            best_epoch = epoch
            print('current best', best_measure, ' at epoch ', best_epoch)

    return best_epoch, best_measure,measure, bestweights,loss_train_list,loss_val_list,class_ap_list,mean_ap_list,accuracy_list


def finetunealllayers():
 # someparameters
    batchsize_tr = 10
    batchsize_test = 10
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
    datasets['train'] = Data_utils(testing=False,Train=True,num=False,transform=data_transforms['train'])
    datasets['val'] =  Data_utils(testing=False,Val=True,num=False,transform=data_transforms['val'])
    datasets['test'] = Data_utils(testing=False,Test=True,num=False,transform=data_transforms['test'])

    dataloaders = {}
    dataloaders['train'] = torch.utils.data.DataLoader(datasets['train'], batch_size=batchsize_tr, shuffle=True)
    dataloaders['val'] = torch.utils.data.DataLoader(datasets['val'], batch_size=batchsize_test, shuffle=False)
    dataloaders['test'] = torch.utils.data.DataLoader(datasets['test'], batch_size=batchsize_test, shuffle=False)

    criterion = torch.nn.CrossEntropyLoss(weight=None, size_average=None, ignore_index=-100, reduce=None,
                                          reduction='mean')

 

    lrates = [0.001,0.01,0.05]
    loss_train_lis,loss_val_lis,class_ap_lis,mean_ap_lis,accuracy_lis=[],[],[],[],[]
    best_hyperparameter = None
    weights_chosen = None
    best_measure = 0
    num_classes=6
    for lr in lrates:
        # modelmodels.resnet18(pretrained=True)
        model = models.googlenet(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, numcl)
        model.fc.reset_parameters()

        model.to(device)
         
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        best_epoch, bestmeasure,measure, bestweights,loss_train_list,loss_val_list,class_ap_list,mean_ap_list,accuracy_list = train_model(dataloader_train=dataloaders['train'],
                                                                           dataloader_val=dataloaders['val'],
                                                                           model=model, losscriterion=criterion,
                                                                           optimizer=optimizer, scheduler=None,
                                                                           num_epochs=maxnumepochs, device=device,num_classes=num_classes)

    
    

        
    
        
        if best_measure < bestmeasure:
            loss_train_lis,loss_val_lis,class_ap_lis,mean_ap_lis,accuracy_lis=loss_train_list,loss_val_list,class_ap_list,mean_ap_list,accuracy_list
            best_hyperparameter = lr
            weights_chosen = bestweights
            best_measure = bestmeasure
            print(f"best hyperparam:{best_hyperparameter}")
   
    #torch.save(model.state_dict(), "model_large.pt")
    model.load_state_dict(weights_chosen)
    torch.save(weights_chosen, "saved_model.pt")
    meanlosses, class_ap, mean_ap_test,accuracy  = evaluate(model=model, dataloader=dataloaders['test'], losscriterion=criterion,
                                      device=device,num_classes=num_classes,test=True,save_name="save_softmax_1_ae")

    print('mAP val', bestmeasure, 'mAP test', mean_ap_test,"Ap pr class",class_ap,"accuracy",accuracy)
    
    import matplotlib.pyplot as plt

    
    num_to_label_dict={ "0":'buildings', "1":'forest', "2":'glacier', "3":'mountain', "4":'sea', "5":'street'}
    # Plot loss curves
    
    # Plot loss curves
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(accuracy_lis, label='Accuracy Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(' Validation Accuracy')
    plt.legend()
    plt.savefig("accuracy_val_train")
    plt.close() 
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(loss_train_lis, label='Train Loss')
    plt.plot(loss_val_lis, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig("train_val_loss")
    plt.close() 
    # Plot mean AP curve
    plt.subplot(1, 2, 1)
    plt.plot(mean_ap_lis, label='Mean AP')
    plt.xlabel('Epoch')
    plt.ylabel('Mean AP')
    plt.title('Mean Average Precision (mAP)')
    plt.legend()
    plt.savefig("mAP")
    plt.close() 

    # Plot AP curves for each class
    num_classes = len(class_ap_lis[0])
    plt.figure(figsize=(12, 8))
    for class_idx in range(num_classes):
        class_ap_values = [class_ap[class_idx] for class_ap in class_ap_list]
        plt.plot(class_ap_values, label=f'Class { num_to_label_dict[str(class_idx)]}')
    plt.xlabel('Epoch')
    plt.ylabel('Average Precision (AP)')
    plt.title('Average Precision (AP) for Each Class')
    plt.legend()
    
    plt.savefig("AP_class")
    plt.close() 
    

finetunealllayers()    
    