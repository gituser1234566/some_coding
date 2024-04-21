import os
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import torchvision.transforms as transforms
import zipfile
import torch
import os
import matplotlib.pyplot as plt
from PIL import Image

from sklearn.model_selection import train_test_split
# Define a Resize transformation

class Data_utils(Dataset):
    def __init__(self,testing:bool,num : bool,Train = False,Val = False,Test = False, transform=None):

        """
    Args:

        root_dir (string): Directory with all the images.
        transform (callable, optional): Optional transform to be applied
            on a sample.
    """
        self.Train=Train
        self.Val=Val
        self.Test=Test
        self.root_dir = os.getcwd()
        self.num=num
        self.image_folder="mandatory1_data.zip"
        
        with zipfile.ZipFile(self.image_folder, 'r') as zip_ref:
            zip_ref.extractall(os.path.join(self.root_dir, "zipped_dir"))
            
        
        self.image_foler_path=os.path.join(self.root_dir,"zipped_dir")
        
        self.image_folder_image_unzipped=os.listdir(self.image_foler_path)
        
        self.image_foler_path = os.path.join(self.image_foler_path,self.image_folder_image_unzipped[0])
        
        self.self_class_folders=os.listdir(self.image_foler_path)
       
        
        self.image_folder_imagedir_path=[os.path.join(self.image_foler_path, folder) for folder in self.self_class_folders]
        
        
        self.label_names = self.self_class_folders
        
        self.label_name_dict = dict(Counter(self.label_names))
        self.label_name_dict={key: idx for idx, key in enumerate(self.label_name_dict)}
        #print(self.label_name_dict)
        self.transform = transform
        self.imgfilenames = []
        self.labels = []
        self.ending = ".PNG"

      
        for i,image_folder in enumerate(self.image_folder_imagedir_path):
            for file in os.listdir(image_folder):
                name = os.path.join(image_folder, file)
                #print(name)
                if file.endswith(".jpg"):
                    name = os.path.join(image_folder, file)
                    label=self.label_name_dict[self.label_names[i]]
                    
                    self.imgfilenames.append(name)
                    self.labels.append(label)
        
        train_set, test_set, label_train, label_test = train_test_split(self.imgfilenames, self.labels, test_size=0.18, random_state=42)
        
        if testing==False:
           if self.Train==True:
              self.imgfilenames_train,self.imgfilenames_val,self.labels_train,self.labels_val=train_test_split(train_set,label_train, test_size=0.15, random_state=42)
              self.imgfilenames,self.labels =self.imgfilenames_train,self.labels_train  
           elif self.Val==True:
               self.imgfilenames_train,self.imgfilenames_val,self.labels_train,self.labels_val=train_test_split(train_set,label_train, test_size=0.15, random_state=42)
               self.imgfilenames,self.labels =self.imgfilenames_val,self.labels_val
           elif self.Test==True:
                self.imgfilenames,self.labels=test_set,label_test   
    
        if num==True:
           self.imgfilenames,self.labels=test_set[0:200],label_test[0:200]     
            
    def __len__(self):
        return len(self.imgfilenames)

    def __getitem__(self, idx):
        
        
        
        image = Image.open(self.imgfilenames[idx]).convert('RGB')

        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)

        sample = {'image': image, 'label': label, 'filename': self.imgfilenames[idx],"idx":idx}

        return sample
    