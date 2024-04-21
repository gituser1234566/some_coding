from smart_open import open
import pandas as pd
import os
from collections import Counter
from matplotlib import pyplot as plt
class plot_NER:
    def __init__(self,files):
        self.files=files

    def __call__(self):
        current_directory = os.getcwd()
        folder_path = os.path.join(current_directory, "plots_NER")
        for fil in self.files:
            print(fil)
            files=self.read_data(fil)
            num_of_counts_per_file=[]
            label_dist_list=[]
           
            file_df=self.read_data(fil)
            
            nan_count_per_row = file_df[1].isna().sum(axis=0)[0].astype(int)
            
        
            total_labels=len(file_df[1])-nan_count_per_row 
           
            label_in= file_df[1].dropna(inplace=False)
            
            label_in=label_in.iloc[:,1].tolist()
           
            label_dist=Counter(label_in)
            self.label_distribution = {label: count / total_labels for label, count in label_dist.items()}
            labels, frequencies = zip(*self.label_distribution.items())
            fig, ax = plt.subplots() 
            ax.bar(labels, frequencies)
            ax.set_xlabel('Labels ')  
            ax.set_ylabel('Frequency')  
            ax.set_title(f"Label Distribution for {file_df[0]}")
             
            fig.subplots_adjust(bottom=0.3) 
            ax.text(0.5, -0.15, f" dataset nr of examples:{nan_count_per_row}",fontsize=10,horizontalalignment='center',verticalalignment='top', transform=ax.transAxes)
            file_name = os.path.join(folder_path,file_df[0][:-10])
            
            plt.savefig(file_name)
            plt.close() 
                
    def read_data(self,file): 
        with open(file,"r") as f:
            df=pd.read_csv(f,sep="\t",names=["words","labels"])
            df_name=file.split("/")[-1]
            
        return df_name,df  
if __name__ == "__main__":

    current_directory = os.getcwd()
    folder_path = os.path.join(current_directory, "data_split")
    files = os.listdir(folder_path)
    files_join=[os.path.join(folder_path,fil) for fil in files]
    plots=plot_NER(files_join)
    plots()
