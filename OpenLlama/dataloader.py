import sys
import torch
from torch.utils.data import Dataset
from utils import read_json

def build_data(datapath, batch_size):
    representation_dataset = MyDataset(datapath)
    data_loader = torch.utils.data.DataLoader(dataset=representation_dataset, batch_size=batch_size,
                                               collate_fn=representation_dataset.collate_fn,shuffle=True)
    return data_loader




class MyDataset(Dataset):
    def __init__(self,path):
        self.data_dir=path
        self.all_data=self.read_data_file(self.data_dir)
        
    def __len__(self):
        return len(self.all_data)
    
    def __getitem__(self, idx):
        data_sample=self.all_data[idx]
        return data_sample
        
    def read_data_file(self, data_dir):
        datafile=data_dir
        all_data=read_json(datafile)
        input_data=all_data['input']
        output_data=all_data['output']
        all_data_sample=[]
        for i in range(len(input_data)):
            input_data_sample=input_data[i]
            output_data_sample=output_data[i]
            data_sample=[]
            data_sample.append(input_data_sample)
            data_sample.append(output_data_sample)
            all_data_sample.append(data_sample)
        return all_data_sample
    
    def collate_fn(self,batch):
            data_sample=batch
            batch_input=torch.tensor([sample[0] for sample in data_sample],dtype=torch.float32)
            batch_output=torch.tensor([sample[1] for sample in data_sample],dtype=torch.float32)
            return batch_input,batch_output