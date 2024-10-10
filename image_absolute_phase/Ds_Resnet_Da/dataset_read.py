import os
import numpy as np
import torch
import scipy.io as sio
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader,random_split
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import random
import math


' Define the reading of the dataset '
class PUDataset(Dataset):
    def __init__(self, ids, dir_input1,dir_gt_phl3,dir_gt_wrap, dir_gt_unwrap,extension='.mat'):
        self.dir_input1= dir_input1
        self.dir_gt_phl3 = dir_gt_phl3
        self.dir_gt_wrap = dir_gt_wrap
        self.dir_gt_unwrap = dir_gt_unwrap
        self.extension = extension
        self.ids = ids # Dataset IDS
        self.data_len = len(self.ids) # Calculate len of data

    ' Ask for input and ground truth'
    def __getitem__(self, index):

        # Get an ID of the input and ground truth
        id_input1 = self.dir_input1 + self.ids[index] + self.extension
        id_gt_phl3= self.dir_gt_phl3 + self.ids[index] + self.extension
        id_gt_wrap = self.dir_gt_wrap + self.ids[index] + self.extension
        id_gt_unwrap = self.dir_gt_unwrap + self.ids[index] + self.extension
        # Open them
        #print(id_input)
        input1 = sio.loadmat(id_input1)
        gt_phl3= sio.loadmat(id_gt_phl3)
    
        gt_wrap = sio.loadmat(id_gt_wrap)
        gt_unwrap = sio.loadmat(id_gt_unwrap)
        input1 = input1['g4'] #inpu
        gt_phl3 = gt_phl3['phl_3']   #gt
        gt_wrap = gt_wrap['phi4']   #gt
        gt_unwrap = gt_unwrap['phase_unwrapped_save']   #gt

        input = torch.from_numpy(input1).float().unsqueeze(0)###（C,H.W）→（1,C,H.W）
        gt_phl3 = torch.from_numpy(gt_phl3).float().unsqueeze(0)

        gt_wrap = torch.from_numpy(gt_wrap).float().unsqueeze(0)
        gt_unwrap = torch.from_numpy(gt_unwrap).float().unsqueeze(0)
        

        #wrapped_phase_4 belongs to【-0.5,0.5】，normalize it by addiing 0.5

        gt_wrap= gt_wrap + 0.5

        return input,gt_phl3,gt_wrap,gt_unwrap 

    ' Length of the dataset '
    def __len__(self):
        return self.data_len
    




' Return the training dataset separated in batches '
def get_dataloaders(dir_input1,dir_gt_phl3,dir_gt_wrap, dir_gt_unwrap, batch_size,val_percent=0.1, test_size=0.1):
    val_percent = val_percent / 100 if val_percent > 1 else val_percent  # Validate a correct percentage
    ids = [f[:-4] for f in os.listdir(dir_input1)] # Read the names of the images
    ids = sorted(ids, key=lambda x: int(x))#对

    dset = PUDataset(ids,dir_input1,dir_gt_phl3, dir_gt_wrap, dir_gt_unwrap) # Get the dataset

    torch.manual_seed(0)   

    train_dataset, val_dataset ,test_dataset= random_split(dset, [0.8,0.1,0.1])
   # Create the dataloaders
    dataloaders = {}
    dataloaders['train'] = DataLoader(train_dataset, batch_size=batch_size,shuffle=True,drop_last = False)
    dataloaders['val'] = DataLoader(val_dataset, batch_size=batch_size,shuffle=True,drop_last = False)
   

    ba=int((len(ids)-1)/100)
    print('ba:',ba)
    dataloaders['test'] = DataLoader(test_dataset, batch_size=ba)

    return dataloaders['train'], dataloaders['val'],dataloaders['test'] 

