import argparse
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt 
import time
from utils import * 
from conformal import ConformalModel
import torch.backends.cudnn as cudnn
import random
from lenet import LeNet 
from nn import NN 
import pandas as pd 

parser = argparse.ArgumentParser(description='Conformalize Torchvision Model on Imagenet')
# parser.add_argument('data', metavar='IMAGENETVALDIR', help='path to Imagenet Val')
parser.add_argument('--batch_size', metavar='BSZ', help='batch size', default=128)
parser.add_argument('--num_workers', metavar='NW', help='number of workers', default=0)
parser.add_argument('--num_calib', metavar='NCALIB', help='number of calibration points', default=10000)
parser.add_argument('--seed', metavar='SEED', help='random seed', default=0)



class MyDataset(torch.utils.data.Dataset):
 
  def __init__(self):
    price_df=pd.DataFrame(np.random.randint(0,9,size=(10000, 4)), columns=list('ABCD')) 
 
    x=price_df.iloc[:,0:3].values
    y=price_df.iloc[:,3].values
 
    self.x_train=torch.tensor(x,dtype=torch.float32)
    self.y_train=torch.tensor(y,dtype=torch.float32)
 
  def __len__(self):
    return len(self.y_train)
   
  def __getitem__(self,idx):
    return self.x_train[idx],self.y_train[idx]

if __name__ == "__main__":
    args = parser.parse_args()
    ### Fix randomness 
    np.random.seed(seed=args.seed)
    torch.manual_seed(args.seed)
    # torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)

    my_data = MyDataset() 

    calib_loader = torch.utils.data.DataLoader(my_data, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(my_data, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    
    cudnn.benchmark = True

    # Get the model 
    model = NN() 
    model = torch.nn.DataParallel(model) 
    model.eval()

    # optimize for 'size' or 'adaptiveness'
    lamda_criterion = 'size'
    # allow sets of size zero
    allow_zero_sets = False 
    # use the randomized version of conformal
    randomized = True 

    # Conformalize model
    model = ConformalModel(model, calib_loader, alpha=0.1, lamda=0, randomized=randomized, allow_zero_sets=allow_zero_sets)

    print("Model calibrated and conformalized! Now evaluate over remaining data.")
    # validate(val_loader, model, print_bool=True)
    validate_new(val_loader, model, print_bool=True)

    print("Complete!")