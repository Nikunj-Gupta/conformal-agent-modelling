import torch, pandas as pd, numpy as np, random 
import torch.nn as nn
import torch.nn.functional as F 
from conformal import ConformalModel
from utils import * 


class CPDataset(torch.utils.data.Dataset):
    def __init__(self, states, actions):
        print(states, actions)
        price_df=pd.DataFrame(np.random.randint(0,9,size=(10000, 4)), columns=list('ABCD')) 

        x=price_df.iloc[:,0:3].values
        y=price_df.iloc[:,3].values

        self.x_train=torch.tensor(x,dtype=torch.float32)
        self.y_train=torch.tensor(y,dtype=torch.float32)

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self,idx):
        return self.x_train[idx],self.y_train[idx] 


'''CP model'''
class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.fc1   = nn.Linear(3, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

"""

CAM: Conformal Action/Agent Modeling 
"""
class CAM:  
    def __init__(
                self, 
                batch_size=128, 
                lr=0.1, 
                momentum=0.9, 
                weight_decay=5e-4, 
                max_epochs=200, 
                val_frac=0.1
            ): 
        # Get the model 
        self.model = NN() 
        self.conformal_model = self.conformalize_model() 

        self.batch_size = batch_size 
        self.lr = lr
        self.momentum = momentum 
        self.weight_decay = weight_decay 
        self.max_epochs = max_epochs 
        self.val_frac = val_frac 
        
    def create_cp_dataset(self, states, actions): 
        cp_data = CPDataset(states, actions) 
        batch_size = batch_size 

        # TODO: Decide: Split into train/test to validate? Train on all prefered. But then how to validate? 
        self.calib_loader = torch.utils.data.DataLoader(cp_data, batch_size=self.batch_size, shuffle=True, pin_memory=True)
        self.val_loader = torch.utils.data.DataLoader(cp_data, batch_size=self.batch_size, shuffle=True, pin_memory=True) 
        # self.calib_loader, self.val_loader = torch.utils.data.random_split(cp_data, [int(cp_data.shape[0]*(1-self.val_frac)), int(cp_data.shape[0]*self.val_frac)])


    def train_cp_model(self): 
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay) 

        for epoch in range(self.max_epochs):
            print('Training conformal model: %d' % epoch)
            print('\nEpoch: %d' % epoch)
            self.model.train(True) 
            train_loss = 0
            correct = 0
            total = 0
            for batch_idx, (inputs, targets) in enumerate(self.calib_loader):
                inputs, targets = inputs, targets
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                print(batch_idx, len(self.calib_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total)) 

    def conformalize_model(self): 
        self.model = torch.nn.DataParallel(self.model) 
        self.model.eval() 

        # optimize for 'size' or 'adaptiveness'
        lamda_criterion = 'size'
        
        # Conformalize model
        conformal_model = ConformalModel(
            self.model, 
            self.calib_loader, 
            alpha=0.1, 
            lamda=0, 
            randomized=True, # use the randomized version of conformal 
            allow_zero_sets=False # allow sets of size zero 
        ) 
        print("Model calibrated and conformalized! Now evaluate over remaining data.")
        return conformal_model 
    
    def get_conformal_actions(self, obs): 
        with torch.no_grad():
            # switch to evaluate mode
            self.conformal_model.eval() 
            output, S = self.conformal_model(obs) 
        return output, S 

    def validate_model(self): 
        validate(self.val_loader, self.model, print_bool=True)
        # validate_new(val_loader, model, print_bool=True)
        print("Complete!")