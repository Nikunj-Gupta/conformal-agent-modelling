import torch, pandas as pd, numpy as np, random 
import torch.nn as nn
import torch.nn.functional as F 
from conformal import ConformalModel
from utils import * 


class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
    
    def clear(self):
        del self.actions[:]
        del self.states[:]

class CPDataset(torch.utils.data.Dataset):
    def __init__(self, states, actions):
        # states = np.array([s.detach().cpu().numpy() for s in states])
        # actions = np.array([a.detach().cpu().numpy() for a in actions])
        states = np.array(states)
        actions = np.array(actions)
        state_dim = states.shape[1] 
        df = pd.DataFrame(states) 
        df["actions"] = actions 

        x=df.iloc[:,0:state_dim].values
        y=df.iloc[:,state_dim].values 

        self.x_train=torch.tensor(x,dtype=torch.float32)
        self.y_train=torch.tensor(y,dtype=torch.long)

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self,idx):
        return self.x_train[idx],self.y_train[idx] 


'''CP model'''
class NN(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=64):
        super(NN, self).__init__()
        self.fc1   = nn.Linear(in_dim, hidden_dim)
        self.fc2   = nn.Linear(hidden_dim, hidden_dim)
        self.fc3   = nn.Linear(hidden_dim, out_dim) 

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
                state_dim, 
                act_dim 
            ): 
        # Get the model 

        self.state_dim = state_dim 
        self.act_dim = act_dim 
        self.model = NN(in_dim=state_dim, out_dim=act_dim) 
        self.buffer = RolloutBuffer() 
        
    def create_cp_dataset(self, batch_size=128, val_frac=0.1, shuffle=True, pin_memory=True): 
        cp_data = CPDataset(self.buffer.states, self.buffer.actions) 
        self.buffer.clear() 

        # TODO: Decide: Split into train/test to validate? Train on all prefered. But then how to validate? 
        self.calib_loader = torch.utils.data.DataLoader(cp_data, batch_size=batch_size, shuffle=shuffle, pin_memory=pin_memory)
        self.val_loader = torch.utils.data.DataLoader(cp_data, batch_size=batch_size, shuffle=True, pin_memory=True) 
        # self.calib_loader, self.val_loader = torch.utils.data.random_split(cp_data, [int(cp_data.shape[0]*(1-self.val_frac)), int(cp_data.shape[0]*self.val_frac)])

        print("Dataset created!")

    def train_cp_model(self, max_epochs=200, lr=0.1, momentum=0.9, weight_decay=5e-4): 
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay) 

        for epoch in range(max_epochs):
            print("Training conformal model")
            print("\nEpoch: %d" % epoch)
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

        
        print("Prediction Model trained!")
        
        self.conformalize_model() 

        return train_loss/(batch_idx+1), 100.*correct/total 

    def conformalize_model(self): 
        self.model = torch.nn.DataParallel(self.model) 
        self.model.eval() 

        # optimize for 'size' or 'adaptiveness'
        lamda_criterion = 'size'
        
        # Conformalize model
        self.conformal_model = ConformalModel(
            self.model, 
            self.calib_loader, 
            alpha=0.1, 
            num_classes=self.act_dim, 
            lamda=0, 
            randomized=True, # use the randomized version of conformal 
            allow_zero_sets=False # allow sets of size zero 
        ) 
        print("Model calibrated and conformalized! Now evaluate over remaining data.")  
    
    def get_conformal_action_predictions(self, obs): 
        # print("Getting conformal action predictions") 
        with torch.no_grad():
            # switch to evaluate mode
            self.conformal_model.eval() 
            output, S = self.conformal_model(obs) 
        return output, S 

    def get_single_action_prediction(self, obs): 
        print("Getting single action predictions") 
        with torch.no_grad():
            # switch to evaluate mode
            self.model.eval() 
            output = self.model(obs) 
        return output  

    def validate_model(self): 
        print("Validating conformal action predictions") 
        top1_avg, top5_avg, coverage_avg, size_avg = validate(self.val_loader, self.conformal_model, print_bool=True)
        # validate_new(val_loader, model, print_bool=True)
        print("Complete!") 
        return top1_avg, top5_avg, coverage_avg, size_avg  

