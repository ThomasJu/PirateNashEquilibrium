import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F

# input:  player id  1
# output:   proposed shared  3
# This neural network supposed to propose shares that 
# can achieve optimal shares of money
class ShareNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 8)
        self.fc4 = nn.Linear(8, 3)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return x
    

# input:  player id  1 and given share 3
# output:   votes  (either yes or no) but it's a probability here
# This neural network supposed to votes for shared proposed by others that 
# can achieve optimal shares of money 
class VotesNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 4)
        self.fc4 = nn.Linear(4, 1)
    
    def forward(self, player_id, proposed_shares):
        x = torch.cat((player_id, proposed_shares))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return x