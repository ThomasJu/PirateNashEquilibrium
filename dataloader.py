import torch
import numpy as np
from torch.utils.data import DataLoader
# Game procedure
# Round 1
def play_games(sharesnn, votesnn, shares_optim, votes_optim, epoch):
    # round 1
    proposed_shares = sharesnn[0].forward(1)
    votes = [votenn.forward(proposed_shares, 1)>0.5 for votenn in votesnn]
    if sum(votes)/len(votes) > 0.5:      # proposed_shares win the election
        conclude_games(proposed_shares, sharesnn, votesnn, shares_optim, votes_optim, 0, epoch)
        return proposed_shares
    
    # round 2 (player 1 solution isn't approved by half the players)
    proposed_shares = sharesnn[1].forward(1)
    votes = [votenn.forward(proposed_shares, 2)>0.5 for votenn in votesnn]
    if sum(votes)/len(votes) > 0.5:      # proposed_shares win the election
        conclude_games(proposed_shares, sharesnn, votesnn, shares_optim, votes_optim, 1, epoch)
        return proposed_shares
    
    # round 3
    proposed_shares = sharesnn[2].forward(1)
    conclude_games(proposed_shares, sharesnn, votesnn, shares_optim, votes_optim, 2, epoch)
    return proposed_shares
    
def conclude_games(proposed_shares, sharesnn, votesnn, shares_optim, votes_optim, i, epoch):
    if epoch % 2 == 0:     # train only proposed share layer
        for votenn in votesnn:
            for param in votenn.parameters():
                param.requires_grad = False
        
        torch.mul(torch.index_select(proposed_shares, 0, torch.tensor([i])), -1).backward(retain_graph=True)
        shares_optim[i].step()
        
        for votenn in votesnn:
            for param in votenn.parameters():
                param.requires_grad = True
    else:
        for sharenn in sharesnn:
            for param in sharenn.parameters():
                param.requires_grad = False
        
        for j in range(3):    
            torch.mul(torch.index_select(proposed_shares, 0, torch.tensor([j])), -1).backward(retain_graph=True)
            votes_optim[i].step()
        
        for sharenn in sharesnn:
            for param in sharenn.parameters():
                param.requires_grad = True

@torch.no_grad()  
def official_game(sharesnn, votesnn):
    # round 1
    print(f'Round 1')
    proposed_shares = sharesnn[0].forward(1)
    print(f'player1 propose {proposed_shares}')    
    votes = [votesnn[i].forward(proposed_shares, 1)>0.5 for i in range(3)]
    print(f'players votes: {votes}')
    if sum(votes)/len(votes) > 0.5:      # proposed_shares win the election
        print(f'players accept player1 policy with proposed_shares {proposed_shares}')
        print('Game Ended')
        return proposed_shares
    
    # round 2 (player 1 solution isn't approved by half the players)
    print(f'Round 2')
    proposed_shares = sharesnn[1].forward(1)
    print(f'player2 propose {proposed_shares}') 
    votes = [votesnn[i].forward(proposed_shares, 2)>0.5 for i in range(1, 3)]
    print(f'players votes: {votes}')
    if sum(votes)/len(votes) > 0.5:      # proposed_shares win the election
        print(f'players accept player2 policy with proposed_shares {proposed_shares}')
        print('Game Ended')
        return proposed_shares
    
    # round 3
    print(f'Round 3')
    proposed_shares = sharesnn[2].forward(1)
    print(f'player3 propose {proposed_shares}')
    print(f'No need to votes only player3 left')
    print('Game Ended')
    return proposed_shares

class PretrainedDataset(torch.utils.data.Dataset):
    def __init__(self, num_example, pirate_idx):
        self.num_example = num_example
        self.pirate_idx = pirate_idx
        self.data = np.random.dirichlet(np.ones(3), size=num_example)

    def __len__(self):
        return self.num_example

    def __getitem__(self, idx):
        point = self.data[idx]
        label = float(self.data[idx][self.pirate_idx] > 0.33333)
        round = idx % 3
        sample = {"data": (point, round), "label": label}
        return sample