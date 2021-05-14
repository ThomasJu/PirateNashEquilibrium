import torch
from models import ShareNN, VotesNN
from train import train_model, pre_train_votes

lr = 0.1
weight_decay = 0.01


sharesnn, votesnn = [ShareNN()] * 3, [VotesNN()] * 3
share_optim = [torch.optim.Adam(sharenn.parameters(), lr=lr, weight_decay=weight_decay) for sharenn in sharesnn]
votes_optim = [torch.optim.Adam(votenn.parameters(), lr=1, weight_decay=weight_decay) for votenn in votesnn]

# sharesnn, votesnn = train_model(sharesnn, share_optim, votesnn, votes_optim, num_epoch=100, collect_cycle=30, device='cpu', verbose=True)
votesnn = pre_train_votes(votesnn, votes_optim, num_epoch=1)