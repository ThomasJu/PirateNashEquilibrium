import torch
from models import ShareNN, VotesNN
from train import train_model

lr = 0.1
weight_decay = 0.01


sharesnn, votesnn = ShareNN(), VotesNN()
share_optim = torch.optim.Adam(sharesnn.parameters(), lr=lr, weight_decay=weight_decay)
votes_optim = torch.optim.Adam(votesnn.parameters(), lr=lr, weight_decay=weight_decay)

sharesnn, votesnn = train_model(sharesnn, share_optim, votesnn, votes_optim, num_epoch=5, collect_cycle=30, device='cpu', verbose=True)