import torch
from models import ShareNN, VotesNN

lr = 0.1
weight_decay = 0.01


sharesnn, votesnn = ShareNN(), VotesNN()
share_optim = torch.optimizer.Adam(sharesnn.parameters(), lr=lr, weight_decay=weight_decay)
votes_optim = torch.optimizer.Adam(votesnn.parameters(), lr=lr, weight_decay=weight_decay)

sharesnn, votesnn = train_model(sharesnn, shares_optim, votesnn, votes_optim, num_epoch=5, collect_cycle=30, device='cpu', verbose=True)