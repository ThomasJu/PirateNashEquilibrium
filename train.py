import time
import torch
import torch.nn as nn
import torch.optim as optimizer
import numpy as np
from dataloader import play_games, official_game

def train_model(sharesnn, shares_optim, votesnn, votes_optim, num_epoch=5, collect_cycle=30, device='cpu', verbose=True):
    # Initialize:
    # -------------------------------------
    if verbose:
        print('------------------------ Start Training ------------------------')
    t_start = time.time()
    num_itr = 0
    for epoch in range(num_epoch):
        ############## Training ##############
        sharesnn.train()
        votesnn.train()
        num_itr += 1
        
        # zero the parameter gradients
        shares_optim.zero_grad()
        votes_optim.zero_grad()
        sharesnn = sharesnn.to(device)
        votesnn = votesnn.to(device)

        # forward
        proposed_shares = play_games(sharesnn, votesnn, shares_optim, votes_optim)

        ###################### End of your code ######################              
        if verbose:
            print('Epoch No. {0}--Iteration No. {1}'.format(epoch + 1, num_itr,))

        ############## Validation ##############
        # now see how much the players has learn from failures
        if epoch % 10 == 0:
            print('aaaa')
            official_game(sharesnn, votesnn)
    
    t_end = time.time()
    if verbose:
        print('Training lasted {0:.2f} minutes'.format((t_end - t_start)/60))
        print('------------------------ Training Done ------------------------')

    return sharesnn, votesnn