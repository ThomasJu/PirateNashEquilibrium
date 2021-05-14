import time
import torch
import numpy as np
from dataloader import play_games, official_game, PretrainedDataset
from torch.utils.data import DataLoader

def train_model(sharesnn, shares_optim, votesnn, votes_optim, num_epoch=5, collect_cycle=30, device='cpu', verbose=True):
    # Initialize:
    # -------------------------------------
    if verbose:
        print('------------------------ Start Training ------------------------')
    t_start = time.time()
    num_itr = 0
    for epoch in range(num_epoch):
        ############## Training ##############    
        [sharenn.train() for sharenn in sharesnn]
        [votenn.train() for votenn in votesnn]
        num_itr += 1
        
        # zero the parameter gradients
        [share_optim.zero_grad() for share_optim in shares_optim]
        [vote_optim.zero_grad() for vote_optim in votes_optim]
        sharesnn = [sharenn.to(device) for sharenn in sharesnn]
        votesnn = [votenn.to(device) for votenn in votesnn]

        # forward
        proposed_shares = play_games(sharesnn, votesnn, shares_optim, votes_optim, epoch)

        ###################### End of your code ######################              
        if verbose:
            print('Epoch No. {0}--Iteration No. {1}'.format(epoch + 1, num_itr,))

        ############## Validation ##############
        # now see how much the players has learn from failures
        if epoch % 10 == 0:
            official_game(sharesnn, votesnn)
    
    t_end = time.time()
    if verbose:
        print('Training lasted {0:.2f} minutes'.format((t_end - t_start)/60))
        print('------------------------ Training Done ------------------------')

    return sharesnn, votesnn

# this function pretrain the votesnn so votes know that it should support
def pre_train_votes(votesnn, votes_optim, num_epoch=5, collect_cycle=30, device='cpu', verbose=True):
    # Initialize:
    # -------------------------------------
    if verbose:
        print('------------------------ Start Training ------------------------')
    t_start = time.time()
    num_itr = 0
    for epoch in range(num_epoch):
        ############## Training ##############    
        [votenn.train() for votenn in votesnn]
        num_itr += 1
        
        # zero the parameter gradients
        [vote_optim.zero_grad() for vote_optim in votes_optim]
        votesnn = [votenn.to(device) for votenn in votesnn]

        # forward
        loss_func = torch.nn.MSELoss()
        for i, (votenn, voteoptim) in enumerate(zip(votesnn, votes_optim)):
            print(i)
            dataset = PretrainedDataset(1000, i)
            pretrain_dataloader = DataLoader(dataset, batch_size=5, shuffle=True)
            for databatch in pretrain_dataloader:
                vote = votenn.forward(databatch['data'][0], databatch['data'][1])
                loss = loss_func(vote, databatch['label']).float()
                print(loss)
                loss.backward()
                voteoptim.step()     
            
        ###################### End of your code ######################              
        if verbose:
            print('Epoch No. {0}--Iteration No. {1}'.format(epoch + 1, num_itr,))

        ############## Validation ##############
        # now see how much the players has learn from failures
        if epoch == num_epoch - 1:
            for i, votenn in enumerate(votesnn):
                print(f'votes {i}: training result')
                for j, data in enumerate(np.random.dirichlet(np.ones(3), size=10)):
                    data = torch.tensor(data, dtype= torch.float, requires_grad= True)
                    vote = votenn.forward(data, j%3)
                    print(f'data = {data.data}, vote = {vote}')
    
    t_end = time.time()
    if verbose:
        print('Training lasted {0:.2f} minutes'.format((t_end - t_start)/60))
        print('------------------------ Training Done ------------------------')

    return votesnn