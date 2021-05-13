import torch
# Game procedure
# Round 1
def play_games(sharesnn, votesnn, shares_optim, votes_optim):
    # round 1
    proposed_shares = sharesnn.forward(1)
    votes = [votesnn.forward(i, proposed_shares, 1) for i in range(3)]
    if sum(votes)/len(votes) > 0.5:      # proposed_shares win the election
        conclude_games(proposed_shares, shares_optim, votes_optim, 0)
        return proposed_shares
    
    # round 2 (player 1 solution isn't approved by half the players)
    proposed_shares = sharesnn.forward(2)
    votes = [votesnn.forward(i, proposed_shares, 2) for i in range(1, 3)]
    if sum(votes)/len(votes) > 0.5:      # proposed_shares win the election
        conclude_games(proposed_shares, shares_optim, votes_optim, 1)
        return proposed_shares
    
    # round 3
    proposed_shares = sharesnn.forward(3)
    conclude_games(proposed_shares, shares_optim, votes_optim, 2)
    return proposed_shares
    
def conclude_games(proposed_shares, shares_optim, votes_optim, i):
    torch.autograd.set_detect_anomaly(True)
    torch.mul(torch.index_select(proposed_shares, 0, torch.tensor([i])), -1).backward(retain_graph=True)
    shares_optim.step()
    votes_optim.step()
    
    # for j in range(3):
    #     torch.mul(torch.index_select(proposed_shares, 0, torch.tensor([j])), -1).backward(retain_graph=True)
    #     votes_optim.step()
    #     if i == j:
    #         shares_optim.step()



@torch.no_grad()  
def official_game(sharesnn, votesnn):
    # round 1
    print(f'Round 1')
    proposed_shares = sharesnn.forward(1)
    print(f'player1 propose {proposed_shares}')    
    votes = [votesnn.forward(i, proposed_shares, 1) for i in range(3)]
    print(f'players votes: {votes}')
    if sum(votes)/len(votes) > 0.5:      # proposed_shares win the election
        print(f'players accept player1 policy with proposed_shares {proposed_shares}')
        print('Game Ended')
        return proposed_shares
    
    # round 2 (player 1 solution isn't approved by half the players)
    print(f'Round 2')
    proposed_shares = sharesnn.forward(2)
    print(f'player2 propose {proposed_shares}') 
    votes = [votesnn.forward(i, proposed_shares, 2) for i in range(1, 3)]
    print(f'players votes: {votes}')
    if sum(votes)/len(votes) > 0.5:      # proposed_shares win the election
        print(f'players accept player2 policy with proposed_shares {proposed_shares}')
        print('Game Ended')
        return proposed_shares
    
    # round 3
    print(f'Round 3')
    proposed_shares = sharesnn.forward(3)
    print(f'player3 propose {proposed_shares}')
    print(f'No need to votes only player3 left')
    print('Game Ended')
    return proposed_shares