import numpy as np
import random

# return list of value of how to share the money
class person:
    def __init__(self, order, num_people):
        self.order = order
        self.num_people = num_people
        
    def give_shares(self):
        share_list = [0] * (self.num_people)
        share_list[self.order] = 1
        return share_list
    
    def votes(self, share):
        return 1
    
    def result(self, result):
        self.result = result
        
def end_games(self, players, proposed_share):
    for player, share in zip(players, proposed_share):
        player.result = share
    

def play_games(p1, p2, p3):
    players = [p1, p2, p3]
    for i in range(len(players)):
        proposed_share = players[i].give_shares()
        votes = [players[j].votes(proposed_share) for j in range(i, len(players))]
        if sum(votes)/len(votes) > 0.5:
            end_games(players, proposed_share)
            break
        
def main():
    p1, p2, p3 = person(0, 3), person(1, 3), person(2, 3)   
    play_games(p1, p2, p3)

if __name__ == "__main__":
    main()

