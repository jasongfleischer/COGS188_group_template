import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque,defaultdict
import torch.nn.functional as F
import numpy as np


class Tile:
    def __init__(self, suit, value):
        self.suit = suit
        self.value = value

    def __repr__(self):
        return f"{self.suit} {self.value}"

class MahjongGame:
    def __init__(self, num_players=4):
        self.num_players = num_players
        self.players = [[] for _ in range(num_players)]
        self.wall = self._create_wall()
        self.discards = [[] for _ in range(num_players)]
        self.current_player = 0

    def _create_wall(self):
        # Create and shuffle a standard set of 144 tiles
        suits = ["mans", "pins", "sticks", "honors"]
        tiles = [Tile(suit, value) for suit in suits for value in range(1, 10)] * 4 #includes flowers
        random.shuffle(tiles)
        return tiles

    def deal_tiles(self):
        # Deal 13 tiles to each player
        for _ in range(13):
            for player in self.players:
                player.append(self.wall.pop())

    def draw_tile(self, player):
        # Player draws a tile from the wall
        tile = self.wall.pop()
        self.players[player].append(tile)
        return tile

    def discard_tile(self, player, tile):
        # Player discards a tile
        self.players[player].remove(tile)
        self.discards[player].append(tile)

    def get_game_state(self):
        #current game state
        state = {
            "hand": [player.copy() for player in self.players],
            "discards": [discard.copy() for discard in self.discards],
            "current_player": self.current_player
        }
        return state
    
    def check_win(self, player):
        # Check if the player's hand is a winning hand
        # Implement winning logic here
        hand = self.players[player]
        if is_winning_hand(hand):
            print(f"Player {player} wins!")
            self.scores[player] += 1000
            return True
        return False
        

    def play_turn(self):
        tile = self.draw_tile(self.current_player)
        print(f"Player {self.current_player} draws {tile}")
        if self.check_win(self.current_player):
            return True  # ゲーム終了
        self.current_player = (self.current_player + 1) % self.num_players
        return False  #


    

def is_straight(tiles):
    """
    Check if the given tiles form a valid straight (chow).
    Tiles must be of the same suit and consecutive in value.
    """
    if len(tiles) != 3:
        return False
    suits = {tile.suit for tile in tiles}
    if len(suits) != 1:  # All tiles must be of the same suit
        return False
    if any(tile.suit == "honors" for tile in tiles):
        return False
    values = sorted([tile.value for tile in tiles])
    return values[0] + 1 == values[1] and values[1] + 1 == values[2]

def is_pon(tiles):
    """
    Check if the given tiles form a valid pung (3 identical tiles).
    """
    if len(tiles) != 3:
        return False
    return all(tile == tiles[0] for tile in tiles)

def is_pair(tiles):
    """
    Check if the given tiles form a valid pair (2 identical tiles).
    """
    if len(tiles) != 2:
        return False
    return tiles[0] == tiles[1]

def find_melds(hand):
    """
    Find all possible melds (straights or pungs) in the hand.
    """
    melds = []
    hand = sorted(hand, key=lambda x: (x.suit, x.value))  # Sort hand for easier processing

    # Check for pungs
    counts = defaultdict(int)
    for tile in hand:
        counts[(tile.suit, tile.value)] += 1
    for key, count in counts.items():
        if count >= 3:
            melds.append([tile for tile in hand if tile.suit == key[0] and tile.value == key[1]][:3])

    # Check for straights
    for i in range(len(hand) - 2):
        group = hand[i:i+3]
        if is_straight(group):
            melds.append(group)

    return melds

def is_winning_hand(hand):
    """
    Check if the hand is a winning hand (4 melds and 1 pair).
    """
    if len(hand) != 14:  # A complete hand has 14 tiles
        return False

    # Find all possible pairs
    counts = defaultdict(int)
    for tile in hand:
        counts[(tile.suit, tile.value)] += 1
    pairs = [key for key, count in counts.items() if count >= 2]

    # Try each pair and see if the remaining tiles can form 4 melds
    for pair_key in pairs:
        remaining_hand = hand.copy()
        # Remove the pair from the hand
        pair_tiles = [tile for tile in remaining_hand if tile.suit == pair_key[0] and tile.value == pair_key[1]][:2]
        for tile in pair_tiles:
            remaining_hand.remove(tile)

        # Find melds in the remaining hand
        melds = find_melds(remaining_hand)
        if len(melds) >= 4:
            return True

    return False


def is_reach_possible(hand):
    """
    Check if the hand is in a state where a player can declare a reach.
    (i.e., the hand is 13 tiles and 1 tile away from being a winning hand)
    """
    if len(hand) != 14: 
        return False
    counts = defaultdict(int)
    for tile in hand:
        counts[(tile.suit, tile.value)] += 1
    pairs = [key for key, count in counts.items() if count >= 2]
    for pair_key in pairs:
        remaining_hand = hand.copy()
        pair_tiles = [tile for tile in remaining_hand if tile.suit == pair_key[0] and tile.value == pair_key[1]][:2]
        for tile in pair_tiles:
            remaining_hand.remove(tile)
        melds = find_melds(remaining_hand)
        if len(melds) == 4:
            return True

    return False
# Example usage
game = MahjongGame()
game.deal_tiles()
game.play_turn()





class GlobalRewardPredictor(nn.Module):
    def __init__(self, input_size):
        super(GlobalRewardPredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()

    def forward(self, state):
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x



def look_ahead_features(state, drawn_tile):
    features = []
    features.append(state["current_player"])#adding current player to features
    current_player_hand = state["hand"][state["current_player"]]
    features.append(len(find_melds(current_player_hand)))#adding current player's melds' number to features
    
    pair_count = 0
    counts = defaultdict(int)
    for tile in current_player_hand:
        counts[(tile.suit, tile.value)] += 1
    for count in counts.values():
        if count >= 2:
            pair_count += 1
    features.append(pair_count)##adding current player's pair's number to features

    discard_count = 0
    for player_discard in state["discards"]:
        for tile in player_discard:
            if tile.value == drawn_tile.value and tile.suit == drawn_tile.suit:
                discard_count += 1   
    features.append(discard_count)#adding number of discarded tiles same as drawn_tile to features

    return torch.FloatTensor(features)

def decide_tile_to_discard(agent, state):
    current_player_hand = state["hand"][state["current_player"]]
    
    best_tile = None
    best_value = -float("inf")  
 
    for tile in current_player_hand:
        state_features = look_ahead_features(state, discarded_tile=tile)
        predicted_value = agent.predict_reward(state_features).item()
        
        if predicted_value > best_value:
            best_value = predicted_value
            best_tile = tile
    
    return best_tile



class GlobalRewardAgent:
    def __init__(self, input_size, learning_rate=0.001):
        self.model = GlobalRewardPredictor(input_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

    def predict_reward(self, state, drawn_tile):
        state_features = look_ahead_features(state, drawn_tile) 
        return self.model(state_features)

    def train(self, state, true_reward):
        self.optimizer.zero_grad()
        predicted_reward = self.predict_reward(state)
        loss = self.criterion(predicted_reward, torch.FloatTensor([true_reward]))
        loss.backward()
        self.optimizer.step()
        return loss.item()

agent = GlobalRewardAgent(input_size)

def can_pong(tiles):


def can_chow(tiles):

    
def extract_features(state, discarded_tile):
    features = []

    features.append(state["current_player"])
  
    current_player_hand = state["hand"][state["current_player"]]
    melds = find_melds(current_player_hand)
    features.append(len(melds)) 
    
    can_pong = False
    if can_pong:
        features.append(can_pong)  #whether we can pong
        
    can_chow = False
    if can_chow:
            features.append(can_chow)
      #whether we can chow  
    return torch.FloatTensor(features)

def decide_action(agent, state, discarded_tile):
    state_features = extract_features(state, discarded_tile)
    action_values = agent.predict_action(state_features).detach().numpy()  #predicted action value

    action = np.argmax(action_values)
    return action#pong or chow