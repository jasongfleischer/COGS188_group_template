import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque,defaultdict
import torch.nn.functional as F
import numpy as np


class Tile:
    order_of_suits = [["mans", "pins", "sticks"],
                      ["東", "南", "西", "北", "白", "發", "中"]]
    
    def __init__(self, suit, value):
        self.suit = suit
        self.value = value
        self.enumerate()

    def __repr__(self):
        return f"{self.suit} {self.value}"

    def __eq__(self, other):
        return self.suit == other.suit and self.value == other.value
    
    def enumerate(self):
        if self.suit in self.order_of_suits[0]:
            self.num = self.value + 4 * self.order_of_suits[0].index(self.suit)
        else:
            self.num = 30 + self.order_of_suits[1].index(self.suit)
    
    def reorder_suits(self, order_of_suits):
        self.order_of_suits = order_of_suits
        self.enumerate()
    

class MahjongGame:
    def __init__(self, num_players=4, device='cpu', silent=False):
        self.num_players = num_players
        self.players = [[] for _ in range(num_players)]

        # create state matrix: player, location, tile type
        self.state_matrix = torch.zeros((self.num_players, 4, 37))
        # location meanings:
        # 0: unknown position
        # 1: in hand, not locked in set
        # 2: in hand, locked in set
        # 3: not in hand, inaccessible

        self.wall = self._create_wall()
        self.discards = [[] for _ in range(num_players)]
        self.current_player = 0
        self.scores = [0] * num_players
        self.silent = silent
        self.device = device
        self.agent = None
        self.last_discard = None

    def toggle_silence(self):
        self.silent = not self.silent

    def _create_wall(self):
        self.state_matrix *= 0 #remove all tiles
        self.state_matrix[:, 0] += 4 #add 4 tiles to unknown

        suits = ["mans", "pins", "sticks"]
        tiles = [Tile(suit, value) for suit in suits for value in range(1, 10)] * 4 #includes flowers
        honors  = ["東", "南", "西", "北", "白", "發", "中"]
        for i in range(4):
            tiles.extend([Tile(suit, 0) for suit in honors])
        random.shuffle(tiles)
        self.last_discard = None
        return tiles

    def deal_tiles(self):
        # Deal 13 tiles to each player
        for _ in range(13):
            for player in range(self.num_players):
                self.draw_tile(player)

    def draw_tile(self, player, do_matrix=True, do_old=True):
        # Player draws a tile from the wall
        tile = self.wall.pop()
        return self._add(player, tile, do_matrix, do_old)

    def _add(self, player, tile, do_matrix=True, do_old=True):
        if do_old:
            self.players[player].append(tile)
        if do_matrix:
            self.state_matrix[player, 0, tile.num] -= 1 #remove from unknown
            self.state_matrix[player, 1, tile.num] += 1 #add to hand
            assert self.state_matrix[player, :, tile.num].sum() == 4
            assert self.state_matrix[player, :, tile.num].min() >= 0
        return tile
    
    def _pong(self, player, tile):
        self.players[player].append(tile)
        for p in range(self.num_players):
            self.state_matrix[p, 3, tile.num] -= 1
            if p == player:
                self.state_matrix[p, 1, tile.num] -= 2
                self.state_matrix[p, 2, tile.num] += 3
                assert self.state_matrix[p, :, tile.num].sum() == 4
                assert self.state_matrix[p, :, tile.num].min() >= 0
            else:
                self.state_matrix[p, 0, tile.num] -= 2
                self.state_matrix[p, 3, tile.num] += 3
                assert self.state_matrix[p, :, tile.num].sum() == 4
                assert self.state_matrix[p, :, tile.num].min() >= 0
    
    def _chow(self, player, tile):
        self.players[player].append(tile)
        for p in range(self.num_players):
            if p == player:
                self.state_matrix[p, 1, tile.num-1] -= 1
                self.state_matrix[p, 0, tile.num] -= 1
                self.state_matrix[p, 1, tile.num+1] -= 1
                new = 2
            else:
                self.state_matrix[p, 0, tile.num-1:tile.num+2] -= 1
                new = 3
            self.state_matrix[p, new, tile.num-1:tile.num+2] += 1
            assert self.state_matrix[p, :, tile.num-1].sum() == 4
            assert self.state_matrix[p, :, tile.num].sum() == 4
            assert self.state_matrix[p, :, tile.num+1].sum() == 4
            assert self.state_matrix[p, :, tile.num-1].min() >= 0
            assert self.state_matrix[p, :, tile.num].min() >= 0
            assert self.state_matrix[p, :, tile.num+1].min() >= 0

    def discard_tile(self, player, tile, do_matrix=False):
        # Player discards a tile
        if tile not in self.players[player]:
            print(f"Error: {tile} not in player's hand!")
            return None
        if do_matrix:
            for p in range(self.num_players):
                if p == player:
                    self.state_matrix[p, 1, tile.num] -= 1
                else:
                    self.state_matrix[p, 0, tile.num] -= 1
                self.state_matrix[p, 3, tile.num] += 1
                assert self.state_matrix[p, :, tile.num].sum() == 4
                assert self.state_matrix[p, :, tile.num].min() >= 0
        self.players[player].remove(tile)
        self.discards[player].append(tile)
        
    def _discard(self, player, tile, do_old=False):
        if do_old:
            self.players[player].remove(tile)
            self.discards[player].append(tile)
        self.state_matrix[player, 1, tile.num] -= 1
        self.state_matrix[player, 0, tile.num] += 1
        assert self.state_matrix[player, :, tile.num].sum() == 4
        assert self.state_matrix[player, :, tile.num].min() >= 0

    def get_game_state(self):
        #current game state
        state = {
            "hand": [sorted(player.copy(), key=lambda x: (x.suit, x.value)) for player in self.players],
            "discards": [sorted(discard.copy(), key=lambda x: (x.suit, x.value)) for discard in self.discards],
            "current_player": self.current_player
        }
        return state
    
    def add_agent(self, agent):
        self.agent = agent

    def check_win(self, player):
        # Check if the player's hand is a winning hand
        # Implement winning logic here
        hand = self.players[player]
        if is_winning_hand(hand):
            print(f"Player {player} wins!")
            self.scores[player] += 1000
            return True
        return False
    
    def check_pong(self, tile):
        for n in range(self.num_players-1):
            player = (self.current_player + n) % self.num_players
            if self.state_matrix[player, 1, tile.num] == 2:
                if not self.silent:
                    print(f"Player {player} forms a Pong with {tile}")
                return player
        return False
    
    def one_turn(self):
        tile = self.last_discard
        if tile:
            pong = self.check_pong(tile)
            if pong:
                self.current_player = pong
                self._pong(self.current_player, tile)
            elif can_chow_from_discard(self.players[self.current_player], tile):
                self._chow(self.current_player, tile)
                if not self.silent:
                    print(f"Player {self.current_player} forms a chow with {tile}")
            else:
                tile = self.draw_tile(self.current_player)
                if not self.silent:
                    print(f"Player {self.current_player} draws {tile}")
        else:
            tile = self.draw_tile(self.current_player)
            if not self.silent:
                print(f"Player {self.current_player} draws {tile}")
        
        if self.check_win(self.current_player):
            return True, self.current_player

        discard = self.decide_tile_to_discard(self.current_player)
        self.discard_tile(self.current_player, discard, True)
        self.current_player += 1
        self.current_player %= self.num_players
        self.last_discard = discard
        return False, self.current_player
    
    def decide_tile_to_discard(self, p):
        best_tile = None
        best_value = -float("inf")

        with torch.no_grad():
            for tile in self.players[p]:
                self._discard(p, tile)
                predicted_value = self.agent.predict_reward(self.state_matrix[p:p+1]).item()
                if predicted_value > best_value:
                    best_tile = tile
                self._add(p, tile, do_old=False)

        if self.state_matrix[p, best_tile.num, 1] == 0:
            print(f"Error: {best_tile} not found in player's hand!")
            return None
        return best_tile

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
    #print(counts)
    test_hand = hand
    for (suit, value), num_times in counts.items():
        test_hand = hand.copy()
        if num_times >= 2:
            test_hand.remove(Tile(suit, value))
            test_hand.remove(Tile(suit, value))
            counts[(suit, value)] -= 2
            if can_form_meld(test_hand):
                return True
            test_hand.append(Tile(suit, value))
            test_hand.append(Tile(suit, value))
            counts[(suit, value)] += 2
    return False


def is_pon(tiles):
    """
    Check if the given tiles form a valid pung (3 identical tiles).
    """
    if len(tiles) != 3:
        return False
    return all(tile == tiles[0] for tile in tiles)


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


class GlobalRewardPredictor(nn.Module):
    def __init__(self, input_size, net_type='lin'):
        super(GlobalRewardPredictor, self).__init__()
        if net_type == 'lin':
            self.model = nn.Sequential(
                nn.Flatten(),
                nn.Linear(input_size, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
            )
        else:
            self.model = nn.Sequential(
                nn.Conv1d(4, 8, 4), # applies learned kernel function to locations for each tile type
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(8*input_size, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Softmax()
            )

        self.flatten = nn.Flatten()

    def forward(self, state):
        x = self.model(state)
        return x


class GlobalRewardAgent:
    def __init__(self, input_size, learning_rate=0.001, net_type='lin', device='cpu'):
        self.device = device
        self.model = GlobalRewardPredictor(input_size, net_type).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

    def predict_reward(self, state_features):
        return self.model(state_features)

    def train(self, state, true_reward, drawn_tile):
        state_features = look_ahead_features(state, state['current_player'], drawn_tile)
        predicted_reward = self.predict_reward(state_features)#input state_features into NN
        loss = self.criterion(predicted_reward, torch.FloatTensor([true_reward]))
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.item()
    
    def matrix_train(self, state_matrix, true_reward):
        predicted_reward = self.predict_reward(state_matrix)
        loss = self.criterion(predicted_reward, torch.FloatTensor([true_reward]))
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.item()
    
    def get_reward(self, player, scores):
        reward = 2 * scores[player] - sum(scores)

    def calculate_reward(self, state, action, player, drawn_tile, is_win=False):
        """
        Calculate the reward based on the current state and the action taken.
        """
        reward = 0
        player_hand = state['hand'][player]
        # Winning reward
        if is_win:
            reward += 1000  # Reward for winning the game
        
        # Reward for forming a Pong or Chow
        if action == "Pong" and is_pon(player_hand):
            reward += 100  # Reward for forming a Pong
        elif action == "Chow" and can_chow_from_discard(player_hand, drawn_tile):
            reward += 50  # Reward for forming a Chow
        
        # Penalty for discarding important tiles (e.g., if a tile is essential for completing a hand)
        if action == "Discard" and self.is_important_tile(drawn_tile, player_hand):
            reward -= 100  # Penalty for discarding a critical tile
        
        # Small penalty for taking too long to make a decision
        reward -= 5  # A small penalty for taking a longer turn (to encourage efficiency)
        
        return reward

    def is_important_tile(self, tile, hand):
        """
        Determine if a tile is critical for completing the hand.
        This can be based on the strategy, such as if a tile is required to complete a set.
        """
        # A simplistic approach: check if the tile completes a potential meld
        for meld in find_melds(hand):
            for tile_in_meld in meld:
                if tile_in_meld == tile:
                    return False  # If the tile is already part of a meld, it's not important to discard
        
        # If the tile is not part of any meld, it might be important
        return True
   

def can_chow_from_discard(hand, discarded_tile):
    """
    Check if the player can form a Chow (consecutive 3 tiles of the same suit) using a discarded tile.
    """
    if discarded_tile.suit == "honors":
        return False  # Honors tiles cannot form a Chow
    
    # Group the hand by suit and check if we can form a Chow with the discarded tile
    grouped_by_suit = defaultdict(list)
    for tile in hand:
        if tile.suit != "honors":  # Honors tiles can't form a Chow
            grouped_by_suit[tile.suit].append(tile.value)

    # For each suit, check if a Chow can be formed with the discarded tile
    for suit, values in grouped_by_suit.items():
        if discarded_tile.suit == suit:
            values.append(discarded_tile.value)  # Add the discarded tile
            values.sort()  # Sort the values to check for sequences
            
            # Look for a sequence of 3 consecutive tiles
            for i in range(len(values) - 2):
                if values[i] + 1 == values[i + 1] and values[i + 1] + 1 == values[i + 2]:
                    return True  # Found a valid Chow
    return False