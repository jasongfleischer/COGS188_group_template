import numpy as np
import random
from collections import defaultdict
from Monte_Carlo_Environment import OnePlayerMahjongGame

class MonteCarlo:
    def __init__(self, env: OnePlayerMahjongGame, gamma: float = 1.0, epsilon: float = 0.1, Q0: float = 0.0, max_episode_size : int = 100):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q0 = Q0
        self.max_episode_size = max_episode_size
        self.Q = defaultdict(int)
        self.C = defaultdict(int)
        self.greedy_policy = defaultdict(int)
        self.egreedy_policy = defaultdict(int)

    def create_target_policy(self):
        for state in self.Q:
            self.greedy_policy[state] = self.Q[state]


    def create_behavior_policy(self):
        for state in self.greedy_policy:
           best_action = self.egreedy_selection(state)
           self.egreedy_policy[state] = self.Q[state]   

        
    def egreedy_selection(self, state):
        if np.random.rand() < self.epsilon:
            tile_index = np.random.randint(0, len(self.env.player))
            best_action = self.env.player[tile_index]

        else:
            best_value = -np.inf
            for tile in self.env.player:
                value = self.Q[(tile.suit, tile.value)]
                if value > best_value:
                    best_value = value
                    best_action = tile
        return best_action
    
    def greedy_selection(self):
        best_value = -np.inf
        for tile in self.env.player:
            best_action = tile
            tuple_tile = (tile.suit, tile.value)
            value = self.Q[tuple_tile]
            if value > best_value:
                best_value = value
                best_action = tile
        return best_action

    def generate_egreedy_episode(self):
        state = self.env.reset()
        self.wall = self.env._create_wall()
        self.env.draw_tile()
        counter = 0
        path = []
        while counter < 119:
            action = self.egreedy_selection(state)
            state, action, reward = self.env.take_turn(action)
            path.append((state, action, reward))
            if self.env.is_winning_hand(state):
                break
            state = self.env.player
            counter += 1
            self.env.draw_tile()

        return path

        
    
    def generate_greedy_episode(self):
        state = self.env.reset()
        self.wall = self.env._create_wall()
        self.env.draw_tile()
        counter = 0
        path = []
        self.create_target_policy()
        while counter < 119:
            action = self.greedy_selection()
            state, action, reward = self.env.take_turn(action)
            path.append((state, action, reward))
            if self.env.is_winning_hand(state):
                break 
            state = self.env.player
            counter += 1
            self.env.draw_tile()
        return path
    
    def update_offpolicy(self, episode):
        G = 0.0
        W = 1.0

        for t in range(len(episode)-1, -1, -1):
            __, state, reward = episode[t]
            G = self.gamma*G + reward
            state = (state.suit, state.value)
            self.C[state] += W
            self.Q[state] = self.Q[state] + (W/(self.C[state]))*(G - self.Q[state])
            if self.greedy_policy[state] == 0:
                continue
            W *= self.egreedy_policy[state]/self.greedy_policy[state]
            if W == 0:
                break
        self.create_target_policy()
        self.create_behavior_policy()


