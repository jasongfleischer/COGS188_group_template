import numpy as np
import random
from collections import defaultdict
from monte_carlo_control import OnePlayerMahjongGame

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
        """
        Loop through all states in the self.Q dictionary. 
        1. determine the greedy policy for that state
        2. create a probability vector that is all 0s except for the greedy action where it is 1
        3. store that probability vector in self.target_policy[state]

        Args: none
        Returns: none, stores new policy in self.target_policy
        """
        # Your code here
        for state in self.Q:
            self.greedy_policy[state] = self.Q[state]


    def create_behavior_policy(self):
        for state in self.greedy_policy:
            #Do epsilon greedy action selection
           best_action = self.egreedy_selection(state)
           self.egreedy_policy[state] = self.Q[state]   

            
            #Assigns the value to the egreedy policy
        # for state in self.Q:
        #     best_action = np.argmax(self.greedy_policy[state])
        #     action_prob = np.full(len(self.greedy_policy[state]), self.epsilon / len(self.greedy_policy[state]))
        #     action_prob[best_action] += 1 - self.epsilon
        #     self.egreedy_policy[state] = action_prob
        # for state, action_probs in self.greedy_policy.items():
        #     num_actions = len(action_probs)
        #     greedy_action = max(action_probs, key=action_probs.get)  # Find the action with the highest probability
            
        #     # Initialize all actions with epsilon-distributed probability
        #     self.egreedy_policy[state] = {action: self.epsilon / num_actions for action in action_probs}
            
        #     # Assign the greedy action the additional probability mass
        #     self.egreedy_policy[state][greedy_action] += (1 - self.epsilon)
   


        
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
        #Initialize variables 
        state = self.env.reset()
        self.wall = self.env._create_wall()
        self.env.draw_tile()
        counter = 0
        path = []
        #Begin going through episode
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
        #Initialize variables 
        state = self.env.reset()
        self.wall = self.env._create_wall()
        self.env.draw_tile()
        counter = 0
        path = []
        #print(self.Q)
        self.create_target_policy()
        #Begin going through episode
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
        #Initialize G, W values
        G = 0.0
        W = 1.0

        #Follow Sutton and Barton's pseudocode
        for t in range(len(episode)-1, -1, -1):
            __, state, reward = episode[t]
            # print(episode[t])
            G = self.gamma*G + reward
            # if (state, action) not in self.C:
            #     self.C[state][action] = 0.0
            #print(action)
            # if not isinstance(state, tuple):
            #     state = tuple(state.tolist())
            state = (state.suit, state.value)
            self.C[state] += W
            self.Q[state] = self.Q[state] + (W/(self.C[state]))*(G - self.Q[state])
            # self.greedy_policy[state][np.argmax(self.Q[(state, action)])] = 1
            if self.greedy_policy[state] == 0:
                continue
            W *= self.egreedy_policy[state]/self.greedy_policy[state]
            if W == 0:
                break
        self.create_target_policy()
        self.create_behavior_policy()


