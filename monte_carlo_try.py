import numpy as np
import random
from collections import defaultdict
from monte_carlo_control import OnePlayerMahjongGame

class MonteCarlo:
    def __init__(self, env: OnePlayerMahjongGame, gamma: float = 1.0, epsilon: float = 0.1, Q0: float = 0.0, max_episode_size : int = 1000):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q0 = Q0
        self.max_episode_size = max_episode_size
        self.Q = defaultdict(lambda: np.full(env.unique_tiles, 0))
        self.C = defaultdict(lambda: np.zeros(env.n_actions))
        self.greedy_policy = defaultdict(lambda: np.full(env.unique_tiles, 1 / env.unique_tiles))
        self.egreedy_policy = defaultdict(lambda: np.full(env.unique_tiles, 1 / env.unique_tiles))

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
            best_action = np.argmax(self.Q[state])
            action_prob = np.zeros(len(self.Q[state]))
            action_prob[best_action] = 1
            self.greedy_policy[state] = action_prob


    def create_behavior_policy(self):
        """
        Loop through all states in the self.target_policy dictionary. 
        Using that greedy probability vector, and self.epsilon, 
        calculate the epsilon greedy behavior probability vector and store it in self.behavior_policy[state]
        
        Args: none
        Returns: none, stores new policy in self.target_policy
        """
        for state in self.greedy_policy:
            #Do epsilon greedy action selection
            if np.random.rand() < self.epsilon:
                best_action = np.random.randint(0, self.env.unique_tiles)
            else:
                best_action = np.argmax(self.greedy_policy[state])
            
            #Assigns the value to the egreedy policy
            action_prob = np.zeros(len(self.greedy_policy[state]))
            action_prob[best_action] = 1
            self.egreedy_policy[state] = action_prob   
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
        """
        Select an action proportional to the probabilities of epsilon-greedy encoded in self.behavior_policy
        HINT: 
        - check out https://www.w3schools.com/python/ref_random_choices.asp
        - note that random_choices returns a numpy array, you want a single int
        - make sure you are using the probabilities encoded in self.behavior_policy 

        Args: state (string): the current state in which to choose an action
        Returns: action (int): an action index between 0 and self.env.n_actions
        """
        if np.random.rand() < self.epsilon:
            tile_index = np.random.randint(0, len(self.env.player))
            best_action = self.env.player[tile_index]

        else:
            best_value = -np.inf
            for tile in self.env.player:
                value = self.Q[(tile.suit, tile.value)]
                print(value)
                if value > best_value:
                    best_value = value
                    best_action = tile
        return best_action
    
    def greedy_selection(self):
        best_value = -np.inf
        for tile in self.env.player:
            value = self.Q[tile]
            if value > best_value:
                best_value = value
                best_action = tile
        return best_action

    def generate_egreedy_episode(self):
        """
        Generate an episode using the epsilon-greedy behavior policy. Will not go longer than self.max_episode_size
        
        Hints: 
        - need to setup and use self.env methods and attributes
        - use self.egreedy_selection() above as a helper function
        - use the behavior e-greedy policy attribute aleady calculated (do not update policy here!)
        
        Returns:
            list: The generated episode, which is a list of (state, action, reward) tuples.
        """
        #Initialize variables 
        state = self.env.reset()
        print(state)
        counter = 0
        path = []
        #Begin going through episode
        while counter < self.max_episode_size:
            action = self.egreedy_selection(state)
            state, action, reward = self.env.take_turn(action)
            path.append((state, action, reward))
            state = self.env.player
            counter += 1
            if self.env.is_winning_hand():
                break

        return path

        
    
    def generate_greedy_episode(self):
        """
        Generate an episode using the greedy target policy. Will not go longer than self.max_episode_size
        Note: this function is not used during learning, its only for evaluating the target policy
        
        Hints: 
        - need to setup and use self.env methods and attributes
        - use the greedy policy attribute aleady calculated (do not update policy here!)

        Returns:
            list: The generated episode, which is a list of (state, action, reward) tuples.
        """
        #Initialize variables 
        state = self.env.reset()
        state = tuple(self.env.get_state())
        counter = 0
        path = []
        #print(self.Q)
        self.create_target_policy()
        #Begin going through episode
        while counter < self.max_episode_size:
            action = self.greedy_selection
            state, action, reward = self.env.take_turn(action)
            path.append((state, action, reward))
            state = self.env.player
            counter += 1
            if self.env.is_winning_hand():
                break 
        return path
    
    def update_offpolicy(self, episode):
        """
        Update the Q-values using every visit weighted importance sampling. 
        See Figure 5.9, p. 134 of Sutton and Barto 2nd ed.
        
        Args: episode (list): An episode generated by the behavior policy; a list of (state, action, reward) tuples.
        Returns: none
        """
        #Initialize G, W values
        G = 0.0
        W = 1.0

        #Follow Sutton and Barton's pseudocode
        for t in range(len(episode)-1, -1, -1):
            state, action, reward = episode[t]
            G = self.gamma*G + reward
            # if (state, action) not in self.C:
            #     self.C[state][action] = 0.0
            #print(action)
            if not isinstance(state, tuple):
                state = tuple(state.tolist())
            self.C[state] += W
            self.Q[state] = self.Q[state] + (W/(self.C[state]))*(G - self.Q[state])
            # self.greedy_policy[state][np.argmax(self.Q[(state, action)])] = 1
            if self.greedy_policy[state][action] == 0:
                continue
            W *= self.egreedy_policy[state]/self.greedy_policy[state]
            if W == 0:
                break
        self.create_target_policy()
        self.create_behavior_policy()


