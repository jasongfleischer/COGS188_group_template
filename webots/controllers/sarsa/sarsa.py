import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from webots_env import WebotsCarEnv


NUM_BINS = [10, 10, 10, 10, 10, 10]  # 6 discrete dimensions
STATE_LIMITS = [
    (0, 250),     # speed
    (-100, 100),  # gps_x
    (-100, 100),  # gps_y
    (0, 100),     # lidar_dist
    (-180, 180),  # lidar_angle
    (0, 50)       # lane_deviation (example limit)
]


class SARSA:
    def __init__(self, env, alpha=0.1, gamma=0.99, epsilon=0.1, bins=10):
        self.env = env
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount factor
        self.epsilon = epsilon  # exploration rate
        
        # discretize state space
        self.state_bins = {key: bins for key in self.env.state_space.spaces.keys()}
        self.q_table = np.zeros(tuple(self.state_bins.values()) + (env.action_space.shape[0],))
        
    # TODO
    def discretize_state(self, state):
        pass
    
    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()
        else:
            discrete_state = self.discretize_state(state)
            print(f"Q-table shape: {self.q_table.shape}, Expected indices: {discrete_state}")

            return self.q_table[discrete_state].argmax()
        
    def train(self, episodes=500):
        for episode in range(episodes):
            state = self.env.reset()
            action = self.choose_action(state)
            
            done = False
            total_reward = 0
            
            while not done:
                next_state, reward, done = self.env.step(action)
                next_action = self.choose_action(next_state)
                
                self.q_table[state][action] += self.alpha * (
                    reward + self.gamma * self.q_table[next_state][next_action] - self.q_table[state][action]
                )
                
                state, action = next_state, next_action
                total_reward += reward
            
            print(f"Episode {episode+1}, Total Reward: {total_reward}")

env = WebotsCarEnv()
sarsa_agent = SARSA(env)
sarsa_agent.train()

