import numpy as np
import random
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
print(sys.path)

import gymnasium as gym
from collections import defaultdict
from webots_env import WebotsCarEnv

class MonteCarloAgent:
    def __init__(self, action_space, gamma=0.99, epsilon=0.1): # tuning parameters: gamma and epsilon
        self.gamma = gamma  # discount factor
        self.epsilon = epsilon  # exploration rate
        self.action_space = action_space # stores avaible actions
        self.returns = defaultdict(list)  # stores rewards for each pair of state and action
        self.q_table = defaultdict(lambda: np.zeros(len(action_space)))  # initialize Q values

    def select_action(self, state): # epsilon-greedy policy
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(range(len(self.action_space))) # random exploration
        else:
            return np.argmax(self.q_table[state]) # greedy exploitation

    def update_policy(self, episode): # update the Q-table
        states, actions, rewards = zip(*episode) # get episod data
        g = 0  # initialize return
        visited_states = set() # keep track of visited states

        for t in reversed(range(len(states))): # iterate backwards over the episode
            g = self.gamma * g + rewards[t]  # calculate return here
            state_action = (states[t], actions[t])

            if state_action not in visited_states:  # first-visit Monte Carlo update
                self.returns[state_action].append(g) # store returns
                self.q_table[states[t]][actions[t]] = np.mean(self.returns[state_action]) # get the average return then update Q-value
                visited_states.add(state_action) # mark visited states

# train the MC agent
def train_monte_carlo(env, agent, episodes=2000): # tuning parameter: episodes number
    for episode in range(episodes): 
        state, _ = env.reset() # reset the environment
        episode_data = [] # store episode data
        done = False

        while not done:
            action = agent.select_action(str(state)) # select action
            next_state, reward, done, _, _ = env.step(action) # take action in the environment
            episode_data.append((str(state), action, reward)) # store episode data
            state = next_state # update state

        agent.update_policy(episode_data)  # Update policy from the episode
        print(f"Episode {episode} completed.")
    
    return agent

if __name__ == "__main__":
    env = WebotsCarEnv() # initialize the environment
    action_space = [-1.0, -0.5, 0.0, 0.5, 1.0]  # discretize continuous actions. tuning parameter: action space
    mc_agent = MonteCarloAgent(action_space=action_space) # initialize the agent
    mc_agent = train_monte_carlo(env, mc_agent, episodes=5000)
    
