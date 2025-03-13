import numpy as np
import random
from collections import defaultdict
import mahjong_environment

def __init__(self, env: mahjong_environment, gamma: float = 1.0, epsilon: float = 0.1, Q0: float = 0.0, max_episode_size : int = 1000):
    self.env = env
    self.gamma = gamma
    self.epsilon = epsilon
    self.Q0 = Q0
    self.max_episode_size = max_episode_size
    self.Q = defaultdict(lambda: np.full(env.n_actions, 0))
    self.C = defaultdict(lambda: np.zeros(env.n_actions))
    self.greedy_policy = defaultdict(lambda: np.full(env.n_actions, 1 / env.n_actions))
    self.egreedy_policy = defaultdict(lambda: np.full(env.n_actions, 1 / env.n_actions))

