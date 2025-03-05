# The base code was constructed with the help of DeepSeek AI (OracleAgent class, NormalAgent class, and most of the methods as a
# starter code)

#have OracleAgent inherit #NormalAgent

import torch
import torch.nn as nn
import torch.optim as optim

class OracleAgent:
    def __init__(self, state_size, action_size, hidden_size=128):
        self.state_size = state_size  # Size of the full game state
        self.action_size = action_size  # Number of possible actions
        self.model = self.build_model(hidden_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def build_model(self, hidden_size):
        # Define a neural network for the oracle agent
        model = nn.Sequential(
            nn.Linear(self.state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.action_size),
            nn.Softmax(dim=-1)  # Output a probability distribution over actions
        )
        return model

    def act(self, state):
        # Convert state to a tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        # Get action probabilities from the model
        action_probs = self.model(state_tensor)
        # Sample an action from the probability distribution
        action = torch.multinomial(action_probs, num_samples=1).item()
        return action

    def train(self, states, actions, rewards):
        # Convert inputs to tensors
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)

        # Forward pass
        action_probs = self.model(states)
        # Compute the loss (negative log likelihood)
        loss = -torch.log(action_probs.gather(1, actions.unsqueeze(1))) * rewards
        loss = loss.mean()

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        #make sure to use the loss value for training; either pass it into the optimizer or make it aware of loss. Have
        #agent inherit from a Torch class nn.Module

class NormalAgent:
    def __init__(self, state_size, action_size, hidden_size=128):
        self.state_size = state_size  # Size of the observable game state
        self.action_size = action_size  # Number of possible actions
        self.model = self.build_model(hidden_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def build_model(self, hidden_size):
        # Define a neural network for the normal agent
        model = nn.Sequential(
            nn.Linear(self.state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.action_size),
            nn.Softmax(dim=-1)  # Output a probability distribution over actions
        )
        return model

    def act(self, state):
        # Convert state to a tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        # Get action probabilities from the model
        action_probs = self.model(state_tensor)
        # Sample an action from the probability distribution
        action = torch.multinomial(action_probs, num_samples=1).item()
        return action

    def train(self, states, actions, rewards):
        # Convert inputs to tensors
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards) #should be taking from the global reward prediction from ryota's code

        # Forward pass
        action_probs = self.model(states)
        # Compute the loss (negative log likelihood)
        loss = -torch.log(action_probs.gather(1, actions.unsqueeze(1))) * rewards
        loss = loss.mean()

        # Backward pass
        self.optimizer.zero_g

    def tile_to_index(tile):
        """
        Map a tile to a unique index (0-33).
        tile: A tuple of (suit, value), e.g., ("manzu", 1) or ("honors", "east").
        """
        suit, value = tile

        if suit == "mans":
            return value - 1  # 0-8
        elif suit == "pins":
            return 9 + (value - 1)  # 9-17
        elif suit == "sticks":
            return 18 + (value - 1)  # 18-26
        elif suit == "honors":
            if value == "east":
                return 27
            elif value == "south":
                return 28
            elif value == "west":
                return 29
            elif value == "north":
                return 30
            elif value == "white":
                return 31
            elif value == "green":
                return 32
            elif value == "red":
                return 33
        raise ValueError(f"Invalid tile: {tile}")

    def encode_state(hand, discards, wall, other_hands):
        # One-hot encode the hand
        hand_encoded = [0] * 34  # 34 unique tile types
        for tile in hand:
            hand_encoded[tile_to_index(tile)] += 1

        # One-hot encode the discards
        discards_encoded = [0] * 34
        for tile in discards:
            discards_encoded[tile_to_index(tile)] += 1

        # Combine into a single state vector
        state = hand_encoded + discards_encoded
        return state
    
    def train_normal_agent(env, oracle_agent, normal_agent, episodes):
        for episode in range(episodes):
            state = env.reset()
            done = False
            while not done:
                # Oracle agent provides guidance
                oracle_action = oracle_agent.act(state)

                # Normal agent learns from oracle
                normal_action = normal_agent.act(state)
                reward = env.step(normal_action)

                # Train the normal agent using oracle guidance
                normal_agent.train((state, oracle_action, reward))

                # Progressively reduce oracle's information
                if episode > episodes // 2:
                    state = reduce_oracle_information(state, episode, episodes)

                state, done = env.get_next_state()

    def reduce_oracle_information(state, episode, episodes):
    # Gradually remove information from the oracle's state
        if episode > episodes // 2:
            state.pop('wall', None)  # Remove knowledge of the wall
        if episode > 3 * episodes // 4:
            state.pop('other_hands', None)  # Remove knowledge of other players' hands
        return state
        