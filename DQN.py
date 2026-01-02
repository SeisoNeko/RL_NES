import torch as T
import torch.nn.functional as F
import numpy as np
import random
from collections import deque

class ReplayMemory:                                                              # Store and sample training data
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)                                     # Use deque to store data, set maxlen to ensure oldest experiences are removed when capacity is reached
                                                                                 # self.memory is a deque storing tuples, each representing an experience in format (s,a,r,n_s,d)
    def push(self, state, action, reward, next_state, done):                     # Add experience (state, action, reward, next_state, done) to memory
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)                           # batch = Randomly sample batch_size items from memory
        states, actions, rewards, next_states, dones = zip(*batch)               # zip(*batch) groups data of the same type (e.g., states, actions) from multiple experiences
                                                                                 # e.g. if batch samples 32 items (s,a,r,n_s,d), then zip(*batch) results in states=[s1...s32], actions=[a1...a32]
        return np.stack(states), actions, rewards, np.stack(next_states), dones  # np.stack converts states and next_states to NumPy arrays for easier computation

    def __len__(self):                                                           # Number of experiences currently stored in memory
        return len(self.memory)

class DQN:
    def __init__(self,
                 model,
                 state_dim, action_dim,
                 learning_rate, gamma,
                 epsilon, target_update, device):
        self.device = device
        self.action_dim = action_dim

        self.gamma = gamma
        self.epsilon = epsilon
        self.target_update = target_update
        self.update_count = 0

        # Initialize [Q-net] and target [Q-net]
        self.model = model
        self.q_net = self._build_net(state_dim, action_dim)                      # [Q-net], the actual training network (updated in real-time)
        self.tgt_q_net = self._build_net(state_dim, action_dim)                  # target [Q-net], used for stable training (delayed update)
        self.tgt_q_net.load_state_dict(self.q_net.state_dict())                  # Copy weights from [Q-net] to target [Q-net]

        # Optimizer
        self.optimizer = T.optim.Adam(self.q_net.parameters(), lr=learning_rate)

    # Define neural network constructor
    def _build_net(self, state_dim, action_dim):                                 # Initialize neural network based on passed model architecture (self.model) and move to device
        return self.model(state_dim,action_dim).to(self.device)

    # Action selection function
    def take_action(self, state):
        # Exploration Unknown Policy (Explore)
        if np.random.rand() < self.epsilon:                                      # Generate random float (0-1), if less than epsilon (exploration probability), execute random action
            return np.random.randint(self.action_dim)                            # Generate a random integer in range [0, self.action_dim)

        # Exploitation Known Policy (Exploit)                                    # If random float > epsilon, execute action based on inference (Exploit)
        state_x = T.tensor([state], dtype=T.float32, device=self.device)         # Convert single state to PyTorch tensor
        with T.no_grad():
            q_values = self.q_net(state_x)

            # CRITICAL FIX FOR TETRIS:
            # Use argmax to pick the action with the highest Q-value deterministically.
            # Do not use softmax sampling here.
            action = T.argmax(q_values, dim=1).item()

            return action

    # Loss function calculation
    def get_loss(self, states, actions, rewards, next_states, dones):
        # Get current Q-values
        actions = actions.unsqueeze(1)
        q_val = self.q_net(states).gather(1, actions).squeeze(1)                 # Calculate current Q-value

        # Get maximum expected Q-values
        next_q_val = self.tgt_q_net(next_states).max(dim=1)[0]                   # Calculate maximum target Q-value

        # Compute target Q-values [custom-reward]
        q_target = rewards + self.gamma * next_q_val * (1 - dones.float())       # Calculate target Q-value

        return T.nn.functional.mse_loss(q_val, q_target.detach())                # Calculate loss using Mean Squared Error (MSE)

    def train_per_step(self, state_dict):
        # Convert one trajectory(s,a,r,n_s) to tensor
        states,actions,rewards,next_states,dones = self._state_2_tensor(state_dict)  # Convert data stored in Python structures to PyTorch tensors

        # Compute loss
        loss = self.get_loss(states, actions, rewards, next_states, dones)
        self.optimizer.zero_grad()                                               # Clear accumulated gradients before each update
        loss.backward()                                                          # Backpropagate using calculated loss to compute gradients for each parameter
        self.optimizer.step()                                                    # Update [Q-net] parameters using calculated gradients

        if self.update_count % self.target_update == 0:                          # target_update defined in runs.py (e.g., 50) (update frequency)
            self.tgt_q_net.load_state_dict(self.q_net.state_dict())              # Periodically copy [Q-net] parameters to target [Q-net]

        self.update_count += 1

    def _state_2_tensor(self,state_dict):                                        # Convert data in an experience trajectory (s,a,r,n_s,d) to PyTorch tensors
        states      = T.tensor(state_dict['states'], dtype=T.float32, device=self.device)
        actions     = T.tensor(state_dict['actions'], dtype=T.long, device=self.device)
        rewards     = T.tensor(state_dict['rewards'], dtype=T.float32, device=self.device)
        next_states = T.tensor(state_dict['next_states'], dtype=T.float32, device=self.device)
        dones       = T.tensor(state_dict['dones'], dtype=T.float32, device=self.device)

        return states,actions,rewards,next_states,dones