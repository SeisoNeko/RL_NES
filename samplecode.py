import pdb
from nes_py.wrappers import JoypadSpace
from gym_tetris.actions import MOVEMENT
import gym_tetris
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )

    def forward(self, x):
        return self.net(x)
    
def preprocess_state(state):
    return np.array(state).flatten()

def select_action(state, policy_net, epsilon, action_dim, ):
    if random.random() < epsilon:
        return random.randrange(action_dim)

    with torch.no_grad():
        state = torch.FloatTensor(state)
        q_values = policy_net(state)
    return q_values.argmax().item()

def train():
    env = gym_tetris.make('TetrisA-v3')
    env = JoypadSpace(env, MOVEMENT)
    
    state = preprocess_state(env.reset())
    state_dim = state.shape[0]
    action_dim = env.action_space.n
    
    policy_net = DQN(state_dim, action_dim).to(device)
    target_net = DQN(state_dim, action_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)
    memory = deque(maxlen=10000)
    batch_size = 64
    gamma = 0.99
    epsilon = 1.0
    epsilon_min = 0.05
    epsilon_decay = 0.995
    update_target = 100
    
    for episode in range(300):
        state = preprocess_state(env.reset())
        total_reward = 0
        done = False
        
        while not done:
            action = select_action(state, policy_net, epsilon, action_dim)
            next_state, reward, done, _ = env.step(action)
            next_state = preprocess_state(next_state)
            memory.append((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward
            
            if len(memory) >= batch_size:
                batch = random.sample(memory, batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)
                
                states = torch.from_numpy(np.array(states)).float().to(device)
                actions = torch.from_numpy(np.array(actions)).long().unsqueeze(1).to(device)
                rewards = torch.from_numpy(np.array(rewards)).float().unsqueeze(1).to(device)
                next_states = torch.from_numpy(np.array(next_states)).float().to(device)
                dones = torch.from_numpy(np.array(dones)).float().unsqueeze(1).to(device)
                
                q_values = policy_net(states).gather(1, actions)
                next_q_values = target_net(next_states).max(1)[0].unsqueeze(1)
                expected_q_values = rewards + (gamma * next_q_values * (1 - dones))
                
                loss = nn.MSELoss()(q_values, expected_q_values)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            # env.render()
        
        
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        if episode % update_target == 0:
            target_net.load_state_dict(policy_net.state_dict())
        print(f"Episode {episode}, Total Reward: {total_reward}, Epsilon: {epsilon}")
    env.close()
    
if __name__ == "__main__":
    train()