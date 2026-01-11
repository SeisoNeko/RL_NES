import math
import random
import numpy as np
from itertools import count
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from engine import TetrisEngine

# --- Configuration for 18-Hour Deadline ---
BATCH_SIZE = 512           # Larger batch for stable gradients
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 100000         # Explore for about 2-3 hours, then exploit
TARGET_UPDATE = 1000       # Sync target network every 1000 steps
MEMORY_SIZE = 50000        # Fits in RAM, enough history
LR = 1e-3                  # Standard Adam learning rate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {device}")

# --- 1. The Model (Same BasicFF but cleaned up) ---
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, output_size)

    def forward(self, x):
        # Flatten input: (Batch, 1, H, W) -> (Batch, Input_Size)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# --- 2. Fast Replay Memory ---
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# --- 3. Setup ---
width, height = 10, 20
env = TetrisEngine(width, height)
n_actions = len(env.value_action_map)
input_size = width * height # 200

policy_net = DQN(input_size, n_actions).to(device)
target_net = DQN(input_size, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=LR)
memory = ReplayMemory(MEMORY_SIZE)

steps_done = 0

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return

    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch
    batch = list(zip(*transitions))

    state_batch = torch.cat(batch[0])
    action_batch = torch.cat(batch[1])
    reward_batch = torch.cat(batch[2])
    next_state_batch = torch.cat(batch[3])
    done_batch = torch.cat(batch[4])

    # Compute Q(s_t, a)
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected Q values are 0 for terminal states (done = 1)
    next_state_values = target_net(next_state_batch).max(1)[0].detach()
    expected_state_action_values = (next_state_values * (1 - done_batch)) * GAMMA + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # Gradient clipping for stability
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

# --- 4. Main Training Loop ---
print("Starting Fast Training...")
best_score = 0

for i_episode in range(100000): # Infinite loop basically
    # Initialize the environment
    board = env.clear()
    # Convert board (10x20 boolean) to float tensor (1, 1, 10, 20)
    state = torch.tensor(board, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    
    total_reward = 0
    
    for t in count():
        action = select_action(state)
        
        # Step env
        next_board, reward, done = env.step(action.item())
        total_reward += reward

        # Process next state
        next_state = torch.tensor(next_board, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
        reward_tensor = torch.tensor([reward], device=device, dtype=torch.float32)
        done_tensor = torch.tensor([float(done)], device=device, dtype=torch.float32)

        # Store in memory
        memory.push(state, action, reward_tensor, next_state, done_tensor)

        # Move to next state
        state = next_state

        # --- CRITICAL FIX: Train EVERY STEP ---
        optimize_model()
        
        # Update Target Network
        if steps_done % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if done:
            break

    # Logging
    if i_episode % 10 == 0:
        print(f"Episode {i_episode} | Score: {total_reward:.1f} | Epsilon: {EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY):.4f}")
    
    # Save Model
    if total_reward > best_score:
        best_score = total_reward
        torch.save(policy_net.state_dict(), "tetris_cnn.pt")
        print(f"--> New Best Model Saved! Score: {best_score}")