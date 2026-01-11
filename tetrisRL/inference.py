import time
import curses
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from engine import TetrisEngine

# --- 1. Re-define the Model (Must match fast_train.py exactly) ---
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, output_size)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

def main(stdscr):
    # Setup Curses
    curses.curs_set(0) # Hide cursor
    stdscr.nodelay(True) # Non-blocking input

    # Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize Engine
    width, height = 10, 20
    env = TetrisEngine(width, height)
    
    # Initialize Model
    input_size = width * height
    n_actions = len(env.value_action_map)
    model = DQN(input_size, n_actions).to(device)
    
    # Load Weights
    try:
        model.load_state_dict(torch.load("tetris_cnn.pt", map_location=device))
        model.eval() # Set to evaluation mode (no dropout, etc)
    except FileNotFoundError:
        print("Error: tetris_cnn.pt not found. Run fast_train.py first!")
        return

    # Play Loop
    state_board = env.clear()
    total_score = 0
    
    while True:
        # Prepare Input
        # Convert board to tensor (1, 1, 10, 20)
        state_tensor = torch.tensor(state_board, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
        
        # Predict Action
        with torch.no_grad():
            action = model(state_tensor).max(1)[1].item()
        
        # Step Environment
        state_board, reward, done = env.step(action)
        total_score += reward
        
        # Render
        stdscr.clear()
        stdscr.addstr(0, 0, f"Score: {total_score}")
        stdscr.addstr(2, 0, str(env))
        stdscr.refresh()
        
        # Control Speed
        time.sleep(0.05) # Adjust this to make it faster/slower
        
        if done:
            stdscr.addstr(height + 4, 0, "GAME OVER! Restarting...")
            stdscr.refresh()
            time.sleep(2)
            state_board = env.clear()
            total_score = 0
            
        # Exit condition (Press 'q')
        key = stdscr.getch()
        if key == ord('q'):
            break

if __name__ == "__main__":
    curses.wrapper(main)