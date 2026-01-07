import os
import numpy as np
import torch
import gym
import gym_tetris
import argparse
from nes_py.wrappers import JoypadSpace
from gym_tetris.actions import MOVEMENT
from gym.wrappers import StepAPICompatibility, RecordVideo
from pyvirtualdisplay import Display

# Import modules from our project
from model import CustomNN
from DQN import DQN
from wrapper import TetrisWrapper  # Ensure wrapper.py is in the same directory

# ========== Config ===========
# Modify this to the model weights path you want to test
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default=os.path.join("ckpt_test", "best.pth"),
                    help='Path to the trained model weights')
parser.add_argument('--episodes', type=int, default=5,
                    help='Number of episodes to run for evaluation')
parser.add_argument('--visualize', action='store_true', default=False,
                    help='Whether to visualize the gameplay')
args = parser.parse_args()

MODEL_PATH = os.path.join("ckpt_test", args.model)
VISUALIZE = args.visualize
TOTAL_EPISODES = args.episodes        # Number of episodes to play
FPS = 30                  # Limit display speed (otherwise it's too fast to see)

# Hardware setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OBS_SHAPE = (223,)
N_ACTIONS = len(MOVEMENT)

piece_dict = {
    0: "T",
    1: "J",
    2: "Z",
    3: "O",
    4: "S",
    5: "L",
    6: "I"
}

# =============================================================================
# 1. Environment Setup (Must be identical to training logic)
# =============================================================================
# Note: We only open 1 environment for evaluation, not AsyncVectorEnv
env = gym_tetris.make('TetrisA-v0')

# Fix API compatibility
if isinstance(env, gym.wrappers.TimeLimit):
    env = env.env
env = StepAPICompatibility(env, output_truncation_bool=False)
env = JoypadSpace(env, MOVEMENT)

env = TetrisWrapper(env, skip=4)

if not VISUALIZE:
    display = Display(visible=0, size=(1400, 900))
    display.start()
    video_folder = os.path.join("videos", "eval")
    print(f"Visualization disabled. Recording video to: {video_folder}")
    env = RecordVideo(env, video_folder, episode_trigger=lambda x: True, name_prefix="eval-run")

print(f"Evaluation Environment Initialized: {env}")

# =============================================================================
# 2. Model Loading
# =============================================================================
dqn = DQN(
    model=CustomNN,
    state_dim=OBS_SHAPE,
    action_dim=N_ACTIONS,
    learning_rate=0.0001,
    gamma=0.99,
    epsilon=0.001,       # Set very low (e.g. 0.001) to follow learned policy almost entirely, but keep a little randomness to avoid infinite loops
    target_update=1000,
    device=device
)

if os.path.exists(MODEL_PATH):
    try:
        print(f"Loading model from {MODEL_PATH}...")
        model_weights = torch.load(MODEL_PATH, map_location=device)
        dqn.q_net.load_state_dict(model_weights)
        dqn.q_net.eval() # Set to evaluation mode
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Failed to load model: {e}")
        exit()
else:
    print(f"Error: Model file not found at {MODEL_PATH}")
    # Not forcing exit here, if you just want to see random actions, comment out the line below
    exit()

# =============================================================================
# 3. Test Loop
# =============================================================================
for episode in range(1, TOTAL_EPISODES + 1):

    # Reset returns (state, info) -> we only need state
    state, _ = env.reset()

    # Dimension handling:
    state = np.expand_dims(state, axis=0)

    done = False
    total_reward = 0
    steps = 0

    while not done:
        if VISUALIZE:
            env.render()
            # Simple speed limit for human eye
            # import time
            # time.sleep(1/FPS)

        # Prepare input Tensor
        state_input = np.expand_dims(state, axis=0)
        state_tensor = torch.tensor(state_input, dtype=torch.float32, device=device)

        # Predict action
        with torch.no_grad():
            # Calculate Q values directly
            q_values = dqn.q_net(state_tensor)
            # Pick the largest (Argmax)
            action = torch.argmax(q_values, dim=1).item()

        # Execute action
        # Wrapper returns 5 values (state, reward, done, truncated, info)
        next_state, reward, terminated, truncated, info = env.step(action)

        """ if 'board' in info:
            # ANSI escape code to clear screen and move cursor to home for real-time effect
            print("\033[H\033[J", end="")
            print("--- Board State ---")
            for row in info['board']:
                print("".join(["[]" if x else " ." for x in row]))
            print("-------------------")
            print(f"Current Piece shape: {piece_dict.get(np.argmax(info.get('current_piece', [0]*7)), 'Unknown')}")
            print(f"Current Piece rotation: {np.argmax(info.get('current_rotation', [0]*4))}")
            print(f"Next Piece shape: {piece_dict.get(np.argmax(info.get('next_piece', [0]*7)), 'Unknown')}")
            print(f"Current X: {info.get('curr_x', 'N/A')}, Current Y: {info.get('curr_y', 'N/A')}") """

        done = terminated or truncated

        # Dimension handling Next State
        next_state = np.expand_dims(next_state, axis=0)

        state = next_state
        total_reward += reward
        steps += 1

        # Set a limit to avoid getting stuck in some cases
        # if steps > 5000:
        #     done = True

    print(f"Episode {episode}/{TOTAL_EPISODES} | Steps: {steps} | Total Reward: {total_reward:.2f} | Lines: {info.get('number_of_lines', 0)}")

env.close()