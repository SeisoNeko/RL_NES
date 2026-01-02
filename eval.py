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

# 引用我們專案中的模組
from model import CustomCNN
from DQN import DQN
from wrapper import TetrisWrapper  # 必須確保 wrapper.py 在同一目錄下

# ========== Config ===========
# 請修改這裡為你想要測試的模型權重路徑
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
TOTAL_EPISODES = args.episodes        # 測試玩幾場
FPS = 30                  # 限制顯示速度 (不然電腦太快會看不清楚)

# 硬體設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OBS_SHAPE = (219,)
N_ACTIONS = len(MOVEMENT)

# =============================================================================
# 1. 環境設置 (必須跟訓練時的邏輯一模一樣)
# =============================================================================
# 注意：評估時我們只開 "1個" 環境，不用 AsyncVectorEnv
env = gym_tetris.make('TetrisA-v0')

# 修正 API 兼容性
if isinstance(env, gym.wrappers.TimeLimit):
    env = env.env
env = StepAPICompatibility(env, output_truncation_bool=False)
env = JoypadSpace(env, MOVEMENT)

# 套用全能 Wrapper (負責切圖、灰階、SkipFrame)
# 這樣我們拿到的 state 就已經是 (84, 84) 的乾淨圖了
env = TetrisWrapper(env, skip=4)

if not VISUALIZE:
    display = Display(visible=0, size=(1400, 900))
    display.start()
    video_folder = os.path.join("videos", "eval")
    print(f"Visualization disabled. Recording video to: {video_folder}")
    env = RecordVideo(env, video_folder, episode_trigger=lambda x: True, name_prefix="eval-run")

print(f"Evaluation Environment Initialized: {env}")

# =============================================================================
# 2. 模型載入
# =============================================================================
dqn = DQN(
    model=CustomCNN,
    state_dim=OBS_SHAPE,
    action_dim=N_ACTIONS,
    learning_rate=0.0001,
    gamma=0.99,
    epsilon=0.001,       # 設為極低 (例如 0.001) 讓它幾乎完全依照學到的策略，但保留一點點隨機性避免死循環
    target_update=1000,
    device=device
)

if os.path.exists(MODEL_PATH):
    try:
        print(f"Loading model from {MODEL_PATH}...")
        model_weights = torch.load(MODEL_PATH, map_location=device)
        dqn.q_net.load_state_dict(model_weights)
        dqn.q_net.eval() # 設定為評估模式
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Failed to load model: {e}")
        exit()
else:
    print(f"Error: Model file not found at {MODEL_PATH}")
    # 這裡不強制退出，如果你只是想看隨機動作的效果，可以註解掉下面這行
    exit()

# =============================================================================
# 3. 測試迴圈
# =============================================================================
for episode in range(1, TOTAL_EPISODES + 1):

    # Reset 回傳 (state, info) -> 我們只需要 state
    state, _ = env.reset()

    # 維度處理:
    # Wrapper 回傳的 state 是 (84, 84)
    # 我們需要轉成 (1, 1, 84, 84) 給 CNN 吃
    # 步驟 1: 加 Channel -> (1, 84, 84)
    state = np.expand_dims(state, axis=0)
    # 步驟 2: 加 Batch (在此變數中暫存，等等轉 Tensor) -> (1, 1, 84, 84)

    done = False
    total_reward = 0
    steps = 0

    while not done:
        if VISUALIZE:
            env.render()
            # 簡單的限速機制，讓肉眼跟得上
            # import time
            # time.sleep(1/FPS)

        # 準備輸入 Tensor (1, 1, 84, 84)
        state_input = np.expand_dims(state, axis=0)
        state_tensor = torch.tensor(state_input, dtype=torch.float32, device=device)

        # 預測動作
        with torch.no_grad():
            # 直接算 Q 值
            q_values = dqn.q_net(state_tensor)
            # 選最大的 (Argmax)
            action = torch.argmax(q_values, dim=1).item()

        # 執行動作
        # Wrapper 回傳 5 個值 (state, reward, done, truncated, info)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # 維度處理 Next State
        # next_state 從 (84, 84) -> (1, 84, 84)
        next_state = np.expand_dims(next_state, axis=0)

        state = next_state
        total_reward += reward
        steps += 1

        # 為了避免某些情況卡死，可以設個上限
        if steps > 5000:
            done = True

    print(f"Episode {episode}/{TOTAL_EPISODES} | Steps: {steps} | Total Reward: {total_reward:.2f} | Lines: {info.get('number_of_lines', 0)}")

env.close()