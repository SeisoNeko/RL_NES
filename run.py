import os
import argparse
import numpy as np
import random
import torch
import torch.nn as nn
import cv2
from tqdm import tqdm

import gym_tetris
from nes_py.wrappers import JoypadSpace
from gym_tetris.actions import MOVEMENT

from utils import preprocess_frame
from reward import *
from model import CustomCNN
from DQN import DQN, ReplayMemory



# ========== config ===========
#env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')   #
#env = JoypadSpace(env, SIMPLE_MOVEMENT)
import gym
from gym.wrappers import StepAPICompatibility

# skip frame speed up
class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip=4):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = False
        # Repeat the action for 'skip' frames
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info

# 1) make（這裡可能會自動包 TimeLimit）
env = gym_tetris.make('TetrisA-v0')

# 2) 🔑 拆掉 TimeLimit（不拆一定炸 expected 5 got 4）
if isinstance(env, gym.wrappers.TimeLimit):
    env = env.env

# 3) 固定成舊 step API（回 4-tuple）
env = StepAPICompatibility(env, output_truncation_bool=False)

# 4) 再包 JoypadSpace
env = JoypadSpace(env, MOVEMENT)

env = SkipFrame(env, skip=4)

print("Final env:", env)

#========= basic train config==============================================
parser = argparse.ArgumentParser(description="Train DQN on Tetris")
parser.add_argument("--lr", type=float, default=0.00001, help="Learning rate")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
parser.add_argument("--memory_size", type=int, default=10000, help="Replay memory size")
parser.add_argument("--epsilon_end", type=float, default=0.3, help="Final epsilon value")
parser.add_argument("--target_update", type=int, default=50, help="Target network update frequency")
parser.add_argument("--total_timesteps", type=int, default=2000000, help="Total training timesteps")
parser.add_argument("--visualize", type=bool, default=False, help="Render the environment")
parser.add_argument("--load_checkpoint", type=str, default=None, help="Path to checkpoint to load")

args = parser.parse_args()

LR = args.lr
BATCH_SIZE = args.batch_size
GAMMA = args.gamma
MEMORY_SIZE = args.memory_size
EPSILON_END = args.epsilon_end
TARGET_UPDATE = args.target_update
TOTAL_TIMESTEPS = args.total_timesteps
VISUALIZE = args.visualize

prev_info = {
    "score": 0,
    "number_of_lines": 0,
    # "holes", "bumps" 會在第一次計算 reward 時自動加入
}
device = torch.device("cuda")


# ========================DQN Initialization==========================================
obs_shape = (1, 84, 84)                         #obs_shape = (1, 84, 84)
n_actions = len(MOVEMENT)                #定義動作空間大小，使用SIMPLE_MOVEMENT中的動作數量（例如向右移動、跳躍等）
if args.load_checkpoint:
    if os.path.exists(args.load_checkpoint):
        print(f"Loading checkpoint from {args.load_checkpoint}")
        dqn.q_net.load_state_dict(torch.load(args.load_checkpoint))
        # Sync target network
        if hasattr(dqn, 'target_q_net'):
            dqn.target_q_net.load_state_dict(dqn.q_net.state_dict())
    else:
        print(f"Checkpoint not found: {args.load_checkpoint}")

model = CustomCNN                               #指定模型架構為CustomCNN用於處理圖像並預測各動作的 Q 值
dqn = DQN(                                      #初始化 DQN agent
    model=model,
    state_dim=obs_shape,                        #狀態空間大小
    action_dim=n_actions,                       #動作空間大小
    learning_rate=LR,                           #學習率
    gamma=GAMMA,                                #折扣因子，用於計算未來獎勵
    epsilon=EPSILON_END,                        #初始探索率
    target_update=TARGET_UPDATE,                #目標網路更新頻率
    device=device
)

memory = ReplayMemory(MEMORY_SIZE)              #創建經驗回放記憶體，用於存儲狀態轉移
step = 0                                        #記錄總步數
best_reward = -float('inf')                     # 儲存最佳累積獎勵Track the best reward in each SAVE_INTERVAL
cumulative_reward = 0                           # 當前時間步的總累積獎勵Track cumulative reward for the current timestep



#=======================訓練開始============================
for timestep in tqdm(range(1, TOTAL_TIMESTEPS + 1), desc="Training"):
    state = env.reset()
    state = preprocess_frame(state)  # 這裡會變成 Crop 後的 84x84
    if timestep == 1:
        cv2.imwrite("debug_state.png", state * 255)

    # 轉成 tensor 格式
    state_input = np.expand_dims(state, axis=0) # (1, 84, 84)
    # state_input = np.expand_dims(state_input, axis=0) # (1, 1, 84, 84)

    done = False
    cumulative_reward = 0

    # 重置 prev_info 的統計數據
    prev_info["holes"] = 0
    prev_info["bumps"] = 0
    prev_info["height"] = 0
    prev_info["score"] = 0
    prev_info["number_of_lines"] = 0

    while not done:
        # 1. AI 決定動作
        action = dqn.take_action(state_input)

        # 2. 執行動作 (Step 1)
        next_raw_state, reward, done, info = env.step(action)

        # === 關鍵修改：強制下落 (Bonus Step) ===
        # 參考 Repo：讓 AI 做完動作後，強制讓方塊掉一格。
        # 這能加快遊戲節奏，避免 AI 在空中無限旋轉。
        # 動作 5 對應 'down' (在 SIMPLE_MOVEMENT 中可能不同，請確認 mapping)
        # 假設 SIMPLE_MOVEMENT 裡 'down' 是索引 3 或其他，這裡用 Tetris 預設邏輯
        # 如果不知道 mapping，可以暫時先不加這段，或者試試 env.step(env.action_space.sample()) 的 'down'
        if not done:
             # 這裡假設 action 0-11 中有一個是純 down。如果不想太複雜，可以先略過這步。
             # 但如果加上去，訓練會快很多。
             pass

        # 3. 預處理畫面 (用於 CNN 和 計算 Reward)
        next_state = preprocess_frame(next_raw_state)
        next_state_input = np.expand_dims(next_state, axis=0) # (1, 84, 84)[C,H,W]

        # 4. 計算高級獎勵 (Holes & Bumps)
        # 注意：我們需要把 next_state 傳進去算洞
        custom_reward, new_stats = calculate_custom_reward(info, reward, prev_info, next_state, env)
        prev_info.update(new_stats)
        prev_info["score"] = info["score"]
        prev_info["number_of_lines"] = info["number_of_lines"]

        # 5. 死亡懲罰
        if done:
            custom_reward -= 10

        # 累積獎勵
        cumulative_reward += custom_reward

        # 6. 存入 Memory
        memory.push(state_input, action, custom_reward, next_state_input, done)

        #==============================Train DQN 當記憶體中樣本數量達到批次大小時，從記憶體中隨機抽取一批樣本進行網路更新
        if len(memory) >= BATCH_SIZE:
            batch = memory.sample(BATCH_SIZE)

            state_dict = {                                       #將這些數據打包為字典格式，方便傳遞給模型進行訓練
                'states': batch[0],
                'actions': batch[1],
                'rewards': batch[2],
                'next_states': batch[3],
                'dones': batch[4],
            }
            dqn.train_per_step(state_dict)                       #train_per_step是DQN中的方法，用於計算損失並更新神經網路的權重

        # Update epsilon
        dqn.epsilon = EPSILON_END               #訓練前就設定:代理的探索能力會立即降低，可能在策略還不完善時過早專注於利用，會影響最終的學習效果

        #================================更新狀態訊息
        prev_info = info
        step += 1

        if VISUALIZE:                                   #渲染當前遊戲畫面
            env.render()

    # Print cumulative reward for the current timestep
    if timestep % 1000 == 0:
        print(f"Timestep {timestep} - Total Reward: {cumulative_reward}")

    #如果當前累積獎勵超過歷史最佳值，保存模型的權重 每次超過最佳值就會保留一次
    #要改成自定義獎勵
    if cumulative_reward > best_reward:
        best_reward = cumulative_reward
        os.makedirs("ckpt_test", exist_ok=True)
        #命名邏輯是採第幾步+最佳獎勵+自訂義獎勵的累積總合
        model_path = os.path.join("ckpt_test",f"step_{timestep}_reward_{int(best_reward)}.pth")
        torch.save(dqn.q_net.state_dict(), model_path)
        print(f"Model saved: {model_path}")

env.close()
