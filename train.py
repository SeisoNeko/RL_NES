import os
import argparse
import numpy as np
import random
import torch
import torch.nn as nn
import cv2
from tqdm import tqdm

import gym_tetris
from gym.wrappers import RecordVideo
from nes_py.wrappers import JoypadSpace
from gym_tetris.actions import MOVEMENT

from utils import preprocess_frame
from reward import *
from wrapper import TetrisWrapper
from model import CustomCNN
from DQN import DQN, ReplayMemory

import gym
from gym.vector import AsyncVectorEnv

# ========== config ===========
import gym
from gym.wrappers import StepAPICompatibility

# ================ Support function ==================
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

class RenderWrapper(gym.Wrapper):
    def step(self, action):
        self.env.render()
        return self.env.step(action)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


def make_env(rank, args):
    def _thunk():
        env = gym_tetris.make('TetrisA-v0')
        # 修正 API
        if isinstance(env, gym.wrappers.TimeLimit):
            env = env.env
        env = StepAPICompatibility(env, output_truncation_bool=False)
        env = JoypadSpace(env, MOVEMENT)

        env = TetrisWrapper(env, skip=4)

        if rank == 0 and args.visualize:
            env = RenderWrapper(env)

        return env
    return _thunk

def main():
    #========= basic train config==============================================
    parser = argparse.ArgumentParser(description="Train DQN on Tetris")
    parser.add_argument("--lr", type=float, default=0.00001, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--memory_size", type=int, default=10000, help="Replay memory size")
    parser.add_argument("--epsilon_end", type=float, default=0.001, help="Final epsilon value")
    parser.add_argument("--target_update", type=int, default=1000, help="Target network update frequency")
    parser.add_argument("--total_timesteps", type=int, default=200000000, help="Total training timesteps")
    parser.add_argument("--visualize", action='store_true', help="Render the environment")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of parallel environments")
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
    NUM_ENVS = args.num_workers

    EPSILON_START = 1.0           # 一開始 100% 隨機
    EPSILON_DECAY = 0.00001      # 每次訓練減少多少

    envs = AsyncVectorEnv([make_env(i, args) for i in range(NUM_ENVS)])

    print(f"Vectorized Envs Created: {NUM_ENVS} parallel games.")

    device = torch.device("cuda")


    # ========================DQN Initialization==========================================
    obs_shape = (223, )
    n_actions = len(MOVEMENT)                #定義動作空間大小，使用SIMPLE_MOVEMENT中的動作數量

    model = CustomCNN                               #指定模型架構為CustomCNN用於處理圖像並預測各動作的 Q 值
    dqn = DQN(                                      #初始化 DQN agent
        model=model,
        state_dim=obs_shape,                        #狀態空間大小
        action_dim=n_actions,                       #動作空間大小
        learning_rate=LR,                           #學習率
        gamma=GAMMA,                                #折扣因子，用於計算未來獎勵
        epsilon=EPSILON_START,                        #初始探索率
        target_update=TARGET_UPDATE,                #目標網路更新頻率
        device=device
    )

    if args.load_checkpoint:
        if os.path.exists(args.load_checkpoint):
            print(f"Loading checkpoint from {args.load_checkpoint}")
            dqn.q_net.load_state_dict(torch.load(args.load_checkpoint))
            # Sync target network
            if hasattr(dqn, 'target_q_net'):
                dqn.target_q_net.load_state_dict(dqn.q_net.state_dict())
        else:
            print(f"Checkpoint not found: {args.load_checkpoint}")

    memory = ReplayMemory(MEMORY_SIZE)              #創建經驗回放記憶體，用於存儲狀態轉移
    step = 0                                        #記錄總步數
    best_reward = -float('inf')                     # 儲存最佳累積獎勵Track the best reward in each SAVE_INTERVAL
    cumulative_reward = 0                           # 當前時間步的總累積獎勵Track cumulative reward for the current timestep



    #=======================訓練開始============================
    state, _ = envs.reset()
    state_input = np.expand_dims(state, axis=1)  # (NUM_ENVS, 1, 203)
    cumulative_reward = np.zeros(NUM_ENVS)

    for timestep in tqdm(range(1, TOTAL_TIMESTEPS + 1, NUM_ENVS), desc="Training"):

        actions = []
        for i in range(NUM_ENVS):
            single_state = state_input[i]
            action = dqn.take_action(single_state)  # Get action for each env
            actions.append(action)

        next_states, rewards, terminateds, truncateds, infos = envs.step(actions)  # Step all envs
        dones = [t or tr for t, tr in zip(terminateds, truncateds)]

        next_states_input = np.expand_dims(next_states, axis=1)

        for i in range(NUM_ENVS):
            current_reward = rewards[i]
            cumulative_reward[i] += current_reward

            memory.push(
                state_input[i],
                actions[i],
                current_reward,
                next_states_input[i],
                dones[i]
            )

            if dones[i]:
                if i==0:
                    print(f"Env 0 Reward: {cumulative_reward[i]}")
                if cumulative_reward[i] > best_reward:
                    best_reward = cumulative_reward[i]
                    os.makedirs("ckpt_test", exist_ok=True)
                    model_path = os.path.join("ckpt_test",f"step_{timestep}_reward_{int(best_reward)}.pth")
                    torch.save(dqn.q_net.state_dict(), model_path)
                    print(f"Model saved: {model_path}")
                cumulative_reward[i] = 0  # Reset for next episode

        states_input = next_states_input

        if len(memory) >= BATCH_SIZE:
            for _ in range(4):  # Train 4 times per timestep
                batch = memory.sample(BATCH_SIZE)
                state_dict = {
                    'states': batch[0],
                    'actions': batch[1],
                    'rewards': batch[2],
                    'next_states': batch[3],
                    'dones': batch[4],
                }
                dqn.train_per_step(state_dict)


        # Update epsilon
        dqn.epsilon = max(EPSILON_END, dqn.epsilon - EPSILON_DECAY)  # Gradually decrease epsilon
        if step % 10000 == 0:
            print(f"Step: {step}, Epsilon: {dqn.epsilon:.4f}")
        step += 1

        # Print cumulative reward for the current timestep
        if timestep % 1000 == 0:
            print(f"Timestep {timestep} - Total Reward: {cumulative_reward}")

        if timestep % 10000 == 0:
            # Save model checkpoint
            os.makedirs("ckpt_test", exist_ok=True)
            model_path = os.path.join("ckpt_test",f"step_{timestep}.pth")
            torch.save(dqn.q_net.state_dict(), model_path)
            print(f"Model checkpoint saved at timestep {timestep}: {model_path}")


    envs.close()

if __name__ == "__main__":
    main()
