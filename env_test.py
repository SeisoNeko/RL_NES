from nes_py.wrappers import JoypadSpace
from gym.wrappers import StepAPICompatibility
import gym_tetris
from gym_tetris.actions import MOVEMENT

env = gym_tetris.make('TetrisA-v3')
env = JoypadSpace(env, MOVEMENT)

env = StepAPICompatibility(env, output_truncation_bool=False)

done = True
for step in range(5000):
    if done:
        state = env.reset()
    state, reward, done, info = env.step(env.action_space.sample())
    # env.render()

    if step % 1000 == 0:
        print(f"Step: {step}, Reward: {reward}, Done: {done}")

env.close()

# cuda test
import torch
print("CUDA available:", torch.cuda.is_available())