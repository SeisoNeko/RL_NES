import gym
import numpy as np
from gym import spaces
from utils import preprocess_frame
from reward import get_board_state, count_holes, count_bumps, get_max_height, calculate_custom_reward

class TetrisWrapper(gym.Wrapper):
    def __init__(self, env, skip=4):
        super().__init__(env)
        self._skip = skip
        self.prev_info = {
            "score": 0,
            "number_of_lines": 0,
            "holes": 0,
            "bumps": 0,
            "height": 0
        }

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(203, ), dtype=np.float32
        )

    def _get_state_vector(self, info):
        """
        Constructs the state vector using functions from reward.py
        """
        # 1. Get the board (20x10 matrix)
        # We pass None for screen_frame because we are using RAM now
        board = get_board_state(info, screen_frame=None, env=self.env)

        # 2. Calculate features
        holes = count_holes(board)
        bumps = count_bumps(board)
        height = get_max_height(board)

        # 3. Flatten board to 1D array (200,)
        board_flat = board.flatten().astype(np.float32)

        # 4. Create feature array (3,)
        # Normalize slightly to help training (optional but recommended)
        features = np.array([holes, bumps, height], dtype=np.float32)

        # 5. Concatenate to make shape (203,)
        state = np.concatenate((board_flat, features))

        return state, board

    def reset(self, **kwargs):
        # Handle cases where inner env returns (obs, info) or just obs
        # This robustness fixes the "too many values to unpack" error
        results = self.env.reset(**kwargs)

        if isinstance(results, tuple):
            obs, info = results
        else:
            obs = results
            info = {} # Empty info if not provided

        # Reset internal stats
        self.prev_info = {
            "score": 0,
            "number_of_lines": 0,
            "holes": 0,
            "bumps": 0,
            "height": 0
        }

        state, current_board = self._get_state_vector(info)
        return state, info

    def step(self, action):
        total_reward = 0.0
        done = False
        info = {}

        # 1. Skip Frame Logic
        for _ in range(self._skip):
            # Handle New vs Old API for step
            step_result = self.env.step(action)

            if len(step_result) == 5:
                obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                obs, reward, done, info = step_result

            total_reward += reward
            if done:
                break

        # 2. Preprocess
        state, current_board = self._get_state_vector(info)


        # 3. Calculate Custom Reward
        custom_reward, new_stats = calculate_custom_reward(
            info,
            total_reward,
            self.prev_info,
            current_frame=None,
            env=self.env,
            current_board=current_board
        )

        self.prev_info.update(new_stats)
        self.prev_info["score"] = info.get("score", 0)
        self.prev_info["number_of_lines"] = info.get("number_of_lines", 0)

        if done:
            custom_reward -= 10

        # Return 4 values (Gym Vector Env usually handles 4 fine, but reset is strict)
        return state, custom_reward, done, False, info