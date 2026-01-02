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

        # Calculate new Observation Space Size
        # 200 (Board) + 3 (Stats) + 7 (Current Piece One-Hot) + 7 (Next Piece One-Hot) + 2 (XY Pos)
        # Total = 219
        self.obs_dim = 219
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.obs_dim,), dtype=np.float32
        )

    def _get_ram(self):
        """Robustly fetch RAM from the environment"""
        if hasattr(self.env, 'ram'):
            return self.env.ram
        elif hasattr(self.env, 'unwrapped') and hasattr(self.env.unwrapped, 'ram'):
            return self.env.unwrapped.ram
        elif hasattr(self.env, 'env'):
            return self.env.env.unwrapped.ram
        raise RuntimeError("RAM not found")

    def _get_one_hot_piece(self, piece_id):
        """Converts piece ID (0-6) to One-Hot Vector (size 7)"""
        vec = np.zeros(7, dtype=np.float32)
        # NES Tetris Piece IDs are usually 0x00 to 0x06 (T, J, Z, O, S, L, I)
        # Sometimes 0x07-0x12 depending on rotation, but usually modulo 7 works for type
        if piece_id is not None and 0 <= piece_id < 255:
            # Simple modulo to handle rotation variants if any
            idx = piece_id % 7
            vec[idx] = 1.0
        return vec

    def _get_state_vector(self, info):
        """
        Constructs the state vector using functions from reward.py
        """

        # 1. Get RAM
        ram = self._get_ram()

        # 2. Get Board (Static Blocks)
        # We use the helper from reward.py, but pass the RAM we found
        board = get_board_state(info, screen_frame=None, env=self.env)

        # 3. Get Moving Piece Info (The Missing Link!)
        curr_piece_id = ram[0x0042]
        next_piece_id = ram[0x00BF]
        curr_x = ram[0x0041]
        curr_y = ram[0x0040]

        # 4. Feature Engineering
        # One-Hot Encode Pieces
        curr_piece_vec = self._get_one_hot_piece(curr_piece_id)
        next_piece_vec = self._get_one_hot_piece(next_piece_id)

        # Normalize Position
        # X is usually 0-9, Y is 0-19
        pos_vec = np.array([curr_x / 10.0, curr_y / 20.0], dtype=np.float32)

        # Existing Stats
        holes = count_holes(board)
        bumps = count_bumps(board)
        height = get_max_height(board)
        stats_vec = np.array([holes/100.0, bumps/100.0, height/20.0], dtype=np.float32)

        # Flatten Board
        board_flat = board.flatten().astype(np.float32)

        # 5. Concatenate Everything
        # [200] + [3] + [7] + [7] + [2] = 219
        state = np.concatenate((
            board_flat,
            stats_vec,
            curr_piece_vec,
            next_piece_vec,
            pos_vec
        ))

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

        state, _ = self._get_state_vector(info)
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