import gym
import numpy as np
from gym import spaces
from utils import preprocess_frame
from reward import count_holes, count_bumps, get_max_height, calculate_custom_reward

shape_dict = {
    0: "T",
    1: "J",
    2: "Z",
    3: "O",
    4: "S",
    5: "L",
    6: "I"
}

class TetrisWrapper(gym.Wrapper):
    def __init__(self, env, skip=4):
        super().__init__(env)
        self.skip = skip
        self.prev_info = {
            "score": 0,
            "number_of_lines": 0,
            "holes": 0,
            "bumps": 0,
            "height": 0,
            "is_game_over": False
        }

        # Calculate new Observation Space Size
        # 200 (Board) + 3 (Stats) + 7 (Current Piece One-Hot) + 4 rotation + 7 (Next Piece One-Hot) + 2 (XY Pos)
        # Total = 223
        self.obs_dim = 223
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

    def _get_board_state(self, env=None):
        """
        Use RAM to read the board state, which is more accurate than visual recognition and does not count falling blocks as holes.
        Note: This requires passing the env object.
        """
        if env is None:
            raise ValueError("Environment (env) must be passed to read RAM state.")

        ram = None
        curr_env = env

        for _ in range(10):
            if hasattr(curr_env, 'ram'):
                ram = curr_env.ram
                break
            elif hasattr(curr_env, 'unwrapped') and hasattr(curr_env.unwrapped, 'ram'):
                ram = curr_env.unwrapped.ram
                break
            elif hasattr(curr_env, 'env'):
                curr_env = curr_env.env
            else:
                break

        if ram is None:
            raise RuntimeError("Could not find NES RAM in the environment wrappers.")

        board_ram = ram[0x0400:0x04C8]

        board_matrix = np.zeros((20, 10), dtype=int)

        for i in range(200):
            row = i // 10
            col = i % 10

            if board_ram[i] != 239 and board_ram[i] != 0:
                board_matrix[row, col] = 1

        return board_matrix

    def _get_one_hot_piece(self, piece_id):
        """
        Maps the 19 NES Tetris orientation states to 7 Piece Types.
        Reference from tetris_env.py _PIECE_ORIENTATION_TABLE:
        0-3: T, 4-7: J, 8-9: Z, 10: O, 11-12: S, 13-16: L, 17-18: I
        """
        vec = np.zeros(7, dtype=np.float32)

        if piece_id is None:
            return vec

        piece_type = -1

        # Exact mapping based on the ROM internal table
        if 0 <= piece_id <= 3:
            piece_type = 0   # T
        elif 4 <= piece_id <= 7:
            piece_type = 1   # J
        elif 8 <= piece_id <= 9:
            piece_type = 2   # Z
        elif piece_id == 10:
            piece_type = 3   # O
        elif 11 <= piece_id <= 12:
            piece_type = 4   # S
        elif 13 <= piece_id <= 16:
            piece_type = 5   # L
        elif 17 <= piece_id <= 18:
            piece_type = 6   # I

        if piece_type != -1:
            vec[piece_type] = 1.0

        return vec

    def _get_rotation_one_hot(self, piece_id):
        """
        Extracts rotation (0, 1, 2, 3) from the raw Piece ID.
        Returns a One-Hot vector of size 4.
        """
        vec = np.zeros(4, dtype=np.float32)
        if piece_id is None: return vec

        rotation_idx = 0

        # Logic derived from tetris_env.py ranges
        if 0 <= piece_id <= 3:   # T (4 states)
            rotation_idx = piece_id - 0
        elif 4 <= piece_id <= 7: # J (4 states)
            rotation_idx = piece_id - 4
        elif 8 <= piece_id <= 9: # Z (2 states)
            rotation_idx = piece_id - 8
        elif piece_id == 10:     # O (1 state)
            rotation_idx = 0
        elif 11 <= piece_id <= 12: # S (2 states)
            rotation_idx = piece_id - 11
        elif 13 <= piece_id <= 16: # L (4 states)
            rotation_idx = piece_id - 13
        elif 17 <= piece_id <= 18: # I (2 states)
            rotation_idx = piece_id - 17

        # Safety clamp just in case
        if 0 <= rotation_idx < 4:
            vec[rotation_idx] = 1.0

        return vec

    def _get_state_vector(self, info):
        """
        Constructs the state vector using functions from reward.py
        """

        # 1. Get RAM
        ram = self._get_ram()

        # 2. Get Board (Static Blocks)
        # We use the helper from reward.py, but pass the RAM we found
        board = self._get_board_state(env=self.env)

        # 3. Get Moving Piece Info (The Missing Link!)
        curr_piece_id = ram[0x0042]
        next_piece_id = ram[0x00BF]
        curr_x = ram[0x0040]
        curr_y = ram[0x0041]

        # 4. Feature Engineering
        # One-Hot Encode Pieces
        curr_piece_vec = self._get_one_hot_piece(curr_piece_id)
        curr_rot_vec = self._get_rotation_one_hot(curr_piece_id)
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
        # [200] + [3] + [7] + [4] + [7] + [2] = 223
        state = np.concatenate((
            board_flat,     #200
            stats_vec,      #3
            curr_piece_vec, #7
            curr_rot_vec,   #4
            next_piece_vec, #7
            pos_vec         #2
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
            "height": 0,
            "is_game_over": False
        }

        state, _ = self._get_state_vector(info)
        return state, info

    def step(self, action):
        # print("DEBUG: TetrisWrapper Step with Skip =", self.skip)
        total_reward = 0.0
        done = False
        info = {}

        for _ in range(self.skip):
            # Step the inner environment
            step_result = self.env.step(action)

            # Handle Gym 0.26+ tuple unpacking (5 values) vs Old Gym (4 values)
            if len(step_result) == 5:
                obs, reward, terminated, truncated, inner_info = step_result
                current_done = terminated or truncated
            else:
                obs, reward, current_done, inner_info = step_result
            
            # Accumulate the standard game reward (score for lines)
            total_reward += reward
            
            # Update info to the latest frame's info
            info = inner_info
            
            if current_done:
                done = True
                break

        if done:
            info['is_game_over'] = True

        # 2. Get State Vector
        state, current_board = self._get_state_vector(info)
        info["board"] = current_board

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

        # Note: Death penalty is now handled entirely in reward.py,
        # so we do not subtract anything here.

        return state, custom_reward, done, False, info