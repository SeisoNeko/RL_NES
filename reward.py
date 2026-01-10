import numpy as np
import cv2

# =============================================================================
# Auxiliary calculation functions (analyze board state from image)
# =============================================================================

def count_holes(board):
    """Calculate number of holes in the board"""
    holes = 0
    rows, cols = board.shape

    # Scan each column
    for c in range(cols):
        block_found = False
        for r in range(rows):
            if board[r, c] == 1:
                block_found = True
            elif block_found and board[r, c] == 0:
                # If there is a block above but it is empty below -> this is a hole
                holes += 1
    return holes

def count_bumps(board):
    """Calculate surface roughness (Bumps) of the board"""
    # Calculate the height of each column
    rows, cols = board.shape
    col_heights = []

    for c in range(cols):
        h = 0
        for r in range(rows):
            if board[r, c] == 1:
                h = rows - r # Height is calculated from the bottom
                break
        col_heights.append(h)

    # Calculate sum of height differences between adjacent columns
    bumps = 0
    for i in range(len(col_heights) - 1):
        bumps += abs(col_heights[i] - col_heights[i+1])

    return bumps

def get_max_height(board):
    rows, cols = board.shape
    for r in range(rows):
        if np.sum(board[r, :]) > 0:
            return rows - r
    return 0

# =============================================================================
# Main Reward Calculator
# =============================================================================

def calculate_custom_reward(info, base_reward, prev_info, current_frame=None, env=None, current_board = None):
    """
    :param info: Gym info dict
    :param base_reward: Original reward returned by Gym
    :param prev_info: Info from the previous step (includes old holes/bumps stats)
    :param current_frame: Processed frame (84x84), used to calculate new holes/bumps --> aborted
    :param env: Environment instance (TetrisEnv), used to get board state --> aborted
    :param current_board: Current board binary matrix (0/1)
    :return: (total_reward, new_info_dict)
    """

    # 1. Get current board data
    if current_board is not None:
        board = current_board
    else:
        raise ValueError("current_board must be provided to calculate_custom_reward")

    current_holes = count_holes(board)
    current_bumps = count_bumps(board)
    current_height = get_max_height(board)

    # If it is the first frame, initialize prev_info first
    if "holes" not in prev_info:
        prev_info["holes"] = current_holes
        prev_info["bumps"] = current_bumps
        prev_info["height"] = current_height

    total_reward = 0.0

    is_board_changed = (
        current_holes != prev_info["holes"] or
        current_bumps != prev_info["bumps"] or
        current_height != prev_info["height"] or
        info['number_of_lines'] > prev_info['number_of_lines']
    )

    lines_diff = info['number_of_lines'] - prev_info['number_of_lines']

    if lines_diff > 0:
        # Non-linear rewards to encourage Tetris (4 lines)
        if lines_diff == 1:
            total_reward += 10.0
        elif lines_diff == 2:
            total_reward += 30.0
        elif lines_diff == 3:
            total_reward += 60.0
        elif lines_diff == 4:
            total_reward += 100.0
        else:
            total_reward += lines_diff * 25.0  # Extra lines beyond 4

        # print(f"DEBUG: Cleared {lines_diff} lines! Reward: {total_reward}")

    if is_board_changed:

        """# 1. Survival / Placement (Starvation Mode)
        total_reward += 0.1

        # 2. Holes (The Enemy)
        # If holes increase, punish HARD.
        holes_diff = prev_info["holes"] - current_holes
        if holes_diff < 0:
            # Created new holes. Penalty: -10 per hole.
            total_reward += holes_diff * 5.0
        elif holes_diff > 0:
            # Filled holes. Small reward.
            total_reward += holes_diff * 3.0

        # 3. Bumps (Roughness)
        bumps_diff = prev_info["bumps"] - current_bumps
        total_reward += bumps_diff * 0.5

        # 4. Height Penalty
        total_reward -= current_height * 0.1"""
        # Balanced simpler reward scheme
        total_reward += base_reward

    # -------------------------------------------------------------------------
    # C. Game Over Penalty (Always check)
    # -------------------------------------------------------------------------
    if info.get('is_game_over', False):
        total_reward -= 50.0

    # -------------------------------------------------------------------------
    # Update info
    # -------------------------------------------------------------------------
    new_stats = {
        "holes": current_holes,
        "bumps": current_bumps,
        "height": current_height
    }

    return total_reward, new_stats