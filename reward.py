import numpy as np
import cv2

# =============================================================================
# 輔助計算函數 (從圖像中分析盤面狀態)
# =============================================================================

def count_holes(board):
    """計算盤面中的空洞數 (Holes)"""
    holes = 0
    rows, cols = board.shape

    # 對每一列(column)進行掃描
    for c in range(cols):
        block_found = False
        for r in range(rows):
            if board[r, c] == 1:
                block_found = True
            elif block_found and board[r, c] == 0:
                # 如果上面已經有磚塊，但下面是空的 -> 這是洞
                holes += 1
    return holes

def count_bumps(board):
    """計算盤面表面的崎嶇度 (Bumps)"""
    # 計算每一列的高度
    rows, cols = board.shape
    col_heights = []

    for c in range(cols):
        h = 0
        for r in range(rows):
            if board[r, c] == 1:
                h = rows - r # 高度是從底部算上來
                break
        col_heights.append(h)

    # 計算相鄰列的高度差總和
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
# 主獎勵計算器
# =============================================================================

def calculate_custom_reward(info, base_reward, prev_info, current_frame=None, env=None, current_board = None):
    """
    :param info: Gym info dict
    :param base_reward: Gym 原始回傳的 reward
    :param prev_info: 上一步的 info (包含舊的 holes/bumps 統計)
    :param current_frame: 處理過的畫面 (84x84)，用於計算新的 holes/bumps --> aborted
    :param env: 環境實例 (TetrisEnv)，用於取得盤面狀態 --> aborted
    :param current_board: 當前的盤面二值矩陣 (0/1)
    :return: (total_reward, new_info_dict)
    """

    # 1. 取得當前盤面數據
    if current_board is not None:
        board = current_board
    else:
        raise ValueError("current_board must be provided to calculate_custom_reward")

    current_holes = count_holes(board)
    current_bumps = count_bumps(board)
    current_height = get_max_height(board)

    # 若是第一幀，先初始化 prev_info
    if "holes" not in prev_info:
        prev_info["holes"] = current_holes
        prev_info["bumps"] = current_bumps
        prev_info["height"] = current_height

    total_reward = 0

    is_board_changed = (
        current_holes != prev_info["holes"] or
        current_bumps != prev_info["bumps"] or
        current_height != prev_info["height"] or
        info['number_of_lines'] > prev_info['number_of_lines']
    )

    # -------------------------------------------------------------------------
    # A. 基礎分數與行數獎勵 (沿用之前的邏輯)
    # -------------------------------------------------------------------------
    score_diff = info['score'] - prev_info['score']
    if score_diff > 0:
        total_reward += score_diff * 0.1  # 基礎分數獎勵

    lines_diff = info['number_of_lines'] - prev_info['number_of_lines']
    if lines_diff > 0:
        total_reward += lines_diff * 30.0 # 強力獎勵消行
        if lines_diff >= 2:
            total_reward += lines_diff * 5.0  # 額外獎勵多行消除
        print(f"Cleared {lines_diff} lines! +{lines_diff * 20.0} reward.")

    # -------------------------------------------------------------------------
    # B. 進階策略獎勵 (參考該 Repo)
    # -------------------------------------------------------------------------

    if is_board_changed:

        total_reward += 5.0
        # 1. 填補空洞獎勵
        holes_diff = prev_info["holes"] - current_holes
        total_reward += holes_diff * 0.2
        # print(f"Holes reduced by {holes_diff}, +{holes_diff * 0.2} reward.")

        # 2. 平整度獎勵
        bumps_diff = prev_info["bumps"] - current_bumps
        total_reward += bumps_diff * 0.5
        # print(f"Bumps reduced by {bumps_diff}, +{bumps_diff * 0.5} reward.")

        # 3. 高度懲罰 (越高扣越多)
        if current_height > 10:
            height_diff = current_height - prev_info["height"]
            if height_diff > 0:
                height_penalty = height_diff * 1.0
                total_reward -= height_penalty
                # print(f"Height increased by {height_diff}, -{height_penalty} penalty.")

    # -------------------------------------------------------------------------
    # 更新 info 供下一步使用
    # -------------------------------------------------------------------------
    new_stats = {
        "holes": current_holes,
        "bumps": current_bumps,
        "height": current_height
    }

    # Clip reward 避免梯度爆炸
    # total_reward = np.clip(total_reward, -15, 100)

    return total_reward, new_stats