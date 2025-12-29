import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import cv2

# Env state
# info = {
#     "x_pos",  # (int) The player's horizontal position in the level.
#     "y_pos",  # (int) The player's vertical position in the level.
#     "score",  # (int) The current score accumulated by the player.
#     "coins",  # (int) The number of coins the player has collected.
#     "time",   # (int) The remaining time for the level.
#     "flag_get",  # (bool) True if the player has reached the end flag (level completion).
#     "life"   # (int) The number of lives the player has left.
# }


# # simple actions_dim = 7
# SIMPLE_MOVEMENT = [
#     ["NOOP"],       # Do nothing.
#     ["right"],      # Move right.
#     ["right", "A"], # Move right and jump.
#     ["right", "B"], # Move right and run.
#     ["right", "A", "B"], # Move right, run, and jump.
#     ["A"],          # Jump straight up.
#     ["left"],       # Move left.
# ]
#-----------------------------------------------------------------------------
#獎勵函數
'''
get_coin_reward         : 根據硬幣數量變化提供額外獎勵

'''
'''
環境資訊 (info)
1."x_pos": 水平位置，用於判斷角色的前進情況
2."y_pos": 垂直位置，用於分析跳躍或下落行為
3."score": 玩家目前的遊戲分數
4."coins": 收集到的硬幣數量
5."time": 剩餘時間
5."flag_get": 是否到達終點旗幟（遊戲完成）
6."life": 玩家剩餘的生命數
'''

#===============to do===============================請自定義獎勵函數 至少7個(包含提供的)
#例子:用來獎勵玩家蒐集硬幣的行為
def get_coin_reward(info, reward, prev_info):
    #寫下蒐集到硬幣會對應多少獎勵
    total_reward = reward                                         #獲得目前已有的獎勵數量

    total_reward += (info['coins'] - prev_info['coins']) * 10     #這裡是定義，如果玩家有蒐集到硬幣，則獎勵加10(這裡是可以自己去定義獎勵要給多少的)
    return total_reward

# 2. [補全] 用來鼓勵玩家進行跳躍或高度變化 (避免掉坑，鼓勵越過障礙)
def distance_y_offset_reward(info, reward, prev_info):
    y_reward = 0
    # 如果高度變高，給予微小獎勵 (鼓勵嘗試跳躍)
    # 注意：不要給太大，否則 Mario 會變成只會在原地一直跳的袋鼠
    if info['y_pos'] > prev_info['y_pos']:
        y_reward += 0.5

    # 這裡可以加一個判斷：如果 y_pos < 75 (通常是掉進坑裡的閾值)，則不給獎勵甚至給懲罰
    # 但通常死亡懲罰會處理掉坑的情況，所以這裡專注於鼓勵"向上"
    return reward + y_reward

# 3. [補全] 用來鼓勵玩家前進，懲罰原地停留或後退
def distance_x_offset_reward(info, reward, prev_info):
    x_reward = 0
    dist = info['x_pos'] - prev_info['x_pos']

    if dist > 0:
        # 前進給予獎勵 (這是最重要的訊號，權重可以高一點)
        x_reward += dist * 1.0
    elif dist == 0:
        # 原地不動給予微小懲罰 (逼迫他動起來)
        x_reward -= 0.1
    else:
        # 後退給予懲罰 (視情況而定，有時候為了躲怪需要後退，所以懲罰不要太重)
        x_reward -= 0.5

    return reward + x_reward

# 4. [補全] 用來鼓勵玩家提高分數（例如擊敗敵人、頂磚塊)
def monster_score_reward(info, reward, prev_info):
    # 分數增加可能來自：吃金幣、殺怪、頂磚塊、通關
    score_diff = info['score'] - prev_info['score']

    # 我們已經有金幣獎勵了，為了避免重複獎勵(Double Dipping)，我們可以嘗試扣除金幣的分數
    # 假設一個金幣通常是 200 分 (視遊戲版本而定，這裡假設單純獎勵分數增加)
    if score_diff > 0:
        # 這裡給予分數變化的 10% 作為獎勵，避免數值過大掩蓋了"前進"的重要性
        return reward + (score_diff * 0.1)

    return reward

# 5. [補全] 用來鼓勵玩家完成關卡（到達終點旗幟）
def final_flag_reward(info, reward):
    flag_reward = 0
    if info['flag_get']:
        # 這是一個稀疏獎勵(Sparse Reward)，一旦發生要給予巨大的肯定
        flag_reward += 500
        # 也可以加上剩餘時間的獎勵
        # flag_reward += info['time'] * 0.5
    return reward + flag_reward

# 6. [新增] 死亡懲罰 (Life Loss Penalty)
# 原因：訓練 Agent 活下去是最基本的，如果死掉了要給予重罰
def death_penalty(info, reward, prev_info):
    death_r = 0
    if info['life'] < prev_info['life']:
        # 死掉一條命扣 50 分
        death_r -= 50
    return reward + death_r

# 7. [新增] 時間流逝懲罰 (Time Penalty)
# 原因：鼓勵 Agent 盡快完成任務，不要在原地發呆浪費時間
def time_penalty(info, reward, prev_info):
    # 每一幀都扣除微小的分數，這在 RL 中稱為 "Living Penalty"
    # 這會給予 Agent 一種"急迫感"
    return reward - 0.01

# 8. [新增] 卡死/無效動作懲罰 (Stuck Penalty)
# 原因：如果 Mario 卡在管子前一直走不過去，需要給予懲罰讓他嘗試別的動作(如跳躍)
# 註：這需要比較複雜的邏輯(例如記錄過去10幀的位置)，這裡先用簡單版
def stuck_penalty(info, reward, prev_info):
    # 如果沒死、沒贏、且 x 座標完全沒變、y 座標也完全沒變
    if (info['x_pos'] == prev_info['x_pos'] and
        info['y_pos'] == prev_info['y_pos'] and
        not info['flag_get']):
        return reward - 0.05
    return reward

# =============================================================================
# 總獎勵計算器 (Wrapper 裡呼叫這個)
# =============================================================================

def calculate_custom_reward(info, base_reward, prev_info):
    """
    將所有自定義獎勵函數串接起來
    :param info: 當前幀的資訊
    :param base_reward: Gym 環境原始返回的獎勵 (通常基於 x 的移動)
    :param prev_info: 上一幀的資訊 (用於計算變化量)
    """

    # 如果沒有上一幀的資訊（剛 Reset），直接回傳原始獎勵
    if prev_info is None:
        return base_reward

    # 初始化總獎勵 (你可以選擇忽略 base_reward 從 0 開始，或者在 base_reward 基礎上疊加)
    # 建議：Gym-super-mario-bros 預設的 reward 其實就是 x 的移動距離。
    # 如果你想完全掌控，建議 total_reward = 0，然後依賴上面的 distance_x_offset_reward

    total_reward = 0  # 這裡我們選擇從 0 開始重新定義，比較乾淨

    # 依序呼叫各個獎勵函數
    total_reward = get_coin_reward(info, total_reward, prev_info)
    total_reward = distance_y_offset_reward(info, total_reward, prev_info)
    total_reward = distance_x_offset_reward(info, total_reward, prev_info)
    total_reward = monster_score_reward(info, total_reward, prev_info)
    total_reward = final_flag_reward(info, total_reward)
    total_reward = death_penalty(info, total_reward, prev_info)
    total_reward = time_penalty(info, total_reward, prev_info)
    # total_reward = stuck_penalty(info, total_reward, prev_info) # 視需求開啟

    # 將數值正規化 (這對於某些 RL 演算法如 PPO 很重要，避免梯度爆炸)
    # 這裡只做簡單的限制，將單步獎勵限制在 -15 到 15 之間
    total_reward = np.clip(total_reward, -15, 15)

    return total_reward