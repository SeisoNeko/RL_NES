"""
Functions for preprocessing state image and creating the additional info vector live here
"""

import numpy as np
from matplotlib import pyplot as plt


def rgb2gray(rgb):
    """
    Changes rgb image to grayscale
    """
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


def bw_pixel(gray):
    """
    Changes a grayscale pixel to only black or white
    """
    return gray if gray == 0 else 1


def crop_clean_state(state):
    """
    Crops only the tetris board from the state image and makes it black and white
    """
    cropped = state[47:207, 95:175]
    cropped = cropped / 255.0
    gray_cropped = rgb2gray(cropped)
    bw_vec = np.vectorize(bw_pixel)
    bw_cropped = np.expand_dims(np.array(list(map(bw_vec, gray_cropped))), axis=2)
    # plt.imshow(bw_cropped, interpolation='nearest', cmap="gray")
    # plt.show()
    return bw_cropped


# Mapping of piece names to ints
piece_map = {
    "Sh": 0,
    "Sv": 1,
    "Zh": 2,
    "Zv": 3,
    "Ih": 4,
    "Iv": 5,
    "O": 6,
    "Ld": 7,
    "Ll": 8,
    "Lr": 9,
    "Lu": 10,
    "Jd": 11,
    "Jl": 12,
    "Jr": 13,
    "Ju": 14,
    "Td": 15,
    "Tl": 16,
    "Tr": 17,
    "Tu": 18,
    None: 19,
}


def extra_feats(info):
    """
    Extract the additional feature vector from the info dict
    """
    feats = []
    feats.append(piece_map[info["current_piece"]])
    feats.append(piece_map[info["next_piece"]])
    feats.append(info["score"])
    feats.append(info["number_of_lines"])
    feats.append(info["board_height"])
    return np.expand_dims(np.array(feats), axis=0)