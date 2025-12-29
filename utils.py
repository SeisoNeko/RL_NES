import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import cv2

def preprocess_frame(frame):
    cropprd = frame[47:207, 95:175]  # Crop to play area
    gray = cv2.cvtColor(cropprd, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)  # Binarize
    resized = cv2.resize(binary, (84, 84), interpolation=cv2.INTER_AREA)  # Resize to 84x84
    normalized = resized.astype(np.float32) / 255.0  # Normalize pixel values to [0, 1]

    return normalized  # Return preprocessed frame