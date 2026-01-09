import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomNN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(CustomNN, self).__init__()

        # input_shape comes in as (223,) from the wrapper
        # But for this wrapper, the input dimension is simply the length of the vector.

        # Determine input size
        if isinstance(input_shape, int):
            self.input_dim = input_shape
        elif isinstance(input_shape, tuple):
             # If it comes as (223,), take the first element
            self.input_dim = input_shape[0]
        else:
            self.input_dim = 223 # Fallback

        self.fc1 = nn.Linear(self.input_dim, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, num_actions)

    def forward(self, x):
        # x shape might be (Batch, 1, 223) or (Batch, 223)
        # We need to flatten it to (Batch, 223) just in case
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

