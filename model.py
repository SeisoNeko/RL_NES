import torch
import torch.nn as nn
import torch.nn.functional as F

# Basic Block
# class Basic_C2D_Block(nn.Module):
#     def __init__(self, in_dim, out_dim, k_size, stride, is_BN):
#         super(Basic_C2D_Block, self).__init__()
#         self.conv_1 = nn.Conv2d(
#             in_dim, out_dim, kernel_size=k_size, stride=stride, padding=k_size // 2
#         )
#         self.bn_1 = nn.BatchNorm2d(out_dim) if is_BN else nn.Identity()
#         self.lrelu = nn.LeakyReLU(inplace=False)

#     def forward(self, x):
#         y = self.conv_1(x)
#         y = self.bn_1(y)
#         return self.lrelu(y)

# # Residual Block
# class Res_C2D_Block(nn.Module):
#     def __init__(self, in_dim, out_dim, num_blocks, stride=1):
#         super(Res_C2D_Block, self).__init__()

#         layers = []
#         for i in range(num_blocks):
#             layers.append(
#                 Basic_C2D_Block(
#                     in_dim=in_dim if i == 0 else out_dim,
#                     out_dim=out_dim,
#                     k_size=3,
#                     stride=stride if i == 0 else 1,
#                     is_BN=False,
#                 )
#             )
#         self.blocks = nn.Sequential(*layers)

#         self.adjust_residual = None
#         if in_dim != out_dim or stride != 1:
#             self.adjust_residual = nn.Sequential(
#                 nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=stride, padding=0, bias=False),
#                 nn.BatchNorm2d(out_dim),
#             )

#     def forward(self, x):
#         residual = x
#         if self.adjust_residual:
#             residual = self.adjust_residual(x)

#         y = self.blocks(x)
#         y += residual
#         return nn.LeakyReLU(inplace=False)(y)

class CustomCNN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(CustomCNN, self).__init__()

        # input_shape comes in as (203,) from the wrapper
        # But for this wrapper, the input dimension is simply the length of the vector.

        # Determine input size
        if isinstance(input_shape, int):
            self.input_dim = input_shape
        elif isinstance(input_shape, tuple):
             # If it comes as (203,), take the first element
            self.input_dim = input_shape[0]
        else:
            self.input_dim = 203 # Fallback

        self.fc1 = nn.Linear(self.input_dim, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, num_actions)

    def forward(self, x):
        # x shape might be (Batch, 1, 203) or (Batch, 203)
        # We need to flatten it to (Batch, 203) just in case
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

