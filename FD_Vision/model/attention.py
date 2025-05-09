import torch
import torch.nn as nn


class CrossAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv1d = nn.Conv1d(1, 1, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, path1, path2):
        # Compute attention weights
        f = path1 + path2
        gap = self.gap(f)
        std = torch.std(f, dim=(2, 3), keepdim=True)
        attention = self.sigmoid(self.conv1d((gap + std).squeeze(-1).unsqueeze(1)))
        attention = attention.unsqueeze(-1).unsqueeze(-1)

        # Apply attention
        path1_out = attention * path1 + path2
        path2_out = (1 - attention) * path2 + path1

        return path1_out, path2_out