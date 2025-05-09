import torch
import torch.nn as nn
from mamba_ssm import Mamba


class CSSMBlock(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()

        # Initial convolution block
        self.conv_block = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)
        )

        self.norm = nn.LayerNorm(dim)

        # Mamba module for sequence modeling
        self.mamba = Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )

    def forward(self, x):
        # Convolution path
        conv_out = self.conv_block(x) + x  # Residual connection

        # Prepare for Mamba (sequence modeling)
        B, C, H, W = conv_out.shape
        x_reshaped = conv_out.permute(0, 2, 3, 1).reshape(B, H * W, C)
        x_norm = self.norm(x_reshaped)

        # Mamba processing
        mamba_out = self.mamba(x_norm)
        mamba_out = mamba_out.reshape(B, H, W, C).permute(0, 3, 1, 2)

        # Combine paths
        out = conv_out * mamba_out + mamba_out

        return out