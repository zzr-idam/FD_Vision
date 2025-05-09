import torch
import torch.nn as nn
import torch.fft as fft
from .cssm_block import CSSMBlock
from .attention import CrossAttention


class FDVMNet(nn.Module):
    def __init__(self, num_blocks=8, num_channels=48):
        super().__init__()

        # Initial downsampling
        self.downsample = nn.Sequential(
            nn.Conv2d(3, num_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        # Dual-path processing blocks
        self.phase_path = nn.ModuleList([CSSMBlock(num_channels) for _ in range(num_blocks)])
        self.amplitude_path = nn.ModuleList([CSSMBlock(num_channels) for _ in range(num_blocks)])

        # Cross-attention modules
        self.cross_attentions = nn.ModuleList([CrossAttention(num_channels) for _ in range(num_blocks)])

        # Upsampling and refinement
        self.upsample = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(num_channels, 3, kernel_size=3, stride=1, padding=1)
        )

        self.refine_net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        # Initial feature extraction
        f0 = self.downsample(x)

        # Fourier transform to get phase and amplitude
        fft_features = fft.fft2(f0, dim=(-2, -1))
        phase = torch.angle(fft_features)
        amplitude = torch.abs(fft_features)

        # Process through dual paths
        phase_feat = phase
        amplitude_feat = amplitude

        for phase_block, amp_block, cross_attn in zip(self.phase_path,
                                                      self.amplitude_path,
                                                      self.cross_attentions):
            phase_feat = phase_block(phase_feat)
            amplitude_feat = amp_block(amplitude_feat)

            # Cross-attention between paths
            phase_feat, amplitude_feat = cross_attn(phase_feat, amplitude_feat)

        # Combine features and inverse Fourier transform
        combined = phase_feat + amplitude_feat
        restored = fft.ifft2(torch.polar(combined, phase), dim=(-2, -1)).real

        # Upsample and refine
        out = self.upsample(restored)
        out = self.refine_net(out + x)  # Residual connection

        return out