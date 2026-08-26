"""
AudioForge Flagship 45.6M Parameter Generative Adversarial Speech Specialist.
Features:
- High-Capacity Residual & Dilated Speech Generator (~37.0M params)
- Spectral-Normalized Multi-Scale PatchGAN Discriminator (~8.6M params, HiFi-GAN style)
- Lipschitz-constrained adversarial critic to prevent discriminator overpower/gradient vanishing
- Direct Complex STFT Phase & Harmonic Formant Inpainting
"""

from typing import Dict, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


# =========================================================================
# 1. FLAGSHIP RESIDUAL DILATED GENERATOR (~37.0M Parameters)
# =========================================================================

class DilatedResidualBlock(nn.Module):
    """Residual block with dilated convolutions for long-range temporal acoustic context"""

    def __init__(self, channels: int, dilation: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(channels)
        self.act1 = nn.LeakyReLU(0.2, inplace=True)

        self.conv2 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(channels)
        self.act2 = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act2(out + res)


class HighResSpeechGenerator(nn.Module):
    """
    37.0M Parameter Deep Residual Speech Generator.
    Operates on complex STFT spectrograms with dilated receptive fields for phoneme inpainting.
    """

    def __init__(self, base_channels: int = 48):
        super().__init__()
        self.base_channels = base_channels

        # Initial Feature Projection (2 -> 48)
        self.in_conv = nn.Sequential(
            nn.Conv2d(2, base_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Encoder Stages (Downsampling)
        # Stage 1: 48 -> 96
        self.down1_pool = nn.MaxPool2d(2)
        self.down1_conv = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            DilatedResidualBlock(base_channels * 2, dilation=1),
        )

        # Stage 2: 96 -> 192
        self.down2_pool = nn.MaxPool2d(2)
        self.down2_conv = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
            DilatedResidualBlock(base_channels * 4, dilation=2),
        )

        # Stage 3: 192 -> 384
        self.down3_pool = nn.MaxPool2d(2)
        self.down3_conv = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 8),
            nn.LeakyReLU(0.2, inplace=True),
            DilatedResidualBlock(base_channels * 8, dilation=4),
        )

        # Bottleneck: 384 -> 768 (Deep Semantic Latent Space)
        self.down4_pool = nn.MaxPool2d(2)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_channels * 8, base_channels * 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 16),
            nn.LeakyReLU(0.2, inplace=True),
            DilatedResidualBlock(base_channels * 16, dilation=1),
            DilatedResidualBlock(base_channels * 16, dilation=2),
        )

        # Decoder Stages (Upsampling with Skip Connections)
        # Stage 3 Up: (768 + 384) -> 384
        self.up3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.up3_conv = nn.Sequential(
            nn.Conv2d(base_channels * 16 + base_channels * 8, base_channels * 8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 8),
            nn.LeakyReLU(0.2, inplace=True),
            DilatedResidualBlock(base_channels * 8, dilation=2),
        )

        # Stage 2 Up: (384 + 192) -> 192
        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.up2_conv = nn.Sequential(
            nn.Conv2d(base_channels * 8 + base_channels * 4, base_channels * 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
            DilatedResidualBlock(base_channels * 4, dilation=1),
        )

        # Stage 1 Up: (192 + 96) -> 96
        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.up1_conv = nn.Sequential(
            nn.Conv2d(base_channels * 4 + base_channels * 2, base_channels * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            DilatedResidualBlock(base_channels * 2, dilation=1),
        )

        # Final Stage Up: (96 + 48) -> 48
        self.up0 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.up0_conv = nn.Sequential(
            nn.Conv2d(base_channels * 2 + base_channels, base_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Output Complex STFT Projection: 48 -> 2 [Real, Imag]
        self.out_conv = nn.Conv2d(base_channels, 2, kernel_size=1)

    def _pad_and_cat(self, upsampled: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        diffY = skip.size()[2] - upsampled.size()[2]
        diffX = skip.size()[3] - upsampled.size()[3]
        upsampled = F.pad(upsampled, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        return torch.cat([skip, upsampled], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        x0 = self.in_conv(x)                      # [B, 48, F, T]
        x1 = self.down1_conv(self.down1_pool(x0)) # [B, 96, F/2, T/2]
        x2 = self.down2_conv(self.down2_pool(x1)) # [B, 192, F/4, T/4]
        x3 = self.down3_conv(self.down3_pool(x2)) # [B, 384, F/8, T/8]

        # Bottleneck
        b = self.bottleneck(self.down4_pool(x3))  # [B, 768, F/16, T/16]

        # Decoder
        u3 = self.up3_conv(self._pad_and_cat(self.up3(b), x3))
        u2 = self.up2_conv(self._pad_and_cat(self.up2(u3), x2))
        u1 = self.up1_conv(self._pad_and_cat(self.up1(u2), x1))
        u0 = self.up0_conv(self._pad_and_cat(self.up0(u1), x0))

        return self.out_conv(u0)


# =========================================================================
# 2. SPECTRAL-NORMALIZED MULTI-SCALE PATCHGAN DISCRIMINATOR (~8.6M Params)
# =========================================================================

class DiscriminatorScale(nn.Module):
    """Single scale patch discriminator with Spectral Normalization (HiFi-GAN standard)"""

    def __init__(self):
        super().__init__()
        self.conv1 = spectral_norm(nn.Conv2d(2, 64, kernel_size=5, stride=2, padding=2))
        self.act1 = nn.LeakyReLU(0.2, inplace=True)

        self.conv2 = spectral_norm(nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2))
        self.act2 = nn.LeakyReLU(0.2, inplace=True)

        self.conv3 = spectral_norm(nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2))
        self.act3 = nn.LeakyReLU(0.2, inplace=True)

        self.conv4 = spectral_norm(nn.Conv2d(256, 512, kernel_size=5, stride=2, padding=2))
        self.act4 = nn.LeakyReLU(0.2, inplace=True)

        self.out_conv = spectral_norm(nn.Conv2d(512, 1, kernel_size=3, padding=1))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        features = []
        x1 = self.act1(self.conv1(x))
        features.append(x1)

        x2 = self.act2(self.conv2(x1))
        features.append(x2)

        x3 = self.act3(self.conv3(x2))
        features.append(x3)

        x4 = self.act4(self.conv4(x3))
        features.append(x4)

        score = self.out_conv(x4)
        return score, features


class MultiScaleSpeechDiscriminator(nn.Module):
    """
    Multi-Scale PatchGAN Discriminator with Spectral Normalization.
    Judges full-resolution spectrogram patches + 2x downscaled global harmonic structures.
    """

    def __init__(self):
        super().__init__()
        self.disc_full = DiscriminatorScale()
        self.disc_down = DiscriminatorScale()
        self.downsample = nn.AvgPool2d(2)

    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[List[torch.Tensor]]]:
        score1, feat1 = self.disc_full(x)
        score2, feat2 = self.disc_down(self.downsample(x))
        return [score1, score2], [feat1, feat2]


# Aliases
SpeechSpecialistUNet = HighResSpeechGenerator
SmallUNet = HighResSpeechGenerator


if __name__ == "__main__":
    gen = HighResSpeechGenerator(base_channels=48)
    disc = MultiScaleSpeechDiscriminator()
    g_params = sum(p.numel() for p in gen.parameters() if p.requires_grad)
    d_params = sum(p.numel() for p in disc.parameters() if p.requires_grad)
    print(f"Generator Params:     {g_params:,} (~{g_params*4/1024/1024:.2f} MB)")
    print(f"Discriminator Params: {d_params:,} (~{d_params*4/1024/1024:.2f} MB)")
    print(f"Total Pipeline:       {g_params + d_params:,} (~{(g_params+d_params)*4/1024/1024:.2f} MB)")
