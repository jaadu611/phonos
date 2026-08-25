"""
Universal SmallUNet Architecture with FiLM Prompt Conditioning.
Core backbone for all AudioForge domain specialists (Battlefield, Office, Urban, Nature).
"""

from typing import Dict, List, Literal, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(Conv2D -> BatchNorm -> ReLU) * 2"""

    def __init__(self, in_channels: int, out_channels: int, mid_channels: int = None):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv with skip connection"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(
            x1,
            [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2],
        )

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class FiLMConditioner(nn.Module):
    """
    Generates feature-wise scale (gamma) and shift (beta) modulations
    based on the requested sound stem.
    """

    def __init__(self, num_classes: int = 32, embed_dim: int = 128, feature_dim: int = 256):
        super().__init__()
        self.embedding = nn.Embedding(num_classes, embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, feature_dim * 2),
        )

    def forward(self, class_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        emb = self.embedding(class_ids)
        mod = self.mlp(emb)
        gamma, beta = torch.chunk(mod, 2, dim=-1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return gamma + 1.0, beta


class SmallUNet(nn.Module):
    """
    Universal Prompt-Conditioned Specialist U-Net.
    Extracts and inpaints any requested sound stem from dense acoustic mixtures.
    """

    def __init__(
        self,
        n_fft: int = 512,
        hop_length: int = 160,
        base_channels: int = 16,
        num_classes: int = 32,
        mode: Literal["generate", "mask"] = "generate",
    ):
        super().__init__()

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.mode = mode
        self.base_channels = base_channels
        self.num_classes = num_classes
        self.stem_map: Dict[str, int] = {}

        # Encoder
        self.inc = DoubleConv(2, base_channels)
        self.down1 = Down(base_channels, base_channels * 2)
        self.down2 = Down(base_channels * 2, base_channels * 4)
        self.down3 = Down(base_channels * 4, base_channels * 8)

        # Bottleneck (256 channels for base_channels=16)
        self.down4 = Down(base_channels * 8, base_channels * 16)

        # FiLM Prompt Conditioning (supports up to num_classes dynamically)
        self.film = FiLMConditioner(
            num_classes=num_classes,
            embed_dim=128,
            feature_dim=base_channels * 16,
        )

        # Decoder
        self.up1 = Up(base_channels * 16 + base_channels * 8, base_channels * 8)
        self.up2 = Up(base_channels * 8 + base_channels * 4, base_channels * 4)
        self.up3 = Up(base_channels * 4 + base_channels * 2, base_channels * 2)
        self.up4 = Up(base_channels * 2 + base_channels, base_channels)

        # Output projection
        self.outc = nn.Conv2d(base_channels, 2, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def set_stem_map(self, stem_map: Dict[str, int]):
        self.stem_map = stem_map

    def parse_prompt(self, prompt: Union[torch.Tensor, str, List[str], int], batch_size: int, device: torch.device) -> torch.Tensor:
        if prompt is None:
            return torch.zeros(batch_size, dtype=torch.long, device=device)

        if isinstance(prompt, torch.Tensor):
            return prompt.to(device=device, dtype=torch.long)

        if isinstance(prompt, int):
            return torch.full((batch_size,), prompt, dtype=torch.long, device=device)

        if isinstance(prompt, str):
            p = prompt.lower().strip()
            cid = self.stem_map.get(p, 0)
            if cid == 0 and p not in ["speech", "voice", "clean"]:
                # Fuzzy keyword matching
                for key, val in self.stem_map.items():
                    if key in p or p in key:
                        cid = val
                        break
            return torch.full((batch_size,), cid, dtype=torch.long, device=device)

        if isinstance(prompt, (list, tuple)):
            cids = [self.parse_prompt(p, 1, device).item() for p in prompt]
            return torch.tensor(cids, dtype=torch.long, device=device)

        return torch.zeros(batch_size, dtype=torch.long, device=device)

    def forward(
        self,
        x: torch.Tensor,
        prompt: Optional[Union[torch.Tensor, str, List[str], int]] = None,
    ) -> torch.Tensor:
        batch_size = x.shape[0]
        prompt_tensor = self.parse_prompt(prompt, batch_size, x.device)

        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        # Bottleneck
        x5 = self.down4(x4)

        # FiLM Conditioning
        gamma, beta = self.film(prompt_tensor)
        x5 = gamma * x5 + beta

        # Decoder
        u1 = self.up1(x5, x4)
        u2 = self.up2(u1, x3)
        u3 = self.up3(u2, x2)
        u4 = self.up4(u3, x1)

        raw_output = self.outc(u4)

        if self.mode == "generate":
            return raw_output
        else:
            mask = self.sigmoid(raw_output)
            return mask * x
