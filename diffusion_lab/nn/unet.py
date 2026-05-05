"""
diffusion_lab/nn/unet.py
Small U-Net for 28×28 MNIST generative model experiments.

Architecture
────────────
Encoder:
    Conv2d(1 → C)  →  ResBlock(C → 2C, stride=2)  →  ResBlock(2C → 4C, stride=2)

Bottleneck:
    ResBlock(4C → 4C)   ← time-conditioned via AdaGN

Decoder (with skip connections):
    Upsample + ResBlock(4C+4C → 2C)  ← skip from encoder
    Upsample + ResBlock(2C+2C →  C)  ← skip from encoder

Head:
    GroupNorm → SiLU → Conv2d(C → out_channels)

Time conditioning (Adaptive Group Normalization, AdaGN):
────────────────────────────────────────────────────────
Each ResBlock applies AdaGN:
    y = GroupNorm(x)
    AdaGN(x, t) = scale(t) · y + shift(t)
where scale, shift ∈ ℝᶜ are predicted by a shared time-embedding MLP.
This is identical to the conditioning in DDPM (Ho et al., 2020) and
Stable Diffusion.

For 28×28 MNIST with C=32:
    Encoder feature maps: 28×28, 14×14, 7×7
    Parameters ≈ 400K  (fast to train on CPU, ~5 min per epoch)
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from diffusion_lab.nn.mlp import SinusoidalEmbedding

__all__ = ["SmallUNet"]


# ---------------------------------------------------------------------------
# Adaptive Group Normalization
# ---------------------------------------------------------------------------

class AdaGN(nn.Module):
    """
    Adaptive Group Normalization.

    Applies GroupNorm then modulates with time-dependent scale and shift:
        AdaGN(x, emb) = scale(emb) · GN(x) + shift(emb)

    Parameters
    ----------
    channels      : number of feature channels
    time_emb_dim  : dimension of the time embedding
    num_groups    : groups for GroupNorm (default 8)
    """

    def __init__(
        self,
        channels: int,
        time_emb_dim: int,
        num_groups: int = 8,
    ) -> None:
        super().__init__()
        self.norm  = nn.GroupNorm(num_groups=min(num_groups, channels),
                                  num_channels=channels,
                                  eps=1e-6, affine=False)
        # Project time embedding to (scale, shift) pair
        self.proj  = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, channels * 2),
        )

    def forward(self, x: Tensor, t_emb: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x     : (B, C, H, W)
        t_emb : (B, time_emb_dim)

        Returns
        -------
        out : (B, C, H, W)
        """
        # scale, shift: (B, C) each
        ss              = self.proj(t_emb)               # (B, 2C)
        scale, shift    = ss.chunk(2, dim=1)             # (B, C), (B, C)
        # Reshape for broadcasting over H, W
        scale = scale[:, :, None, None]                  # (B, C, 1, 1)
        shift = shift[:, :, None, None]
        return self.norm(x) * (1.0 + scale) + shift


# ---------------------------------------------------------------------------
# ResBlock
# ---------------------------------------------------------------------------

class ResBlock(nn.Module):
    """
    Residual block with optional downsampling (stride=2) and AdaGN conditioning.

    Structure:
        AdaGN → SiLU → Conv2d → AdaGN → SiLU → Dropout → Conv2d
        + residual projection if in_channels ≠ out_channels

    Parameters
    ----------
    in_channels   : input channels
    out_channels  : output channels
    time_emb_dim  : time embedding dimension
    stride        : 1 = same resolution, 2 = halve H and W
    dropout       : dropout probability (default 0 for small models)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        stride: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.adagn1 = AdaGN(in_channels, time_emb_dim)
        self.conv1  = nn.Conv2d(in_channels, out_channels,
                                kernel_size=3, stride=stride, padding=1, bias=False)
        self.adagn2 = AdaGN(out_channels, time_emb_dim)
        self.drop   = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.conv2  = nn.Conv2d(out_channels, out_channels,
                                kernel_size=3, stride=1, padding=1, bias=False)

        # Residual projection: match spatial resolution and channels
        if stride == 1 and in_channels == out_channels:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Conv2d(in_channels, out_channels,
                                  kernel_size=1, stride=stride, bias=False)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: Tensor, t_emb: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x     : (B, in_channels, H, W)
        t_emb : (B, time_emb_dim)

        Returns
        -------
        out : (B, out_channels, H', W')  where H'=H/stride
        """
        h = F.silu(self.adagn1(x, t_emb))
        h = self.conv1(h)
        h = F.silu(self.adagn2(h, t_emb))
        h = self.drop(h)
        h = self.conv2(h)
        return h + self.skip(x)


# ---------------------------------------------------------------------------
# Small U-Net
# ---------------------------------------------------------------------------

class SmallUNet(nn.Module):
    """
    Lightweight U-Net for 28×28 image generation.

    Suitable for MNIST diffusion experiments; trainable on CPU
    in ~5–20 minutes depending on hardware.

    Parameters
    ----------
    in_channels   : input image channels (1 for grayscale MNIST)
    out_channels  : output channels (same as in_channels for diffusion)
    base_channels : C in the architecture diagram (default 32)
    time_embed_dim: sinusoidal embedding dimension
    dropout       : dropout in ResBlocks (default 0.1)
    num_classes   : if set, adds a learned class embedding summed into the
                    time embedding for class-conditional generation.
                    Index 0 is reserved as the "null / unconditional" token
                    used for classifier-free guidance dropout.
                    Pass ``num_classes=K`` to support classes 1…K
                    (class 0 = unconditional).
    """

    def __init__(
        self,
        in_channels:    int = 1,
        out_channels:   int = 1,
        base_channels:  int = 32,
        time_embed_dim: int = 128,
        dropout:        float = 0.1,
        num_classes:    int | None = None,
    ) -> None:
        super().__init__()
        C  = base_channels
        E  = time_embed_dim

        # ── Time embedding ──────────────────────────────────────────────
        self.time_embed = nn.Sequential(
            SinusoidalEmbedding(E),
            nn.Linear(E, E * 4),
            nn.SiLU(),
            nn.Linear(E * 4, E),
        )
        te = E  # alias for ResBlock construction

        # ── Class conditioning (optional) ───────────────────────────────
        # num_classes + 1 rows: row 0 is the null/unconditional embedding.
        if num_classes is not None:
            self.class_embed = nn.Embedding(num_classes + 1, E)
            nn.init.normal_(self.class_embed.weight, std=0.02)
        else:
            self.class_embed = None

        # ── Encoder ─────────────────────────────────────────────────────
        # stem: (B,1,28,28) → (B,C,28,28)
        self.stem = nn.Conv2d(in_channels, C, kernel_size=3, padding=1, bias=False)

        # enc1: (B,C,28,28) → (B,2C,14,14)
        self.enc1 = ResBlock(C,   2*C, te, stride=2, dropout=dropout)

        # enc2: (B,2C,14,14) → (B,4C,7,7)
        self.enc2 = ResBlock(2*C, 4*C, te, stride=2, dropout=dropout)

        # ── Bottleneck ──────────────────────────────────────────────────
        self.mid1 = ResBlock(4*C, 4*C, te, dropout=dropout)
        self.mid2 = ResBlock(4*C, 4*C, te, dropout=dropout)

        # ── Decoder (with skip concat) ──────────────────────────────────
        # dec2: concat(mid, enc2_skip) (B,8C,7,7) → (B,2C,14,14)
        self.up2  = nn.Upsample(scale_factor=2, mode="nearest")
        self.dec2 = ResBlock(4*C + 4*C, 2*C, te, dropout=dropout)

        # dec1: concat(dec2, enc1_skip) (B,4C,14,14) → (B,C,28,28)
        self.up1  = nn.Upsample(scale_factor=2, mode="nearest")
        self.dec1 = ResBlock(2*C + 2*C, C, te, dropout=dropout)

        # ── Output head ─────────────────────────────────────────────────
        self.head = nn.Sequential(
            nn.GroupNorm(num_groups=min(8, C), num_channels=C, eps=1e-6),
            nn.SiLU(),
            nn.Conv2d(C, out_channels, kernel_size=3, padding=1),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, x: Tensor, t: Tensor, c: Tensor | None = None) -> Tensor:
        """
        Parameters
        ----------
        x : (B, in_channels, 28, 28) noisy image in [-1, 1]
        t : (B,) time variable (float, normalized to [0,1])
        c : (B,) optional integer class labels in {0, …, num_classes}.
            0 = unconditional / null token (for CFG training dropout).
            Only used when the model was built with ``num_classes`` set.

        Returns
        -------
        out : (B, out_channels, 28, 28)
        """
        # Time embedding
        t_emb = self.time_embed(t)          # (B, E)

        # Class conditioning: add class embedding to time embedding
        if self.class_embed is not None and c is not None:
            t_emb = t_emb + self.class_embed(c)   # (B, E)

        # Encoder
        h0    = self.stem(x)                # (B, C,   28, 28)
        h1    = self.enc1(h0, t_emb)        # (B, 2C,  14, 14)
        h2    = self.enc2(h1, t_emb)        # (B, 4C,   7,  7)

        # Bottleneck
        h     = self.mid1(h2, t_emb)        # (B, 4C,   7,  7)
        h     = self.mid2(h,  t_emb)

        # Decoder
        h     = torch.cat([h, h2], dim=1)   # (B, 8C,   7,  7)  ← skip from enc2
        h     = self.up2(h)                 # (B, 8C,  14, 14)
        h     = self.dec2(h, t_emb)         # (B, 2C,  14, 14)

        h     = torch.cat([h, h1], dim=1)   # (B, 4C,  14, 14)  ← skip from enc1
        h     = self.up1(h)                 # (B, 4C,  28, 28)
        h     = self.dec1(h, t_emb)         # (B,  C,  28, 28)

        return self.head(h)                 # (B, out, 28, 28)
