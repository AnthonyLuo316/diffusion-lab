"""
diffusion_lab/nn/mlp.py
Time-conditioned MLP for 2-D generative model experiments.

Architecture
------------
    x ∈ ℝᵈ, t ∈ ℝ
        → sinusoidal_embed(t) ∈ ℝᴱ
        → Linear(d + E → H)  →  SiLU
        → [Linear(H → H)  →  SiLU] × (depth - 2)
        → Linear(H → d_out)
    output += x   (residual skip, useful for score / velocity heads)

The sinusoidal embedding follows the DDPM / Transformer convention:
    embed(t)_{2i}   = sin(t / 10000^{2i/E})
    embed(t)_{2i+1} = cos(t / 10000^{2i/E})
where t should be passed as a *normalized* scalar in [0, 1] (or any range
the user finds convenient; the embedding handles it).
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
from torch import Tensor

__all__ = ["SinusoidalEmbedding", "TimeMLP"]


class SinusoidalEmbedding(nn.Module):
    """
    Sinusoidal positional embedding for a scalar time variable t.

    Maps t: (B,) → embed: (B, dim).

    Parameters
    ----------
    dim       : embedding dimension (must be even)
    max_period: controls the maximum frequency (default 10 000, as in DDPM)
    """

    def __init__(self, dim: int = 128, max_period: float = 10_000.0) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"SinusoidalEmbedding dim must be even, got {dim}.")
        self.dim = dim
        self.max_period = max_period

    def forward(self, t: Tensor) -> Tensor:
        """
        Parameters
        ----------
        t : (B,) float tensor, any range (commonly [0, 1] or [0, T])

        Returns
        -------
        embed : (B, dim) float tensor
        """
        half = self.dim // 2
        # freqs shape: (half,)
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(half, device=t.device, dtype=t.dtype)
            / half
        )
        # args shape: (B, half)
        args = t[:, None] * freqs[None, :]
        embed = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return embed                        # (B, dim)


class TimeMLP(nn.Module):
    """
    Time-conditioned MLP for 2-D generative models.

    The network maps (x, t) → output of the same spatial dimension as x.
    A residual connection from x to the output stabilises score / velocity
    heads that are expected to be "corrections" to the input.

    Parameters
    ----------
    in_dim        : spatial input dimension (default 2 for 2-D data)
    out_dim       : output dimension (default = in_dim)
    hidden        : width of each hidden layer
    depth         : total number of linear layers (including input & output)
    time_embed_dim: dimension of the sinusoidal time embedding
    residual      : if True, add a skip x → output (default True)
    """

    def __init__(
        self,
        in_dim: int = 2,
        out_dim: int | None = None,
        hidden: int = 256,
        depth: int = 4,
        time_embed_dim: int = 128,
        residual: bool = True,
    ) -> None:
        super().__init__()
        if out_dim is None:
            out_dim = in_dim
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.residual = residual and (in_dim == out_dim)

        self.time_embed = SinusoidalEmbedding(time_embed_dim)

        # Build MLP layers
        # Input: concat [x, t_embed]  →  hidden
        # Middle: hidden → hidden  (depth - 2 layers)
        # Output: hidden → out_dim
        layers: list[nn.Module] = []
        in_features = in_dim + time_embed_dim
        for i in range(depth):
            out_features = out_dim if i == depth - 1 else hidden
            layers.append(nn.Linear(in_features, out_features))
            if i < depth - 1:
                layers.append(nn.SiLU())
            in_features = hidden
        self.net = nn.Sequential(*layers)

        # Residual projection (used only when in_dim != out_dim)
        if residual and in_dim != out_dim:
            self.proj = nn.Linear(in_dim, out_dim, bias=False)
        else:
            self.proj = None

        self._init_weights()

    def _init_weights(self) -> None:
        """Kaiming uniform init for all Linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                    nn.init.uniform_(m.bias, -bound, bound)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x : (B, in_dim) spatial input
        t : (B,) time variable (float, any range)

        Returns
        -------
        out : (B, out_dim)
        """
        t_emb = self.time_embed(t)              # (B, time_embed_dim)
        h = torch.cat([x, t_emb], dim=-1)       # (B, in_dim + time_embed_dim)
        out = self.net(h)                        # (B, out_dim)
        if self.residual:
            skip = x if self.proj is None else self.proj(x)
            out = out + skip
        return out
