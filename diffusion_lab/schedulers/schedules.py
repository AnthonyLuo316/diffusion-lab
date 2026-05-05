"""
diffusion_lab/schedulers/schedules.py
Noise schedules for discrete-time diffusion models.

Math reference: quickguide Ch. 2

VP (variance-preserving) forward process:
    q(x_t | x_0) = N(√ᾱ_t x_0, (1 − ᾱ_t) I)
where ᾱ_t = ∏_{s=1}^t α_s = ∏_{s=1}^t (1 − β_s).

VE (variance-exploding) forward process (for Score SDE / SMLD):
    q(x_t | x_0) = N(x_0, σ_t² I)
with σ_t following a geometric schedule.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import torch
from torch import Tensor

__all__ = [
    "Schedule",
    "linear_vp_schedule",
    "cosine_vp_schedule",
    "ve_schedule",
]


@dataclass
class Schedule:
    """
    Noise schedule container.  All tensors have shape (T+1,) indexed from 0.

    Index convention
    ----------------
    - Index 0 corresponds to the clean data state:
        alpha_bar[0] = 1,  sigma[0] = 0
    - Index t ∈ {1, …, T} corresponds to the noisy state at step t.

    Fields
    ------
    T         : number of diffusion steps
    alpha_bar : (T+1,)  ᾱ_t = ∏_{s=1}^t (1−β_s),  ᾱ_0 = 1
    beta      : (T+1,)  β_t,  β_0 = 0 (unused sentinel)
    alpha     : (T+1,)  α_t = 1 − β_t
    sigma     : (T+1,)  √(1 − ᾱ_t),   noise std at step t
    log_snr   : (T+1,)  log(ᾱ_t / (1 − ᾱ_t)),  signal-to-noise ratio
    """

    T:         int
    alpha_bar: Tensor   # (T+1,)
    beta:      Tensor   # (T+1,)
    alpha:     Tensor   # (T+1,)
    sigma:     Tensor   # (T+1,)  = sqrt(1 − alpha_bar)
    log_snr:   Tensor   # (T+1,)

    def to(self, device) -> "Schedule":
        """Move all tensors to device, return self."""
        self.alpha_bar = self.alpha_bar.to(device)
        self.beta      = self.beta.to(device)
        self.alpha     = self.alpha.to(device)
        self.sigma     = self.sigma.to(device)
        self.log_snr   = self.log_snr.to(device)
        return self

    # ------------------------------------------------------------------
    # Convenience look-ups (accept integer or (B,) tensor indices)
    # ------------------------------------------------------------------

    def get(self, key: str, t: Tensor) -> Tensor:
        """
        Fetch schedule values at time indices t.

        Parameters
        ----------
        key : one of 'alpha_bar', 'beta', 'alpha', 'sigma', 'log_snr'
        t   : (B,) long tensor with values in {0, …, T}

        Returns
        -------
        vals : (B,) tensor
        """
        arr: Tensor = getattr(self, key)
        return arr[t]

    def broadcast(self, key: str, t: Tensor, ndim: int) -> Tensor:
        """
        Fetch schedule values at t and reshape to (B, 1, …, 1) for broadcasting.

        Parameters
        ----------
        key   : schedule field name
        t     : (B,) long tensor
        ndim  : total number of dimensions of the target tensor (e.g. 4 for images)

        Returns
        -------
        vals : (B, 1, …, 1)  with ndim−1 trailing singleton dims
        """
        vals = self.get(key, t)                 # (B,)
        shape = (-1,) + (1,) * (ndim - 1)
        return vals.view(shape)


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def _make_schedule(beta: Tensor) -> Schedule:
    """
    Build a Schedule dataclass from a beta sequence.

    beta : (T+1,) with beta[0] = 0 (sentinel), beta[1..T] are the actual values.
    """
    T = len(beta) - 1
    alpha     = 1.0 - beta
    alpha_bar = torch.cumprod(alpha, dim=0)     # ᾱ_t = ∏_{s=0}^t α_s
    # Clamp for numerical stability
    alpha_bar = alpha_bar.clamp(1e-8, 1.0)
    sigma     = (1.0 - alpha_bar).clamp(0.0).sqrt()
    log_snr   = torch.log(alpha_bar) - torch.log(1.0 - alpha_bar + 1e-8)
    return Schedule(
        T=T,
        alpha_bar=alpha_bar,
        beta=beta,
        alpha=alpha,
        sigma=sigma,
        log_snr=log_snr,
    )


def linear_vp_schedule(
    T: int = 1000,
    beta_start: float = 1e-4,
    beta_end:   float = 0.02,
) -> Schedule:
    """
    Linear β schedule from Ho et al. (2020) — DDPM.

    β_t linearly interpolates from beta_start to beta_end.

    Parameters
    ----------
    T          : number of diffusion steps
    beta_start : β_1  (first non-zero beta)
    beta_end   : β_T
    """
    # beta[0] = 0 (sentinel for t=0 / clean data), beta[1..T] are the schedule
    betas = torch.zeros(T + 1)
    betas[1:] = torch.linspace(beta_start, beta_end, T)
    return _make_schedule(betas)


def cosine_vp_schedule(
    T: int = 1000,
    s: float = 0.008,
) -> Schedule:
    """
    Cosine β schedule from Nichol & Dhariwal (2021) — improved DDPM.

    Defines ᾱ_t directly:
        f(t) = cos²(π/2 · (t/T + s) / (1 + s))
        ᾱ_t  = f(t) / f(0)
        β_t  = 1 − ᾱ_t / ᾱ_{t−1},  clipped to [0, 0.999]

    Parameters
    ----------
    T : number of steps
    s : small offset to prevent β_T from being too large (default 0.008)
    """
    steps = T + 1
    t = torch.linspace(0, T, steps)
    f = torch.cos((t / T + s) / (1 + s) * math.pi / 2) ** 2
    alpha_bar = f / f[0]
    alpha_bar = alpha_bar.clamp(1e-8, 1.0)

    # Derive beta from alpha_bar
    betas = torch.zeros(steps)
    betas[1:] = (1.0 - alpha_bar[1:] / alpha_bar[:-1]).clamp(0.0, 0.999)

    sigma   = (1.0 - alpha_bar).clamp(0.0).sqrt()
    log_snr = torch.log(alpha_bar) - torch.log(1.0 - alpha_bar + 1e-8)
    return Schedule(
        T=T,
        alpha_bar=alpha_bar,
        beta=betas,
        alpha=1.0 - betas,
        sigma=sigma,
        log_snr=log_snr,
    )


def ve_schedule(
    T: int = 1000,
    sigma_min: float = 0.01,
    sigma_max: float = 50.0,
) -> Schedule:
    """
    Variance-Exploding (VE) geometric noise schedule for SMLD / NCSN.

    σ_t = σ_min · (σ_max / σ_min)^{t / T},  t = 0, 1, …, T.

    In the VE convention ᾱ_t = 1 for all t (no signal scaling), so the
    forward process is:  q(x_t | x_0) = N(x_0, σ_t² I).

    Parameters
    ----------
    T         : number of noise levels
    sigma_min : smallest noise std  (≈ σ at t=0, near-clean)
    sigma_max : largest noise std   (≈ σ at t=T, pure noise)
    """
    t = torch.arange(T + 1, dtype=torch.float32)
    sigma = sigma_min * (sigma_max / sigma_min) ** (t / T)  # (T+1,)
    alpha_bar = torch.ones(T + 1)                            # no signal scaling
    beta      = torch.zeros(T + 1)
    alpha     = torch.ones(T + 1)
    log_snr   = -2.0 * sigma.log()                          # log(1/σ²)
    return Schedule(
        T=T,
        alpha_bar=alpha_bar,
        beta=beta,
        alpha=alpha,
        sigma=sigma,
        log_snr=log_snr,
    )
