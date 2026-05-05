"""
diffusion_lab/models/inverse.py
Linear inverse-problem operators and the DPS sampler.

Reference: Chung et al. 2022 — "Diffusion Posterior Sampling for General
Noisy Inverse Problems", ICLR 2023.  (https://arxiv.org/abs/2209.14687)

Theory
------
We observe y = A(x₀) + η, where:
  - x₀ ∈ ℝᴰ  is the unknown clean image
  - A : ℝᴰ → ℝᴹ  is a known linear (or differentiable) forward operator
  - η ~ N(0, σ_y² I)  is additive Gaussian noise

DPS approximates the intractable posterior score by combining:
  1. The unconditional diffusion score  ∇_{x_t} log p_t(x_t)
  2. A likelihood gradient via the Tweedie approximation:

     ∇_{x_t} log p(y | x_t)  ≈  -ζ/‖r‖ · ∇_{x_t} ‖r‖²

  where r = y - A(x̂₀) is the measurement residual and x̂₀ is the Tweedie
  denoised estimate  x̂₀ = (x_t - σ_t · ε̂_θ(x_t, t)) / √ᾱ_t.

The guided reverse step is:

    x_{t-1} = DDPM_step(x_t, t)  -  (ζ/‖r‖) · ∇_{x_t} ‖r‖²

where the gradient is computed via autograd through the denoiser network.

Operators provided
------------------
  RandomMaskOperator    — random-pixel inpainting (iid Bernoulli mask)
  BoxMaskOperator       — rectangular centre-crop inpainting
  GaussianBlurOperator  — isotropic Gaussian convolution
  SuperResolutionOperator — bicubic downsampling / upsampling

All operators share the `LinearOperator` base class interface:
    A(x)   — apply the forward operator
    A_pinv(y, x_shape) — pseudo-inverse / adjoint (for display only)
"""

from __future__ import annotations

import math
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

__all__ = [
    "LinearOperator",
    "RandomMaskOperator",
    "BoxMaskOperator",
    "GaussianBlurOperator",
    "SuperResolutionOperator",
    "dps_sample",
]


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class LinearOperator:
    """
    Abstract base for differentiable forward operators A : ℝᴰ → ℝᴹ.

    All subclasses must implement ``forward(x)``.  The pseudo-inverse
    ``A_pinv`` defaults to the transpose (valid for orthogonal operators).
    """

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    def A_pinv(self, y: Tensor, x_shape: tuple) -> Tensor:
        """
        Pseudo-inverse / adjoint A†.

        Default implementation returns the masked/blurred output zero-padded
        back to x_shape.  Override in subclasses for exact pinverse.
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Random-pixel inpainting
# ---------------------------------------------------------------------------

class RandomMaskOperator(LinearOperator):
    """
    Random-pixel inpainting: A(x) = mask ⊙ x.

    Parameters
    ----------
    keep_prob : probability of keeping each pixel (default 0.5)
    seed      : random seed for mask generation (reproducible across calls)
    """

    def __init__(self, keep_prob: float = 0.5, seed: int = 0) -> None:
        self.keep_prob = keep_prob
        self.seed      = seed
        self._mask: Tensor | None = None

    def _get_mask(self, x: Tensor) -> Tensor:
        if self._mask is None or self._mask.shape != x.shape:
            gen = torch.Generator(device=x.device).manual_seed(self.seed)
            self._mask = (
                torch.rand(x.shape, generator=gen, device=x.device) < self.keep_prob
            ).float()
        return self._mask.to(x.device)

    def forward(self, x: Tensor) -> Tensor:
        """Apply mask: corrupted(x) = mask ⊙ x."""
        return x * self._get_mask(x)

    def A_pinv(self, y: Tensor, x_shape: tuple) -> Tensor:
        """For a mask operator, A† = A (self-adjoint)."""
        return y   # already in image space


# ---------------------------------------------------------------------------
# Box (centre-crop) inpainting
# ---------------------------------------------------------------------------

class BoxMaskOperator(LinearOperator):
    """
    Centre-box inpainting: zeros out a rectangular region.

    Parameters
    ----------
    box_size : side length of the square masked region (pixels)
    """

    def __init__(self, box_size: int = 14) -> None:
        self.box_size = box_size

    def forward(self, x: Tensor) -> Tensor:
        """Zero out the centre box_size × box_size region."""
        H, W  = x.shape[-2], x.shape[-1]
        y     = x.clone()
        r0    = (H - self.box_size) // 2
        c0    = (W - self.box_size) // 2
        y[..., r0:r0 + self.box_size, c0:c0 + self.box_size] = 0.0
        return y

    def A_pinv(self, y: Tensor, x_shape: tuple) -> Tensor:
        return y


# ---------------------------------------------------------------------------
# Gaussian blur
# ---------------------------------------------------------------------------

class GaussianBlurOperator(LinearOperator):
    """
    Isotropic Gaussian convolution: A(x) = G_σ * x.

    Parameters
    ----------
    kernel_size : filter size (must be odd)
    sigma       : Gaussian standard deviation (pixels)
    channels    : number of image channels (for depthwise conv)
    """

    def __init__(
        self,
        kernel_size: int = 7,
        sigma: float = 2.0,
        channels: int = 1,
    ) -> None:
        self.kernel_size = kernel_size
        self.sigma       = sigma
        self.channels    = channels
        self._kernel: Tensor | None = None

    def _get_kernel(self, device: torch.device) -> Tensor:
        if self._kernel is None:
            ks  = self.kernel_size
            ax  = torch.arange(ks, dtype=torch.float32) - ks // 2
            g1d = torch.exp(-ax ** 2 / (2 * self.sigma ** 2))
            g2d = torch.outer(g1d, g1d)
            g2d = g2d / g2d.sum()
            # (C_out, C_in/groups, kH, kW)
            self._kernel = g2d.expand(self.channels, 1, ks, ks)
        return self._kernel.to(device)

    def forward(self, x: Tensor) -> Tensor:
        """Apply Gaussian blur (depthwise convolution)."""
        k   = self._get_kernel(x.device)
        pad = self.kernel_size // 2
        return F.conv2d(x, k, padding=pad, groups=self.channels)

    def A_pinv(self, y: Tensor, x_shape: tuple) -> Tensor:
        """A† for blur is also blur (symmetric kernel), used for display only."""
        return self.forward(y)


# ---------------------------------------------------------------------------
# Super-resolution (bicubic downsampling)
# ---------------------------------------------------------------------------

class SuperResolutionOperator(LinearOperator):
    """
    Bicubic downsampling: A(x) = downsample(x, 1/scale).

    The pseudo-inverse upsamples back to the original resolution.

    Parameters
    ----------
    scale : downsampling factor (integer; e.g. 4 → 7×7 from 28×28 MNIST)
    """

    def __init__(self, scale: int = 4) -> None:
        self.scale = scale

    def forward(self, x: Tensor) -> Tensor:
        """Bicubic downsampling by `scale`."""
        B, C, H, W = x.shape
        return F.interpolate(
            x,
            size=(H // self.scale, W // self.scale),
            mode="bicubic",
            align_corners=False,
        )

    def A_pinv(self, y: Tensor, x_shape: tuple) -> Tensor:
        """Bicubic upsampling back to original resolution."""
        _, _, H, W = x_shape
        return F.interpolate(y, size=(H, W), mode="bicubic", align_corners=False)


# ---------------------------------------------------------------------------
# DPS Sampler
# ---------------------------------------------------------------------------

def dps_sample(
    ddpm,
    y:            Tensor,
    operator_A:   LinearOperator,
    zeta:         float = 1.0,
    device:       str | torch.device = "cpu",
    sample_shape: tuple | None = None,
    return_chain: bool = False,
    chain_stride: int = 100,
    verbose:      bool = False,
) -> Tensor | list[Tensor]:
    """
    Diffusion Posterior Sampling (DPS) — Chung et al. 2022.

    Runs the DDPM reverse process with a per-step likelihood gradient correction:

        x_{t-1} = DDPM_step(x_t, t)  −  (ζ / ‖r‖) · ∇_{x_t} ‖r‖²

    where r = y − A(x̂₀) is the measurement residual and x̂₀ is obtained via
    the Tweedie formula from the network's noise prediction.

    Parameters
    ----------
    ddpm         : trained DDPM model (must expose .network, .schedule, .prediction)
    y            : (B, ...) measurement tensor on `device`.  The shape of y is
                   used to determine the sample shape unless ``sample_shape`` is
                   given explicitly.
    operator_A   : forward operator A (must be differentiable w.r.t. x₀)
    zeta         : step-size for the likelihood gradient (default 1.0)
    device       : target device
    sample_shape : explicit ``(B, C, H, W)`` for the latent image.  Required
                   when measurement and image spaces differ, e.g. super-resolution
                   where y has shape (B, 1, 7, 7) but x₀ should be (B, 1, 28, 28).
                   If ``None`` (default), inferred from ``y.shape``.
    return_chain : if True, return list of intermediate snapshots
    chain_stride : snapshot every this many reverse steps
    verbose      : print progress every 100 steps

    Returns
    -------
    x0    : (B, C, H, W) posterior sample  OR
    chain : list[Tensor]  (only if return_chain=True)

    Notes
    -----
    The gradient ∇_{x_t} ‖r‖² is computed by differentiating through the
    network with a fresh ``requires_grad_(True)`` leaf at each step.  The
    outer DDPM step loop runs under ``torch.no_grad()`` except for the
    gradient sub-computation.
    """
    device   = torch.device(device)
    sched    = ddpm.schedule
    T        = sched.T
    if sample_shape is not None:
        B, C, H, W = sample_shape
    else:
        B, C, H, W = y.shape
    y        = y.to(device)

    x     = torch.randn(B, C, H, W, device=device)
    chain = [x.clone().cpu()] if return_chain else []

    for t in range(T, 0, -1):
        t_tensor = torch.full((B,), t, dtype=torch.long, device=device)
        t_norm   = t_tensor.float() / T

        ab_t  = sched.alpha_bar[t].to(device)
        sig_t = sched.sigma[t].to(device)

        # ── Gradient sub-step: enable autograd through the denoiser ───────
        x_in = x.detach().requires_grad_(True)

        eps_hat = ddpm.network(x_in, t_norm)

        if ddpm.prediction == "epsilon":
            x0_hat = (x_in - sig_t * eps_hat) / ab_t.sqrt()
        elif ddpm.prediction == "x0":
            x0_hat = eps_hat
        else:   # "v"
            x0_hat = ab_t.sqrt() * x_in - sig_t * eps_hat

        x0_hat = x0_hat.clamp(-5.0, 5.0)

        # Measurement residual and likelihood loss ‖y − A(x̂₀)‖²
        residual  = y - operator_A(x0_hat)
        loss_ll   = (residual ** 2).flatten(1).sum(dim=1).mean()
        loss_ll.backward()

        grad_xt  = x_in.grad.detach()                         # (B, C, H, W)
        norm_r   = residual.detach().flatten(1).norm(dim=1)   # (B,)
        norm_r   = norm_r.mean().clamp(min=1e-8)

        # ── Standard DDPM ancestral step (no grad) ────────────────────────
        with torch.no_grad():
            ab_tm1  = sched.alpha_bar[t - 1] if t > 1 else torch.tensor(1.0)
            beta_t  = sched.beta[t].to(device)
            alpha_t = sched.alpha[t].to(device)

            c1 = ab_tm1.sqrt().to(device) * beta_t / (1.0 - ab_t)
            c2 = alpha_t.sqrt() * (1.0 - ab_tm1.to(device)) / (1.0 - ab_t)

            def _r(v):
                return v.view(-1, *([1] * (x.ndim - 1))).expand_as(x)

            mu = _r(c1) * x0_hat.detach() + _r(c2) * x.detach()

            if t > 1:
                beta_tilde = beta_t * (1.0 - ab_tm1.to(device)) / (1.0 - ab_t)
                x_prev = mu + _r(beta_tilde.sqrt()) * torch.randn_like(x)
            else:
                x_prev = mu

            # ── DPS likelihood gradient correction ───────────────────────
            x_prev = x_prev - (zeta / norm_r) * grad_xt
            x = x_prev

        if return_chain and t % chain_stride == 0:
            chain.append(x.clone().cpu())
        if verbose and t % 100 == 0:
            print(f'  t={t:4d}  ‖r‖={norm_r.item():.4f}')

    if return_chain:
        chain.append(x.clone().cpu())
        return chain
    return x
