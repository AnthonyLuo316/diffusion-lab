"""
diffusion_lab/models/edm.py
Elucidating the Design Space of Diffusion Models (EDM).

Reference: Karras et al. 2022 — "Elucidating the Design Space of Diffusion-Based
Generative Models", NeurIPS 2022. (https://arxiv.org/abs/2206.00364)

Theory
------
EDM reorganizes diffusion model training and sampling around the noise level σ as
the primary coordinate.  The clean-data marginal is:

    x = x₀ + σ·n,    n ~ N(0, I),    σ ∈ [σ_min, σ_max]

Preconditioning (Karras 2022, Sec. 5)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
A raw neural network F_θ(·) is poorly conditioned when called directly with
noisy input x and noise level σ.  EDM wraps F_θ in a preconditioned denoiser:

    D_θ(x, σ) = c_skip(σ)·x  +  c_out(σ)·F_θ(c_in(σ)·x, c_noise(σ))

where the four scalar preconditioning functions are chosen so that:
  - inputs and outputs have unit variance at every noise level,
  - the skip connection provides a near-optimal prior at large σ,
  - gradients flow through a well-scaled residual.

For the VP variant (used here, consistent with DDPM/DDIM in this codebase):

    σ_data = 0.5  (assumed std of the training data, i.e. images in [-1,1])

    c_skip(σ)  = σ_data² / (σ² + σ_data²)
    c_out(σ)   = σ · σ_data / sqrt(σ² + σ_data²)
    c_in(σ)    = 1 / sqrt(σ² + σ_data²)
    c_noise(σ) = (1/4) · ln(σ)            [log-scale for the time MLP]

Training loss (Karras 2022, Eq. 5)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Sample noise levels from a log-normal distribution during training:

    ln σ ~ N(P_mean, P_std²),    P_mean = -1.2,  P_std = 1.2

The loss for a single training example is:

    L(θ) = λ(σ) · ‖D_θ(x + σ·n, σ) − x‖²

where the weighting λ(σ) = (σ² + σ_data²) / (σ · σ_data)² balances contribution
across noise levels.

Heun 2nd-order ODE sampler (Karras 2022, Alg. 2)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
EDM uses a 2nd-order Heun method (predictor-corrector) to integrate the
probability flow ODE:

    dx/dσ = (x − D_θ(x, σ)) / σ    (σ is the "time" variable, decreasing)

Starting from x_T ~ N(0, σ_max²·I) and integrating from σ_max → σ_min:

  For each step σ_i → σ_{i+1} (with σ_{i+1} < σ_i):

    d_i    = (x_i − D_θ(x_i, σ_i)) / σ_i        [derivative at σ_i]
    x̂     = x_i + (σ_{i+1} − σ_i) · d_i          [Euler step]
    d̂     = (x̂ − D_θ(x̂, σ_{i+1})) / σ_{i+1}   [derivative at σ_{i+1}]
    x_{i+1} = x_i + (σ_{i+1} − σ_i) · (d_i + d̂)/2  [Heun correction]

  (At the final step to σ=0, the Heun correction is skipped and the output is
   simply D_θ(x, σ_last), since D_θ at σ→0 is the identity.)

Stochastic sampler (with churn, Karras 2022, Alg. 2 extended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Inject noise at each step before the Heun update:

    σ̂_i = σ_i · (1 + γ_i)^{1/2},  γ_i = min(S_churn / N, sqrt(2)−1)
    x̂_i = x_i + sqrt(σ̂_i² − σ_i²) · randn_like(x_i)

then use σ̂_i instead of σ_i as the starting noise level.

Noise schedule
~~~~~~~~~~~~~~~
The default EDM σ schedule is a power-law spacing:

    σ_i = (σ_max^(1/ρ) + i/(N-1) · (σ_min^(1/ρ) − σ_max^(1/ρ)))^ρ

where ρ = 7 concentrates steps near σ_min (where quality matters most).
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
from torch import Tensor

__all__ = ["EDMPrecon", "EDMSampler"]


# ---------------------------------------------------------------------------
# Preconditioning wrapper
# ---------------------------------------------------------------------------

class EDMPrecon(nn.Module):
    """
    EDM-style preconditioned denoiser wrapping a raw network F_θ.

    Given a network F_θ that maps (scaled_input, c_noise) → output, this
    module implements:

        D_θ(x, σ) = c_skip(σ)·x + c_out(σ)·F_θ(c_in(σ)·x, c_noise(σ))

    Parameters
    ----------
    network   : raw network F_θ with signature forward(x, t, c=None)
                where t = c_noise(σ) is the log-scale noise coordinate
    sigma_data: assumed standard deviation of training data (default 0.5
                for images normalised to [-1, 1])

    Notes
    -----
    The four preconditioning scalars are:

        c_skip(σ)  = σ_data² / (σ² + σ_data²)
        c_out(σ)   = σ · σ_data / sqrt(σ² + σ_data²)
        c_in(σ)    = 1 / sqrt(σ² + σ_data²)
        c_noise(σ) = (1/4) · ln(σ)
    """

    def __init__(self, network: nn.Module, sigma_data: float = 0.5) -> None:
        super().__init__()
        self.network    = network
        self.sigma_data = sigma_data

    # ------------------------------------------------------------------
    # Preconditioning scalars
    # ------------------------------------------------------------------

    def c_skip(self, sigma: Tensor) -> Tensor:
        """Skip-connection weight: σ_data² / (σ² + σ_data²)."""
        sd2 = self.sigma_data ** 2
        return sd2 / (sigma ** 2 + sd2)

    def c_out(self, sigma: Tensor) -> Tensor:
        """Output scale: σ · σ_data / sqrt(σ² + σ_data²)."""
        sd2 = self.sigma_data ** 2
        return sigma * self.sigma_data / (sigma ** 2 + sd2).sqrt()

    def c_in(self, sigma: Tensor) -> Tensor:
        """Input scale: 1 / sqrt(σ² + σ_data²)."""
        sd2 = self.sigma_data ** 2
        return 1.0 / (sigma ** 2 + sd2).sqrt()

    def c_noise(self, sigma: Tensor) -> Tensor:
        """
        Log-scale noise coordinate: (1/4) · ln(σ).

        Maps σ ∈ (0, ∞) to ℝ; the factor 1/4 keeps values in a comfortable
        range for the sinusoidal embedding.
        """
        return 0.25 * sigma.log()

    # ------------------------------------------------------------------
    # Loss weighting
    # ------------------------------------------------------------------

    def loss_weight(self, sigma: Tensor) -> Tensor:
        """
        Per-sample loss weight λ(σ) = (σ² + σ_data²) / (σ · σ_data)².

        Derived by requiring the expected gradient magnitude to be roughly
        constant across noise levels.
        """
        sd2 = self.sigma_data ** 2
        return (sigma ** 2 + sd2) / (sigma * self.sigma_data) ** 2

    # ------------------------------------------------------------------
    # Forward: denoising
    # ------------------------------------------------------------------

    def forward(
        self,
        x:     Tensor,
        sigma: Tensor,
        c:     Tensor | None = None,
    ) -> Tensor:
        """
        Preconditioned denoiser D_θ(x, σ, c).

        Parameters
        ----------
        x     : (B, C, H, W) noisy image x = x₀ + σ·n
        sigma : (B,) noise levels (one per sample in the batch)
        c     : (B,) optional class labels (passed through to network)

        Returns
        -------
        denoised : (B, C, H, W) estimate of x₀
        """
        # Reshape scalar multipliers for broadcasting: (B, 1, 1, 1)
        def _b(v: Tensor) -> Tensor:
            return v.view(-1, *([1] * (x.ndim - 1)))

        cs  = _b(self.c_skip(sigma))
        co  = _b(self.c_out(sigma))
        ci  = _b(self.c_in(sigma))
        cn  = self.c_noise(sigma)        # (B,) — fed as "time" to network

        # Scaled network input and log-scale time coordinate
        x_in = ci * x                    # (B, C, H, W)

        # Network forward
        F_out = self.network(x_in, cn, c)

        return cs * x + co * F_out

    # ------------------------------------------------------------------
    # Training loss (log-normal σ sampling)
    # ------------------------------------------------------------------

    def loss(
        self,
        x0:      Tensor,
        labels:  Tensor | None = None,
        P_mean:  float = -1.2,
        P_std:   float = 1.2,
    ) -> Tensor:
        """
        EDM training loss with log-normal noise schedule.

        Samples σ ~ log-Normal(P_mean, P_std²) per training example, then
        computes the weighted MSE:

            L = λ(σ) · ‖D_θ(x₀ + σ·n, σ) − x₀‖²

        Parameters
        ----------
        x0     : (B, C, H, W) clean training images in [-1, 1]
        labels : (B,) optional integer class labels (for conditional training)
        P_mean : mean of ln σ (default −1.2 from Karras 2022)
        P_std  : std of ln σ (default 1.2 from Karras 2022)

        Returns
        -------
        loss : scalar Tensor
        """
        B      = x0.shape[0]
        device = x0.device

        # Sample noise levels: ln σ ~ N(P_mean, P_std²)
        ln_sigma = P_mean + P_std * torch.randn(B, device=device)
        sigma    = ln_sigma.exp()                  # (B,)

        # Corrupt input
        noise = torch.randn_like(x0)
        x_noisy = x0 + sigma.view(-1, *([1] * (x0.ndim - 1))) * noise  # (B, C, H, W)

        # Preconditioned denoiser prediction
        x0_hat = self.forward(x_noisy, sigma, labels)

        # Per-sample MSE
        mse = ((x0_hat - x0) ** 2).flatten(1).mean(dim=1)   # (B,)

        # Loss weighting λ(σ)
        wt = self.loss_weight(sigma)                          # (B,)

        return (wt * mse).mean()


# ---------------------------------------------------------------------------
# EDM noise schedule
# ---------------------------------------------------------------------------

def edm_sigma_schedule(
    num_steps: int,
    sigma_min: float = 0.002,
    sigma_max: float = 80.0,
    rho:       float = 7.0,
    device:    str | torch.device = "cpu",
) -> Tensor:
    """
    EDM power-law noise schedule (Karras 2022, Eq. 5).

    Produces a decreasing sequence of noise levels:

        σ_i = (σ_max^(1/ρ) + i/(N−1) · (σ_min^(1/ρ) − σ_max^(1/ρ)))^ρ

    for i = 0, 1, …, N−1, with an appended σ = 0 at the end.

    Parameters
    ----------
    num_steps : number of sampling steps N
    sigma_min : smallest noise level (default 0.002)
    sigma_max : largest noise level (default 80, roughly N(0,σ²) ≈ pure noise)
    rho       : curvature parameter (default 7 concentrates near σ_min)
    device    : target device

    Returns
    -------
    sigmas : (num_steps + 1,) tensor, sigmas[-1] = 0
    """
    # Compute in float64 on CPU for numerical precision, then move to device
    steps   = torch.arange(num_steps, dtype=torch.float64)
    inv_rho = 1.0 / rho
    sigmas  = (
        sigma_max ** inv_rho
        + steps / (num_steps - 1) * (sigma_min ** inv_rho - sigma_max ** inv_rho)
    ) ** rho
    # Append σ=0 as the terminal state
    sigmas  = torch.cat([sigmas, sigmas.new_zeros(1)])
    return sigmas.float().to(device)  # convert to float32, move to target device


# ---------------------------------------------------------------------------
# EDM Heun Sampler
# ---------------------------------------------------------------------------

class EDMSampler(nn.Module):
    """
    2nd-order Heun sampler for EDM (Karras 2022, Algorithm 2).

    Integrates the probability flow ODE

        dx/dσ = (x − D_θ(x, σ)) / σ

    from σ_max to σ_min using 2nd-order Heun steps.

    Optionally adds stochastic churn noise between steps (Sec. 5 of the paper).

    Parameters
    ----------
    model      : EDMPrecon (or any denoiser with forward(x, sigma, c))
    sigma_min  : minimum noise level (default 0.002)
    sigma_max  : maximum noise level (default 80.0)
    rho        : schedule curvature (default 7)
    S_churn    : churn noise injection strength (0 = deterministic)
    S_min      : minimum σ for churn (default 0)
    S_max      : maximum σ for churn (default inf)
    S_noise    : std scaling of injected noise (default 1.003)
    """

    def __init__(
        self,
        model:     EDMPrecon,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        rho:       float = 7.0,
        S_churn:   float = 0.0,
        S_min:     float = 0.0,
        S_max:     float = float("inf"),
        S_noise:   float = 1.003,
    ) -> None:
        super().__init__()
        self.model     = model
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho       = rho
        self.S_churn   = S_churn
        self.S_min     = S_min
        self.S_max     = S_max
        self.S_noise   = S_noise

    @torch.no_grad()
    def sample(
        self,
        shape:       tuple,
        num_steps:   int = 50,
        labels:      Tensor | None = None,
        device:      str | torch.device = "cpu",
        return_chain: bool = False,
        chain_stride: int = 10,
    ) -> Tensor | list[Tensor]:
        """
        Generate samples using 2nd-order Heun integration.

        Parameters
        ----------
        shape        : (B, C, H, W)
        num_steps    : number of ODE integration steps N
        labels       : (B,) optional class labels
        device       : target device
        return_chain : if True, return list of intermediate states
        chain_stride : snapshot every this many steps

        Returns
        -------
        x  : (B, C, H, W) final samples  OR
        chain : list[Tensor]  (only if return_chain=True)
        """
        self.model.eval()
        device = torch.device(device)

        # Build sigma schedule: σ₀ > σ₁ > … > σ_N = 0
        sigmas = edm_sigma_schedule(
            num_steps, self.sigma_min, self.sigma_max, self.rho, device=device
        )  # (N+1,)

        # Initial sample x₀ ~ N(0, σ_max²·I)
        x = torch.randn(shape, device=device) * sigmas[0]
        if labels is not None:
            labels = labels.to(device)

        chain = [x.clone().cpu()] if return_chain else []

        for i in range(num_steps):
            sigma_i   = sigmas[i]
            sigma_next = sigmas[i + 1]

            # ----------------------------------------------------------
            # Optional stochastic churn (inject noise)
            # ----------------------------------------------------------
            if self.S_churn > 0 and self.S_min <= sigma_i <= self.S_max:
                gamma    = min(self.S_churn / num_steps, math.sqrt(2) - 1)
                sigma_hat = sigma_i * (1.0 + gamma)
                x = x + (sigma_hat ** 2 - sigma_i ** 2).sqrt() * self.S_noise * torch.randn_like(x)
            else:
                sigma_hat = sigma_i

            # ----------------------------------------------------------
            # Heun step
            # ----------------------------------------------------------
            s_i_batch = sigma_hat.expand(shape[0])     # (B,)

            # 1st derivative: d = (x − D_θ(x, σ)) / σ
            D_i = self.model(x, s_i_batch, labels)
            d_i = (x - D_i) / sigma_hat

            # Euler predictor: x̂ = x + (σ_{i+1} − σ_hat) · d_i
            x_hat = x + (sigma_next - sigma_hat) * d_i

            if sigma_next > 0:
                # 2nd derivative at x̂, σ_{i+1}
                s_next_batch = sigma_next.expand(shape[0])   # (B,)
                D_hat = self.model(x_hat, s_next_batch, labels)
                d_hat = (x_hat - D_hat) / sigma_next

                # Heun corrector: average of two derivatives
                x = x + (sigma_next - sigma_hat) * 0.5 * (d_i + d_hat)
            else:
                # Terminal step: use the denoiser output directly
                x = D_i

            if return_chain and i % chain_stride == 0:
                chain.append(x.clone().cpu())

        if return_chain:
            chain.append(x.clone().cpu())
            return chain
        return x

    @torch.no_grad()
    def sample_euler(
        self,
        shape:     tuple,
        num_steps: int = 50,
        labels:    Tensor | None = None,
        device:    str | torch.device = "cpu",
    ) -> Tensor:
        """
        1st-order Euler sampler (for comparison / ablation).

        Uses the same σ schedule but only the predictor step (no Heun correction).
        """
        self.model.eval()
        device = torch.device(device)

        sigmas = edm_sigma_schedule(
            num_steps, self.sigma_min, self.sigma_max, self.rho, device=device
        )
        x = torch.randn(shape, device=device) * sigmas[0]
        if labels is not None:
            labels = labels.to(device)

        for i in range(num_steps):
            sigma_i    = sigmas[i]
            sigma_next = sigmas[i + 1]
            s_batch    = sigma_i.expand(shape[0])

            D_i = self.model(x, s_batch, labels)

            if sigma_next > 0:
                d_i = (x - D_i) / sigma_i
                x   = x + (sigma_next - sigma_i) * d_i
            else:
                x = D_i

        return x
