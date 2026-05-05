"""
diffusion_lab/models/score_sde.py
Score-based generative models via SDEs — Song et al. (2021).

Math reference: quickguide Ch. 3

Two complementary views are implemented:

1.  NCSN (Noise Conditional Score Network, discrete)
    ─────────────────────────────────────────────────
    Noise levels: σ_1 > σ_2 > … > σ_L  (geometric schedule, VE)

    DSM loss (quickguide Thm 3.3, Vincent 2011):
        L_DSM(θ, σ_i) = E_{x_0, ε}[ ‖ s_θ(x_0 + σ_i ε, σ_i) + ε/σ_i ‖² ]

    NCSN multi-scale objective (quickguide Def 3.5):
        L_NCSN(θ) = Σ_i λ(σ_i) · L_DSM(θ, σ_i),   λ(σ_i) = σ_i²

    Optimal score: s*(x̃, σ) = −(x̃ − x_0)/σ² = −ε/σ

    Annealed Langevin dynamics (quickguide Def 3.4):
        For each σ_i (decreasing), run K steps:
            x_{k+1} = x_k + (α_i/2) · s_θ(x_k, σ_i) + √α_i · z_k,  z_k ~ N(0,I)

2.  VE-SDE (continuous-time, Song et al. 2021)
    ────────────────────────────────────────────
    Forward SDE:
        dx = √(d[σ²(t)]/dt) · dW

    Marginal:  q(x_t | x_0) = N(x_0, σ²(t) I)
    Geometric: σ(t) = σ_min · (σ_max/σ_min)^t,  t ∈ [0,1]

    Score matching loss (continuous DSM):
        L_VE = E_{t, x_0, ε}[ σ²(t) · ‖ s_θ(x_t, t) + ε/σ(t) ‖² ]

    Reverse SDE (Andersen–Léa / time-reversed Itô):
        dx = −g²(t) · s_θ(x, t) dt + g(t) dW̄
        g(t) = σ(t) · √(2 log(σ_max/σ_min))

    Probability flow ODE (quickguide §3.5):
        dx = −½ g²(t) · s_θ(x, t) dt
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
from torch import Tensor

__all__ = ["NCSN", "VE_SDE"]


# ---------------------------------------------------------------------------
# NCSN — Noise Conditional Score Network  (discrete)
# ---------------------------------------------------------------------------

class NCSN(nn.Module):
    """
    Noise Conditional Score Network with multi-scale DSM training objective
    and annealed Langevin dynamics sampler.

    Parameters
    ----------
    network    : score network s_θ(x, σ).
                 Must expose forward(x, t) → same shape as x,
                 where t carries the *normalized* σ in [0,1].
    sigmas     : (L,) decreasing sequence of noise levels (float tensor).
                 Typically geometric: σ_i = σ_min · (σ_max/σ_min)^{(L-i)/(L-1)}
    """

    def __init__(
        self,
        network: nn.Module,
        sigmas: Tensor,
    ) -> None:
        super().__init__()
        self.network = network
        self.register_buffer("sigmas", sigmas.float())

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def loss(self, x0: Tensor) -> Tensor:
        """
        Multi-scale NCSN DSM loss.

        For each sample in the batch, uniformly sample a noise level σ_i,
        perturb x_0, and compute the weighted DSM objective.

        L_NCSN = E_{i, x_0, ε}[ σ_i² · ‖ s_θ(x̃, σ_i) + ε/σ_i ‖² ]

        Returns
        -------
        loss : scalar Tensor
        """
        B      = x0.shape[0]
        device = x0.device
        L      = len(self.sigmas)

        # Sample random noise-level indices
        idx    = torch.randint(0, L, (B,), device=device)           # (B,)
        sigma  = self.sigmas[idx]                                    # (B,)

        # Perturb: x̃ = x_0 + σ_i · ε
        eps    = torch.randn_like(x0)
        sig_bc = sigma.view((-1,) + (1,) * (x0.ndim - 1))          # (B,1,…)
        x_tilde = x0 + sig_bc * eps

        # Score network: t_input = σ_i / σ_max (normalized to [0,1])
        sigma_max = self.sigmas[0]
        t_input   = sigma / sigma_max                                # (B,)
        score_pred = self.network(x_tilde, t_input)                  # (B,*)

        # DSM target: s*(x̃, σ) = −ε / σ
        target = -eps / sig_bc                                       # (B,*)

        # Per-sample MSE, weighted by σ²  (λ(σ) = σ²)
        mse  = ((score_pred - target) ** 2).flatten(1).mean(dim=1)  # (B,)
        wt   = sigma ** 2                                            # (B,)
        return (wt * mse).mean()

    # ------------------------------------------------------------------
    # Annealed Langevin dynamics sampler
    # ------------------------------------------------------------------

    @torch.no_grad()
    def annealed_langevin(
        self,
        shape: tuple,
        device: str | torch.device = "cpu",
        n_steps_per_level: int = 100,
        step_lr: float = 2e-5,
        return_chain: bool = False,
        chain_levels: list[int] | None = None,
    ) -> Tensor | list[Tensor]:
        """
        Annealed Langevin dynamics: run Langevin at each σ_i in decreasing order.

        Update rule (quickguide Def 3.4):
            α_i  = step_lr · σ_i² / σ_L²      (noise-adaptive step)
            x_{k+1} = x_k + (α_i/2) · s_θ(x_k, σ_i) + √α_i · z_k

        Parameters
        ----------
        shape             : sample shape (B, *), e.g. (512, 2)
        device            : target device
        n_steps_per_level : Langevin steps K at each noise level
        step_lr           : base step size (α_1 at highest σ)
        return_chain      : if True, return snapshots after each level
        chain_levels      : indices of noise levels to snapshot (default: all)

        Returns
        -------
        x_final : (B, *) samples after all levels
        chain   : list[Tensor]  (only if return_chain=True)
        """
        self.eval()
        device    = torch.device(device)
        sigma_max = self.sigmas[0]
        L         = len(self.sigmas)

        x = torch.randn(shape, device=device) * self.sigmas[0]  # start at highest σ

        chain = []
        if chain_levels is None:
            chain_levels = list(range(L))

        for i, sigma in enumerate(self.sigmas):
            sigma = sigma.to(device)
            # Adaptive step size: α_i = step_lr * (σ_i / σ_L)²
            alpha_i = step_lr * (sigma / self.sigmas[-1]) ** 2

            t_input = (sigma / sigma_max).unsqueeze(0).expand(shape[0])  # (B,)

            for _ in range(n_steps_per_level):
                score = self.network(x, t_input)                 # (B,*)
                drift = 0.5 * alpha_i * score
                noise = alpha_i.sqrt() * torch.randn_like(x)
                x     = x + drift + noise

            if return_chain and i in chain_levels:
                chain.append(x.clone().cpu())

        if return_chain:
            return chain
        return x


# ---------------------------------------------------------------------------
# VE-SDE — Variance-Exploding SDE  (continuous-time)
# ---------------------------------------------------------------------------

class VE_SDE(nn.Module):
    """
    Continuous-time VE score model with reverse SDE and probability flow ODE.

    Forward process: dx = g(t) dW,   g(t) = σ(t) √(2 log(σ_max/σ_min))
    Marginal:        q_t(x|x_0) = N(x_0, σ²(t) I)
    σ(t) = σ_min · (σ_max/σ_min)^t,   t ∈ [0,1]

    Parameters
    ----------
    network   : score network s_θ(x, t),  t ∈ [0,1]
    sigma_min : σ at t=0  (near-clean data)
    sigma_max : σ at t=1  (pure noise)
    """

    def __init__(
        self,
        network: nn.Module,
        sigma_min: float = 0.01,
        sigma_max: float = 50.0,
    ) -> None:
        super().__init__()
        self.network   = network
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self._log_ratio = math.log(sigma_max / sigma_min)

    # ------------------------------------------------------------------
    # SDE coefficients
    # ------------------------------------------------------------------

    def sigma(self, t: Tensor) -> Tensor:
        """σ(t) = σ_min · (σ_max/σ_min)^t,  shape: same as t."""
        return self.sigma_min * (self.sigma_max / self.sigma_min) ** t

    def g(self, t: Tensor) -> Tensor:
        """
        Diffusion coefficient g(t) = σ(t) · √(2 log(σ_max/σ_min)).
        This is the derivative form: d[σ²]/dt = 2 σ(t) σ'(t) = g²(t).
        """
        return self.sigma(t) * math.sqrt(2.0 * self._log_ratio)

    # ------------------------------------------------------------------
    # Training loss  (continuous DSM)
    # ------------------------------------------------------------------

    def loss(self, x0: Tensor) -> Tensor:
        """
        Continuous VE-SDE DSM loss:

        L = E_{t~U[0,1], x_0, ε}[ σ²(t) · ‖ s_θ(x_t, t) + ε/σ(t) ‖² ]

        The σ²(t) weighting follows Song et al. (2021), ensuring the loss
        is roughly scale-equalized across noise levels.

        Returns
        -------
        loss : scalar Tensor
        """
        B      = x0.shape[0]
        device = x0.device

        # Sample t ~ U[ε, 1] (avoid t=0 where σ→0)
        t      = torch.rand(B, device=device) * (1.0 - 1e-3) + 1e-3   # (B,)
        sigma_t = self.sigma(t)                                         # (B,)

        eps    = torch.randn_like(x0)
        sig_bc = sigma_t.view((-1,) + (1,) * (x0.ndim - 1))
        x_t    = x0 + sig_bc * eps

        score_pred = self.network(x_t, t)                               # (B,*)
        target     = -eps / sig_bc                                      # (B,*)

        mse = ((score_pred - target) ** 2).flatten(1).mean(dim=1)      # (B,)
        wt  = sigma_t ** 2                                              # (B,)
        return (wt * mse).mean()

    # ------------------------------------------------------------------
    # Reverse SDE sampler (Euler–Maruyama)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def reverse_sde_sample(
        self,
        shape: tuple,
        device: str | torch.device = "cpu",
        n_steps: int = 500,
        return_chain: bool = False,
        chain_stride: int = 50,
    ) -> Tensor | list[Tensor]:
        """
        Euler–Maruyama discretization of the reverse-time SDE.

        Reverse SDE (quickguide §3.5):
            dx = −g²(t) · s_θ(x, t) dt + g(t) dW̄

        Discretized (from t=1 to t=0):
            x_{t−Δt} = x_t + g²(t) · s_θ(x_t, t) · Δt + g(t) · √Δt · z

        Parameters
        ----------
        shape       : (B, *) output shape
        device      : target device
        n_steps     : number of Euler steps
        return_chain: return intermediate states
        chain_stride: snapshot every this many steps
        """
        self.eval()
        device = torch.device(device)
        dt     = 1.0 / n_steps
        t_seq  = torch.linspace(1.0, dt, n_steps, device=device)  # t from 1→Δt

        x     = torch.randn(shape, device=device) * self.sigma_max
        chain = []

        for step_i, t_val in enumerate(t_seq):
            t_batch = t_val.expand(shape[0])                      # (B,)
            score   = self.network(x, t_batch)                    # (B,*)
            g_t     = self.g(t_val)
            g2      = g_t ** 2

            drift = g2 * score * dt
            noise = g_t * math.sqrt(dt) * torch.randn_like(x)
            x     = x + drift + noise

            if return_chain and step_i % chain_stride == 0:
                chain.append(x.clone().cpu())

        if return_chain:
            chain.append(x.cpu())
            return chain
        return x

    # ------------------------------------------------------------------
    # Predictor-Corrector (PC) sampler
    # ------------------------------------------------------------------

    @torch.no_grad()
    def pc_sample(
        self,
        shape: tuple,
        device: str | torch.device = "cpu",
        n_steps: int = 500,
        n_corrector: int = 1,
        target_snr: float = 0.16,
        return_chain: bool = False,
        chain_stride: int = 50,
    ) -> Tensor | list[Tensor]:
        """
        Predictor-Corrector (PC) sampler (Song et al., 2021, Algorithm 2).

        Each denoising iteration:
            Predictor  — one Euler–Maruyama step of the reverse SDE.
            Corrector  — n_corrector Langevin MCMC steps at σ(t−Δt).

        Langevin step size (SNR-adaptive, Song et al. SMLD corrector):
            ε = 2 · (r · ‖z‖ / ‖s_θ‖)²   where r = target_snr.
        Update: x ← x + ε · s_θ(x,t) + √(2ε) · z

        Parameters
        ----------
        shape        : (B, *) output shape
        n_steps      : number of predictor (denoising) steps
        n_corrector  : Langevin correction steps per predictor step
        target_snr   : SNR target for adaptive step size (0.16 for MNIST)
        return_chain : whether to return intermediate states
        chain_stride : snapshot every this many steps
        """
        self.eval()
        device = torch.device(device)
        dt     = 1.0 / n_steps
        t_seq  = torch.linspace(1.0, dt, n_steps, device=device)

        x     = torch.randn(shape, device=device) * self.sigma_max
        chain = []

        for step_i, t_val in enumerate(t_seq):
            B       = shape[0]
            t_batch = t_val.expand(B)

            # ── Predictor: one reverse SDE Euler–Maruyama step ──────────
            score = self.network(x, t_batch)
            g_t   = self.g(t_val)
            drift = g_t ** 2 * score * dt
            noise = g_t * math.sqrt(dt) * torch.randn_like(x)
            x     = x + drift + noise

            # ── Corrector: Langevin MCMC at σ(t − Δt) ───────────────────
            t_cur       = (t_val - dt).clamp(min=1e-3)
            t_cur_batch = t_cur.expand(B)

            for _ in range(n_corrector):
                z       = torch.randn_like(x)
                score_c = self.network(x, t_cur_batch)

                # SNR-adaptive step size
                score_norm = score_c.flatten(1).norm(dim=1).mean()
                noise_norm = z.flatten(1).norm(dim=1).mean()
                eps        = 2.0 * (target_snr * noise_norm / (score_norm + 1e-8)) ** 2

                x = x + eps * score_c + (2.0 * eps).sqrt() * z

            if return_chain and step_i % chain_stride == 0:
                chain.append(x.clone().cpu())

        if return_chain:
            chain.append(x.cpu())
            return chain
        return x

    # ------------------------------------------------------------------
    # Probability flow ODE sampler (Euler)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def ode_sample(
        self,
        shape: tuple,
        device: str | torch.device = "cpu",
        n_steps: int = 200,
        return_chain: bool = False,
        chain_stride: int = 20,
    ) -> Tensor | list[Tensor]:
        """
        Euler discretization of the probability flow ODE.

        ODE (quickguide §3.5, half the diffusion coefficient):
            dx/dt = −½ g²(t) · s_θ(x, t)

        This ODE has the same marginals as the SDE but is deterministic.

        Parameters
        ----------
        shape       : (B, *) output shape
        n_steps     : number of Euler steps (fewer needed vs SDE)
        """
        self.eval()
        device = torch.device(device)
        dt     = 1.0 / n_steps
        t_seq  = torch.linspace(1.0, dt, n_steps, device=device)

        x     = torch.randn(shape, device=device) * self.sigma_max
        chain = []

        for step_i, t_val in enumerate(t_seq):
            t_batch = t_val.expand(shape[0])
            score   = self.network(x, t_batch)
            g_t     = self.g(t_val)
            drift   = 0.5 * g_t ** 2 * score * dt     # ODE: half the SDE drift
            x       = x + drift

            if return_chain and step_i % chain_stride == 0:
                chain.append(x.clone().cpu())

        if return_chain:
            chain.append(x.cpu())
            return chain
        return x
