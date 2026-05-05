"""
diffusion_lab/models/ddpm.py
Denoising Diffusion Probabilistic Model — Ho et al. (2020).

Math reference: quickguide Ch. 2

Forward process (VP)
--------------------
    q(x_t | x_{t-1}) = N(√α_t x_{t-1}, β_t I)

Marginal (closed form, quickguide Thm 2.1):
    q(x_t | x_0) = N(√ᾱ_t x_0, (1−ᾱ_t) I)
    ↔  x_t = √ᾱ_t x_0 + √(1−ᾱ_t) ε,  ε ∼ N(0,I)

Denoising posterior (tractable given x_0, quickguide §2.3):
    q(x_{t-1} | x_t, x_0) = N(μ̃_t(x_t, x_0), β̃_t I)
    μ̃_t = (√ᾱ_{t-1} β_t x_0  +  √α_t (1−ᾱ_{t-1}) x_t) / (1−ᾱ_t)
    β̃_t = β_t (1−ᾱ_{t-1}) / (1−ᾱ_t)

Three prediction targets (Tweedie triad, quickguide Thm 2.2):
    ε-prediction:  L_ε  = E‖ε − ε_θ(x_t, t)‖²
    x_0-prediction: L_x = E‖x_0 − x̂_0(x_t, t)‖²
    v-prediction:  L_v  = E‖v − v_θ(x_t, t)‖²
        where v = √ᾱ_t ε − √(1−ᾱ_t) x_0  (velocity parameterization)

Conversion between targets via Tweedie:
    x̂_0 = (x_t − √(1−ᾱ_t) ε_θ) / √ᾱ_t
    ε̂   = (x_t − √ᾱ_t x̂_0) / √(1−ᾱ_t)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from diffusion_lab.schedulers.schedules import Schedule

__all__ = ["DDPM"]


class DDPM(nn.Module):
    """
    DDPM forward process, training loss, and ancestral sampler.

    Parameters
    ----------
    network    : time-conditioned network (e.g. TimeMLP or SmallUNet).
                 Must expose forward(x, t_normalized) → same shape as x.
    schedule   : a Schedule object (use linear_vp_schedule or cosine_vp_schedule)
    prediction : target parameterization: 'epsilon', 'x0', or 'v'
    loss_weight: 'uniform'  — equal weight on all t  (DDPM default)
                 'snr'      — weight by SNR  (often better for x0-pred)
                 'min_snr'  — min-SNR weighting (Hang et al. 2023)
    min_snr_gamma: γ for min-SNR weighting (default 5)
    """

    PREDICTION_TYPES = ("epsilon", "x0", "v")

    def __init__(
        self,
        network: nn.Module,
        schedule: Schedule,
        prediction: str = "epsilon",
        loss_weight: str = "uniform",
        min_snr_gamma: float = 5.0,
    ) -> None:
        super().__init__()
        if prediction not in self.PREDICTION_TYPES:
            raise ValueError(f"prediction must be one of {self.PREDICTION_TYPES}")
        self.network       = network
        self.schedule      = schedule
        self.prediction    = prediction
        self.loss_weight   = loss_weight
        self.min_snr_gamma = min_snr_gamma

    # ------------------------------------------------------------------
    # Forward process utilities
    # ------------------------------------------------------------------

    def q_sample(
        self,
        x0: Tensor,
        t: Tensor,
        noise: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """
        Sample x_t ~ q(x_t | x_0) in one shot (closed-form marginal).

        x_t = √ᾱ_t · x_0 + √(1−ᾱ_t) · ε,   ε ~ N(0,I)   [quickguide Eq.(2.5)]

        Parameters
        ----------
        x0    : (B, *) clean data
        t     : (B,) int tensor, values in {1, …, T}
        noise : optional (B, *) pre-sampled noise; drawn from N(0,I) if None

        Returns
        -------
        xt    : (B, *) noisy sample
        noise : (B, *) the noise ε that was added
        """
        sched = self.schedule
        if noise is None:
            noise = torch.randn_like(x0)
        ab  = sched.broadcast("alpha_bar", t, x0.ndim)   # (B, 1, …)
        sig = sched.broadcast("sigma",     t, x0.ndim)   # (B, 1, …)
        xt  = ab.sqrt() * x0 + sig * noise
        return xt, noise

    # ------------------------------------------------------------------
    # Prediction / target conversion (Tweedie triad)
    # ------------------------------------------------------------------

    def _predict_x0_eps_v(
        self,
        xt: Tensor,
        t: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Run the network and return all three quantities (x̂_0, ε̂, v̂).

        All three are in bijection via quickguide Thm 2.2 (Tweedie):
            x̂_0 = (x_t − σ_t · ε̂) / √ᾱ_t
            ε̂   = (x_t − √ᾱ_t · x̂_0) / σ_t
            v̂   = √ᾱ_t · ε̂ − σ_t · x̂_0
        """
        sched = self.schedule
        # t is 1-indexed; normalize to [0,1] for the network
        t_norm = t.float() / sched.T

        net_out = self.network(xt, t_norm)                  # (B, *)

        ab  = sched.broadcast("alpha_bar", t, xt.ndim)
        sig = sched.broadcast("sigma",     t, xt.ndim)

        if self.prediction == "epsilon":
            eps_hat = net_out
            x0_hat  = (xt - sig * eps_hat) / ab.sqrt()
            v_hat   = ab.sqrt() * eps_hat - sig * x0_hat
        elif self.prediction == "x0":
            x0_hat  = net_out
            eps_hat = (xt - ab.sqrt() * x0_hat) / sig
            v_hat   = ab.sqrt() * eps_hat - sig * x0_hat
        else:  # "v"
            v_hat   = net_out
            x0_hat  = ab.sqrt() * xt - sig * v_hat
            eps_hat = (xt - ab.sqrt() * x0_hat) / sig

        return x0_hat, eps_hat, v_hat

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def _loss_weight(self, t: Tensor) -> Tensor:
        """Per-sample loss weight w(t), shape (B,)."""
        if self.loss_weight == "uniform":
            return torch.ones(len(t), device=t.device)
        snr = self.schedule.get("log_snr", t).exp()        # SNR_t = ᾱ_t / (1−ᾱ_t)
        if self.loss_weight == "snr":
            return snr
        # min-SNR: w = min(SNR, γ) / SNR  → clips large-SNR steps
        return torch.minimum(snr, torch.full_like(snr, self.min_snr_gamma)) / snr

    def loss(self, x0: Tensor) -> Tensor:
        """
        Compute the (weighted) diffusion loss on a batch of clean data.

        1. Sample t ~ Uniform{1, …, T}
        2. Sample x_t via q_sample
        3. Run network, compute ‖prediction − target‖² per sample
        4. Apply loss weight w(t) and return mean

        Returns
        -------
        loss : scalar Tensor
        """
        B = x0.shape[0]
        sched = self.schedule
        device = x0.device

        # Sample random timesteps
        t = torch.randint(1, sched.T + 1, (B,), device=device)  # in {1,…,T}

        # Noisy sample
        xt, noise = self.q_sample(x0, t)

        # Network prediction → Tweedie triad
        x0_hat, eps_hat, v_hat = self._predict_x0_eps_v(xt, t)

        # Target
        if self.prediction == "epsilon":
            target, pred = noise, eps_hat
        elif self.prediction == "x0":
            target, pred = x0, x0_hat
        else:  # "v"
            ab  = sched.broadcast("alpha_bar", t, x0.ndim)
            sig = sched.broadcast("sigma",     t, x0.ndim)
            v_target = ab.sqrt() * noise - sig * x0
            target, pred = v_target, v_hat

        # Per-sample MSE (mean over spatial dims)
        mse = ((pred - target) ** 2).flatten(1).mean(dim=1)   # (B,)

        # Loss weighting
        w = self._loss_weight(t)                               # (B,)
        return (w * mse).mean()

    # ------------------------------------------------------------------
    # Ancestral sampler  p_θ(x_{t-1} | x_t)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def p_sample(self, xt: Tensor, t: int) -> Tensor:
        """
        One reverse step: sample x_{t-1} ~ p_θ(x_{t-1} | x_t).

        Uses the mean of q(x_{t-1} | x_t, x̂_0) (the denoising posterior)
        plus optional noise σ̃_t · z.

        Posterior mean (quickguide §2.3):
            μ̃_t = (√ᾱ_{t-1} β_t x̂_0 + √α_t (1−ᾱ_{t-1}) x_t) / (1−ᾱ_t)

        Posterior variance:
            β̃_t = β_t (1−ᾱ_{t-1}) / (1−ᾱ_t)

        Parameters
        ----------
        xt : (B, *) noisy sample at step t
        t  : integer time step (1 ≤ t ≤ T)

        Returns
        -------
        x_{t-1} : (B, *) sample at step t-1
        """
        sched  = self.schedule
        device = xt.device
        B      = xt.shape[0]

        t_tensor = torch.full((B,), t, dtype=torch.long, device=device)
        x0_hat, _, _ = self._predict_x0_eps_v(xt, t_tensor)
        # Clip x̂_0 to data range for stability
        x0_hat = x0_hat.clamp(-5.0, 5.0)

        # Schedule values at t and t-1
        ab_t   = sched.alpha_bar[t]
        ab_tm1 = sched.alpha_bar[t - 1] if t > 1 else torch.tensor(1.0)
        beta_t = sched.beta[t]
        alpha_t = sched.alpha[t]

        # Posterior mean μ̃_t
        c1   = ab_tm1.sqrt() * beta_t / (1.0 - ab_t)
        c2   = alpha_t.sqrt() * (1.0 - ab_tm1) / (1.0 - ab_t)
        # Reshape for broadcasting
        def _r(v):
            return v.view((-1,) + (1,) * (xt.ndim - 1)).expand_as(xt)

        mu   = _r(c1.to(device)) * x0_hat + _r(c2.to(device)) * xt

        # Posterior variance β̃_t  (set to 0 at t=1)
        if t > 1:
            beta_tilde = beta_t * (1.0 - ab_tm1) / (1.0 - ab_t)
            noise      = torch.randn_like(xt)
            return mu + _r(beta_tilde.sqrt().to(device)) * noise
        else:
            return mu

    @torch.no_grad()
    def sample(
        self,
        shape: tuple,
        device: str | torch.device = "cpu",
        return_chain: bool = False,
        chain_stride: int = 100,
    ) -> Tensor | list[Tensor]:
        """
        Full reverse chain: x_T ~ N(0,I) → x_0.

        Parameters
        ----------
        shape        : shape of samples (B, *), e.g. (64, 2) or (8, 1, 28, 28)
        device       : target device
        return_chain : if True, return a list of intermediate x_t snapshots
        chain_stride : snapshot every this many steps (only if return_chain=True)

        Returns
        -------
        x0       : (B, *) final samples  (if return_chain=False)
        chain    : list of Tensors       (if return_chain=True, last element = x0)
        """
        self.eval()
        device = torch.device(device)
        sched  = self.schedule.to(device)
        xt     = torch.randn(shape, device=device)

        chain = []
        for t in range(sched.T, 0, -1):
            xt = self.p_sample(xt, t)
            if return_chain and t % chain_stride == 0:
                chain.append(xt.clone().cpu())

        if return_chain:
            chain.append(xt.cpu())
            return chain
        return xt
