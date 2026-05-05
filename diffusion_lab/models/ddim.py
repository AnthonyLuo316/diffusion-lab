"""
diffusion_lab/models/ddim.py
DDIM — Denoising Diffusion Implicit Models (Song et al., 2021).

Math reference: quickguide Ch. 2 §DDIM

Key idea: the DDPM forward marginals q(x_t | x_0) are compatible with an
entire *family* of inference processes parametrized by a free schedule
σ_τ ≥ 0.  The DDIM update uses the same trained network as DDPM, but
can skip most timesteps and uses:

    x_{t-1} = √ᾱ_{t-1} · x̂_0(x_t)
            + √(1−ᾱ_{t-1} − σ²_τ) · ε_θ(x_t, t)
            + σ_τ · z,    z ~ N(0, I)

where x̂_0 = (x_t − √(1−ᾱ_t) · ε_θ) / √ᾱ_t   (Tweedie, quickguide Thm 2.2).

Special cases:
- σ_τ = 0              → fully *deterministic* ODE sampler (DDIM)
- σ_τ = √β̃_t          → recovers DDPM ancestral sampling
- 0 < σ_τ < √β̃_t      → intermediate stochasticity controlled by η:
                          σ_τ = η · √(β̃_t),   η ∈ [0, 1]
"""

from __future__ import annotations

import torch
from torch import Tensor

from diffusion_lab.models.ddpm import DDPM

__all__ = ["DDIMSampler"]


class DDIMSampler:
    """
    DDIM sampler that wraps a trained DDPM (no retraining required).

    Parameters
    ----------
    ddpm      : trained DDPM instance
    eta       : stochasticity level in [0, 1].
                eta=0 → deterministic DDIM ODE.
                eta=1 → recovers DDPM-level stochasticity.
    num_steps : number of denoising steps (< ddpm.schedule.T for speed-up)
    """

    def __init__(
        self,
        ddpm: DDPM,
        eta: float = 0.0,
        num_steps: int = 50,
    ) -> None:
        self.ddpm      = ddpm
        self.eta       = eta
        self.num_steps = num_steps

    def _make_timestep_seq(self, T: int) -> list[int]:
        """
        Build a sub-sequence of T timesteps for accelerated sampling.

        Returns a list of length num_steps, descending, ending at 1.
        Example: T=1000, num_steps=50 → [980, 960, …, 20, 0] (every 20th)

        We include t=0 as a sentinel for the final step (goes to t_prev=0).
        """
        # Evenly spaced indices from T down to 1, length = num_steps
        seq = torch.linspace(T, 1, self.num_steps).round().long().tolist()
        return seq

    @torch.no_grad()
    def sample(
        self,
        shape: tuple,
        device: str | torch.device = "cpu",
        return_chain: bool = False,
    ) -> Tensor | list[Tensor]:
        """
        Run DDIM sampling.

        Parameters
        ----------
        shape        : output shape, e.g. (64, 2) or (8, 1, 28, 28)
        device       : target device
        return_chain : if True, return all intermediate x_t snapshots

        Returns
        -------
        x0    : (B, *) final samples         (return_chain=False)
        chain : list[Tensor]                 (return_chain=True, last = x0)
        """
        self.ddpm.eval()
        device  = torch.device(device)
        sched   = self.ddpm.schedule.to(device)
        T       = sched.T
        B       = shape[0]

        # Build sub-sequence: [t_S, t_{S-1}, …, t_1]  (descending)
        seq     = self._make_timestep_seq(T)
        # Each t_prev is the *next* (lower) t in the sequence; final step goes to 0
        seq_prev = seq[1:] + [0]

        xt    = torch.randn(shape, device=device)
        chain = []

        for t_cur, t_prev in zip(seq, seq_prev):
            t_tensor = torch.full((B,), t_cur, dtype=torch.long, device=device)

            # Get network prediction → all three Tweedie quantities
            x0_hat, eps_hat, _ = self.ddpm._predict_x0_eps_v(xt, t_tensor)
            x0_hat = x0_hat.clamp(-5.0, 5.0)

            # Schedule values at t_cur and t_prev
            ab_cur  = sched.alpha_bar[t_cur]
            ab_prev = sched.alpha_bar[t_prev] if t_prev > 0 else torch.tensor(1.0, device=device)
            beta_tilde = (
                (1.0 - ab_prev) / (1.0 - ab_cur) * (1.0 - ab_cur / ab_prev)
            ).clamp(0.0)

            sigma_tau = self.eta * beta_tilde.sqrt()

            # Direction pointing to x_t  (Eq. quickguide Ch.2 DDIM section)
            # "predicted x_t direction"  = √(1−ᾱ_{t-1} − σ²_τ) · ε_θ
            coeff_eps = (1.0 - ab_prev - sigma_tau ** 2).clamp(0.0).sqrt()

            def _r(v):
                return v.view((-1,) + (1,) * (xt.ndim - 1)).expand_as(xt)

            xt_prev = (
                _r(ab_prev.sqrt())  * x0_hat
                + _r(coeff_eps)     * eps_hat
                + _r(sigma_tau)     * torch.randn_like(xt)
            )
            xt = xt_prev

            if return_chain:
                chain.append(xt.clone().cpu())

        if return_chain:
            return chain
        return xt
