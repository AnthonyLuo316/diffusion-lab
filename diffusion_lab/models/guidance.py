"""
diffusion_lab/models/guidance.py
Classifier-Free Guidance (CFG) for conditional DDPM.

Math reference: quickguide Ch. 9 (Extensions → Guidance)

Theory
------
Classifier-free guidance (Ho & Salimans 2022) jointly trains a single
network for both conditional and unconditional generation by randomly
dropping class labels during training (replacing them with a null token).

At inference, the conditional and unconditional scores are combined:

    ε_guided(x_t, t, c) = ε_uncond(x_t, t) + w · (ε_cond(x_t, t, c) − ε_uncond(x_t, t))

where w ≥ 0 is the guidance scale:
    w = 0 : unconditional generation (ignores c)
    w = 1 : standard conditional generation
    w > 1 : amplified guidance (sharper/more class-typical, lower diversity)

This is equivalent to a higher-temperature classifier guidance where
∇ log p(c|x_t) ∝ ε_cond − ε_uncond (up to a known prefactor).

Usage
-----
1. Build a SmallUNet with ``num_classes=K``:
       net = SmallUNet(..., num_classes=10)
2. Wrap in CondDDPM:
       model = CondDDPM(net, schedule, p_uncond=0.1)
3. Train with ``model.loss(x0, labels)``
4. Sample with CFGSampler:
       sampler = CFGSampler(model, guidance_scale=3.0)
       x0      = sampler.sample(shape, labels, device)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from diffusion_lab.models.ddpm import DDPM

__all__ = ["CondDDPM", "CFGSampler"]


# ---------------------------------------------------------------------------
# Conditional DDPM — adds class-label dropout for CFG training
# ---------------------------------------------------------------------------

class CondDDPM(DDPM):
    """
    Class-conditional DDPM with classifier-free guidance training.

    Extends DDPM by:
    - Accepting integer class labels in {1, …, K} at training time
    - Randomly replacing labels with 0 (null/unconditional token) with
      probability ``p_uncond`` each sample in the batch
    - Passing labels to the network's optional ``c`` argument

    Parameters
    ----------
    network    : SmallUNet (or any net) built with ``num_classes=K``
    schedule   : VP noise schedule (same as DDPM)
    p_uncond   : probability of dropping the class label each sample
                 (Ho & Salimans 2022 use 0.1–0.2)
    prediction : 'eps', 'x0', or 'v'  (passed to DDPM base class)
    loss_weight: 'uniform', 'snr', or 'min_snr'
    """

    def __init__(
        self,
        network:     nn.Module,
        schedule,
        p_uncond:    float = 0.1,
        prediction:  str   = "eps",
        loss_weight: str   = "uniform",
    ) -> None:
        super().__init__(network=network, schedule=schedule,
                         prediction=prediction, loss_weight=loss_weight)
        self.p_uncond = p_uncond

    # ------------------------------------------------------------------
    # Training — forward pass with label dropout
    # ------------------------------------------------------------------

    def loss(self, x0: Tensor, labels: Tensor) -> Tensor:
        """
        Conditional diffusion loss with CFG label dropout.

        Parameters
        ----------
        x0     : (B, C, H, W) clean images in [-1, 1]
        labels : (B,) integer class labels in {1, …, K}
                 (0 is reserved as the null token)

        Returns
        -------
        loss : scalar Tensor
        """
        B      = x0.shape[0]
        device = x0.device
        sched  = self.schedule

        # Sample random time steps
        t    = torch.randint(1, sched.T + 1, (B,), device=device)
        noise = torch.randn_like(x0)
        xt, _ = self.q_sample(x0, t, noise)

        # CFG label dropout: replace each label with 0 (null) independently
        drop_mask = (torch.rand(B, device=device) < self.p_uncond)
        c = labels.clone()
        c[drop_mask] = 0   # null token → unconditional

        # Network forward with class conditioning
        t_norm   = t.float() / sched.T
        net_out  = self.network(xt, t_norm, c)

        # Compute per-sample loss (same as DDPM, using inherited helpers)
        ab  = sched.broadcast("alpha_bar", t, x0.ndim)
        sig = sched.broadcast("sigma",     t, x0.ndim)

        if self.prediction == "epsilon":
            target = noise
        elif self.prediction == "x0":
            target = x0
        else:  # "v"
            target = ab.sqrt() * noise - sig * x0

        mse = ((net_out - target) ** 2).flatten(1).mean(dim=1)
        wt  = self._loss_weight(t)
        return (wt * mse).mean()

    # ------------------------------------------------------------------
    # Conditional network call (used by CFGSampler)
    # ------------------------------------------------------------------

    def _net_cfg(self, xt: Tensor, t_norm: Tensor, c: Tensor, w: float) -> Tensor:
        """
        Compute the guided network output:

            out_guided = out_uncond + w · (out_cond − out_uncond)

        Parameters
        ----------
        xt     : (B, C, H, W) noisy image
        t_norm : (B,) normalised time in [0, 1]
        c      : (B,) integer class labels in {1, …, K}
        w      : guidance scale ≥ 0

        Returns
        -------
        out_guided : (B, C, H, W)
        """
        if w == 0.0:
            null_c = torch.zeros_like(c)
            return self.network(xt, t_norm, null_c)

        # Run both conditional and unconditional in one batched forward pass
        # to avoid two separate network calls
        B        = xt.shape[0]
        null_c   = torch.zeros_like(c)
        c_batch  = torch.cat([c,      null_c], dim=0)   # (2B,)
        xt_batch = torch.cat([xt,     xt],     dim=0)   # (2B, C, H, W)
        t_batch  = torch.cat([t_norm, t_norm], dim=0)   # (2B,)

        out_batch = self.network(xt_batch, t_batch, c_batch)  # (2B,*)
        out_cond, out_uncond = out_batch[:B], out_batch[B:]

        return out_uncond + w * (out_cond - out_uncond)


# ---------------------------------------------------------------------------
# CFG Sampler — ancestral sampling with classifier-free guidance
# ---------------------------------------------------------------------------

class CFGSampler(nn.Module):
    """
    Classifier-free guidance ancestral sampler for CondDDPM.

    Parameters
    ----------
    model          : trained CondDDPM
    guidance_scale : w in the CFG formula (0 = unconditional, >1 = amplified)
    """

    def __init__(self, model: CondDDPM, guidance_scale: float = 3.0) -> None:
        super().__init__()
        self.model         = model
        self.guidance_scale = guidance_scale

    @torch.no_grad()
    def sample(
        self,
        shape: tuple,
        labels: Tensor,
        device: str | torch.device = "cpu",
        return_chain: bool = False,
        chain_stride: int = 100,
    ) -> Tensor | list[Tensor]:
        """
        Generate samples conditioned on ``labels``.

        Parameters
        ----------
        shape        : (B, C, H, W) — must match labels.shape[0]
        labels       : (B,) integer class labels {1, …, K}
        device       : target device
        return_chain : if True, return list of intermediate snapshots
        chain_stride : snapshot every this many reverse steps

        Returns
        -------
        x0    : (B, C, H, W) generated samples
        chain : list[Tensor]  (only if return_chain=True)
        """
        self.model.eval()
        device = torch.device(device)
        sched  = self.model.schedule
        T      = sched.T
        B      = shape[0]
        labels = labels.to(device)
        w      = self.guidance_scale

        x     = torch.randn(shape, device=device)
        chain = [x.clone().cpu()] if return_chain else []

        for t in range(T, 0, -1):
            t_tensor = torch.full((B,), t, dtype=torch.long, device=device)
            t_norm   = t_tensor.float() / T

            # ----------------------------------------------------------
            # Guided network prediction
            # ----------------------------------------------------------
            out_guided = self.model._net_cfg(x, t_norm, labels, w)

            # Convert to x0_hat (same Tweedie logic as DDPM.p_sample)
            ab  = sched.broadcast("alpha_bar", t_tensor, x.ndim)
            sig = sched.broadcast("sigma",     t_tensor, x.ndim)

            if self.model.prediction == "epsilon":
                eps_hat = out_guided
                x0_hat  = (x - sig * eps_hat) / ab.sqrt()
            elif self.model.prediction == "x0":
                x0_hat  = out_guided
                eps_hat = (x - ab.sqrt() * x0_hat) / sig
            else:  # "v"
                v_hat   = out_guided
                x0_hat  = ab.sqrt() * x - sig * v_hat
                eps_hat = (x - ab.sqrt() * x0_hat) / sig

            x0_hat = x0_hat.clamp(-5.0, 5.0)

            # ----------------------------------------------------------
            # Denoising posterior mean + noise
            # ----------------------------------------------------------
            ab_t   = sched.alpha_bar[t]
            ab_tm1 = sched.alpha_bar[t - 1] if t > 1 else torch.tensor(1.0)
            beta_t = sched.beta[t]
            alpha_t = sched.alpha[t]

            c1 = ab_tm1.sqrt() * beta_t / (1.0 - ab_t)
            c2 = alpha_t.sqrt() * (1.0 - ab_tm1) / (1.0 - ab_t)

            def _r(v):
                return v.to(device).view((-1,) + (1,) * (x.ndim - 1)).expand_as(x)

            mu = _r(c1) * x0_hat + _r(c2) * x

            if t > 1:
                beta_tilde = beta_t * (1.0 - ab_tm1) / (1.0 - ab_t)
                x = mu + _r(beta_tilde.sqrt()) * torch.randn_like(x)
            else:
                x = mu

            if return_chain and t % chain_stride == 0:
                chain.append(x.clone().cpu())

        if return_chain:
            return chain
        return x

    @torch.no_grad()
    def sample_ddim(
        self,
        shape: tuple,
        labels: Tensor,
        device: str | torch.device = "cpu",
        num_steps: int = 50,
        eta: float = 0.0,
    ) -> Tensor:
        """
        DDIM sampler with CFG — deterministic (η=0) or stochastic (η=1).

        Parameters
        ----------
        num_steps : number of DDIM sub-steps (≪ T)
        eta       : 0 = deterministic, 1 = DDPM-variance stochasticity
        """
        self.model.eval()
        device = torch.device(device)
        sched  = self.model.schedule
        T      = sched.T
        B      = shape[0]
        labels = labels.to(device)
        w      = self.guidance_scale

        # Build evenly-spaced sub-sequence τ₁ > τ₂ > … > τ_S
        tau = torch.linspace(T, 1, num_steps, dtype=torch.long)

        x = torch.randn(shape, device=device)

        for i in range(len(tau)):
            t_cur  = int(tau[i])
            t_prev = int(tau[i + 1]) if i + 1 < len(tau) else 0

            t_tensor = torch.full((B,), t_cur, dtype=torch.long, device=device)
            t_norm   = t_tensor.float() / T

            out_guided = self.model._net_cfg(x, t_norm, labels, w)

            ab_t   = sched.alpha_bar[t_cur].to(device)
            sig_t  = sched.sigma[t_cur].to(device)
            ab_tm1 = (sched.alpha_bar[t_prev].to(device)
                      if t_prev > 0 else torch.tensor(1.0, device=device))

            if self.model.prediction == "epsilon":
                eps_hat = out_guided
            elif self.model.prediction == "x0":
                x0_hat_tmp = out_guided
                eps_hat    = (x - ab_t.sqrt() * x0_hat_tmp) / sig_t
            else:
                v_hat   = out_guided
                x0_hat_tmp = ab_t.sqrt() * x - sig_t * v_hat
                eps_hat    = (x - ab_t.sqrt() * x0_hat_tmp) / sig_t

            x0_hat = (x - sig_t * eps_hat) / ab_t.sqrt()
            x0_hat = x0_hat.clamp(-5.0, 5.0)

            # DDIM variance
            beta_tilde = sched.beta[t_cur].to(device) * (1.0 - ab_tm1) / (1.0 - ab_t)
            sigma_tau  = eta * beta_tilde.sqrt()

            # Direction toward x_t
            coeff_eps = (1.0 - ab_tm1 - sigma_tau ** 2).clamp(min=0.0).sqrt()
            x = ab_tm1.sqrt() * x0_hat + coeff_eps * eps_hat
            if eta > 0.0 and t_prev > 0:
                x = x + sigma_tau * torch.randn_like(x)

        return x
