"""
diffusion_lab/models/flow_matching.py
Conditional Flow Matching (CFM) — Lipman et al. (2022); Liu et al. (2022).

Math reference: quickguide Ch. 4

Theory summary
--------------
A flow model defines a time-dependent vector field v_t : ℝᵈ × [0,1] → ℝᵈ
whose ODE  dx/dt = v_t(x)  pushes source p_0 = N(0,I) to target p_1 = p_data.

Flow Matching loss (marginal, intractable directly):
    L_FM(θ) = E_{t, x~p_t}[ ‖v_θ(x, t) − u_t(x)‖² ]

Conditional Flow Matching (CFM, tractable):
    L_CFM(θ) = E_{t, x_1~p_data, x_0~N(0,I)}[ ‖v_θ(x_t, t) − u_t(x_t|x_0,x_1)‖² ]

with SAME gradient as L_FM (quickguide Thm 4.3, marginalization trick).

OT-Gaussian path (affine, straight-line couplings):
    x_t  = (1 − (1−σ_min)t) x_0 + t x_1             [mean path]
    u_t(x_t|x_0,x_1) = x_1 − (1 − σ_min) x_0        [constant velocity]

This is the simplest and empirically best variant (Liu et al. "Rectified Flow",
Lipman et al. "OT-CFM"), because the flow lines are straight → fewer ODE steps.

Note on sigma_min
-----------------
Adding a small σ_min > 0 keeps the conditional path non-degenerate at t=1:
    x_t = (1 − (1−σ_min)t) x_0 + t x_1
At t=1: x_1 = σ_min x_0 + x_1  → x_1 with tiny noise from x_0.
For very small σ_min (< 1e-3) the effect is negligible and can be set to 0.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

__all__ = ["CFM"]


class CFM(nn.Module):
    """
    Conditional Flow Matching model with OT-Gaussian (straight-line) paths.

    Parameters
    ----------
    network   : velocity network v_θ(x, t).
                forward(x: (B,*), t: (B,)) → (B,*)
    sigma_min : small variance added to path at t=1 for conditioning stability.
                Default 1e-4 (essentially zero for practical purposes).
    """

    def __init__(
        self,
        network: nn.Module,
        sigma_min: float = 1e-4,
    ) -> None:
        super().__init__()
        self.network   = network
        self.sigma_min = sigma_min

    # ------------------------------------------------------------------
    # Conditional path utilities
    # ------------------------------------------------------------------

    def sample_path(
        self,
        x0: Tensor,
        x1: Tensor,
        t: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """
        Compute x_t and the conditional velocity u_t(x_t | x_0, x_1).

        OT-Gaussian affine path (quickguide §4.4):
            x_t  = (1 − (1−σ_min) t) x_0 + t x_1
            u_t  = x_1 − (1 − σ_min) x_0    (independent of t — straight line!)

        Parameters
        ----------
        x0 : (B, *) source sample x_0 ~ N(0, I)
        x1 : (B, *) target sample x_1 ~ p_data
        t  : (B,)   time in [0, 1]

        Returns
        -------
        xt : (B, *)  interpolated point at time t
        ut : (B, *)  conditional velocity at x_t
        """
        t_bc = t.view((-1,) + (1,) * (x1.ndim - 1))          # (B, 1, …)
        c0   = 1.0 - (1.0 - self.sigma_min) * t_bc            # coeff of x0
        xt   = c0 * x0 + t_bc * x1
        ut   = x1 - (1.0 - self.sigma_min) * x0               # constant velocity
        return xt, ut

    # ------------------------------------------------------------------
    # Training loss
    # ------------------------------------------------------------------

    def loss(self, x1: Tensor) -> Tensor:
        """
        CFM regression loss.

        Algorithm:
            1. Sample t ~ Uniform[0, 1]
            2. Sample x_0 ~ N(0, I)  (source)
            3. Compute x_t, u_t  via straight-line interpolation
            4. Return E[ ‖v_θ(x_t, t) − u_t‖² ]

        Returns
        -------
        loss : scalar Tensor
        """
        B      = x1.shape[0]
        device = x1.device

        t  = torch.rand(B, device=device)                     # (B,)
        x0 = torch.randn_like(x1)                             # source

        xt, ut = self.sample_path(x0, x1, t)
        v_pred = self.network(xt, t)                          # (B,*)

        return ((v_pred - ut) ** 2).flatten(1).mean()

    # ------------------------------------------------------------------
    # ODE sampler
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample(
        self,
        shape: tuple,
        device: str | torch.device = "cpu",
        n_steps: int = 100,
        method: str = "euler",
        return_chain: bool = False,
        chain_stride: int = 10,
    ) -> Tensor | list[Tensor]:
        """
        Solve the flow ODE from t=0 to t=1 (source → data).

        dx/dt = v_θ(x, t)

        Parameters
        ----------
        shape        : output shape (B, *), e.g. (1024, 2)
        device       : target device
        n_steps      : number of ODE integration steps
        method       : 'euler' or 'rk4'
        return_chain : if True, return intermediate x_t snapshots
        chain_stride : snapshot every this many steps

        Returns
        -------
        x1   : (B, *) generated samples at t=1
        chain: list[Tensor]  (only if return_chain=True)
        """
        self.eval()
        device = torch.device(device)
        dt     = 1.0 / n_steps
        B      = shape[0]

        x     = torch.randn(shape, device=device)   # x_0 ~ N(0, I)
        chain = [x.clone().cpu()] if return_chain else []

        t_vals = torch.linspace(0.0, 1.0 - dt, n_steps, device=device)

        for step_i, t_val in enumerate(t_vals):
            t_batch = t_val.expand(B)              # (B,)

            if method == "euler":
                v  = self.network(x, t_batch)
                x  = x + dt * v

            elif method == "rk4":
                # Classic 4th-order Runge-Kutta
                k1 = self.network(x,             t_batch)
                k2 = self.network(x + 0.5*dt*k1, (t_val + 0.5*dt).expand(B))
                k3 = self.network(x + 0.5*dt*k2, (t_val + 0.5*dt).expand(B))
                k4 = self.network(x +     dt*k3, (t_val +     dt).expand(B))
                x  = x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            else:
                raise ValueError(f"Unknown method '{method}'. Use 'euler' or 'rk4'.")

            if return_chain and (step_i + 1) % chain_stride == 0:
                chain.append(x.clone().cpu())

        if return_chain:
            return chain
        return x

    # ------------------------------------------------------------------
    # Trajectory visualization helper
    # ------------------------------------------------------------------

    @torch.no_grad()
    def trajectories(
        self,
        x0_fixed: Tensor,
        n_steps: int = 50,
    ) -> Tensor:
        """
        Compute ODE trajectories for a fixed set of source points x_0.

        Parameters
        ----------
        x0_fixed : (N, d) source points (already fixed, not sampled)
        n_steps  : integration steps

        Returns
        -------
        traj : (n_steps+1, N, d)  position at each time step
        """
        self.eval()
        device = x0_fixed.device
        dt     = 1.0 / n_steps
        x      = x0_fixed.clone()
        traj   = [x.clone()]

        t_vals = torch.linspace(0.0, 1.0 - dt, n_steps, device=device)
        for t_val in t_vals:
            t_batch = t_val.expand(x.shape[0])
            v = self.network(x, t_batch)
            x = x + dt * v
            traj.append(x.clone())

        return torch.stack(traj, dim=0)   # (n_steps+1, N, d)
