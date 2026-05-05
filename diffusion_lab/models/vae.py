"""
diffusion_lab/models/vae.py
Variational Autoencoder (VAE) — Kingma & Welling (2014).

Math reference: quickguide Ch. 1

ELBO
----
    L(θ, φ; x)  =  E_{q_φ(z|x)}[log p_θ(x|z)]  −  KL(q_φ(z|x) ‖ p(z))

where p(z) = N(0, I),  q_φ(z|x) = N(μ_φ(x), diag(σ²_φ(x))).

Reparameterization trick:
    z = μ_φ(x) + σ_φ(x) ⊙ ε,    ε ~ N(0, I)

Analytic KL for diagonal Gaussians (quickguide Prop 1.4):
    KL(N(μ, σ²I) ‖ N(0, I)) = ½ Σᵢ (μᵢ² + σᵢ² − log σᵢ² − 1)
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

__all__ = ["Encoder", "Decoder", "VAE"]


# ---------------------------------------------------------------------------
# Sub-networks
# ---------------------------------------------------------------------------

class Encoder(nn.Module):
    """
    Amortised inference network q_φ(z|x).

    Maps x ∈ ℝᵈ  →  (μ_φ(x), log σ²_φ(x)) ∈ ℝᵏ × ℝᵏ.

    Parameters
    ----------
    in_dim   : input dimension d
    latent   : latent dimension k
    hidden   : width of hidden layers
    depth    : number of hidden layers (depth ≥ 1)
    """

    def __init__(
        self,
        in_dim: int,
        latent: int,
        hidden: int = 256,
        depth: int = 3,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        d = in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.SiLU()]
            d = hidden
        self.trunk = nn.Sequential(*layers)
        self.mu_head    = nn.Linear(hidden, latent)
        self.logvar_head = nn.Linear(hidden, latent)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """
        Parameters
        ----------
        x : (B, in_dim)

        Returns
        -------
        mu     : (B, latent)  — posterior mean
        logvar : (B, latent)  — log σ², clamped to [-10, 10] for stability
        """
        h = self.trunk(x)
        mu     = self.mu_head(h)
        logvar = self.logvar_head(h).clamp(-10.0, 10.0)
        return mu, logvar


class Decoder(nn.Module):
    """
    Generative network p_θ(x|z).

    Models p_θ(x|z) = N(μ_θ(z), σ²_rec · I) with fixed σ_rec.
    The reconstruction loss is therefore  ‖x − μ_θ(z)‖² / (2 σ²_rec).

    Parameters
    ----------
    latent  : latent dimension k
    out_dim : output dimension d
    hidden  : width of hidden layers
    depth   : number of hidden layers (depth ≥ 1)
    sigma   : fixed reconstruction standard deviation (default 0.1)
    """

    def __init__(
        self,
        latent: int,
        out_dim: int,
        hidden: int = 256,
        depth: int = 3,
        sigma: float = 0.1,
    ) -> None:
        super().__init__()
        self.sigma = sigma
        layers: list[nn.Module] = []
        d = latent
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.SiLU()]
            d = hidden
        layers.append(nn.Linear(hidden, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z: Tensor) -> Tensor:
        """
        Parameters
        ----------
        z : (B, latent)

        Returns
        -------
        mu_theta : (B, out_dim)  — decoded mean
        """
        return self.net(z)


# ---------------------------------------------------------------------------
# VAE
# ---------------------------------------------------------------------------

class VAE(nn.Module):
    """
    Variational Autoencoder for continuous data.

    For 2-D toy data use latent=2 (enables direct latent-space plotting).
    For MNIST use latent=8 or 16.

    Parameters
    ----------
    in_dim  : input dimension
    latent  : latent dimension k
    hidden  : hidden layer width
    depth   : depth of encoder / decoder MLPs
    beta    : KL weight  (β-VAE, Higgins et al. 2017).  β=1 is standard VAE.
    rec_sigma : fixed decoder std σ_rec
    """

    def __init__(
        self,
        in_dim: int = 2,
        latent: int = 2,
        hidden: int = 256,
        depth: int = 3,
        beta: float = 1.0,
        rec_sigma: float = 0.1,
    ) -> None:
        super().__init__()
        self.in_dim  = in_dim
        self.latent  = latent
        self.beta    = beta

        self.encoder = Encoder(in_dim, latent, hidden, depth)
        self.decoder = Decoder(latent, in_dim, hidden, depth, sigma=rec_sigma)

    # ------------------------------------------------------------------
    # Core computations
    # ------------------------------------------------------------------

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """
        Returns posterior parameters (μ, log σ²) for each input x.

        Parameters
        ----------
        x : (B, in_dim)

        Returns
        -------
        mu, logvar : each (B, latent)
        """
        return self.encoder(x)

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Sample z ~ q_φ(z|x) via the reparameterization trick.

        z = μ + σ ⊙ ε,  ε ~ N(0, I).

        Returns
        -------
        z : (B, latent)
        """
        if self.training:
            std = (0.5 * logvar).exp()          # σ = exp(log σ² / 2)
            eps = torch.randn_like(std)
            return mu + std * eps
        else:
            return mu                            # use mean at test time

    def decode(self, z: Tensor) -> Tensor:
        """
        Returns reconstructed mean μ_θ(z).

        Parameters
        ----------
        z : (B, latent)

        Returns
        -------
        x_recon : (B, in_dim)
        """
        return self.decoder(z)

    # ------------------------------------------------------------------
    # ELBO and loss
    # ------------------------------------------------------------------

    @staticmethod
    def _kl_diagonal_gaussian(mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Analytic KL(N(μ, σ²I) ‖ N(0, I)) per sample, summed over latent dims.

        KL = ½ Σᵢ (μᵢ² + σᵢ² − log σᵢ² − 1)
           = ½ Σᵢ (μᵢ² + exp(logvar_i) − logvar_i − 1)

        Returns
        -------
        kl : (B,)  per-sample KL divergences
        """
        # quickguide Ch.1, closed-form KL for diagonal Gaussians
        kl = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1.0)
        return kl.sum(dim=-1)                   # sum over latent dim → (B,)

    def elbo(self, x: Tensor) -> Tensor:
        """
        Compute the mean ELBO over the batch (higher = better).

        L = E_q[log p_θ(x|z)] − β · KL(q_φ(z|x) ‖ p(z))

        The reconstruction term uses a Gaussian likelihood:
            log p_θ(x|z) = −‖x − μ_θ(z)‖² / (2 σ²_rec) + const

        Returns
        -------
        elbo : scalar Tensor
        """
        mu, logvar = self.encode(x)
        z          = self.reparameterize(mu, logvar)
        x_recon    = self.decode(z)

        sigma2 = self.decoder.sigma ** 2
        # Gaussian log-likelihood (per sample, summed over dim)
        rec = -0.5 * ((x - x_recon) ** 2).sum(dim=-1) / sigma2
        kl  = self._kl_diagonal_gaussian(mu, logvar)

        return (rec - self.beta * kl).mean()    # scalar

    def loss(self, x: Tensor) -> Tensor:
        """
        Training loss = − ELBO  (lower = better, compatible with Trainer).

        Returns
        -------
        loss : scalar Tensor
        """
        return -self.elbo(x)

    # ------------------------------------------------------------------
    # Inference / generation helpers
    # ------------------------------------------------------------------

    def reconstruct(self, x: Tensor) -> Tensor:
        """
        Encode x and decode the posterior mean (no sampling).

        Returns
        -------
        x_recon : (B, in_dim)
        """
        mu, _ = self.encode(x)
        return self.decode(mu)

    @torch.no_grad()
    def sample(self, n: int, device: str | torch.device = "cpu") -> Tensor:
        """
        Ancestral sampling: z ~ p(z) = N(0, I),  x ~ p_θ(x|z).

        Returns
        -------
        x_samples : (n, in_dim)
        """
        z = torch.randn(n, self.latent, device=device)
        return self.decode(z)

    @torch.no_grad()
    def interpolate(
        self,
        x_a: Tensor,
        x_b: Tensor,
        steps: int = 10,
    ) -> Tensor:
        """
        Latent-space linear interpolation between two inputs.

        Returns
        -------
        interp : (steps, in_dim)  decoded points along z_a → z_b
        """
        mu_a, _ = self.encode(x_a[None])       # (1, latent)
        mu_b, _ = self.encode(x_b[None])
        alphas  = torch.linspace(0, 1, steps, device=x_a.device)
        zs      = (1 - alphas[:, None]) * mu_a + alphas[:, None] * mu_b
        return self.decode(zs)
