"""
diffusion_lab/configs.py
Centralised hyperparameter configuration for all models and experiments.

Design
------
All configs are plain Python dataclasses — no YAML, no JSON, no extra
dependencies.  This gives full IDE autocompletion and type checking while
remaining trivially overridable:

    cfg = DDPMConfig()               # defaults
    cfg = DDPMConfig(T=500)          # override one field
    cfg = DDPMConfig(**my_dict)      # from a dict

Notebooks import the relevant config class and pass fields directly to
model/trainer constructors.  This makes the magic numbers in each notebook
explicit and co-located with the code that uses them.

Usage
-----
    from diffusion_lab.configs import DDPMConfig, TrainConfig

    # --- build from config ---
    cfg   = DDPMConfig()
    train = TrainConfig(epochs=20)

    schedule = VPSchedule(T=cfg.T, schedule=cfg.schedule)
    net      = SmallUNet(**cfg.unet_kwargs())
    model    = DDPM(network=net, schedule=schedule,
                    prediction=cfg.prediction, loss_weight=cfg.loss_weight)

    # --- quick override ---
    cfg2 = DDPMConfig(T=500, schedule='linear')
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

__all__ = [
    "TrainConfig",
    "UNetConfig",
    "DDPMConfig",
    "DDIMConfig",
    "NCSNConfig",
    "VESDEConfig",
    "CFMConfig",
    "CondDDPMConfig",
    "EDMConfig",
    "DPSConfig",
]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    """
    Generic training hyperparameters shared by all models.

    Parameters
    ----------
    epochs      : number of training epochs
    lr          : peak learning rate (AdamW)
    batch_size  : mini-batch size
    grad_clip   : max gradient norm (None = disabled)
    lr_schedule : 'cosine' | 'constant'
    seed        : global random seed
    device      : 'cuda' | 'cpu' | 'auto'  ('auto' = cuda if available)
    checkpoint_dir: path to save .pt checkpoints
    """
    epochs:         int   = 30
    lr:             float = 3e-4
    batch_size:     int   = 256
    grad_clip:      float = 1.0
    lr_schedule:    str   = "cosine"
    seed:           int   = 0
    device:         str   = "auto"
    checkpoint_dir: str   = "../checkpoints"


# ---------------------------------------------------------------------------
# Neural network backbone
# ---------------------------------------------------------------------------

@dataclass
class UNetConfig:
    """
    SmallUNet architecture hyperparameters.

    Parameters
    ----------
    in_channels    : input image channels (1 for MNIST grayscale)
    out_channels   : output channels (= in_channels for diffusion)
    base_channels  : C in the encoder/decoder width schedule
    time_embed_dim : sinusoidal time embedding dimension
    dropout        : dropout probability in ResBlocks
    num_classes    : if set, enables class conditioning (index 0 = null token)
    """
    in_channels:    int         = 1
    out_channels:   int         = 1
    base_channels:  int         = 32
    time_embed_dim: int         = 128
    dropout:        float       = 0.1
    num_classes:    int | None  = None

    def as_kwargs(self) -> dict[str, Any]:
        """Return kwargs suitable for ``SmallUNet(**cfg.as_kwargs())``."""
        return asdict(self)


# ---------------------------------------------------------------------------
# DDPM / DDIM
# ---------------------------------------------------------------------------

@dataclass
class DDPMConfig:
    """
    DDPM + VP schedule hyperparameters.

    Parameters
    ----------
    T          : number of diffusion timesteps
    schedule   : 'cosine' | 'linear'
    prediction : 'epsilon' | 'x0' | 'v'
    loss_weight: 'uniform' | 'snr' | 'min_snr'
    unet       : UNet architecture config
    train      : training config
    """
    T:           int        = 1000
    schedule:    str        = "cosine"
    prediction:  str        = "epsilon"
    loss_weight: str        = "uniform"
    unet:        UNetConfig = field(default_factory=UNetConfig)
    train:       TrainConfig = field(default_factory=TrainConfig)


@dataclass
class DDIMConfig:
    """
    DDIM sampler hyperparameters.

    Parameters
    ----------
    num_steps : number of DDIM sub-steps (≪ T; e.g. 50)
    eta       : stochasticity: 0 = deterministic, 1 = DDPM-variance
    """
    num_steps: int   = 50
    eta:       float = 0.0


# ---------------------------------------------------------------------------
# Score SDE / NCSN
# ---------------------------------------------------------------------------

@dataclass
class NCSNConfig:
    """
    NCSN (Noise Conditional Score Network) hyperparameters.

    Parameters
    ----------
    L          : number of noise scales
    sigma_min  : smallest noise level
    sigma_max  : largest noise level
    n_steps_lg : Langevin steps per noise level
    step_lr    : Langevin step size (used as ε in annealed Langevin)
    """
    L:          int   = 10
    sigma_min:  float = 0.01
    sigma_max:  float = 1.0
    n_steps_lg: int   = 100
    step_lr:    float = 2e-5
    train:      TrainConfig = field(default_factory=TrainConfig)


@dataclass
class VESDEConfig:
    """
    VE-SDE hyperparameters (continuous-time SMLD).

    Parameters
    ----------
    sigma_min  : σ_min for the VE schedule
    sigma_max  : σ_max for the VE schedule
    N          : number of discretisation steps for reverse SDE/ODE
    """
    sigma_min: float = 0.01
    sigma_max: float = 5.0
    N:         int   = 1000
    train:     TrainConfig = field(default_factory=TrainConfig)


# ---------------------------------------------------------------------------
# Flow Matching
# ---------------------------------------------------------------------------

@dataclass
class CFMConfig:
    """
    Conditional Flow Matching (OT-Gaussian path) hyperparameters.

    Parameters
    ----------
    sigma_min  : std of the Gaussian conditional distribution q(x₁|x₀)
                 (tiny value → straight-line OT path in the limit σ→0)
    num_steps  : Euler/RK4 ODE steps at sampling time
    solver     : 'euler' | 'rk4'
    """
    sigma_min: float = 1e-4
    num_steps: int   = 100
    solver:    str   = "euler"
    unet:      UNetConfig  = field(default_factory=UNetConfig)
    train:     TrainConfig = field(default_factory=TrainConfig)


# ---------------------------------------------------------------------------
# Classifier-Free Guidance
# ---------------------------------------------------------------------------

@dataclass
class CondDDPMConfig:
    """
    Conditional DDPM with classifier-free guidance.

    Parameters
    ----------
    p_uncond       : label drop probability during training
    guidance_scale : w in the CFG formula at inference
    num_classes    : number of label classes (0 = null token reserved)
    ddpm           : base DDPM config (T, schedule, etc.)
    """
    p_uncond:       float = 0.15
    guidance_scale: float = 3.0
    num_classes:    int   = 10    # MNIST digits 0–9 → classes 1–10
    ddpm:           DDPMConfig   = field(default_factory=DDPMConfig)


# ---------------------------------------------------------------------------
# EDM
# ---------------------------------------------------------------------------

@dataclass
class EDMConfig:
    """
    EDM preconditioning and Heun sampler hyperparameters.

    Parameters
    ----------
    sigma_data : assumed std of clean images (0.5 for [-1,1] images)
    P_mean     : mean of ln σ training distribution
    P_std      : std of ln σ training distribution
    sigma_min  : minimum noise level (Heun schedule)
    sigma_max  : maximum noise level (Heun schedule)
    rho        : power-law schedule curvature (default 7)
    num_steps  : default number of Heun steps at sampling time
    S_churn    : stochastic churn strength (0 = deterministic)
    unet       : UNet architecture config
    train      : training config
    """
    sigma_data: float = 0.5
    P_mean:     float = -1.2
    P_std:      float = 1.2
    sigma_min:  float = 0.002
    sigma_max:  float = 80.0
    rho:        float = 7.0
    num_steps:  int   = 50
    S_churn:    float = 0.0
    unet:       UNetConfig  = field(default_factory=UNetConfig)
    train:      TrainConfig = field(default_factory=TrainConfig)


# ---------------------------------------------------------------------------
# DPS (Diffusion Posterior Sampling)
# ---------------------------------------------------------------------------

@dataclass
class DPSConfig:
    """
    DPS sampler hyperparameters.

    Parameters
    ----------
    zeta         : likelihood gradient step size
    operator     : name of the forward operator
                   ('random_mask' | 'box_mask' | 'gaussian_blur' | 'super_res')
    keep_prob    : for 'random_mask': fraction of pixels kept
    box_size     : for 'box_mask': side length of masked square
    blur_kernel  : for 'gaussian_blur': (kernel_size, sigma)
    sr_scale     : for 'super_res': downsampling factor
    """
    zeta:       float = 1.0
    operator:   str   = "random_mask"
    keep_prob:  float = 0.5
    box_size:   int   = 14
    blur_kernel: tuple = (7, 2.0)
    sr_scale:   int   = 4

    def build_operator(self, channels: int = 1):
        """Instantiate the configured LinearOperator."""
        from diffusion_lab.models.inverse import (
            RandomMaskOperator, BoxMaskOperator,
            GaussianBlurOperator, SuperResolutionOperator,
        )
        if self.operator == "random_mask":
            return RandomMaskOperator(keep_prob=self.keep_prob)
        elif self.operator == "box_mask":
            return BoxMaskOperator(box_size=self.box_size)
        elif self.operator == "gaussian_blur":
            ks, sig = self.blur_kernel
            return GaussianBlurOperator(kernel_size=ks, sigma=sig, channels=channels)
        elif self.operator == "super_res":
            return SuperResolutionOperator(scale=self.sr_scale)
        else:
            raise ValueError(f"Unknown operator: {self.operator!r}")
