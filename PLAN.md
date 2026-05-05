# Diffusion Lab — Implementation Plan

> A modular codebase that turns the theory in `quickguide.tex` into runnable,
> interactive experiments.  Architecture mirrors the
> [facebookresearch/flow\_matching](https://github.com/facebookresearch/flow_matching)
> library: a **shared core library** (`diffusion_lab/`) plus per-framework
> **Jupyter notebooks** (`notebooks/`).

---

## 0. Motivation & Scope

| Framework | Key reference (quickguide chapter) | What the code must do |
|-----------|------------------------------------|-----------------------|
| VAE | Ch. 1 — ELBO, reparameterization | Encode, decode, sample latent space |
| DDPM | Ch. 2 — VP marginal, ε/x₀/v-prediction | Forward diffusion, ELBO training, ancestral sampling |
| DDIM | Ch. 2 §DDIM — non-Markovian inference | Deterministic and stochastic sampler with stride τ |
| Score SDE (SMLD/NCSN) | Ch. 3 — DSM, Langevin, VE-SDE | Score matching, annealed Langevin, reverse SDE/ODE |
| Flow Matching (CFM) | Ch. 4 — CFM, OT path | Vector field regression, Euler/RK4 ODE sampler |

**Datasets.** All toy experiments use **2-D data** (spiral, two-moons, checkerboard).
MNIST is used for DDPM and Flow Matching to show pixel-space generation.

---

## 1. Repository Layout

```
code_claude/
├── PLAN.md                    ← this file
├── README.md
├── requirements.txt
├── pyproject.toml
│
├── diffusion_lab/             # installable core library
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── toy.py             # spiral / moons / checkerboard generators
│   │   └── mnist.py           # MNIST DataLoader wrapper
│   ├── nn/
│   │   ├── __init__.py
│   │   ├── mlp.py             # time-conditioned MLP (2-D experiments)
│   │   └── unet.py            # small U-Net (MNIST experiments)
│   ├── schedulers/
│   │   ├── __init__.py
│   │   └── schedules.py       # VP linear/cosine, VE geometric schedules
│   ├── models/
│   │   ├── __init__.py
│   │   ├── vae.py             # VAE encoder + decoder + ELBO
│   │   ├── ddpm.py            # DDPM forward process + loss + sampler
│   │   ├── ddim.py            # DDIM sampler (wraps DDPM)
│   │   ├── score_sde.py       # NCSN loss + annealed Langevin + VE-SDE
│   │   └── flow_matching.py   # CFM loss + ODE sampler
│   ├── training/
│   │   ├── __init__.py
│   │   └── trainer.py         # generic Trainer class
│   └── utils/
│       ├── __init__.py
│       └── viz.py             # 2-D density plots, sample grids, loss curves
│
├── notebooks/
│   ├── 01_vae_2d.ipynb
│   ├── 02_ddpm_ddim_2d.ipynb
│   ├── 03_score_sde_2d.ipynb
│   ├── 04_flow_matching_2d.ipynb
│   ├── 05_ddpm_mnist.ipynb
│   └── 06_flow_matching_mnist.ipynb
│
└── scripts/                   # optional CLI training scripts
    ├── train_vae.py
    ├── train_ddpm.py
    ├── train_score.py
    └── train_flow.py
```

---

## 2. Core Library — Module-by-Module Spec

### 2.1 `diffusion_lab/data/toy.py`

Provides a unified `make_dataset(name, n_samples, seed)` factory.

| Dataset | Generation logic |
|---------|-----------------|
| `"spiral"` | Archimedean spiral with additive Gaussian noise (two arms) |
| `"moons"` | `sklearn.datasets.make_moons` wrapper |
| `"checkerboard"` | Uniform samples on alternating 2×2 squares (4×4 grid) |

**Interface:**
```python
def make_dataset(name: str, n: int = 10_000, seed: int = 0) -> np.ndarray:
    """Returns (n, 2) float32 array, roughly in [-4, 4]²."""

def get_dataloader(name, n, batch_size, seed) -> DataLoader:
    """Wraps make_dataset as a torch DataLoader with infinite cycling."""
```

All datasets are returned **normalized** to roughly unit variance.

---

### 2.2 `diffusion_lab/nn/mlp.py`

A **time-conditioned MLP** sufficient for 2-D generative modeling.

**Architecture:**
```
x ∈ ℝᵈ, t ∈ ℝ  →  [sinusoidal embed(t)] ∈ ℝᴱ
                    concatenate [x, embed(t)] ∈ ℝ^{d+E}
                    Linear → SiLU → Linear → SiLU → Linear
                    → output ∈ ℝᵈ
```

**Key design choices:**
- Sinusoidal time embedding (identical to the one in DDPM / Transformer positional encoding):
  `embed(t)_i = sin(t / 10000^{2i/E})` for even i, cosine for odd i.
- Hidden dimension H = 256, depth = 4 by default.
- Residual skip connection from input x to output (useful for score/velocity head).

**Class signature:**
```python
class TimeMLP(nn.Module):
    def __init__(self, in_dim: int = 2, out_dim: int = 2,
                 hidden: int = 256, depth: int = 4,
                 time_embed_dim: int = 128): ...

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        """x: (B, in_dim), t: (B,) in [0, 1] or [0, T].  Returns (B, out_dim)."""
```

---

### 2.3 `diffusion_lab/nn/unet.py`

A small U-Net for 28×28 MNIST images.

**Architecture** (inspired by DDPM paper, stripped to minimum):
```
Encoder: Conv2d(1→32) → ResBlock(32→64, stride=2) → ResBlock(64→128, stride=2)
Bottleneck: ResBlock(128→128) + time-conditioning via AdaGN
Decoder: Upsample + ResBlock(128→64) + skip → Upsample + ResBlock(64→32) + skip
Head: Conv2d(32→1)
```

Time conditioning: the bottleneck and each decoder ResBlock receive
`t_emb = MLP(sinusoidal_embed(t))` via **Adaptive Group Normalization (AdaGN)**:
`AdaGN(h, t) = t_scale · GroupNorm(h) + t_shift`.

**Class signature:**
```python
class SmallUNet(nn.Module):
    def __init__(self, in_channels: int = 1, base_channels: int = 32,
                 time_embed_dim: int = 128): ...

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        """x: (B,1,28,28), t: (B,).  Returns same shape."""
```

---

### 2.4 `diffusion_lab/schedulers/schedules.py`

Encapsulates the noise schedule as a single object, following the convention
`αt` = signal coefficient, `σt` = noise coefficient.

```python
@dataclass
class Schedule:
    T: int                          # number of discrete steps
    alpha_bar: Tensor               # (T+1,) — ᾱ₀=1, ᾱ_T≈0
    beta: Tensor                    # (T,)
    alpha: Tensor                   # (T,)  = 1 - beta
    sigma: Tensor                   # (T,)  = sqrt(1 - alpha_bar)
    log_snr: Tensor                 # (T,)  = log(alpha_bar / sigma²)
```

**Factory functions:**
```python
def linear_vp_schedule(T=1000, beta_start=1e-4, beta_end=0.02) -> Schedule
def cosine_vp_schedule(T=1000, s=0.008) -> Schedule   # Nichol & Dhariwal 2021
def ve_schedule(T=1000, sigma_min=0.01, sigma_max=50.0) -> Schedule  # geometric
```

**Mathematical correspondence (from quickguide):**
- VP (DDPM): `q(xₜ|x₀) = N(√ᾱₜ x₀, (1−ᾱₜ)I)` → `alpha_bar[t] = ᾱₜ`
- VE (SMLD): `q(xₜ|x₀) = N(x₀, σₜ²I)` → `alpha_bar[t] = 1`, `sigma[t] = σₜ`

---

### 2.5 `diffusion_lab/models/vae.py`

**Math grounding (quickguide Ch. 1):**

ELBO decomposition:
```
L(θ,φ;x) = E_{q_φ(z|x)}[log p_θ(x|z)] − KL(q_φ(z|x) ‖ p(z))
```
Reparameterization: `z = μ_φ(x) + σ_φ(x) ⊙ ε`, `ε ~ N(0,I)`.

Analytic KL for diagonal Gaussians:
```
KL(N(μ,σ²I) ‖ N(0,I)) = ½ Σᵢ (μᵢ² + σᵢ² − log σᵢ² − 1)
```

**Classes:**
```python
class Encoder(nn.Module):
    """MLP: x ∈ ℝᵈ → (μ_φ, log σ²_φ) ∈ ℝᵏ × ℝᵏ"""

class Decoder(nn.Module):
    """MLP: z ∈ ℝᵏ → μ_θ ∈ ℝᵈ  (Gaussian likelihood, fixed σ=0.1)"""

class VAE(nn.Module):
    def elbo(self, x) -> Tensor:          # returns scalar ELBO (higher = better)
    def loss(self, x) -> Tensor:          # returns -ELBO
    def reconstruct(self, x) -> Tensor:   # mean reconstruction
    def sample(self, n) -> Tensor:        # ancestral: z~N(0,I) → decode
    def encode(self, x) -> tuple[Tensor, Tensor]:   # (μ, log σ²)
```

**Latent dimension**: k=2 for 2-D data (allows direct latent-space plotting),
k=8 for MNIST.

---

### 2.6 `diffusion_lab/models/ddpm.py`

**Math grounding (quickguide Ch. 2):**

Forward marginal: `q(xₜ|x₀) = N(√ᾱₜ x₀, (1−ᾱₜ)I)`

Denoising posterior (tractable given x₀):
```
q(xₜ₋₁|xₜ,x₀) = N(μ̃ₜ, β̃ₜ I)
μ̃ₜ = (√ᾱₜ₋₁ βₜ x₀ + √αₜ (1−ᾱₜ₋₁) xₜ) / (1−ᾱₜ)
β̃ₜ = βₜ (1−ᾱₜ₋₁) / (1−ᾱₜ)
```

Three equivalent prediction targets (Tweedie triad, quickguide Thm 2):
```
ε-prediction:  L_ε = E[‖ε − ε_θ(xₜ, t)‖²]
x₀-prediction: L_x = E[‖x₀ − x̂₀(xₜ, t)‖²]
v-prediction:  L_v = E[‖v − v_θ(xₜ, t)‖²],  v = √ᾱₜ ε − √(1−ᾱₜ) x₀
```

**Interface:**
```python
class DDPM:
    def __init__(self, network: nn.Module, schedule: Schedule,
                 prediction: str = "epsilon"):  # "epsilon" | "x0" | "v"

    def q_sample(self, x0, t, noise=None) -> Tensor:
        """Sample xₜ ~ q(xₜ|x₀) in one shot (closed-form marginal)."""

    def loss(self, x0) -> Tensor:
        """Sample t uniformly, compute xₜ, return ‖pred − target‖²."""

    @torch.no_grad()
    def p_sample(self, xt, t) -> Tensor:
        """One step of ancestral sampling p_θ(xₜ₋₁|xₜ)."""

    @torch.no_grad()
    def sample(self, shape, device) -> Tensor:
        """Full reverse chain: xₜ → xₜ₋₁ → … → x₀."""
```

---

### 2.7 `diffusion_lab/models/ddim.py`

**Math grounding (quickguide Ch. 2 §DDIM):**

DDIM update rule (general σ_τ parameterization):
```
xₜ₋₁ = √ᾱₜ₋₁ · x̂₀(xₜ) + √(1−ᾱₜ₋₁ − σ²τ) · ε_θ(xₜ,t) + σ_τ · ε
```
where `x̂₀ = (xₜ − √(1−ᾱₜ) ε_θ) / √ᾱₜ` (Tweedie).
Setting `σ_τ = 0` gives the **deterministic** DDIM ODE sampler.

**Interface:**
```python
class DDIMSampler:
    def __init__(self, ddpm: DDPM, eta: float = 0.0,
                 num_steps: int = 50):
        """eta=0 → deterministic; eta=1 → recovers DDPM variance."""

    @torch.no_grad()
    def sample(self, shape, device,
               timestep_seq: list[int] | None = None) -> Tensor:
        """Sub-sequence sampling with user-chosen stride."""
```

DDIM wraps the trained DDPM network without retraining — it only changes
the sampling trajectory.

---

### 2.8 `diffusion_lab/models/score_sde.py`

**Math grounding (quickguide Ch. 3):**

**DSM loss** (Thm 3 — Vincent 2011 / Song & Ermon 2019):
```
L_DSM(θ, σᵢ) = E_{x₀,ε}[‖s_θ(x₀ + σᵢε, σᵢ) + ε/σᵢ‖²]
```
Multi-scale NCSN loss with geometric schedule σ₁ > … > σ_L:
```
L_NCSN(θ) = Σᵢ λ(σᵢ) · L_DSM(θ, σᵢ),   λ(σᵢ) = σᵢ²
```
Optimal score: `s*_θ(x̃, σ) = −(x̃ − x₀) / σ²  =  −ε/σ`.

**Annealed Langevin dynamics** (Def 3.4 in quickguide):
For each noise level σᵢ (decreasing), run K Langevin steps:
```
xₖ₊₁ = xₖ + (αᵢ/2) s_θ(xₖ, σᵢ) + √αᵢ · zₖ,   zₖ ~ N(0,I)
```

**VE-SDE forward process** (Song et al. 2021):
`dx = √(d[σ²]/dt) dW`,  i.e.,  `σ(t) = σ_min (σ_max/σ_min)^t`

**Interface:**
```python
class NCSN:
    def __init__(self, network: nn.Module,
                 sigmas: Tensor):            # decreasing geometric schedule

    def loss(self, x0) -> Tensor:            # NCSN multi-scale DSM loss

    @torch.no_grad()
    def annealed_langevin(self, shape, device,
                          n_steps_per_level: int = 100,
                          step_size: float = 2e-5) -> Tensor:

class VE_SDE:
    """Continuous-time VE score model + reverse SDE / probability flow ODE."""

    def __init__(self, network: nn.Module,
                 sigma_min: float = 0.01, sigma_max: float = 50.0): ...

    def sde_coeffs(self, t) -> tuple[Tensor, Tensor]:
        """Returns f(t), g(t) for dx = f dt + g dW."""

    def loss(self, x0) -> Tensor:            # DSM with continuous σ(t)

    @torch.no_grad()
    def reverse_sde_sample(self, shape, device, n_steps=500) -> Tensor:
        """Euler-Maruyama discretization of reverse SDE."""

    @torch.no_grad()
    def ode_sample(self, shape, device, n_steps=200) -> Tensor:
        """Euler discretization of probability flow ODE."""
```

---

### 2.9 `diffusion_lab/models/flow_matching.py`

**Math grounding (quickguide Ch. 4):**

Conditional flow matching (CFM) loss:
```
L_CFM = E_{t,x₀,x₁}[‖v_θ(xₜ, t) − uₜ(xₜ|x₀, x₁)‖²]
```
OT-Gaussian path (affine conditional flow):
```
xₜ = (1−t) x₀ + t x₁,   t ∈ [0,1]
uₜ(xₜ|x₀,x₁) = x₁ − x₀        (constant velocity along straight line)
```
with `x₀ ~ N(0,I)` (source) and `x₁ ~ p_data` (target).

This is **Conditional Flow Matching with OT couplings** — the simplest
and best-performing variant from Lipman et al. 2022 / Liu et al. 2022.

**Interface:**
```python
class CFM:
    def __init__(self, network: nn.Module, sigma_min: float = 1e-4):
        """sigma_min: small variance added to path for conditioning."""

    def sample_path(self, x0, x1, t) -> tuple[Tensor, Tensor]:
        """Returns (xₜ, uₜ): point and conditional velocity at time t."""

    def loss(self, x1) -> Tensor:
        """Sample t~U[0,1], x0~N(0,I), compute CFM regression loss."""

    @torch.no_grad()
    def sample(self, shape, device,
               n_steps: int = 100,
               method: str = "euler") -> Tensor:
        """ODE solve from t=0 to t=1: Euler or RK4."""
```

---

### 2.10 `diffusion_lab/training/trainer.py`

A generic, minimal Trainer that handles:
- Optimizer (`AdamW`, configurable lr / weight decay)
- Training loop with `tqdm` progress bar
- Loss logging (`train_losses: list[float]`)
- Checkpointing (save/load `.pt` files)
- Callback hook `on_epoch_end(trainer, epoch)` for notebook visualization

```python
class Trainer:
    def __init__(self, model, dataloader, optimizer=None,
                 device="cpu", checkpoint_dir=None): ...

    def train(self, n_epochs: int,
              callback_every: int = 10,
              callback: Callable | None = None) -> list[float]: ...

    def save(self, path: str): ...
    def load(self, path: str): ...
```

---

### 2.11 `diffusion_lab/utils/viz.py`

All visualization helpers needed by the notebooks.

```python
# 2D density / sample plots
def plot_samples(samples, ax=None, title="", alpha=0.4, s=5): ...
def plot_density(model_or_fn, grid_range=(-4,4), n_grid=200,
                 ax=None, cmap="viridis"): ...
def compare_panels(data, generated, titles=("Data", "Generated")): ...

# Training diagnostics
def plot_loss_curve(losses, log_scale=False, ax=None): ...

# Diffusion-specific
def plot_forward_chain(ddpm, x0, steps=(0,250,500,750,1000)): ...
def plot_reverse_chain(ddpm_or_ncsn, shape, device, n_frames=8): ...

# Latent space (VAE)
def plot_latent_space(vae, x, labels=None, ax=None): ...

# MNIST
def show_grid(images, nrow=8, title=""): ...
```

---

## 3. Notebooks — Content Spec

### `01_vae_2d.ipynb`
1. **Data**: generate 2-D spiral, moons, checkerboard (animated comparison).
2. **Model setup**: `VAE` with 2-D latent, `Encoder`/`Decoder` as 3-layer MLPs.
3. **Training**: 500 epochs, ELBO loss curve, reconstruction quality.
4. **Visualization**:
   - Latent space scatter (encode all data, plot z coloured by class).
   - Posterior spread: `σ_φ(x)` across the manifold.
   - Interpolation in latent space: walk z₀ → z₁ → decode.
   - Random generation: grid of samples from `z ~ N(0,I)`.

### `02_ddpm_ddim_2d.ipynb`
1. **Schedule**: linear VP, T=1000. Plot `ᾱₜ`, SNR.
2. **Forward chain**: visualize `q(xₜ|x₀)` at t=0,250,500,750,1000.
3. **Training**: ε-prediction, 1000 epochs. Loss curve.
4. **DDPM sampling**: full T=1000 reverse chain animation (15 frames).
5. **DDIM sampling**: same trained model, stride τ ∈ {100, 50, 20, 10}.
   Plot: sample quality vs. NFE (number of function evaluations).
6. **Ablation**: compare ε-pred vs. x₀-pred vs. v-pred (3 training runs).

### `03_score_sde_2d.ipynb`
1. **Score field**: visualize true score `∇log p(x)` computed via KDE.
2. **DSM loss**: show the denoising regression interpretation geometrically.
3. **NCSN training**: L=10 noise levels, geometric σ schedule.
4. **Annealed Langevin sampling**: animate per-level chains.
5. **VE-SDE**: continuous-time variant. Compare reverse SDE vs. ODE sampler.
6. **Comparison**: NCSN vs. VE-SDE samples side by side.

### `04_flow_matching_2d.ipynb`
1. **OT path**: visualize straight-line interpolants `xₜ = (1−t)x₀ + tx₁`.
2. **Conditional velocity field**: plot `uₜ(xₜ|x₀,x₁)` on a grid.
3. **Training**: CFM loss, 1000 epochs.
4. **Marginal velocity field**: plot `v_θ(·,t)` at t=0.1, 0.5, 0.9.
5. **ODE sampling**: Euler (100 steps) and RK4 (20 steps). Show trajectories.
6. **Comparison**: flow matching vs. DDIM-50 on sample quality.

### `05_ddpm_mnist.ipynb`
1. MNIST 28×28, normalize to [−1,1].
2. `SmallUNet`, cosine schedule T=1000, ε-prediction.
3. Train ~50 epochs (convergence on single GPU in ~20 min).
4. DDIM-100 samples shown as 8×8 grid.
5. Progressive denoising strip (x_T → … → x_0) for 5 examples.

### `06_flow_matching_mnist.ipynb`
1. `SmallUNet` as velocity network.
2. CFM with OT paths, T-embedding scaled to [0,1].
3. Euler-100 samples shown as 8×8 grid.
4. NFE sweep: Euler-{10, 20, 50, 100} quality comparison.
5. Interpolation in data space via ODE trajectories.

---

## 4. Implementation Order (Phases)

### Phase 1 — Scaffold & Data (Week 1)
- [ ] `pyproject.toml` + `requirements.txt`
- [ ] `diffusion_lab/data/toy.py` (spiral, moons, checkerboard)
- [ ] `diffusion_lab/utils/viz.py` (basic `plot_samples`, `plot_loss_curve`)
- [ ] `diffusion_lab/nn/mlp.py` (TimeMLP with sinusoidal embed)
- [ ] `diffusion_lab/training/trainer.py`
- [ ] Smoke-test notebook: load data, plot, check MLP forward pass

### Phase 2 — VAE (Week 1–2)
- [ ] `diffusion_lab/models/vae.py`
- [ ] `notebooks/01_vae_2d.ipynb`
- [ ] Verify: latent scatter clusters match data structure, ELBO ≤ 0 and increasing

### Phase 3 — DDPM + DDIM (Week 2)
- [ ] `diffusion_lab/schedulers/schedules.py` (linear VP, cosine VP)
- [ ] `diffusion_lab/models/ddpm.py`
- [ ] `diffusion_lab/models/ddim.py`
- [ ] `notebooks/02_ddpm_ddim_2d.ipynb`
- [ ] Verify: at T=1000, `q(xₜ|x₀)` is near-Gaussian; DDPM samples match data

### Phase 4 — Score SDE (Week 3)
- [ ] `diffusion_lab/schedulers/schedules.py` VE schedule
- [ ] `diffusion_lab/models/score_sde.py` (NCSN + VE-SDE)
- [ ] `notebooks/03_score_sde_2d.ipynb`
- [ ] Verify: learned score aligns with KDE score at low σ; Langevin converges

### Phase 5 — Flow Matching (Week 3–4)
- [ ] `diffusion_lab/models/flow_matching.py`
- [ ] `notebooks/04_flow_matching_2d.ipynb`
- [ ] Verify: Euler ODE with >50 steps produces clean samples; trajectory is ~straight

### Phase 6 — MNIST (Week 4)
- [ ] `diffusion_lab/nn/unet.py` (SmallUNet with AdaGN)
- [ ] `diffusion_lab/data/mnist.py`
- [ ] `notebooks/05_ddpm_mnist.ipynb`
- [ ] `notebooks/06_flow_matching_mnist.ipynb`
- [ ] Verify: FID not required; visual quality should show clear digit structure

---

## 5. Dependencies

```
# requirements.txt
torch>=2.1.0
torchvision>=0.16.0
numpy>=1.24
scipy>=1.10
scikit-learn>=1.3   # make_moons
matplotlib>=3.7
tqdm>=4.65
jupyter>=1.0
ipywidgets>=8.0     # interactive sliders in notebooks
einops>=0.7         # convenient tensor ops in UNet
```

Optional (for ODE solvers beyond Euler):
```
torchdiffeq>=0.2.3  # dopri5, rk4 adaptive solvers
```

---

## 6. Design Conventions

1. **Time convention**: t is always an integer index in `{1,…,T}` for DDPM/DDIM/NCSN,
   and a float in `[0,1]` for flow matching.  Networks always receive a float
   `t_normalized = t / T` so the same sinusoidal embedding works everywhere.

2. **Device agnostic**: all modules accept `device` as argument or use
   `next(self.parameters()).device` internally.

3. **No global state**: schedules and hyperparameters live on the model, not in module-level
   variables.  This makes multi-run notebook experiments clean.

4. **Reproducibility**: all stochastic operations accept an optional `generator` or `seed`
   argument so notebooks can produce deterministic figures.

5. **Type hints everywhere**: all public functions annotated with `Tensor`, `int`, `float`, etc.
   This is important for Cursor / Copilot assistance.

6. **Math-code correspondence**: every non-trivial formula in the implementation
   carries a docstring reference like `# Eq. (quickguide 2.7)`.

---

## 7. Testing Strategy

A minimal `tests/` directory (pytest) covering:

| Test file | What it checks |
|-----------|---------------|
| `test_data.py` | Dataset shapes, value ranges |
| `test_schedules.py` | `ᾱ_T ≈ 0`, monotonicity, `ᾱ_0 = 1` |
| `test_ddpm.py` | `q_sample` variance matches schedule; loss decreases over 50 steps |
| `test_vae.py` | KL term ≥ 0; ELBO ≤ 0 |
| `test_flow.py` | Path interpolant endpoints; loss finite and decreasing |
| `test_score.py` | DSM loss finite; optimal score closed-form check on Gaussian |

---

## 8. Reference Implementations Consulted

- [facebookresearch/flow\_matching](https://github.com/facebookresearch/flow_matching) — library structure, CFM math
- Ho et al. 2020 (DDPM) official TF → ported to PyTorch
- Song et al. 2021 (Score SDE) — [yang-song/score\_sde\_pytorch](https://github.com/yang-song/score_sde_pytorch)
- Kingma & Welling 2014 (VAE)

---

*Plan version 1.0 — 2026-05-04*
