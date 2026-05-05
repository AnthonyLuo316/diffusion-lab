# diffusion-lab

A modular PyTorch reference implementation of generative diffusion model frameworks,
written as the companion codebase to the lecture notes
*Understanding Theories of Diffusion Models*.

> **Lecture notes**: [`docs/quickguide.pdf`](docs/quickguide.pdf)

The library covers VAE, DDPM/DDIM, Score SDE (NCSN/VE-SDE), Flow Matching (CFM),
Classifier-Free Guidance (CFG), EDM preconditioning, inverse problems (DPS),
and the unifying probability-flow ODE perspective.

All toy experiments use 2-D datasets (spiral, two-moons, checkerboard) that train
in seconds on CPU; MNIST experiments show full pixel-space generation.

---

## Quick start

```bash
# Clone and install (editable)
git clone https://github.com/<your-handle>/diffusion-lab.git
cd diffusion-lab
pip install -e .

# Run the smoke test
jupyter notebook notebooks/00_smoke_test.ipynb

# Or run pytest (requires torch in your env)
pytest tests/ -v
```

**Python ≥ 3.10, PyTorch ≥ 2.1** are required.  No GPU needed for the 2-D notebooks;
a GPU or Apple Silicon accelerates the MNIST notebooks but is not required.

---

## Repository layout

```
diffusion_lab/              # installable core library
│
├── configs.py              # dataclass hyperparameter configs for all models
│
├── data/
│   ├── toy.py              # 2-D toy datasets (spiral, moons, checkerboard, …)
│   └── mnist.py            # MNIST DataLoader (auto-downloads to data/)
│
├── nn/
│   ├── mlp.py              # TimeMLP + SinusoidalEmbedding  (2-D experiments)
│   └── unet.py             # SmallUNet (~400K params, 28×28 MNIST)
│                           #   └─ AdaGN time/class conditioning
│
├── schedulers/
│   └── schedules.py        # VPSchedule (cosine / linear) + VESchedule
│
├── models/
│   ├── vae.py              # VAE: encoder, decoder, reparameterisation, ELBO
│   ├── ddpm.py             # DDPM: q_sample, p_sample, ε/x₀/v-prediction
│   ├── ddim.py             # DDIMSampler: deterministic & stochastic sub-sampling
│   ├── score_sde.py        # NCSN (DSM loss, annealed Langevin)
│   │                       # VE_SDE (reverse SDE + probability flow ODE)
│   ├── flow_matching.py    # CFM: OT-Gaussian path, Euler/RK4 sampler
│   ├── guidance.py         # CondDDPM (CFG label dropout) + CFGSampler
│   ├── edm.py              # EDMPrecon (c_skip/out/in/noise), Heun sampler
│   └── inverse.py          # Linear operators (mask/blur/SR) + dps_sample
│
├── training/
│   └── trainer.py          # Generic Trainer (AdamW, cosine LR, checkpointing)
│
└── utils/
    └── viz.py              # Plotting helpers for 2-D density and sample grids

notebooks/                  # Jupyter demo notebooks (one per topic)
tests/                      # pytest shape-correctness tests
```

---

## Notebooks

| # | Notebook | Framework | Experiment |
|---|----------|-----------|------------|
| 00 | `00_smoke_test` | — | Import and instantiation sanity check |
| 01 | `01_vae_2d` | **VAE** | ELBO, latent space, reconstruction on toy 2-D data |
| 02 | `02_ddpm_ddim_2d` | **DDPM + DDIM** | VP cosine schedule, ε/x₀/v-prediction, DDIM NFE sweep |
| 03 | `03_score_sde_2d` | **Score SDE** | DSM loss, annealed Langevin, reverse SDE vs PF-ODE |
| 04 | `04_flow_matching_2d` | **CFM** | OT-Gaussian path, Euler/RK4 solvers, morphing |
| 05 | `05_ddpm_mnist` | **DDPM** | SmallUNet on MNIST, DDIM-{10…200} quality sweep |
| 06 | `06_flow_matching_mnist` | **CFM** | SmallUNet velocity net, Euler/RK4 comparison |
| 07 | `07_score_sde_mnist` | **NCSN + VE-SDE** | Score field visualisation, Langevin chains |
| 08 | `08_inverse_problems` | **DPS** | Inpainting / deblurring / super-resolution on MNIST |
| 09 | `09_guidance` | **CFG** | Guidance scale sweep, class grid, class-accuracy vs w |
| 10 | `10_edm_unification` | **EDM + Unification** | Preconditioning scalars, Heun vs Euler, DDIM=Euler(PF-ODE) |

---

## Key mathematical concepts

| Concept | File | Reference |
|---------|------|-----------|
| VP marginal $q(x_t \mid x_0) = \mathcal{N}(\sqrt{\bar\alpha_t}\,x_0,\,(1-\bar\alpha_t)I)$ | `schedulers/schedules.py` | Ho et al. 2020 |
| Tweedie denoising: $\hat x_0 = (x_t - \sigma_t\,\hat\varepsilon) / \sqrt{\bar\alpha_t}$ | `models/ddpm.py` | Robbins 1956 |
| DDIM as Euler(PF-ODE) at $\eta=0$ | `models/ddim.py` | Song et al. 2021 |
| Denoising Score Matching (DSM) | `models/score_sde.py` | Vincent 2011 |
| OT-Gaussian conditional flow | `models/flow_matching.py` | Lipman et al. 2022 |
| CFG guided score: $\hat\varepsilon + w(\hat\varepsilon_c - \hat\varepsilon)$ | `models/guidance.py` | Ho & Salimans 2022 |
| EDM preconditioning $(c_\text{skip}, c_\text{out}, c_\text{in}, c_\text{noise})$ | `models/edm.py` | Karras et al. 2022 |
| DPS likelihood gradient: $-(\zeta/\|r\|)\nabla_{x_t}\|r\|^2$ | `models/inverse.py` | Chung et al. 2022 |

---

## Configuration

All models expose a dataclass config in `diffusion_lab/configs.py`.
Import it to avoid magic numbers in notebooks:

```python
from diffusion_lab.configs import DDPMConfig, TrainConfig, EDMConfig

cfg   = DDPMConfig(T=1000, schedule='cosine', prediction='epsilon')
train = TrainConfig(epochs=30, lr=3e-4, batch_size=256)
```

---

## Running tests

```bash
pytest tests/ -v
```

The test suite checks output shapes for every model (UNet, DDPM, DDIM, CFG,
EDM, inverse operators, configs).  All tests run on CPU with random weights
and complete in under 30 seconds.

---

## Checkpoints

Trained `.pt` checkpoints are **not** included in the repository (see `.gitignore`).
Run each MNIST notebook from top to bottom; checkpoints are saved to `checkpoints/`
and automatically reloaded on subsequent runs.

---

## License

MIT
