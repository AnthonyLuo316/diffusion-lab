"""
diffusion_lab/utils/viz.py
Visualization helpers for 2-D generative model experiments.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.axes import Axes
from typing import Callable, Sequence

import torch
from torch import Tensor

__all__ = [
    "plot_samples",
    "plot_density",
    "compare_panels",
    "plot_loss_curve",
    "plot_forward_chain",
    "plot_reverse_chain",
    "plot_latent_space",
    "show_grid",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_numpy(x) -> np.ndarray:
    if isinstance(x, Tensor):
        return x.detach().cpu().float().numpy()
    return np.asarray(x, dtype=np.float32)


# ---------------------------------------------------------------------------
# 2-D scatter / density
# ---------------------------------------------------------------------------

def plot_samples(
    samples,
    ax: Axes | None = None,
    title: str = "",
    alpha: float = 0.35,
    s: float = 4,
    color: str = "steelblue",
    xlim: tuple | None = None,
    ylim: tuple | None = None,
) -> Axes:
    """Scatter plot of 2-D samples array (N, 2)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 4))
    xy = _to_numpy(samples)
    ax.scatter(xy[:, 0], xy[:, 1], alpha=alpha, s=s, c=color, rasterized=True)
    ax.set_title(title)
    ax.set_aspect("equal")
    if xlim: ax.set_xlim(xlim)
    if ylim: ax.set_ylim(ylim)
    return ax


def plot_density(
    score_fn: Callable[[Tensor, Tensor], Tensor] | None = None,
    log_prob_fn: Callable[[Tensor], Tensor] | None = None,
    grid_range: tuple = (-4, 4),
    n_grid: int = 200,
    ax: Axes | None = None,
    cmap: str = "viridis",
    title: str = "",
    device: str = "cpu",
) -> Axes:
    """
    Plot the (unnormalized) density or score magnitude on a 2-D grid.

    Pass exactly one of score_fn or log_prob_fn.

    score_fn     : (x: (N,2) Tensor, t: (N,) Tensor) → (N,2)  score ∇ log p
    log_prob_fn  : (x: (N,2) Tensor)                 → (N,)   log density
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 4))

    lo, hi = grid_range
    xs = torch.linspace(lo, hi, n_grid, device=device)
    ys = torch.linspace(lo, hi, n_grid, device=device)
    gx, gy = torch.meshgrid(xs, ys, indexing="xy")
    grid = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1)  # (n_grid², 2)

    with torch.no_grad():
        if log_prob_fn is not None:
            vals = log_prob_fn(grid).reshape(n_grid, n_grid)
            z = vals.cpu().numpy()
        elif score_fn is not None:
            t = torch.zeros(grid.shape[0], device=device)
            s = score_fn(grid, t)          # (N, 2)
            vals = s.norm(dim=-1).reshape(n_grid, n_grid)
            z = vals.cpu().numpy()
        else:
            raise ValueError("Provide score_fn or log_prob_fn.")

    ax.contourf(xs.cpu(), ys.cpu(), z, levels=50, cmap=cmap)
    ax.set_title(title)
    ax.set_aspect("equal")
    return ax


def compare_panels(
    *arrays,
    titles: Sequence[str] | None = None,
    figsize_per: tuple = (4, 4),
    alpha: float = 0.35,
    s: float = 4,
) -> plt.Figure:
    """Side-by-side scatter panels for an arbitrary number of sample arrays."""
    n = len(arrays)
    if titles is None:
        titles = [f"Panel {i+1}" for i in range(n)]
    fig, axes = plt.subplots(1, n, figsize=(figsize_per[0] * n, figsize_per[1]))
    if n == 1:
        axes = [axes]
    for ax, arr, title in zip(axes, arrays, titles):
        plot_samples(arr, ax=ax, title=title, alpha=alpha, s=s)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Training diagnostics
# ---------------------------------------------------------------------------

def plot_loss_curve(
    losses: list[float] | np.ndarray,
    log_scale: bool = False,
    ax: Axes | None = None,
    title: str = "Training loss",
    color: str = "tab:blue",
    smooth: int = 1,
) -> Axes:
    """
    Plot a training loss curve.

    Parameters
    ----------
    losses    : sequence of per-step or per-epoch losses
    log_scale : use log y-axis
    smooth    : running-average window (1 = no smoothing)
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 3))
    losses = np.asarray(losses, dtype=np.float32)
    if smooth > 1:
        kernel = np.ones(smooth) / smooth
        losses_smooth = np.convolve(losses, kernel, mode="valid")
    else:
        losses_smooth = losses
    ax.plot(losses_smooth, color=color, linewidth=1.5)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    if log_scale:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    return ax


# ---------------------------------------------------------------------------
# Diffusion-specific
# ---------------------------------------------------------------------------

def plot_forward_chain(
    samples,
    noisy_list: list,
    t_labels: Sequence[int | str] | None = None,
    figsize: tuple = (3, 3),
) -> plt.Figure:
    """
    Visualize the forward diffusion chain.

    Parameters
    ----------
    samples     : (N, 2) clean data
    noisy_list  : list of (N, 2) arrays at different noise levels
    t_labels    : labels for each panel (e.g. [0, 250, 500, 750, 1000])
    """
    panels = [samples] + list(noisy_list)
    n = len(panels)
    if t_labels is None:
        t_labels = [f"t={i}" for i in range(n)]
    fig, axes = plt.subplots(1, n, figsize=(figsize[0] * n, figsize[1]))
    if n == 1:
        axes = [axes]
    for ax, arr, label in zip(axes, panels, t_labels):
        plot_samples(arr, ax=ax, title=str(label))
    fig.tight_layout()
    return fig


def plot_reverse_chain(
    frames: list,
    t_labels: Sequence[int | str] | None = None,
    figsize: tuple = (3, 3),
) -> plt.Figure:
    """
    Visualize a sequence of intermediate reverse-process samples.

    frames : list of (N, 2) or (N, 1, H, W) arrays
    """
    n = len(frames)
    if t_labels is None:
        t_labels = [f"step {i}" for i in range(n)]
    fig, axes = plt.subplots(1, n, figsize=(figsize[0] * n, figsize[1]))
    if n == 1:
        axes = [axes]
    for ax, frame, label in zip(axes, frames, t_labels):
        arr = _to_numpy(frame)
        if arr.ndim == 4:   # image batch (B, 1, H, W)
            # show first image
            ax.imshow(arr[0, 0], cmap="gray", vmin=-1, vmax=1)
            ax.axis("off")
        else:
            plot_samples(arr, ax=ax, title=str(label))
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# VAE latent space
# ---------------------------------------------------------------------------

def plot_latent_space(
    z: "np.ndarray | Tensor",
    labels: "np.ndarray | None" = None,
    ax: Axes | None = None,
    title: str = "Latent space",
    cmap: str = "tab10",
    alpha: float = 0.5,
    s: float = 6,
) -> Axes:
    """Scatter of 2-D latent codes, optionally coloured by label."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    z = _to_numpy(z)
    if labels is not None:
        labels = np.asarray(labels)
        scatter = ax.scatter(z[:, 0], z[:, 1], c=labels, cmap=cmap,
                             alpha=alpha, s=s, rasterized=True)
        plt.colorbar(scatter, ax=ax)
    else:
        ax.scatter(z[:, 0], z[:, 1], alpha=alpha, s=s,
                   color="steelblue", rasterized=True)
    ax.set_title(title)
    ax.set_aspect("equal")
    return ax


# ---------------------------------------------------------------------------
# MNIST / image grid
# ---------------------------------------------------------------------------

def show_grid(
    images,
    nrow: int = 8,
    title: str = "",
    figsize: tuple | None = None,
    value_range: tuple = (-1, 1),
) -> plt.Figure:
    """
    Display a batch of images as a grid.

    images : (B, 1, H, W) or (B, H, W) tensor / ndarray, values in value_range
    """
    imgs = _to_numpy(images)
    if imgs.ndim == 4:
        imgs = imgs[:, 0]               # (B, H, W)
    # normalize to [0,1]
    lo, hi = value_range
    imgs = (imgs - lo) / (hi - lo + 1e-8)
    imgs = imgs.clip(0, 1)

    B, H, W = imgs.shape
    ncol = nrow
    nrow_actual = (B + ncol - 1) // ncol
    if figsize is None:
        figsize = (ncol * 1.2, nrow_actual * 1.2)
    fig, axes = plt.subplots(nrow_actual, ncol, figsize=figsize)
    axes = np.array(axes).reshape(-1)
    for i, ax in enumerate(axes):
        if i < B:
            ax.imshow(imgs[i], cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
    if title:
        fig.suptitle(title, y=1.01)
    fig.tight_layout()
    return fig
