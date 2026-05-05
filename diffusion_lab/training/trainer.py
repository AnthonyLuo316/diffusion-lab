"""
diffusion_lab/training/trainer.py
Generic Trainer for all diffusion-lab models.

Usage
-----
    trainer = Trainer(
        model   = ddpm,
        loader  = get_dataloader("spiral", batch_size=256),
        lr      = 3e-4,
        device  = "cuda",
    )
    losses = trainer.train(
        n_steps       = 20_000,
        callback_every= 2_000,
        callback      = lambda tr, step: plot_samples(tr.generate(256)),
    )
"""

from __future__ import annotations

import os
import time
from typing import Callable, Any

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

__all__ = ["Trainer"]


class Trainer:
    """
    A minimal, framework-agnostic trainer.

    The model is expected to expose a `.loss(x0) -> Tensor` method
    that returns a scalar loss given a batch of clean data x0.

    Parameters
    ----------
    model          : any object with a `.loss(x0)` method and `.parameters()`
                     (or a dict of submodules for VAE-style dual optimizers)
    loader         : DataLoader yielding (x,) tuples
    lr             : learning rate for AdamW
    weight_decay   : L2 regularization for AdamW
    grad_clip      : max gradient norm (None to disable)
    device         : 'cpu', 'cuda', 'mps', etc.
    checkpoint_dir : directory for saving / loading .pt checkpoints
    """

    def __init__(
        self,
        model: Any,
        loader: DataLoader,
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
        grad_clip: float | None = 1.0,
        device: str | torch.device = "cpu",
        checkpoint_dir: str | None = None,
    ) -> None:
        self.model = model
        self.loader = loader
        self.device = torch.device(device)
        self.grad_clip = grad_clip
        self.checkpoint_dir = checkpoint_dir

        # Move model to device if it has parameters
        if hasattr(model, "to"):
            model.to(self.device)

        # Build optimizer over all learnable parameters
        params = self._collect_params(model)
        self.optimizer = AdamW(params, lr=lr, weight_decay=weight_decay)

        # History
        self.train_losses: list[float] = []
        self.steps_done: int = 0

        if checkpoint_dir is not None:
            os.makedirs(checkpoint_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _collect_params(model) -> list:
        """Collect trainable parameters from model or dict of sub-modules."""
        if isinstance(model, dict):
            params = []
            for v in model.values():
                if hasattr(v, "parameters"):
                    params += list(v.parameters())
            return params
        if hasattr(model, "parameters"):
            return list(model.parameters())
        raise TypeError(f"Cannot collect parameters from {type(model)}.")

    def _infinite_loader(self):
        """Yields batches from loader indefinitely."""
        while True:
            for batch in self.loader:
                yield batch

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        n_steps: int,
        callback_every: int = 1000,
        callback: Callable[["Trainer", int], None] | None = None,
        log_every: int = 100,
    ) -> list[float]:
        """
        Run the training loop for n_steps gradient steps.

        Parameters
        ----------
        n_steps       : total gradient steps to perform
        callback_every: call `callback(trainer, step)` every this many steps
        callback      : optional callable for visualization / logging
        log_every     : print average loss every this many steps

        Returns
        -------
        losses : list of per-step scalar losses
        """
        self.model.train() if hasattr(self.model, "train") else None

        loader_iter = self._infinite_loader()
        pbar = tqdm(total=n_steps, desc="Training", dynamic_ncols=True)

        t0 = time.time()
        for local_step in range(n_steps):
            batch = next(loader_iter)
            # Unpack: DataLoader returns list/tuple of tensors
            if isinstance(batch, (list, tuple)):
                x0 = batch[0]
            else:
                x0 = batch
            x0 = x0.to(self.device)

            self.optimizer.zero_grad()
            loss = self.model.loss(x0)
            loss.backward()

            if self.grad_clip is not None:
                params = self._collect_params(self.model)
                nn.utils.clip_grad_norm_(params, self.grad_clip)

            self.optimizer.step()

            loss_val = loss.item()
            self.train_losses.append(loss_val)
            self.steps_done += 1

            pbar.update(1)
            if (local_step + 1) % log_every == 0:
                recent = self.train_losses[-log_every:]
                avg = sum(recent) / len(recent)
                elapsed = time.time() - t0
                pbar.set_postfix(loss=f"{avg:.4f}", elapsed=f"{elapsed:.1f}s")

            if callback is not None and (local_step + 1) % callback_every == 0:
                self.model.eval() if hasattr(self.model, "eval") else None
                callback(self, self.steps_done)
                self.model.train() if hasattr(self.model, "train") else None

        pbar.close()
        return self.train_losses

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save model state dict and optimizer state to a .pt file."""
        state: dict = {
            "steps_done": self.steps_done,
            "train_losses": self.train_losses,
            "optimizer": self.optimizer.state_dict(),
        }
        # Model state
        if isinstance(self.model, dict):
            state["model"] = {k: v.state_dict() for k, v in self.model.items()
                              if hasattr(v, "state_dict")}
        elif hasattr(self.model, "state_dict"):
            state["model"] = self.model.state_dict()
        torch.save(state, path)

    def load(self, path: str) -> None:
        """Load model state dict and optimizer state from a .pt file."""
        state = torch.load(path, map_location=self.device)
        self.steps_done = state.get("steps_done", 0)
        self.train_losses = state.get("train_losses", [])
        if "optimizer" in state:
            self.optimizer.load_state_dict(state["optimizer"])
        if "model" in state:
            if isinstance(self.model, dict):
                for k, sd in state["model"].items():
                    if k in self.model and hasattr(self.model[k], "load_state_dict"):
                        self.model[k].load_state_dict(sd)
            elif hasattr(self.model, "load_state_dict"):
                self.model.load_state_dict(state["model"])

    def save_checkpoint(self, tag: str = "") -> str:
        """Save to checkpoint_dir/{tag or step}.pt and return the path."""
        if self.checkpoint_dir is None:
            raise RuntimeError("checkpoint_dir was not set.")
        name = tag if tag else f"step_{self.steps_done:07d}"
        path = os.path.join(self.checkpoint_dir, f"{name}.pt")
        self.save(path)
        return path
