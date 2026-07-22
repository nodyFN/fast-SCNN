"""
Learning rate schedulers.

PolyLR
------
Iteration-based polynomial decay:  lr = base_lr × (1 − iter/total_iters)^power

Paper setting: power = 0.9

CosineAnnealingLR
-----------------
Thin wrapper around ``torch.optim.lr_scheduler.CosineAnnealingLR``.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler, CosineAnnealingLR, MultiStepLR


class PolyLR(_LRScheduler):
    """Polynomial LR decay, updated per **iteration** (not per epoch).

    Parameters
    ----------
    optimizer : Optimizer
    total_iters : int
        Total number of optimiser steps across all epochs.
    power : float
        Polynomial power (paper default 0.9).
    min_lr : float
        Minimum learning rate floor.
    last_epoch : int
        For resuming. -1 means start from scratch.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        total_iters: int,
        power: float = 0.9,
        min_lr: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        self.total_iters = max(total_iters, 1)
        self.power = power
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        # Clamp progress to [0, 1] to avoid negative base
        progress = min(self._step_count / self.total_iters, 1.0)
        factor = (1.0 - progress) ** self.power
        return [
            max(base_lr * factor, self.min_lr)
            for base_lr in self.base_lrs
        ]

    def state_dict(self) -> dict:
        state = super().state_dict()
        # Remove hyperparameters from saved state so we don't restore them,
        # allowing the new run's configuration to take precedence
        state.pop("total_iters", None)
        state.pop("power", None)
        state.pop("min_lr", None)
        return state

    def load_state_dict(self, state_dict: dict) -> None:
        # Preserve the new configuration parameters
        saved_attrs = {
            "total_iters": self.total_iters,
            "power": self.power,
            "min_lr": self.min_lr,
        }
        super().load_state_dict(state_dict)
        # Restore them back to the active configuration
        self.__dict__.update(saved_attrs)


def build_scheduler(
    name: str,
    optimizer: Optimizer,
    total_iters: int,
    epochs: int,
    poly_power: float = 0.9,
    cosine_eta_min: float = 1e-6,
    milestones: Optional[list] = None,
    gamma: float = 0.1,
) -> _LRScheduler:
    """Factory to build a scheduler by name.

    Parameters
    ----------
    name : "poly" | "cosine" | "multistep"
    total_iters : total optimizer steps (for PolyLR)
    epochs : total epochs (for CosineAnnealing T_max)
    milestones : list of int, epoch milestones (for MultiStepLR)
    gamma : float, LR decay factor at each milestone (for MultiStepLR)
    """
    name = name.lower()
    if name == "poly":
        return PolyLR(optimizer, total_iters=total_iters, power=poly_power)
    elif name == "cosine":
        return CosineAnnealingLR(optimizer, T_max=epochs, eta_min=cosine_eta_min)
    elif name == "multistep":
        if milestones is None:
            raise ValueError(
                "MultiStepLR requires 'milestones' (list of epoch numbers)"
            )
        return MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
    else:
        raise ValueError(f"Unknown scheduler '{name}'. Choose 'poly', 'cosine', or 'multistep'.")
