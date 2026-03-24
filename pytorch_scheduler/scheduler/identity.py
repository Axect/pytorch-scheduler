"""Identity scheduler — returns the base learning rate unchanged."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pytorch_scheduler.base import BaseScheduler

if TYPE_CHECKING:
    from torch.optim import Optimizer


class IdentityScheduler(BaseScheduler):
    """No-op scheduler that keeps the learning rate constant.

    Useful as a baseline or when a scheduler interface is required
    but no actual scheduling is desired.

    Formula:
        lr(t) = base_lr   for all t
    """

    step_unit = "step"
    needs_total_steps = False
    needs_metric = False

    def __init__(
        self,
        optimizer: Optimizer,
        last_epoch: int = -1,
    ) -> None:
        super().__init__(optimizer, last_epoch=last_epoch)

    def _lr_at(self, step: int, base_lrs: list[float]) -> list[float]:
        return list(base_lrs)
