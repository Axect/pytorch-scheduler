from __future__ import annotations

import fnmatch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch.optim import Optimizer
    from torch.optim.lr_scheduler import LRScheduler

from pytorch_scheduler.scheduler.chebyshev import ChebyshevScheduler
from pytorch_scheduler.scheduler.cosine_annealing import CosineAnnealingWarmupRestarts
from pytorch_scheduler.scheduler.cosine_warmup import CosineWithWarmupScheduler
from pytorch_scheduler.scheduler.flat_cosine import FlatCosineScheduler
from pytorch_scheduler.scheduler.hyperbolic import ExpHyperbolicLRScheduler, HyperbolicLRScheduler
from pytorch_scheduler.scheduler.identity import IdentityScheduler
from pytorch_scheduler.scheduler.inverse_sqrt import InverseSqrtScheduler
from pytorch_scheduler.scheduler.k_decay import KDecayScheduler
from pytorch_scheduler.scheduler.linear_decay import LinearDecayScheduler
from pytorch_scheduler.scheduler.polynomial import PolynomialScheduler
from pytorch_scheduler.scheduler.power_decay import PowerDecayScheduler
from pytorch_scheduler.scheduler.rex import RexScheduler
from pytorch_scheduler.scheduler.slanted_triangular import SlantedTriangularScheduler
from pytorch_scheduler.scheduler.tanh_decay import TanhDecayScheduler
from pytorch_scheduler.scheduler.trapezoidal import TrapezoidalScheduler
from pytorch_scheduler.scheduler.warmup_hold_cosine import WarmupHoldCosineScheduler
from pytorch_scheduler.scheduler.wsd import WSDScheduler

SCHEDULER_LIST: list[type] = [
    RexScheduler,
    InverseSqrtScheduler,
    LinearDecayScheduler,
    TrapezoidalScheduler,
    CosineAnnealingWarmupRestarts,
    WSDScheduler,
    KDecayScheduler,
    TanhDecayScheduler,
    SlantedTriangularScheduler,
    FlatCosineScheduler,
    HyperbolicLRScheduler,
    ExpHyperbolicLRScheduler,
    PolynomialScheduler,
    ChebyshevScheduler,
    PowerDecayScheduler,
    CosineWithWarmupScheduler,
    WarmupHoldCosineScheduler,
    IdentityScheduler,
]

SCHEDULERS: dict[str, type] = {cls.__name__.lower(): cls for cls in SCHEDULER_LIST}

# Shorthand aliases for convenient use in create_scheduler / presets
SCHEDULERS["chebyshev"] = ChebyshevScheduler
SCHEDULERS["cosine_annealing"] = CosineAnnealingWarmupRestarts
SCHEDULERS["cosine_with_warmup"] = CosineWithWarmupScheduler
SCHEDULERS["exp_hyperbolic"] = ExpHyperbolicLRScheduler
SCHEDULERS["flat_cosine"] = FlatCosineScheduler
SCHEDULERS["hyperbolic"] = HyperbolicLRScheduler
SCHEDULERS["identity"] = IdentityScheduler
SCHEDULERS["inverse_sqrt"] = InverseSqrtScheduler
SCHEDULERS["k_decay"] = KDecayScheduler
SCHEDULERS["linear_decay"] = LinearDecayScheduler
SCHEDULERS["polynomial"] = PolynomialScheduler
SCHEDULERS["power_decay"] = PowerDecayScheduler
SCHEDULERS["rex"] = RexScheduler
SCHEDULERS["slanted_triangular"] = SlantedTriangularScheduler
SCHEDULERS["tanh_decay"] = TanhDecayScheduler
SCHEDULERS["trapezoidal"] = TrapezoidalScheduler
SCHEDULERS["warmup_hold_cosine"] = WarmupHoldCosineScheduler
SCHEDULERS["wsd"] = WSDScheduler


def load_scheduler(name: str) -> type:
    """Look up a scheduler class by name (case-insensitive).

    Args:
        name: Scheduler class name (case-insensitive).

    Returns:
        The scheduler class.

    Raises:
        ValueError: If no scheduler with the given name is found.
    """
    key = name.lower()
    if key not in SCHEDULERS:
        available = sorted(SCHEDULERS.keys())
        raise ValueError(f"Unknown scheduler '{name}'. Available schedulers: {available}")
    return SCHEDULERS[key]


def create_scheduler(optimizer: Optimizer, name: str, **kwargs) -> LRScheduler:
    """Factory that creates a scheduler instance.

    Args:
        optimizer: The optimizer to wrap.
        name: Scheduler class name (case-insensitive).
        **kwargs: Additional keyword arguments passed to the scheduler constructor.

    Returns:
        An instantiated LRScheduler.
    """
    cls = load_scheduler(name)
    return cls(optimizer, **kwargs)


def create_scheduler_from_plan(
    optimizer: Optimizer,
    name: str,
    epochs: int,
    steps_per_epoch: int,
    grad_accum_steps: int = 1,
    **kwargs,
) -> LRScheduler:
    """Create a scheduler with automatic total_steps calculation.

    Computes total_steps from training plan parameters, removing the
    error-prone manual calculation.

    Args:
        optimizer: The optimizer to schedule.
        name: Scheduler name.
        epochs: Number of training epochs.
        steps_per_epoch: Steps per epoch (len(dataloader)).
        grad_accum_steps: Gradient accumulation steps (default: 1).
        **kwargs: Additional scheduler parameters.

    Returns:
        A configured scheduler instance.
    """
    total_steps = (epochs * steps_per_epoch) // grad_accum_steps
    return create_scheduler(optimizer, name, total_steps=total_steps, **kwargs)


def get_supported_schedulers(pattern: str = "*") -> list[str]:
    """Return a sorted list of all available scheduler names.

    Args:
        pattern: fnmatch-style glob pattern to filter names.
                 Defaults to '*' which returns all schedulers.

    Returns:
        Sorted list of matching scheduler names.
    """
    names = sorted(SCHEDULERS.keys())
    if pattern == "*":
        return names
    return [name for name in names if fnmatch.fnmatch(name, pattern)]


__all__ = [
    "SCHEDULERS",
    "SCHEDULER_LIST",
    "ChebyshevScheduler",
    "CosineAnnealingWarmupRestarts",
    "CosineWithWarmupScheduler",
    "ExpHyperbolicLRScheduler",
    "FlatCosineScheduler",
    "HyperbolicLRScheduler",
    "IdentityScheduler",
    "InverseSqrtScheduler",
    "KDecayScheduler",
    "LinearDecayScheduler",
    "PolynomialScheduler",
    "PowerDecayScheduler",
    "RexScheduler",
    "SlantedTriangularScheduler",
    "TanhDecayScheduler",
    "TrapezoidalScheduler",
    "WSDScheduler",
    "WarmupHoldCosineScheduler",
    "create_scheduler",
    "create_scheduler_from_plan",
    "get_supported_schedulers",
    "load_scheduler",
]
