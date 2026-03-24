"""Contract tests: universal invariants that all schedulers must satisfy."""

from __future__ import annotations

import math
import warnings

import pytest
import torch

import pytorch_scheduler as ps
from pytorch_scheduler.scheduler import SCHEDULER_LIST

warnings.filterwarnings("ignore")


def make_optimizer(lr: float = 0.1) -> torch.optim.SGD:
    model = torch.nn.Linear(2, 1)
    return torch.optim.SGD(model.parameters(), lr=lr)


def make_multi_optimizer() -> torch.optim.SGD:
    p1 = torch.randn(2, requires_grad=True)
    p2 = torch.randn(3, requires_grad=True)
    return torch.optim.SGD([{"params": [p1], "lr": 0.1}, {"params": [p2], "lr": 0.05}])


# ---------------------------------------------------------------------------
# Per-scheduler constructor kwargs and total_steps for step-through testing.
# These configs are valid and produce well-behaved schedules.
# ---------------------------------------------------------------------------

SCHEDULER_CONFIGS: dict[type, dict] = {
    ps.RexScheduler: {
        "kwargs": {"total_steps": 200},
        "total_steps": 200,
    },
    ps.InverseSqrtScheduler: {
        "kwargs": {"warmup_steps": 50},
        "total_steps": 150,
    },
    ps.LinearDecayScheduler: {
        "kwargs": {"total_steps": 200, "warmup_steps": 20, "min_lr": 0.0},
        "total_steps": 200,
    },
    ps.TrapezoidalScheduler: {
        "kwargs": {"total_steps": 200, "warmup_steps": 20, "decay_steps": 40, "min_lr": 0.0},
        "total_steps": 200,
    },
    ps.CosineAnnealingWarmupRestarts: {
        "kwargs": {"first_cycle_steps": 200, "warmup_steps": 20, "max_lr": 0.1, "min_lr": 0.001},
        "total_steps": 200,
    },
    ps.WSDScheduler: {
        "kwargs": {"total_steps": 200, "warmup_steps": 20, "stable_steps": 100, "min_lr": 0.0},
        "total_steps": 200,
    },
    ps.KDecayScheduler: {
        "kwargs": {"total_steps": 200, "k": 2.0, "min_lr": 0.0},
        "total_steps": 200,
    },
    ps.TanhDecayScheduler: {
        "kwargs": {"total_steps": 200, "steepness": 3.0, "min_lr": 0.0},
        "total_steps": 200,
    },
    ps.SlantedTriangularScheduler: {
        "kwargs": {"total_steps": 200, "cut_frac": 0.1, "ratio": 32.0},
        "total_steps": 200,
    },
    ps.FlatCosineScheduler: {
        "kwargs": {"total_steps": 200, "flat_fraction": 0.7, "min_lr": 0.0},
        "total_steps": 200,
    },
    ps.PolynomialScheduler: {
        "kwargs": {"total_steps": 200, "power": 2.0, "min_lr": 0.0, "warmup_steps": 20},
        "total_steps": 200,
    },
    ps.ChebyshevScheduler: {
        "kwargs": {"total_steps": 200, "min_lr": 0.0},
        "total_steps": 200,
    },
    ps.PowerDecayScheduler: {
        "kwargs": {"total_steps": 200, "warmup_steps": 20, "alpha": 0.5, "min_lr": 0.0},
        "total_steps": 200,
    },
    ps.HyperbolicLRScheduler: {
        "kwargs": {"total_steps": 200, "upper_bound": 500, "min_lr": 1e-6, "warmup_steps": 20},
        "total_steps": 200,
    },
    ps.ExpHyperbolicLRScheduler: {
        "kwargs": {"total_steps": 200, "upper_bound": 500, "min_lr": 1e-6, "warmup_steps": 20},
        "total_steps": 200,
    },
    ps.CosineWithWarmupScheduler: {
        "kwargs": {"total_steps": 200, "warmup_steps": 20, "min_lr": 0.0},
        "total_steps": 200,
    },
    ps.WarmupHoldCosineScheduler: {
        "kwargs": {"total_steps": 200, "warmup_steps": 20, "hold_steps": 80, "min_lr": 0.0},
        "total_steps": 200,
    },
    ps.IdentityScheduler: {
        "kwargs": {},
        "total_steps": 200,
    },
}

# Build parametrize list in same order as SCHEDULER_LIST for consistent naming
SCHEDULER_PARAMS = [pytest.param(cls, SCHEDULER_CONFIGS[cls], id=cls.__name__) for cls in SCHEDULER_LIST]


# ---------------------------------------------------------------------------
# Contract 1: Finite outputs — _lr_at never returns NaN or Inf
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls,config", SCHEDULER_PARAMS)
def test_contract_finite_outputs(cls, config):
    """_lr_at(step, base_lrs) must return finite values for all steps in range."""
    opt = make_optimizer(lr=0.1)
    sched = cls(opt, **config["kwargs"])
    base_lrs = list(sched.base_lrs)
    total = config["total_steps"]

    for step in range(total + 1):
        lrs = sched._lr_at(step, base_lrs)
        for lr in lrs:
            assert math.isfinite(lr), f"{cls.__name__}: non-finite LR {lr!r} at step={step}"


# ---------------------------------------------------------------------------
# Contract 2: Bounded outputs — LR in [0, max(base_lrs)]
# (ChebyshevScheduler can oscillate but should still stay in [min_lr, base_lr])
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls,config", SCHEDULER_PARAMS)
def test_contract_bounded_outputs(cls, config):
    """LR must stay within [0, max(base_lrs)] for all steps in range."""
    opt = make_optimizer(lr=0.1)
    sched = cls(opt, **config["kwargs"])
    base_lrs = list(sched.base_lrs)
    max_base_lr = max(base_lrs)
    total = config["total_steps"]

    for step in range(total + 1):
        lrs = sched._lr_at(step, base_lrs)
        for lr in lrs:
            # CosineAnnealingWarmupRestarts uses its own max_lr
            upper = getattr(sched, "max_lr", max_base_lr)
            assert lr >= -1e-9, f"{cls.__name__}: negative LR {lr} at step={step}"
            assert lr <= upper + 1e-6, f"{cls.__name__}: LR {lr} > upper bound {upper} at step={step}"


# ---------------------------------------------------------------------------
# Contract 3: Step-0 consistency — returns a valid finite value
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls,config", SCHEDULER_PARAMS)
def test_contract_step_0_valid(cls, config):
    """_lr_at(0, base_lrs) must return a valid (finite, non-negative) value."""
    opt = make_optimizer(lr=0.1)
    sched = cls(opt, **config["kwargs"])
    base_lrs = list(sched.base_lrs)

    lrs = sched._lr_at(0, base_lrs)
    assert len(lrs) == len(base_lrs)
    for lr in lrs:
        assert math.isfinite(lr), f"{cls.__name__}: step-0 LR is not finite: {lr!r}"
        assert lr >= 0.0, f"{cls.__name__}: step-0 LR is negative: {lr}"


# ---------------------------------------------------------------------------
# Contract 4: get_last_lr consistency — at any point, get_last_lr() equals
# get_lr() (both reflect the LR at the current last_epoch).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls,config", SCHEDULER_PARAMS)
def test_contract_get_last_lr_consistency(cls, config):
    """get_last_lr() must equal get_lr() at every step."""
    opt = make_optimizer(lr=0.1)
    sched = cls(opt, **config["kwargs"])

    for _ in range(min(20, config["total_steps"])):
        # Both must agree at the same last_epoch
        get_lr_result = sched.get_lr()
        last_lr_result = sched.get_last_lr()
        assert len(last_lr_result) == len(get_lr_result), (
            f"{cls.__name__}: get_last_lr() length {len(last_lr_result)} != get_lr() length {len(get_lr_result)}"
        )
        for a, b in zip(get_lr_result, last_lr_result, strict=True):
            assert abs(a - b) < 1e-9, (
                f"{cls.__name__}: get_last_lr()={b} != get_lr()={a} at last_epoch={sched.last_epoch}"
            )
        sched.step()


# ---------------------------------------------------------------------------
# Contract 5: state_dict round-trip — load_state_dict(state_dict()) is exact
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls,config", SCHEDULER_PARAMS)
def test_contract_state_dict_round_trip(cls, config):
    """load_state_dict(state_dict()) must perfectly preserve scheduler behavior."""
    total = config["total_steps"]
    mid = total // 2

    # Advance original scheduler to midpoint
    opt1 = make_optimizer(lr=0.1)
    sched1 = cls(opt1, **config["kwargs"])
    for _ in range(mid):
        sched1.step()

    saved_state = sched1.state_dict()
    lr_at_mid = sched1.get_last_lr()

    # Load into a fresh scheduler
    opt2 = make_optimizer(lr=0.1)
    sched2 = cls(opt2, **config["kwargs"])
    sched2.load_state_dict(saved_state)

    # LR at load point must match
    lr_loaded = sched2.get_last_lr()
    for a, b in zip(lr_at_mid, lr_loaded, strict=True):
        assert abs(a - b) < 1e-7, f"{cls.__name__}: LR mismatch after state_dict load: {a} vs {b}"

    # Both must produce identical LRs going forward
    remaining = total - mid
    for i in range(remaining):
        sched1.step()
        sched2.step()
        for a, b in zip(sched1.get_last_lr(), sched2.get_last_lr(), strict=True):
            assert abs(a - b) < 1e-7, f"{cls.__name__}: LR diverged at step {mid + i + 1}: {a} vs {b}"


# ---------------------------------------------------------------------------
# Contract 6: Multiple parameter groups — correct number of LRs returned
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls,config", SCHEDULER_PARAMS)
def test_contract_multi_param_groups(cls, config):
    """Scheduler must return one LR per parameter group."""
    opt = make_multi_optimizer()
    sched = cls(opt, **config["kwargs"])

    lrs = sched.get_lr()
    assert len(lrs) == 2, f"{cls.__name__}: expected 2 LRs for 2 param groups, got {len(lrs)}"

    # Step through several iterations
    for _ in range(min(10, config["total_steps"])):
        sched.step()
        lrs = sched.get_last_lr()
        assert len(lrs) == 2, f"{cls.__name__}: param group count mismatch at step {sched.last_epoch}"


# ---------------------------------------------------------------------------
# Contract 7: _lr_at matches get_lr — they must agree at current last_epoch
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls,config", SCHEDULER_PARAMS)
def test_contract_lr_at_matches_get_lr(cls, config):
    """_lr_at(last_epoch, base_lrs) must equal get_lr() for all steps."""
    opt = make_optimizer(lr=0.1)
    sched = cls(opt, **config["kwargs"])

    for _ in range(min(config["total_steps"], 50)):
        get_lr_result = sched.get_lr()
        lr_at_result = sched._lr_at(sched.last_epoch, list(sched.base_lrs))
        for a, b in zip(get_lr_result, lr_at_result, strict=True):
            assert abs(a - b) < 1e-12, f"{cls.__name__}: get_lr()={a} != _lr_at()={b} at last_epoch={sched.last_epoch}"
        sched.step()
