"""Parametrized smoke tests for all 13 scheduler classes."""

from __future__ import annotations

import warnings

import pytest
import torch

import pytorch_scheduler as ps

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Parametrize table: (scheduler_class, constructor_kwargs, total_steps)
# total_steps is the number of .step() calls to simulate.
# For InverseSqrtScheduler there is no total_steps parameter; we run for
# warmup_steps * 3 to cover both phases.
# ---------------------------------------------------------------------------

SCHEDULER_PARAMS = [
    pytest.param(
        ps.RexScheduler,
        {"total_steps": 100},
        100,
        id="RexScheduler",
    ),
    pytest.param(
        ps.InverseSqrtScheduler,
        {"warmup_steps": 50},
        150,  # cover warmup + decay phases
        id="InverseSqrtScheduler",
    ),
    pytest.param(
        ps.LinearDecayScheduler,
        {"total_steps": 100, "warmup_steps": 10, "min_lr": 0.0},
        100,
        id="LinearDecayScheduler",
    ),
    pytest.param(
        ps.TrapezoidalScheduler,
        {"total_steps": 100, "warmup_steps": 10, "decay_steps": 20, "min_lr": 0.0},
        100,
        id="TrapezoidalScheduler",
    ),
    pytest.param(
        ps.CosineAnnealingWarmupRestarts,
        {"first_cycle_steps": 100, "warmup_steps": 10, "max_lr": 0.1, "min_lr": 0.001},
        100,
        id="CosineAnnealingWarmupRestarts",
    ),
    pytest.param(
        ps.WSDScheduler,
        {"total_steps": 100, "warmup_steps": 10, "stable_steps": 50, "min_lr": 0.0},
        100,
        id="WSDScheduler",
    ),
    pytest.param(
        ps.KDecayScheduler,
        {"total_steps": 100, "k": 1.0, "min_lr": 0.0},
        100,
        id="KDecayScheduler",
    ),
    pytest.param(
        ps.TanhDecayScheduler,
        {"total_steps": 100, "steepness": 3.0, "min_lr": 0.0},
        100,
        id="TanhDecayScheduler",
    ),
    pytest.param(
        ps.SlantedTriangularScheduler,
        {"total_steps": 100, "cut_frac": 0.1, "ratio": 32.0},
        100,
        id="SlantedTriangularScheduler",
    ),
    pytest.param(
        ps.FlatCosineScheduler,
        {"total_steps": 100, "flat_fraction": 0.7, "min_lr": 0.0},
        100,
        id="FlatCosineScheduler",
    ),
    pytest.param(
        ps.PolynomialScheduler,
        {"total_steps": 100, "power": 2.0, "min_lr": 0.0, "warmup_steps": 10},
        100,
        id="PolynomialScheduler",
    ),
    pytest.param(
        ps.ChebyshevScheduler,
        {"total_steps": 100, "min_lr": 0.0},
        100,
        id="ChebyshevScheduler",
    ),
    pytest.param(
        ps.PowerDecayScheduler,
        {"total_steps": 100, "warmup_steps": 10, "alpha": 0.5, "min_lr": 0.0},
        100,
        id="PowerDecayScheduler",
    ),
    pytest.param(
        ps.HyperbolicLRScheduler,
        {"total_steps": 100, "upper_bound": 250, "min_lr": 1e-6, "warmup_steps": 10},
        100,
        id="HyperbolicLRScheduler",
    ),
    pytest.param(
        ps.ExpHyperbolicLRScheduler,
        {"total_steps": 100, "upper_bound": 250, "min_lr": 1e-6, "warmup_steps": 10},
        100,
        id="ExpHyperbolicLRScheduler",
    ),
    pytest.param(
        ps.CosineWithWarmupScheduler,
        {"total_steps": 100, "warmup_steps": 10, "min_lr": 0.0},
        100,
        id="CosineWithWarmupScheduler",
    ),
    pytest.param(
        ps.WarmupHoldCosineScheduler,
        {"total_steps": 100, "warmup_steps": 10, "hold_steps": 40, "min_lr": 0.0},
        100,
        id="WarmupHoldCosineScheduler",
    ),
    pytest.param(
        ps.IdentityScheduler,
        {},
        100,
        id="IdentityScheduler",
    ),
]


def _make_optimizer(lr: float = 0.1) -> torch.optim.SGD:
    param = torch.randn(2, requires_grad=True)
    return torch.optim.SGD([param], lr=lr)


def _make_multi_optimizer() -> torch.optim.SGD:
    p1 = torch.randn(2, requires_grad=True)
    p2 = torch.randn(3, requires_grad=True)
    return torch.optim.SGD([{"params": [p1], "lr": 0.1}, {"params": [p2], "lr": 0.01}])


# ---------------------------------------------------------------------------
# Constructability
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls,kwargs,steps", SCHEDULER_PARAMS)
def test_scheduler_constructable(cls, kwargs, steps):
    """Each scheduler should construct without error."""
    opt = _make_optimizer()
    scheduler = cls(opt, **kwargs)
    assert scheduler is not None


# ---------------------------------------------------------------------------
# Full-schedule step-through: no exceptions, all LRs >= 0
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls,kwargs,steps", SCHEDULER_PARAMS)
def test_scheduler_step_through(cls, kwargs, steps):
    """Stepping through the full schedule should not raise exceptions."""
    opt = _make_optimizer()
    scheduler = cls(opt, **kwargs)
    lrs_seen = []
    for _ in range(steps + 1):
        lrs_seen.extend(scheduler.get_last_lr())
        scheduler.step()
    assert len(lrs_seen) > 0, "No LR values collected"


@pytest.mark.parametrize("cls,kwargs,steps", SCHEDULER_PARAMS)
def test_scheduler_lrs_non_negative(cls, kwargs, steps):
    """All learning rates should remain non-negative throughout training."""
    opt = _make_optimizer()
    scheduler = cls(opt, **kwargs)
    for _ in range(steps + 1):
        for lr in scheduler.get_last_lr():
            assert lr >= 0.0, f"Negative LR {lr} encountered at step {scheduler.last_epoch}"
        scheduler.step()


# ---------------------------------------------------------------------------
# Step-0 LR: should be 0 for warmup-starting or base_lr otherwise
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls,kwargs,steps", SCHEDULER_PARAMS)
def test_scheduler_step0_lr_reasonable(cls, kwargs, steps):
    """LR at step 0 should be 0 (warmup) or base_lr (no warmup)."""
    opt = _make_optimizer(lr=0.1)
    scheduler = cls(opt, **kwargs)
    initial_lrs = scheduler.get_last_lr()
    for lr in initial_lrs:
        # Either 0 (warmup start) or within reasonable range of base_lr
        assert lr >= 0.0
        assert lr <= 0.1 + 1e-6, f"LR {lr} unexpectedly large at step 0"


# ---------------------------------------------------------------------------
# CosineAnnealingWarmupRestarts is special: it controls its own LR range.
# At step 0 it starts at min_lr; after the full cycle, it ends near min_lr.
# ---------------------------------------------------------------------------


def test_cosine_annealing_lr_range():
    """CosineAnnealingWarmupRestarts should stay within [min_lr, max_lr]."""
    opt = _make_optimizer()
    scheduler = ps.CosineAnnealingWarmupRestarts(opt, first_cycle_steps=100, warmup_steps=10, max_lr=0.1, min_lr=0.001)
    for _ in range(200):
        for lr in scheduler.get_last_lr():
            assert lr >= 0.001 - 1e-6, f"LR {lr} below min_lr"
            assert lr <= 0.1 + 1e-6, f"LR {lr} above max_lr"
        scheduler.step()


def test_hyperbolic_lr_warmup_starts_at_min_lr():
    """HyperbolicLR with warmup should start at min_lr (not 0)."""
    opt = _make_optimizer(lr=0.1)
    scheduler = ps.HyperbolicLRScheduler(opt, total_steps=100, upper_bound=250, min_lr=1e-4, warmup_steps=10)
    lr0 = scheduler.get_last_lr()[0]
    assert abs(lr0 - 1e-4) < 1e-8, f"HyperbolicLR step-0 LR={lr0}, expected 1e-4"


def test_exp_hyperbolic_lr_warmup_starts_at_min_lr():
    """ExpHyperbolicLR with warmup should start at min_lr (not 0)."""
    opt = _make_optimizer(lr=0.1)
    scheduler = ps.ExpHyperbolicLRScheduler(opt, total_steps=100, upper_bound=250, min_lr=1e-4, warmup_steps=10)
    lr0 = scheduler.get_last_lr()[0]
    assert abs(lr0 - 1e-4) < 1e-8, f"ExpHyperbolicLR step-0 LR={lr0}, expected 1e-4"


# ---------------------------------------------------------------------------
# End-of-schedule LR: decaying schedulers should reach near min_lr
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cls,kwargs,expected_min",
    [
        (ps.RexScheduler, {"total_steps": 100}, 0.0),
        (ps.LinearDecayScheduler, {"total_steps": 100, "min_lr": 0.0}, 0.0),
        (ps.KDecayScheduler, {"total_steps": 100, "min_lr": 0.0}, 0.0),
        (ps.TanhDecayScheduler, {"total_steps": 100, "min_lr": 0.0}, 0.0),
        (ps.FlatCosineScheduler, {"total_steps": 100, "min_lr": 0.0}, 0.0),
        (ps.PolynomialScheduler, {"total_steps": 100, "min_lr": 0.0}, 0.0),
        (ps.PowerDecayScheduler, {"total_steps": 100, "alpha": 2.0, "min_lr": 0.0}, 0.0),
    ],
)
def test_scheduler_end_lr_near_min(cls, kwargs, expected_min):
    """Scheduler LR should reach (or approach) min_lr at total_steps."""
    opt = _make_optimizer()
    scheduler = cls(opt, **kwargs)
    total_steps = kwargs.get("total_steps", 100)
    for _ in range(total_steps + 1):
        scheduler.step()
    final_lr = scheduler.get_last_lr()[0]
    assert final_lr <= expected_min + 1e-4, f"Final LR {final_lr} is not near min_lr={expected_min}"


# ---------------------------------------------------------------------------
# Multi-param group: get_lr() should return correct number of values
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls,kwargs,steps", SCHEDULER_PARAMS)
def test_scheduler_multi_param_group(cls, kwargs, steps):
    """Scheduler should handle multiple param groups and return correct LR count."""
    opt = _make_multi_optimizer()

    # CosineAnnealingWarmupRestarts ignores initial optimizer LRs;
    # all other schedulers derive from base_lrs of the param groups.
    scheduler = cls(opt, **kwargs)
    lr_list = scheduler.get_lr()
    assert len(lr_list) == 2, f"{cls.__name__}.get_lr() returned {len(lr_list)} LRs, expected 2"

    # Step through a few iterations and verify consistency
    for _ in range(10):
        scheduler.step()
        lrs = scheduler.get_last_lr()
        assert len(lrs) == 2


# ---------------------------------------------------------------------------
# No-warmup variants: LR at step 0 should equal base_lr
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cls,kwargs",
    [
        (ps.RexScheduler, {"total_steps": 100}),
        (ps.KDecayScheduler, {"total_steps": 100}),
        (ps.FlatCosineScheduler, {"total_steps": 100}),
        (ps.LinearDecayScheduler, {"total_steps": 100}),  # warmup_steps=0 default
        (ps.PowerDecayScheduler, {"total_steps": 100}),  # warmup_steps=0 default
        (ps.PolynomialScheduler, {"total_steps": 100}),  # warmup_steps=0 default
        (ps.IdentityScheduler, {}),
    ],
)
def test_no_warmup_step0_equals_base_lr(cls, kwargs):
    """Without warmup, LR at step 0 should equal the optimizer's initial LR."""
    opt = _make_optimizer(lr=0.1)
    scheduler = cls(opt, **kwargs)
    lr0 = scheduler.get_last_lr()[0]
    assert abs(lr0 - 0.1) < 1e-6, f"{cls.__name__} step-0 LR={lr0}, expected 0.1"


@pytest.mark.parametrize(
    "cls,kwargs",
    [
        # TanhDecay: uses tanh formula at t=0, which approaches but does not equal 1
        (ps.TanhDecayScheduler, {"total_steps": 100}),
        # ChebyshevScheduler: uses first Chebyshev node, very close to but != 1
        (ps.ChebyshevScheduler, {"total_steps": 100}),
    ],
)
def test_formula_based_step0_near_base_lr(cls, kwargs):
    """Formula-driven schedulers start near (but not exactly) base_lr at step 0."""
    opt = _make_optimizer(lr=0.1)
    scheduler = cls(opt, **kwargs)
    lr0 = scheduler.get_last_lr()[0]
    # Should be > 90% of base_lr but non-negative
    assert lr0 >= 0.0
    assert lr0 <= 0.1 + 1e-6, f"{cls.__name__} step-0 LR={lr0} exceeds base_lr"
    assert lr0 >= 0.09, f"{cls.__name__} step-0 LR={lr0} unexpectedly far below base_lr"


# ---------------------------------------------------------------------------
# Warmup variants: LR at step 0 should be 0
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cls,kwargs",
    [
        (ps.InverseSqrtScheduler, {"warmup_steps": 50}),
        (ps.LinearDecayScheduler, {"total_steps": 100, "warmup_steps": 10}),
        (ps.TrapezoidalScheduler, {"total_steps": 100, "warmup_steps": 10, "decay_steps": 20}),
        (ps.WSDScheduler, {"total_steps": 100, "warmup_steps": 10, "stable_steps": 50}),
        (ps.PolynomialScheduler, {"total_steps": 100, "warmup_steps": 10}),
        (ps.PowerDecayScheduler, {"total_steps": 100, "warmup_steps": 10}),
    ],
)
def test_warmup_step0_is_zero(cls, kwargs):
    """Schedulers with warmup should start from LR=0 at step 0."""
    opt = _make_optimizer(lr=0.1)
    scheduler = cls(opt, **kwargs)
    lr0 = scheduler.get_last_lr()[0]
    assert lr0 == 0.0, f"{cls.__name__} step-0 LR={lr0}, expected 0.0"


# ---------------------------------------------------------------------------
# state_dict round-trip: save mid-schedule, load, compare LRs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls,kwargs,steps", SCHEDULER_PARAMS)
def test_scheduler_state_dict_round_trip(cls, kwargs, steps):
    """Saving and loading state_dict mid-schedule should reproduce LRs."""
    # Original scheduler — step to midpoint
    opt1 = _make_optimizer()
    sched1 = cls(opt1, **kwargs)
    mid = steps // 2
    for _ in range(mid):
        sched1.step()

    saved_state = sched1.state_dict()
    lr_at_mid = sched1.get_last_lr()

    # New scheduler — load saved state
    opt2 = _make_optimizer()
    sched2 = cls(opt2, **kwargs)
    sched2.load_state_dict(saved_state)

    lr_loaded = sched2.get_last_lr()
    for a, b in zip(lr_at_mid, lr_loaded, strict=True):
        assert abs(a - b) < 1e-7, f"LR mismatch after load: {a} vs {b}"

    # Step both forward and compare
    remaining = steps - mid
    for i in range(remaining):
        sched1.step()
        sched2.step()
        for a, b in zip(sched1.get_last_lr(), sched2.get_last_lr(), strict=True):
            assert abs(a - b) < 1e-7, f"LR diverged at step {mid + i + 1}: {a} vs {b}"
