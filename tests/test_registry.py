"""Tests for the scheduler registry: load_scheduler, create_scheduler, get_supported_schedulers."""

from __future__ import annotations

import warnings

import pytest
import torch

from pytorch_scheduler import (
    SCHEDULER_LIST,
    SCHEDULERS,
    create_scheduler,
    get_supported_schedulers,
    load_scheduler,
)

warnings.filterwarnings("ignore")

# Canonical lowercase names as they appear in SCHEDULERS dict
EXPECTED_NAMES = sorted(
    [
        "chebyshev",
        "chebyshevscheduler",
        "cosine_annealing",
        "cosine_with_warmup",
        "cosineannealingwarmuprestarts",
        "cosinewithwarmupscheduler",
        "exp_hyperbolic",
        "exphyperboliclrscheduler",
        "flat_cosine",
        "flatcosinescheduler",
        "hyperbolic",
        "hyperboliclrscheduler",
        "identity",
        "identityscheduler",
        "inverse_sqrt",
        "inversesqrtscheduler",
        "k_decay",
        "kdecayscheduler",
        "linear_decay",
        "lineardecayscheduler",
        "polynomial",
        "polynomialscheduler",
        "power_decay",
        "powerdecayscheduler",
        "rex",
        "rexscheduler",
        "slanted_triangular",
        "slantedtriangularscheduler",
        "tanh_decay",
        "tanhdecayscheduler",
        "trapezoidal",
        "trapezoidalscheduler",
        "warmup_hold_cosine",
        "warmupholdcosinescheduler",
        "wsd",
        "wsdscheduler",
    ]
)


def _make_optimizer(lr: float = 0.1) -> torch.optim.SGD:
    param = torch.randn(2, requires_grad=True)
    return torch.optim.SGD([param], lr=lr)


# ---------------------------------------------------------------------------
# SCHEDULER_LIST / SCHEDULERS dict
# ---------------------------------------------------------------------------


def test_scheduler_list_has_18_entries():
    assert len(SCHEDULER_LIST) == 18


def test_schedulers_dict_has_36_entries():
    assert len(SCHEDULERS) == 36


def test_schedulers_dict_keys_are_lowercase():
    for key in SCHEDULERS:
        assert key == key.lower(), f"Key '{key}' is not lowercase"


# ---------------------------------------------------------------------------
# load_scheduler
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", EXPECTED_NAMES)
def test_load_scheduler_finds_all(name):
    """load_scheduler should successfully find each registered scheduler name."""
    cls = load_scheduler(name)
    assert cls is not None
    assert isinstance(cls, type)


def test_load_scheduler_case_insensitive():
    """load_scheduler should be case-insensitive."""
    cls_lower = load_scheduler("rexscheduler")
    cls_upper = load_scheduler("RexScheduler")
    cls_mixed = load_scheduler("REXSCHEDULER")
    assert cls_lower is cls_upper is cls_mixed


def test_load_scheduler_unknown_raises_value_error():
    """load_scheduler should raise ValueError for unknown names."""
    with pytest.raises(ValueError, match="Unknown scheduler"):
        load_scheduler("NonExistentScheduler")


def test_load_scheduler_error_includes_available():
    """The ValueError message should list available schedulers."""
    with pytest.raises(ValueError) as exc_info:
        load_scheduler("bogus")
    msg = str(exc_info.value)
    assert "Available schedulers" in msg


def test_load_scheduler_returns_class_not_instance():
    """load_scheduler should return the class, not an instance."""
    cls = load_scheduler("RexScheduler")
    assert isinstance(cls, type)


# ---------------------------------------------------------------------------
# create_scheduler
# ---------------------------------------------------------------------------


def test_create_scheduler_returns_instance():
    """create_scheduler should return a working scheduler instance."""
    opt = _make_optimizer()
    scheduler = create_scheduler(opt, "RexScheduler", total_steps=100)
    assert hasattr(scheduler, "step")
    assert hasattr(scheduler, "get_lr")


def test_create_scheduler_case_insensitive():
    """create_scheduler should be case-insensitive."""
    opt1 = _make_optimizer()
    opt2 = _make_optimizer()
    s1 = create_scheduler(opt1, "rexscheduler", total_steps=100)
    s2 = create_scheduler(opt2, "RexScheduler", total_steps=100)
    assert type(s1) is type(s2)


def test_create_scheduler_unknown_raises():
    """create_scheduler should raise ValueError for unknown names."""
    opt = _make_optimizer()
    with pytest.raises(ValueError, match="Unknown scheduler"):
        create_scheduler(opt, "FakeScheduler", total_steps=100)


def test_create_scheduler_passes_kwargs():
    """create_scheduler should pass kwargs to the constructor."""
    opt = _make_optimizer()
    scheduler = create_scheduler(opt, "LinearDecayScheduler", total_steps=200, warmup_steps=20, min_lr=0.001)
    assert scheduler.total_steps == 200  # type: ignore[attr-defined]
    assert scheduler.warmup_steps == 20  # type: ignore[attr-defined]
    assert scheduler.min_lr == 0.001  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    "name,kwargs",
    [
        ("RexScheduler", {"total_steps": 100}),
        ("InverseSqrtScheduler", {"warmup_steps": 50}),
        ("LinearDecayScheduler", {"total_steps": 100}),
        ("KDecayScheduler", {"total_steps": 100}),
        ("ChebyshevScheduler", {"total_steps": 100}),
        (
            "WSDScheduler",
            {"total_steps": 100, "warmup_steps": 10, "stable_steps": 50},
        ),
        (
            "TrapezoidalScheduler",
            {"total_steps": 100, "warmup_steps": 10, "decay_steps": 20},
        ),
    ],
)
def test_create_scheduler_smoke(name, kwargs):
    """Each scheduler should be creatable via create_scheduler and step once."""
    opt = _make_optimizer()
    scheduler = create_scheduler(opt, name, **kwargs)
    scheduler.step()
    lr = scheduler.get_last_lr()
    assert len(lr) == 1
    assert lr[0] >= 0.0


# ---------------------------------------------------------------------------
# get_supported_schedulers
# ---------------------------------------------------------------------------


def test_get_supported_schedulers_returns_all_36():
    """get_supported_schedulers() with default pattern should return 36 names."""
    names = get_supported_schedulers()
    assert len(names) == 36


def test_get_supported_schedulers_sorted():
    """get_supported_schedulers() should return names in sorted order."""
    names = get_supported_schedulers()
    assert names == sorted(names)


def test_get_supported_schedulers_exact_names():
    """get_supported_schedulers() should match expected canonical names."""
    names = get_supported_schedulers()
    assert names == EXPECTED_NAMES


def test_get_supported_schedulers_star_pattern():
    """Pattern '*' (default) should return all schedulers."""
    assert get_supported_schedulers("*") == get_supported_schedulers()


def test_get_supported_schedulers_cosine_pattern():
    """Pattern '*cosine*' should match cosine-related schedulers."""
    matches = get_supported_schedulers("*cosine*")
    assert len(matches) >= 1
    for name in matches:
        assert "cosine" in name


def test_get_supported_schedulers_cosine_returns_known():
    """*cosine* should match both cosine annealing and flat cosine."""
    matches = get_supported_schedulers("*cosine*")
    assert "cosineannealingwarmuprestarts" in matches
    assert "flatcosinescheduler" in matches


def test_get_supported_schedulers_scheduler_suffix():
    """Pattern '*scheduler' should match schedulers ending in 'scheduler'."""
    matches = get_supported_schedulers("*scheduler")
    for name in matches:
        assert name.endswith("scheduler")


def test_get_supported_schedulers_no_match():
    """A pattern matching nothing should return an empty list."""
    matches = get_supported_schedulers("zzz_no_match_*")
    assert matches == []


def test_get_supported_schedulers_returns_list():
    """get_supported_schedulers should return a list."""
    result = get_supported_schedulers()
    assert isinstance(result, list)


def test_get_supported_schedulers_all_strings():
    """All returned names should be strings."""
    for name in get_supported_schedulers():
        assert isinstance(name, str)
