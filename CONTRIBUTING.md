# Contributing to pytorch-scheduler

Thank you for your interest in contributing! This project welcomes new learning rate schedulers, bug fixes, and documentation improvements.

## Adding a New Scheduler

This is the most common contribution. If you've published (or found) a scheduler with a clear formula, you can add it in **4 steps**.

### Step 1: Create the scheduler file

Create `pytorch_scheduler/scheduler/your_scheduler.py`. Use this minimal template:

```python
from __future__ import annotations

import math
from typing import TYPE_CHECKING

from pytorch_scheduler.base.scheduler import BaseScheduler

if TYPE_CHECKING:
    from torch.optim import Optimizer


class YourScheduler(BaseScheduler):
    """One-line description of the schedule.

    Formula:
        lr = ...  (write the closed-form expression here)

    Reference:
        Paper: "Paper Title"
               Author Names, Year
        URL: https://arxiv.org/abs/XXXX.XXXXX
    """

    paper_title = "Paper Title"
    paper_url = "https://arxiv.org/abs/XXXX.XXXXX"
    paper_year = 2025
    needs_total_steps = True  # set False if schedule doesn't need total_steps

    def __init__(
        self,
        optimizer: Optimizer,
        total_steps: int,
        # ... your parameters ...
        last_epoch: int = -1,
    ) -> None:
        # Validate parameters
        if total_steps <= 0:
            raise ValueError(f"total_steps must be positive, got {total_steps}")
        self.total_steps = total_steps
        super().__init__(optimizer, last_epoch=last_epoch)

    def _lr_at(self, step: int, base_lrs: list[float]) -> list[float]:
        """Pure, stateless LR computation. This is the only method you need to implement."""
        if step <= 0:
            return list(base_lrs)
        if step >= self.total_steps:
            return [0.0 for _ in base_lrs]  # or min_lr

        # Your formula here
        t = step / self.total_steps
        factor = ...
        return [base_lr * factor for base_lr in base_lrs]
```

Key rules:
- `_lr_at()` must be **pure and stateless** — use only `step`, `base_lrs`, and `self.*` constructor params
- Do **not** override `get_lr()` — it's inherited from `BaseScheduler`
- Set `paper_title`, `paper_url`, `paper_year` (leave empty strings / 0 if no paper)
- Set `needs_total_steps = True` if the scheduler requires `total_steps`

### Step 2: Register the scheduler

**`pytorch_scheduler/scheduler/__init__.py`** — add 3 things:

```python
# 1. Import
from pytorch_scheduler.scheduler.your_scheduler import YourScheduler

# 2. Add to SCHEDULER_LIST
SCHEDULER_LIST: list[type] = [
    ...,
    YourScheduler,
]

# 3. Add shorthand alias
SCHEDULERS["your_scheduler"] = YourScheduler
```

**`pytorch_scheduler/__init__.py`** — add to imports and `__all__`.

### Step 3: Add tests

At minimum, add a golden test in `tests/test_golden.py`:

```python
class TestYourSchedulerGolden:
    def setup_method(self):
        self.opt = make_optimizer(lr=0.1)
        self.sched = ps.YourScheduler(self.opt, total_steps=1000)
        self.base_lrs = [0.1]

    def test_step_0(self):
        result = self.sched._lr_at(0, self.base_lrs)
        assert result[0] == pytest.approx(0.1, abs=1e-6)  # hand-computed

    def test_midpoint(self):
        result = self.sched._lr_at(500, self.base_lrs)
        assert result[0] == pytest.approx(EXPECTED, abs=1e-6)  # hand-computed from formula
```

Also add a config entry in `tests/test_contracts.py` (`SCHEDULER_CONFIGS`) and `tests/test_schedulers.py` (`SCHEDULER_PARAMS`). The contract suite will automatically test your scheduler for universal invariants (no NaN, bounded output, state_dict round-trip, etc.).

### Step 4: Add documentation

Add your scheduler to `docs/schedulers.md` following the existing format (formula, parameter table, example).

### Checklist

- [ ] `_lr_at()` matches the paper's formula exactly
- [ ] `paper_title`, `paper_url`, `paper_year` are correct
- [ ] Golden test values are hand-computed from the formula, not copied from implementation output
- [ ] Registered in `scheduler/__init__.py` and `pytorch_scheduler/__init__.py`
- [ ] Config added to `test_contracts.py` and `test_schedulers.py`
- [ ] Entry added to `docs/schedulers.md`
- [ ] `uv run ruff check . && uv run ruff format . && uv run pytest` passes

## Other Contributions

### Bug Fixes

If you find a formula that doesn't match its source paper, please open an issue with:
- The scheduler name
- The expected formula (with page/equation number from the paper)
- The actual implementation behavior

### Documentation

Improvements to scheduler cards, examples, or the README are always welcome.

## Development Setup

```bash
git clone https://github.com/Axect/pytorch-scheduler.git
cd pytorch-scheduler
uv sync --extra dev
```

### Running checks

```bash
uv run ruff check .          # lint
uv run ruff format .         # format
uv run pyright               # type check
uv run pytest                # test (544+ tests)
uv run pytest -m "not slow"  # skip Hypothesis property tests
```

## Code Style

- Line length: 119
- Formatter/linter: ruff
- Type checker: pyright (basic mode)
- Use `from __future__ import annotations` in every file
- Use `TYPE_CHECKING` guard for type-only imports
- Single quotes for strings (ruff enforces this)
