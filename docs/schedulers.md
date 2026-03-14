# Scheduler Reference

Detailed API reference for all 17 schedulers, warmup wrapper, and experimental features.

---

## Table of Contents

**Schedulers**

- [CosineWithWarmupScheduler](#cosine-with-warmup) — Cosine annealing with linear warmup
- [WarmupHoldCosineScheduler](#warmup-hold-cosine) — Warmup → hold → cosine decay
- [TrapezoidalScheduler](#trapezoidal) — Warmup → constant → linear decay
- [WSDScheduler](#wsd) — Warmup → stable → decay (MiniCPM)
- [FlatCosineScheduler](#flat-cosine) — Flat then cosine decay
- [LinearDecayScheduler](#linear-decay) — Linear decay to zero (D2Z)
- [CosineAnnealingWarmupRestarts](#cosine-annealing-warm-restarts) — SGDR with warmup (2017)
- [InverseSqrtScheduler](#inverse-sqrt) — Transformer schedule (2017)
- [SlantedTriangularScheduler](#slanted-triangular) — ULMFiT schedule (2018)
- [TanhDecayScheduler](#tanh-decay) — Hypergradient-inspired tanh decay (2018)
- [KDecayScheduler](#k-decay) — k-decay cosine variant (2020)
- [PowerDecayScheduler](#power-decay) — Power-law decay (2020)
- [RexScheduler](#rex) — REX budgeted schedule (2022)
- [HyperbolicLRScheduler](#hyperbolic-lr) — Epoch-insensitive hyperbolic decay (2024)
- [ExpHyperbolicLRScheduler](#exp-hyperbolic-lr) — Log-space hyperbolic decay (2024)
- [PolynomialScheduler](#polynomial) — Polynomial decay with optional cycling
- [ChebyshevScheduler](#chebyshev) — Non-monotonic Chebyshev node schedule

**Wrappers & Utilities**

- [WarmupScheduler](#warmup-scheduler) — Composable warmup wrapper

**Experimental**

- [ScheduleFreeWrapper](#schedule-free-wrapper) — Schedule-free optimizer wrapper
- [SequentialComposer](#sequential-composer) — Chain schedulers at milestones

---

## Schedulers

---

<a id="cosine-with-warmup"></a>
### CosineWithWarmupScheduler

The de facto standard schedule for modern transformer and LLM pretraining. Linear warmup
from 0 to `base_lr`, followed by half-cosine decay to `min_lr`.

**Formula:**

Warmup phase (`0 <= t < warmup_steps`):

```
lr = base_lr * t / warmup_steps
```

Cosine decay phase (`warmup_steps <= t <= total_steps`):

```
lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + cos(π * (t - warmup_steps) / (total_steps - warmup_steps)))
```

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `optimizer` | `Optimizer` | required | Wrapped optimizer |
| `total_steps` | `int` | required | Total number of training steps (must be > 0) |
| `warmup_steps` | `int` | `0` | Steps for linear warmup; must be < `total_steps` |
| `min_lr` | `float` | `0.0` | Minimum LR at end of cosine decay (must be >= 0) |
| `last_epoch` | `int` | `-1` | Index of the last step (-1 = before first step) |

**Example:**

```python
from pytorch_scheduler.scheduler.cosine_warmup import CosineWithWarmupScheduler

scheduler = CosineWithWarmupScheduler(
    optimizer,
    total_steps=10000,
    warmup_steps=500,
    min_lr=1e-6,
)
```

---

<a id="warmup-hold-cosine"></a>
### WarmupHoldCosineScheduler

Three-phase schedule: linear warmup, then a constant hold at peak LR, then cosine decay.
Commonly used in large-scale LLM pretraining (e.g., MiniCPM, LLaMA-style) to maximize
time at peak LR before cooldown.

**Formula:**

Warmup phase (`0 <= t < warmup_steps`):

```
lr = base_lr * t / warmup_steps
```

Hold phase (`warmup_steps <= t < warmup_steps + hold_steps`):

```
lr = base_lr
```

Cosine decay phase (`warmup_steps + hold_steps <= t <= total_steps`):

```
progress = (t - warmup_steps - hold_steps) / (total_steps - warmup_steps - hold_steps)
lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + cos(π * progress))
```

Constraint: `warmup_steps + hold_steps < total_steps`

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `optimizer` | `Optimizer` | required | Wrapped optimizer |
| `total_steps` | `int` | required | Total number of training steps (must be > 0) |
| `warmup_steps` | `int` | `0` | Steps for linear warmup (must be >= 0) |
| `hold_steps` | `int` | `0` | Steps to hold at peak LR (must be >= 0) |
| `min_lr` | `float` | `0.0` | Minimum LR at end of cosine decay (must be >= 0) |
| `last_epoch` | `int` | `-1` | Index of the last step (-1 = before first step) |

**Example:**

```python
from pytorch_scheduler.scheduler.warmup_hold_cosine import WarmupHoldCosineScheduler

scheduler = WarmupHoldCosineScheduler(
    optimizer,
    total_steps=10000,
    warmup_steps=200,
    hold_steps=7000,
    min_lr=1e-6,
)
```

---

<a id="trapezoidal"></a>
### TrapezoidalScheduler

Three-phase trapezoidal schedule: linear warmup, constant plateau, then linear decay.
Constraint: `warmup_steps + decay_steps <= total_steps`.

**Formula:**

Warmup phase (`0 <= t < warmup_steps`):

```
lr = base_lr * t / warmup_steps
```

Constant phase (`warmup_steps <= t < total_steps - decay_steps`):

```
lr = base_lr
```

Linear decay phase (`total_steps - decay_steps <= t <= total_steps`):

```
progress = (t - (total_steps - decay_steps)) / decay_steps
lr = base_lr + (min_lr - base_lr) * progress
```

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `optimizer` | `Optimizer` | required | Wrapped optimizer |
| `total_steps` | `int` | required | Total number of training steps (must be > 0) |
| `warmup_steps` | `int` | required | Steps for linear warmup (must be >= 0) |
| `decay_steps` | `int` | required | Steps for linear decay at the end (must be >= 0) |
| `min_lr` | `float` | `0.0` | Minimum LR at end of decay (must be >= 0) |
| `last_epoch` | `int` | `-1` | Index of the last step (-1 = before first step) |

**Example:**

```python
from pytorch_scheduler.scheduler.trapezoidal import TrapezoidalScheduler

scheduler = TrapezoidalScheduler(
    optimizer,
    total_steps=10000,
    warmup_steps=500,
    decay_steps=2000,
    min_lr=0.0,
)
```

---

<a id="wsd"></a>
### WSDScheduler

Warmup-Stable-Decay schedule from MiniCPM. Three phases: linear warmup, constant stable
phase, then a smooth decay using one of three decay types. The decay function is
configurable (cosine, linear, or square-root).

**Paper:** [MiniCPM: Scaling Large Language Models with Scalable Strategies](https://arxiv.org/abs/2404.06395) — Hu et al., 2024

**Formula:**

Warmup phase (`0 <= t < warmup_steps`):

```
lr = base_lr * t / warmup_steps
```

Stable phase (`warmup_steps <= t < warmup_steps + stable_steps`):

```
lr = base_lr
```

Decay phase (`warmup_steps + stable_steps <= t <= total_steps`), where `p = progress in [0,1]`:

```
cosine: lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + cos(π * p))
linear: lr = base_lr + (min_lr - base_lr) * p
sqrt:   lr = min_lr + (base_lr - min_lr) * (1 - sqrt(p))
```

Constraint: `warmup_steps + stable_steps < total_steps`

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `optimizer` | `Optimizer` | required | Wrapped optimizer |
| `total_steps` | `int` | required | Total number of training steps (must be > 0) |
| `warmup_steps` | `int` | required | Steps for linear warmup (must be >= 0) |
| `stable_steps` | `int` | required | Steps for constant stable phase (must be >= 0) |
| `min_lr` | `float` | `0.0` | Minimum LR at end of decay (must be >= 0) |
| `decay_type` | `Literal["cosine", "linear", "sqrt"]` | `"cosine"` | Shape of the decay curve |
| `last_epoch` | `int` | `-1` | Index of the last step (-1 = before first step) |

**Example:**

```python
from pytorch_scheduler.scheduler.wsd import WSDScheduler

scheduler = WSDScheduler(
    optimizer,
    total_steps=10000,
    warmup_steps=200,
    stable_steps=7800,
    min_lr=1e-6,
    decay_type="cosine",
)
```

---

<a id="flat-cosine"></a>
### FlatCosineScheduler

Holds the LR constant at `base_lr` for a configurable flat fraction, then applies
cosine annealing for the remainder of training.

**Formula:**

```
flat_end = floor(total_steps * flat_fraction)
```

Flat phase (`0 <= t <= flat_end`):

```
lr = base_lr
```

Cosine phase (`flat_end < t <= total_steps`):

```
progress = (t - flat_end) / (total_steps - flat_end)
lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + cos(π * progress))
```

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `optimizer` | `Optimizer` | required | Wrapped optimizer |
| `total_steps` | `int` | required | Total number of training steps (must be > 0) |
| `flat_fraction` | `float` | `0.7` | Fraction of steps for the flat phase; must be in `[0, 1)` |
| `min_lr` | `float` | `0.0` | Minimum LR at end of cosine decay (must be >= 0) |
| `last_epoch` | `int` | `-1` | Index of the last step (-1 = before first step) |

**Example:**

```python
from pytorch_scheduler.scheduler.flat_cosine import FlatCosineScheduler

scheduler = FlatCosineScheduler(
    optimizer,
    total_steps=10000,
    flat_fraction=0.7,
    min_lr=1e-6,
)
```

---

<a id="linear-decay"></a>
### LinearDecayScheduler

Linear Decay to Zero (D2Z) schedule. Optional linear warmup followed by a linear decay
to `min_lr`. From the compute-optimal training scaling laws paper by Hägele et al.

**Paper:** [Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations](https://arxiv.org/abs/2405.18392) — Hägele et al., 2024

**Formula:**

Warmup phase (`0 <= t < warmup_steps`, if `warmup_steps > 0`):

```
lr = base_lr * t / warmup_steps
```

Linear decay phase (`warmup_steps <= t <= total_steps`):

```
progress = (t - warmup_steps) / (total_steps - warmup_steps)
lr = base_lr + (min_lr - base_lr) * progress
```

If `warmup_steps == 0`, the decay starts immediately from step 0 across `total_steps`.

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `optimizer` | `Optimizer` | required | Wrapped optimizer |
| `total_steps` | `int` | required | Total number of training steps (must be > 0) |
| `warmup_steps` | `int` | `0` | Steps for linear warmup; must be < `total_steps` if > 0 |
| `min_lr` | `float` | `0.0` | Minimum LR at end of decay (must be >= 0) |
| `last_epoch` | `int` | `-1` | Index of the last step (-1 = before first step) |

**Example:**

```python
from pytorch_scheduler.scheduler.linear_decay import LinearDecayScheduler

scheduler = LinearDecayScheduler(
    optimizer,
    total_steps=10000,
    warmup_steps=500,
    min_lr=0.0,
)
```

---

<a id="cosine-annealing-warm-restarts"></a>
### CosineAnnealingWarmupRestarts

Extends SGDR (Loshchilov & Hutter, 2017) with optional linear warmup at the start of
each cycle, per-cycle peak LR decay via `gamma`, and variable cycle lengths via
`cycle_mult`. The LR range (`max_lr` / `min_lr`) is managed independently of the
optimizer's initial param-group LRs.

**Paper:** [SGDR: Stochastic Gradient Descent with Warm Restarts](https://arxiv.org/abs/1608.03983) — Loshchilov & Hutter, ICLR 2017

**Formula:**

Within each cycle of length `cycle_len`, with current peak `cur_max_lr`:

Warmup phase (`0 <= step_in_cycle < warmup_steps`):

```
lr = min_lr + (cur_max_lr - min_lr) * step_in_cycle / warmup_steps
```

Cosine phase (`warmup_steps <= step_in_cycle <= cycle_len`):

```
progress = (step_in_cycle - warmup_steps) / (cycle_len - warmup_steps)
lr = min_lr + (cur_max_lr - min_lr) * 0.5 * (1 + cos(π * progress))
```

After each restart:

```
cur_max_lr = max_lr * gamma^cycle
cycle_len  = first_cycle_steps * cycle_mult^cycle  (if cycle_mult != 1.0)
```

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `optimizer` | `Optimizer` | required | Wrapped optimizer |
| `first_cycle_steps` | `int` | required | Steps in the first cycle (must be > 0) |
| `warmup_steps` | `int` | `0` | Steps for linear warmup within each cycle; must be < `first_cycle_steps` |
| `max_lr` | `float` | `0.1` | Peak LR for the first cycle (must be >= `min_lr`) |
| `min_lr` | `float` | `0.001` | Minimum / trough LR (must be >= 0) |
| `cycle_mult` | `float` | `1.0` | Multiplicative factor for cycle length after each restart (must be > 0) |
| `gamma` | `float` | `1.0` | Multiplicative decay applied to `max_lr` after each restart (must be > 0) |
| `last_epoch` | `int` | `-1` | Index of the last step (-1 = before first step) |

**Example:**

```python
from pytorch_scheduler.scheduler.cosine_annealing import CosineAnnealingWarmupRestarts

scheduler = CosineAnnealingWarmupRestarts(
    optimizer,
    first_cycle_steps=2000,
    warmup_steps=200,
    max_lr=1e-3,
    min_lr=1e-5,
    cycle_mult=2.0,
    gamma=0.9,
)
```

---

<a id="inverse-sqrt"></a>
### InverseSqrtScheduler

Inverse square root schedule from "Attention is All You Need". Warmup is built into
the mathematical formula — do **not** wrap this with `WarmupScheduler`.

**Paper:** [Attention is All You Need](https://arxiv.org/abs/1706.03762) — Vaswani et al., NeurIPS 2017

**Formula:**

Unified Vaswani et al. formula:

```
lr = base_lr * min(t^(-0.5), t * warmup_steps^(-1.5)) * sqrt(warmup_steps)
```

Expanded: linear warmup until `t == warmup_steps`, then inverse-sqrt decay:

```
Warmup  (t < warmup_steps): lr ≈ base_lr * t / warmup_steps
Decay   (t >= warmup_steps): lr = base_lr * sqrt(warmup_steps / t)
```

The peak LR equals `base_lr` at step `warmup_steps`.

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `optimizer` | `Optimizer` | required | Wrapped optimizer |
| `warmup_steps` | `int` | required | Warmup steps (must be > 0); also sets the decay reference point |
| `last_epoch` | `int` | `-1` | Index of the last step (-1 = before first step) |

**Example:**

```python
from pytorch_scheduler.scheduler.inverse_sqrt import InverseSqrtScheduler

scheduler = InverseSqrtScheduler(
    optimizer,
    warmup_steps=4000,
)
```

---

<a id="slanted-triangular"></a>
### SlantedTriangularScheduler

Slanted triangular schedule from ULMFiT. A short linear warmup followed by a longer
linear decay. The `ratio` parameter controls how far the LR rises from its starting
point (`base_lr / ratio`) to the peak (`base_lr`) during the warmup phase.

**Paper:** [Universal Language Model Fine-tuning for Text Classification](https://arxiv.org/abs/1801.06146) — Howard & Ruder, 2018

**Formula:**

```
cut_point = floor(total_steps * cut_frac)
```

Warmup phase (`0 <= t < cut_point`):

```
p = t / cut_point
```

Decay phase (`cut_point <= t <= total_steps`):

```
p = 1 - (t - cut_point) / (total_steps - cut_point)
```

Learning rate from `p`:

```
lr = base_lr * (1 + p * (ratio - 1)) / ratio
```

The LR starts at `base_lr / ratio`, rises to `base_lr`, then decays back toward `base_lr / ratio`.

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `optimizer` | `Optimizer` | required | Wrapped optimizer |
| `total_steps` | `int` | required | Total number of training steps (must be > 0) |
| `cut_frac` | `float` | `0.1` | Fraction of steps used for warmup; must be in `(0, 1)` |
| `ratio` | `float` | `32.0` | LR range ratio; LR ranges from `base_lr/ratio` to `base_lr` (must be > 0) |
| `last_epoch` | `int` | `-1` | Index of the last step (-1 = before first step) |

**Example:**

```python
from pytorch_scheduler.scheduler.slanted_triangular import SlantedTriangularScheduler

scheduler = SlantedTriangularScheduler(
    optimizer,
    total_steps=10000,
    cut_frac=0.1,
    ratio=32.0,
)
```

---

<a id="tanh-decay"></a>
### TanhDecayScheduler

Tanh-based decay schedule. The `steepness` parameter controls how sharp the
transition from `base_lr` to `min_lr` is. Higher steepness produces an
abrupt step-function-like transition; lower steepness gives a gradual S-curve.

**Paper:** [Online Learning Rate Adaptation with Hypergradient Descent](https://arxiv.org/abs/1703.04782) — Baydin et al., 2018

**Formula:**

```
t  = step / total_steps          (progress in [0, 1])
lr = min_lr + 0.5 * (base_lr - min_lr) * (1 - tanh(steepness * (2*t - 1)))
```

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `optimizer` | `Optimizer` | required | Wrapped optimizer |
| `total_steps` | `int` | required | Total number of training steps (must be > 0) |
| `steepness` | `float` | `3.0` | Controls sharpness of the sigmoid-like transition (must be > 0) |
| `min_lr` | `float` | `0.0` | Minimum LR at the end of decay (must be >= 0) |
| `last_epoch` | `int` | `-1` | Index of the last step (-1 = before first step) |

**Example:**

```python
from pytorch_scheduler.scheduler.tanh_decay import TanhDecayScheduler

scheduler = TanhDecayScheduler(
    optimizer,
    total_steps=10000,
    steepness=5.0,
    min_lr=1e-6,
)
```

---

<a id="k-decay"></a>
### KDecayScheduler

k-decay cosine schedule. The exponent `k` warps the cosine argument, controlling
when decay happens: `k > 1` keeps LR high longer before a sharp drop; `k < 1`
decays quickly early on; `k == 1` is standard cosine annealing.

**Paper:** [k-decay: A New Method for Learning Rate Schedule](https://arxiv.org/abs/2004.05909) — Zhang & Li, 2020

**Formula:**

```
lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + cos(π * (step / total_steps)^k))
```

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `optimizer` | `Optimizer` | required | Wrapped optimizer |
| `total_steps` | `int` | required | Total number of training steps (must be > 0) |
| `k` | `float` | `1.0` | Cosine argument exponent; `k=1` is standard cosine (must be > 0) |
| `min_lr` | `float` | `0.0` | Minimum LR at the end of decay (must be >= 0) |
| `last_epoch` | `int` | `-1` | Index of the last step (-1 = before first step) |

**Example:**

```python
from pytorch_scheduler.scheduler.k_decay import KDecayScheduler

scheduler = KDecayScheduler(
    optimizer,
    total_steps=10000,
    k=2.0,
    min_lr=1e-6,
)
```

---

<a id="power-decay"></a>
### PowerDecayScheduler

Power-law decay inspired by the Kaplan et al. scaling laws. Decays as
`base_lr * (step / warmup_steps)^(-alpha)` after warmup. `alpha=0.5` reproduces
inverse-square-root decay; `alpha=1.0` gives inverse-linear decay.

**Paper:** [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) — Kaplan et al., 2020

**Formula:**

Warmup phase (`0 <= step < warmup_steps`, if `warmup_steps > 0`):

```
lr = base_lr * step / warmup_steps
```

Power-law decay phase (`step >= warmup_steps`):

```
lr = max(base_lr * (step / warmup_steps)^(-alpha),  min_lr)   # if warmup_steps > 0
lr = max(base_lr * (step + 1)^(-alpha),             min_lr)   # if warmup_steps == 0
```

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `optimizer` | `Optimizer` | required | Wrapped optimizer |
| `total_steps` | `int` | required | Total steps (used for validation; decay is unbounded) |
| `warmup_steps` | `int` | `0` | Steps for linear warmup; must be < `total_steps` |
| `alpha` | `float` | `0.5` | Power-law exponent; 0.5 = inverse sqrt, 1.0 = inverse linear (must be > 0) |
| `min_lr` | `float` | `0.0` | LR floor; LR is clamped to this value (must be >= 0) |
| `last_epoch` | `int` | `-1` | Index of the last step (-1 = before first step) |

**Example:**

```python
from pytorch_scheduler.scheduler.power_decay import PowerDecayScheduler

scheduler = PowerDecayScheduler(
    optimizer,
    total_steps=10000,
    warmup_steps=500,
    alpha=0.5,
    min_lr=1e-7,
)
```

---

<a id="rex"></a>
### RexScheduler

REX (Revisiting Budgeted Training) schedule. A simple closed-form schedule designed for
fixed training budgets. Decays from `base_lr` to 0 with a convex curve that outperforms
cosine on several tasks under the same compute budget.

**Paper:** [Revisiting Budgeted Training with an Improved Schedule](https://arxiv.org/abs/2107.04197) — MLSys 2022

**Formula:**

```
t  = step / total_steps          (t in [0, 1])
lr = base_lr * (1 - t) / (1 - t/2)
```

At `t=0`, `lr = base_lr`. At `t=1`, `lr = 0`. The denominator `(1 - t/2) >= 0.5` ensures
the formula is always well-defined for `t in [0, 1]`.

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `optimizer` | `Optimizer` | required | Wrapped optimizer |
| `total_steps` | `int` | required | Total number of training steps (must be > 0) |
| `last_epoch` | `int` | `-1` | Index of the last step (-1 = before first step) |

**Example:**

```python
from pytorch_scheduler.scheduler.rex import RexScheduler

scheduler = RexScheduler(
    optimizer,
    total_steps=10000,
)
```

---

<a id="hyperbolic-lr"></a>
### HyperbolicLRScheduler

Epoch-insensitive hyperbolic decay. The hyperbolic curve ensures that early LR changes
remain consistent regardless of total training length. The `upper_bound` parameter
controls the curvature; it must be >= `total_steps`.

**Paper:** [HyperbolicLR: Epoch Insensitive Learning Rate Scheduler](https://arxiv.org/abs/2407.15200) — 2024

**Formula:**

Define `N = total_steps - warmup_steps` and `U = upper_bound`. Let `x = step - warmup_steps`.

```
f(x) = sqrt((N - x) / U * (2 - (N + x) / U))
```

Warmup phase (`0 <= step <= warmup_steps`):

```
lr = min_lr + (base_lr - min_lr) * step / warmup_steps
```

Hyperbolic decay phase (`step > warmup_steps`):

```
lr = base_lr + (base_lr - min_lr) * (f(x) - f(0))
```

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `optimizer` | `Optimizer` | required | Wrapped optimizer |
| `total_steps` | `int` | required | Total number of training steps (must be > 0) |
| `upper_bound` | `int` | required | Curvature control; must be >= `total_steps` |
| `min_lr` | `float` | `1e-6` | Minimum LR (must be >= 0) |
| `warmup_steps` | `int` | `0` | Linear warmup steps from `min_lr` to `base_lr`; must be < `total_steps` |
| `last_epoch` | `int` | `-1` | Index of the last step |

**Example:**

```python
from pytorch_scheduler.scheduler.hyperbolic import HyperbolicLRScheduler

scheduler = HyperbolicLRScheduler(
    optimizer,
    total_steps=10000,
    upper_bound=20000,
    min_lr=1e-6,
    warmup_steps=200,
)
```

---

<a id="exp-hyperbolic-lr"></a>
### ExpHyperbolicLRScheduler

Exponential variant of HyperbolicLR. Applies the same hyperbolic curve in log-space,
yielding exponential decay. `min_lr` must be strictly positive (used as a divisor).

**Paper:** [HyperbolicLR: Epoch Insensitive Learning Rate Scheduler](https://arxiv.org/abs/2407.15200) — 2024

**Formula:**

Uses the same `f(x)` as `HyperbolicLRScheduler` (see above).

Warmup phase (`0 <= step <= warmup_steps`):

```
lr = min_lr + (base_lr - min_lr) * step / warmup_steps
```

Exponential hyperbolic decay phase (`step > warmup_steps`):

```
lr = base_lr * (base_lr / min_lr)^(f(x) - f(0))
```

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `optimizer` | `Optimizer` | required | Wrapped optimizer |
| `total_steps` | `int` | required | Total number of training steps (must be > 0) |
| `upper_bound` | `int` | required | Curvature control; must be >= `total_steps` |
| `min_lr` | `float` | `1e-6` | Minimum LR; must be **strictly positive** (used as divisor) |
| `warmup_steps` | `int` | `0` | Linear warmup steps from `min_lr` to `base_lr`; must be < `total_steps` |
| `last_epoch` | `int` | `-1` | Index of the last step |

**Example:**

```python
from pytorch_scheduler.scheduler.hyperbolic import ExpHyperbolicLRScheduler

scheduler = ExpHyperbolicLRScheduler(
    optimizer,
    total_steps=10000,
    upper_bound=20000,
    min_lr=1e-6,
    warmup_steps=200,
)
```

---

<a id="polynomial"></a>
### PolynomialScheduler

Polynomial decay from `base_lr` to `min_lr`. Optional linear warmup. When `cycle=True`,
the schedule restarts after `total_steps`, allowing training to continue beyond the
original duration. `power=1.0` gives linear decay.

**Formula:**

Warmup phase (`0 <= step < warmup_steps`, if `warmup_steps > 0`):

```
lr = base_lr * step / warmup_steps
```

Polynomial decay phase (`warmup_steps <= step <= total_steps`):

```
progress = (step - warmup_steps) / (total_steps - warmup_steps)
lr = (base_lr - min_lr) * (1 - progress)^power + min_lr
```

When `cycle=True`, `step` is replaced by `step % total_steps`.

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `optimizer` | `Optimizer` | required | Wrapped optimizer |
| `total_steps` | `int` | required | Total steps (period when cycling) (must be > 0) |
| `power` | `float` | `1.0` | Polynomial exponent; 1.0 = linear decay (must be > 0) |
| `min_lr` | `float` | `0.0` | Minimum LR at end of decay (must be >= 0) |
| `warmup_steps` | `int` | `0` | Steps for linear warmup; must be < `total_steps` |
| `cycle` | `bool` | `False` | Whether to restart the schedule after `total_steps` |
| `last_epoch` | `int` | `-1` | Index of the last step (-1 = before first step) |

**Example:**

```python
from pytorch_scheduler.scheduler.polynomial import PolynomialScheduler

scheduler = PolynomialScheduler(
    optimizer,
    total_steps=10000,
    power=2.0,
    min_lr=1e-6,
    warmup_steps=200,
)
```

---

<a id="chebyshev"></a>
### ChebyshevScheduler

Non-monotonic schedule based on Chebyshev polynomial nodes. Rather than monotonically
decaying, the LR follows an oscillating pattern theoretically optimal for polynomial
interpolation. The oscillations can help the optimizer escape local minima.

**Formula:**

At step `j`, the j-th Chebyshev node:

```
x_j = cos(π * (2*j + 1) / (2 * total_steps))
```

Learning rate mapped from `x_j` in `[-1, 1]` to `[min_lr, base_lr]`:

```
lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + x_j)
```

After `total_steps`, the final node value is reused.

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `optimizer` | `Optimizer` | required | Wrapped optimizer |
| `total_steps` | `int` | required | Total steps; also the number of Chebyshev nodes (must be > 0) |
| `min_lr` | `float` | `0.0` | Minimum LR in the oscillating range (must be >= 0) |
| `last_epoch` | `int` | `-1` | Index of the last step (-1 = before first step) |

**Example:**

```python
from pytorch_scheduler.scheduler.chebyshev import ChebyshevScheduler

scheduler = ChebyshevScheduler(
    optimizer,
    total_steps=10000,
    min_lr=1e-6,
)
```

---

## Wrappers & Utilities

---

<a id="warmup-scheduler"></a>
### WarmupScheduler

Composable warmup wrapper for any `LRScheduler`. During warmup, scales the LR from
0 to `base_lr` using the specified curve. After warmup, delegates to the wrapped
`base_scheduler` with the step offset removed.

**Note:** Do **not** use this with `InverseSqrtScheduler` — that scheduler has warmup
built into its formula and wrapping it would apply warmup twice.

**Warmup formulas** (where `progress = step / warmup_steps`):

```
linear:      factor = progress
cosine:      factor = 0.5 * (1 - cos(π * progress))
exponential: factor = progress^2
```

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `optimizer` | `Optimizer` | required | Wrapped optimizer |
| `base_scheduler` | `LRScheduler` | required | Scheduler to use after warmup completes |
| `warmup_steps` | `int` | required | Number of warmup steps |
| `warmup_type` | `Literal["linear", "cosine", "exponential"]` | `"linear"` | Shape of the warmup curve |
| `last_epoch` | `int` | `-1` | Index of the last step (-1 = before first step) |

**Example:**

```python
from pytorch_scheduler.base.warmup import WarmupScheduler
from pytorch_scheduler.scheduler.rex import RexScheduler

base = RexScheduler(optimizer, total_steps=10000)
scheduler = WarmupScheduler(
    optimizer,
    base_scheduler=base,
    warmup_steps=500,
    warmup_type="cosine",
)
```

---

## Experimental

---

<a id="schedule-free-wrapper"></a>
### ScheduleFreeWrapper

Schedule-free optimizer wrapper based on online-to-batch conversion (Defazio et al., 2024).
Instead of a decaying LR schedule, maintains two parameter sequences — iterate (`z_t`) for
gradient computation and average (`x_t`) for evaluation — and uses Polyak averaging to track
the optimum without requiring a fixed schedule.

Call `.eval()` before validation/inference to switch to the averaged parameters, and
`.train()` to switch back for gradient steps.

**Paper:** [The Road Less Scheduled](https://arxiv.org/abs/2405.15682) — Defazio et al., 2024

**Formula:**

Polyak average update per step `k`:

```
beta_eff = max(beta, 1 - 1/k)
x_t = beta_eff * x_{t-1} + (1 - beta_eff) * z_t
```

With optional linear warmup applied to the base optimizer's learning rate:

```
lr_effective = initial_lr * min(1, step / warmup_steps)
```

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `optimizer` | `Optimizer` | required | Base optimizer to wrap |
| `warmup_steps` | `int` | `0` | Steps for linear LR warmup (must be >= 0) |
| `beta` | `float` | `0.9` | Polyak averaging momentum; must be in `[0, 1)` |

**Example:**

```python
from pytorch_scheduler.experimental.schedule_free import ScheduleFreeWrapper

wrapped = ScheduleFreeWrapper(optimizer, warmup_steps=200, beta=0.9)

for batch in dataloader:
    wrapped.train()
    loss = model(batch)
    loss.backward()
    wrapped.step()

# Before evaluation:
wrapped.eval()
val_loss = evaluate(model)
```

---

<a id="sequential-composer"></a>
### SequentialComposer

Chains multiple schedulers sequentially, switching between them at specified
step milestones. Each scheduler handles a contiguous segment of training, and step
offsets are automatically subtracted so each scheduler sees its own local step counter.

Unlike PyTorch's built-in `SequentialLR`, this implementation avoids `step(0)` deprecation
issues and correctly handles `BaseScheduler` instances.

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `optimizer` | `Optimizer` | required | Wrapped optimizer (must be shared with all child schedulers) |
| `schedulers` | `list[LRScheduler]` | required | Ordered list of schedulers to chain |
| `milestones` | `list[int]` | required | Step indices where transitions occur; `len(milestones) == len(schedulers) - 1`; must be sorted and non-negative |
| `last_epoch` | `int` | `-1` | Index of the last step (-1 = before first step) |

**Milestone semantics:**

Given `milestones=[100, 500]` with 3 schedulers:

- `scheduler[0]` is active for steps 0–99
- `scheduler[1]` is active for steps 100–499
- `scheduler[2]` is active for steps 500+

Each scheduler's step is offset: `scheduler[i]` receives `step - milestone[i-1]`.

**Example:**

```python
from pytorch_scheduler.experimental.sequential_composer import SequentialComposer
from pytorch_scheduler.scheduler.cosine_warmup import CosineWithWarmupScheduler
from pytorch_scheduler.scheduler.linear_decay import LinearDecayScheduler

warmup_sched = CosineWithWarmupScheduler(optimizer, total_steps=500, warmup_steps=100)
decay_sched = LinearDecayScheduler(optimizer, total_steps=9500)

scheduler = SequentialComposer(
    optimizer,
    schedulers=[warmup_sched, decay_sched],
    milestones=[500],
)
```
