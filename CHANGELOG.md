# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.1] - 2026-03-14

### Fixed
- Resolve all pyright type errors: cast `base_lrs` to `list[float]`, use `isinstance` for `_lr_at` delegation, fix return types in presets and visualization modules

## [0.2.0] - 2026-03-14

### Added
- Pure functional `_lr_at()` core for all schedulers, enabling safe composition
- `CosineWithWarmupScheduler` — the most common LR schedule for modern training
- `WarmupHoldCosineScheduler` — three-phase schedule (warmup → hold → cosine decay)
- Step-semantics metadata (`step_unit`, `needs_total_steps`) on all schedulers
- Runtime configuration validation with clear warning messages
- `create_scheduler_from_plan()` factory for automatic `total_steps` calculation
- Opinionated presets: `llm_pretrain`, `llm_finetune`, `vision_finetune`, `vision_pretrain`, `transfer_small_data`, `budgeted_training`
- `create_from_preset()` for one-line scheduler creation
- Property-based testing with Hypothesis for boundary conditions
- Formula-based golden tests for all paper-referenced schedulers
- Shared contract test suite enforcing universal invariants
- GitHub Actions CI with Python 3.10-3.13 matrix
- Code coverage reporting via Codecov

### Fixed
- README citation errors: KDecayScheduler (2004.04092 → 2004.05909), RexScheduler (2205.04785 → 2107.04197), LinearDecayScheduler (2305.16264 → 2405.18392)
- WarmupScheduler now uses delegation instead of mutating child scheduler state
- SequentialComposer now uses delegation instead of mutating child scheduler state

### Changed
- README restructured as task-oriented decision guide with scheduler cards

## [0.1.0] - 2024-12-01

### Added
- Initial release with 15 LR schedulers
- WarmupScheduler composable wrapper
- SequentialComposer for chaining schedulers
- ScheduleFreeWrapper experimental feature
- Visualization utilities (plot_schedule, compare_schedules)
- Factory API (create_scheduler, load_scheduler, get_supported_schedulers)
