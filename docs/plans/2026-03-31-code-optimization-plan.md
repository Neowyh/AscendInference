# Code Optimization Plan

## Summary

This plan does not change runtime contracts by itself. It organizes the next hardening work into waves so the repository can move from “feature-rich but integration-fragile” to “mergeable and maintainable”.

## Wave A: Baseline Test and Environment Stabilization

Goals:

- recover a trustworthy baseline test signal
- separate environment noise from code defects

Tasks:

- classify the current `pytest tests -q` failures into:
  - Windows temp-directory permission issues
  - logger capture/setup issues
  - error-precedence behavior issues
- update test harness guidance for Windows temp and cache behavior
- decide whether environment-sensitive tests need fixture isolation or documented prerequisites

Acceptance:

- the team can explain every remaining failure as either a known environment prerequisite or a real code issue

## Wave B: Inference API and Validation Precedence

Goals:

- make error ordering deterministic
- align tests with intended runtime semantics

Primary targets:

- `src/api.py`
- `src/inference/base.py`

Tasks:

- define precedence between:
  - module availability
  - ACL availability
  - path validation
- adjust code paths so the precedence is explicit
- update tests to assert the intended contract, not incidental ordering

Acceptance:

- import/ACL/path failure tests no longer depend on unrelated filesystem state

## Wave C: Logging Determinism and Test Robustness

Goals:

- make logger setup repeatable
- remove order-dependent test behavior

Primary target:

- `utils/logger.py`

Tasks:

- decide whether repeated `setup_logger()` calls should reconfigure or return an existing immutable logger
- make handler, level, formatter, and propagation behavior deterministic
- align tests with the chosen behavior

Acceptance:

- logger tests pass reliably regardless of execution order

## Wave D: Evaluation / Registry / Reporting Boundary Cleanup

Goals:

- reduce duplicated modeling and transformation logic
- make the new branch easier to maintain after merge

Primary targets:

- `benchmark/`
- `evaluations/`
- `registry/`
- `reporting/`

Tasks:

- document the canonical owner of:
  - execution-record data
  - report-model data
  - route/tier compatibility data
- reduce duplicate compatibility or normalization code where possible
- decide whether legacy reporters remain first-class or compatibility wrappers

Acceptance:

- contributors can identify one canonical source for report and execution metadata

## Wave E: Documentation and Operator Experience

Goals:

- make docs match reality
- keep future branch reviews cheaper

Primary targets:

- `README.md`
- `docs/implementation-guide.md`
- `docs/README.md`
- `AGENTS.md`

Tasks:

- align CLI examples with actual parser signatures
- document smoke script scope and limits
- document Windows review/testing caveats
- keep `AGENTS.md`, README, and architecture docs synchronized when commands or modules change

Acceptance:

- a reader can follow the docs without tripping over stale command syntax or stale architecture descriptions
