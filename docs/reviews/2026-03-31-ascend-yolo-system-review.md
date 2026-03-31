# `codex/ascend-yolo-system` Branch Review

## Branch Overview

- Reviewed branch: `codex/ascend-yolo-system`
- Review date: `2026-03-31`
- Relationship to `master`:
  - includes the `.worktrees/` housekeeping commit
  - adds a large evaluation/reporting feature set
- Commit span from `master`: 29 commits
- Diff size relative to `master`: 47 changed files, about 4374 insertions and 261 deletions

Key functional additions:

- `evaluations/` for route and input-tier semantics
- `registry/` for model/device/scenario contracts
- `reporting/` for report models, renderers, and archive layout
- smoke evaluation configs and `scripts/run_smoke_eval.py`
- expanded CLI options in `main.py`

## Scope and Evidence

- `git diff --stat master..codex/ascend-yolo-system`
- `git log --oneline master..codex/ascend-yolo-system`
- static review of:
  - `main.py`
  - `config/config.py`
  - `config/validator.py`
  - `benchmark/scenarios.py`
  - `benchmark/reporters.py`
  - `evaluations/tasks.py`
  - `registry/models.py`
  - `reporting/renderers.py`
  - `reporting/archive.py`
  - `scripts/run_smoke_eval.py`
  - `tests/test_reporting.py`

## High Priority Findings

### 1. CLI and documentation semantics are still misaligned

Observed mismatch:

- In `main.py`, `model-bench` accepts models as positional arguments.
- Existing README examples still show usage shaped like `--models ...`.

Impact:

- A user following the docs can invoke the command incorrectly even if the code itself works.
- This becomes worse after merge because the branch introduces many new CLI combinations.

Recommended action:

- Treat command/help/doc synchronization as a pre-merge requirement for this branch.

### 2. Configuration validation semantics are internally inconsistent

Observed mismatch:

- `config/config.py` supports resolution names such as:
  - `640x640`
  - `1k`
  - `1k2k`
  - `4k6k`
  - `6k`
- `config/validator.py` recommends a different list:
  - `320x320`
  - `416x416`
  - `512x512`
  - `608x608`
  - `640x640`
  - `768x768`
  - `800x800`
  - `1024x1024`
  - `1280x1280`

Impact:

- Validation feedback can suggest a naming system that the main config model does not actually use.
- New evaluation tiers depend on `InputTier.runtime_resolution`, so inconsistent resolution vocabulary increases operator confusion and review risk.

Recommended action:

- Normalize the resolution vocabulary across `Config`, validator logic, CLI help text, and docs before merge.

### 3. The branch introduces a large new domain surface without a single explicit acceptance matrix

New domain areas:

- standard evaluation tiers
- remote sensing routes
- registry payload contracts
- report normalization and archive layout
- smoke command generation

Impact:

- The branch has good unit-level coverage in parts, but there is still no single documented acceptance matrix describing:
  - supported command combinations
  - expected archive outputs
  - route/tier compatibility rules
  - what is considered pre-merge blocking vs post-merge hardening

Recommended action:

- Add an acceptance matrix to the merge plan and require smoke-path verification before merge.

## Medium Priority Findings

### 1. `benchmark/reporters.py` now carries two reporting layers at once

Observed shape:

- legacy-style `TextReporter` / `JsonReporter` / `HtmlReporter`
- new unified `render_report()` path that delegates to `reporting/renderers.py`

Impact:

- The branch improves reporting capability, but the coexistence of two report models increases maintenance cost and future drift risk.

Recommended action:

- Decide whether the legacy reporter classes remain supported API or should become compatibility wrappers around the unified report model.

### 2. `scripts/run_smoke_eval.py` is useful, but it is a command builder more than a full smoke framework

What it does well:

- loads sample configs
- builds `main.py` commands
- prints or runs them

Current limitation:

- it does not itself validate output archive shape, result files, or report semantics after execution

Impact:

- Teams may overestimate its verification value if it is labeled as a smoke solution without follow-up checks.

Recommended action:

- Keep it, but document it as command orchestration plus optional execution, not complete acceptance verification.

### 3. Data model layering is improved, but still easy to duplicate

Examples:

- `BenchmarkResult`
- `ExecutionRecord`
- report model dicts
- registry asset and input-spec models

Impact:

- The branch is clearly moving toward stronger contracts, but there is still risk of duplicated or partially overlapping model transformations.

Recommended action:

- Consolidate ownership of canonical execution/report data in a follow-up optimization wave.

## Low Priority Findings

### 1. The branch adds helpful documentation and PR notes, but repository documentation is becoming fragmented

Observed additions:

- implementation docs
- smoke configs
- PR brief under `docs/superpowers/prs/`

Impact:

- Helpful in the short term, but future contributors may not know which file is source of truth.

Recommended action:

- Define a simple doc hierarchy:
  - README for operator entry
  - `docs/implementation-guide.md` for architecture
  - `AGENTS.md` for repository-specific contributor guidance

## Test and Validation Conclusion

- This branch is substantial and not suitable for “merge first, stabilize later”.
- Static design quality is promising:
  - stronger contracts
  - better report normalization
  - clearer route/tier modeling
- Pre-merge confidence is still limited by:
  - baseline red tests
  - missing end-to-end acceptance matrix
  - command/doc drift
  - unresolved config vocabulary inconsistency

## Merge Risk

- Risk level: Medium-High
- Main risk driver is not obvious code chaos; it is integration surface area.
- Safe merge requires a controlled pre-merge hardening pass.

## Recommended Actions

1. Align README, CLI help, and implementation docs with the real command signatures.
2. Normalize resolution vocabulary across config, validator, and evaluation-tier logic.
3. Define and execute a documented acceptance matrix for standard, remote, and strategy evaluation paths.
4. Decide the future of legacy reporter classes versus the unified report model.
5. Merge only after baseline stabilization and branch-specific verification are both complete.
