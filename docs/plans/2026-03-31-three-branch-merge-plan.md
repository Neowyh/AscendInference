# Three-Branch Merge Plan

## Summary

- Final target branch: `master`
- Reviewed branches:
  - `master`
  - `codex/worktree-bootstrap`
  - `codex/ascend-yolo-system`
- Recommended merge order:
  1. apply the `codex/worktree-bootstrap` housekeeping change
  2. stabilize `master`
  3. harden and then merge `codex/ascend-yolo-system`

## Pre-Merge Checklist

### Baseline

- Confirm the target working tree is clean enough for review and merge operations.
- Re-run `pytest tests -q` and record:
  - environment-specific failures
  - reproducible code-level failures
- Separate baseline stabilization issues from branch-specific issues.

### `codex/worktree-bootstrap`

- Confirm `.worktrees/` should be ignored across the team.
- Confirm no tracked assets are expected inside `.worktrees/`.

### `codex/ascend-yolo-system`

- Verify command and doc signatures for:
  - `model-bench`
  - `strategy-bench`
  - `extreme-bench`
- Verify acceptance coverage exists for:
  - standard evaluation tiers
  - remote sensing routes
  - strategy evaluation
  - report generation and archive layout
- Confirm config vocabulary consistency:
  - `Config` supported resolutions
  - validator messages
  - input-tier runtime mapping

## Merge Sequence

### Phase 1: absorb `codex/worktree-bootstrap`

Recommended approach:

- Keep the `.worktrees/` ignore behavior.
- Prefer a direct cherry-pick or equivalent single-change merge.
- Retire the branch after the ignore rule lands.

Why:

- The branch has no meaningful independent integration value.
- `codex/ascend-yolo-system` already contains the same commit.

### Phase 2: stabilize `master`

Required before the feature merge:

- fix or quarantine baseline test failures
- document Windows temp-directory permission noise
- align logger setup behavior and test expectations
- define error-precedence semantics for import, ACL, and path validation

Why:

- Without a trustworthy green or at least classified baseline, feature-branch failures remain ambiguous.

### Phase 3: harden `codex/ascend-yolo-system`

Pre-merge blocking items:

- resolve command/doc mismatch
- resolve resolution vocabulary inconsistency
- define acceptance matrix for new evaluation/reporting capabilities
- confirm branch verification on top of stabilized baseline

Recommended merge style:

- use a focused PR into `master`
- keep review centered on integration seams, not just file count

## Expected Conflict Areas

- `.gitignore`
  - low risk, already understood
- `README.md`
  - likely conflicts due to command documentation drift
- `docs/implementation-guide.md`
  - likely conflicts if baseline docs are updated during stabilization
- `config/config.py` and `config/validator.py`
  - moderate risk due to resolution and evaluation semantics
- `benchmark/` and `main.py`
  - moderate risk due to CLI and report-path integration

## Required Verification Commands After Merge

Minimum commands:

```bash
pytest tests -q
python main.py --help
python main.py model-bench --help
python main.py strategy-bench --help
python main.py extreme-bench --help
python scripts/run_smoke_eval.py --mode standard
python scripts/run_smoke_eval.py --mode remote
python scripts/run_smoke_eval.py --mode strategy
```

If Ascend hardware is available, run hardware-backed smoke commands with `--run` and verify generated report/archive outputs.

## Rollback and Split Strategy

- If the feature merge fails verification, revert or pause at PR scope, not by force-resetting shared history.
- If the branch is too large for one confident merge, split follow-up integration into:
  - evaluation contracts and registry layer
  - report normalization and archive layer
  - smoke tooling and docs

## Recommended Commit / PR Shape

Recommended sequence:

1. `chore: ignore local worktrees`
2. `chore: stabilize baseline tests and logging semantics`
3. `docs: align CLI and evaluation documentation`
4. `feat: merge evaluation, registry, and reporting system`
5. `chore: add post-merge acceptance coverage and archive verification`

## Merge Recommendation

- Merge `codex/worktree-bootstrap` content, not the branch identity.
- Merge `codex/ascend-yolo-system` only after baseline stabilization and feature hardening are complete.
