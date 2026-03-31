# `master` Branch Review

## Branch Overview

- Reviewed branch: `master`
- Review date: `2026-03-31`
- Role in merge plan: target baseline branch
- Observed local relationship:
  - `codex/worktree-bootstrap` is one commit ahead of `master`
  - `codex/ascend-yolo-system` is a large feature branch built on top of the same extra `.gitignore` commit

## Scope and Evidence

- Static inspection of current repository layout and core runtime modules
- `git log --oneline master..codex/worktree-bootstrap`
- `git log --oneline master..codex/ascend-yolo-system`
- `pytest tests -q` on the current working tree

## High Priority Findings

### 1. Baseline tests are not green, so `master` is not a reliable merge target without a stabilization pass

Evidence from `pytest tests -q` on `2026-03-31`:

- Result: `201 passed / 13 failed / 4 errors`
- Environment noise:
  - `PermissionError` under `C:\Users\18222\AppData\Local\Temp\pytest-of-18222`
  - `.pytest_cache` write warnings inside the workspace
- Real regression-shaped failures:
  - `src/api.py` validates file paths before import availability checks
  - `src/inference/base.py` validates file paths before ACL availability checks
  - `utils/logger.py` logger reuse and capture behavior do not match test expectations

Why this matters:

- Merging any feature branch into a red baseline makes it harder to separate new defects from pre-existing ones.
- Current failures already blur the boundary between environment problems and code behavior problems.

Recommended action:

- Treat baseline test stabilization as merge gate zero.
- Split stabilization into:
  - environment-specific test harness fixes
  - runtime/test semantic alignment fixes

### 2. Documentation baseline is stale relative to the actual code layout

Observed drift:

- Root `AGENTS.md` describes the older architecture and still centers the project on inference-only modules.
- The repository already contains newer package structure such as:
  - `src/inference/`
  - `src/strategies/`
  - `benchmark/`
  - richer docs under `docs/`
- The working tree `AGENTS.md` file is untracked, which means branch consumers currently have no guaranteed, versioned agent guidance on `master`.

Why this matters:

- Review, automation, and contributor behavior will be guided by outdated structure assumptions.
- Future branch review quality will suffer if the agent instructions are not versioned with the repo.

Recommended action:

- Version the updated `AGENTS.md` on the target branch as part of the merge-preparation work.

## Medium Priority Findings

### 1. Test expectations and runtime ordering are drifting apart

Examples from the current branch behavior:

- `InferenceAPI.inference_image()` and `inference_batch()` validate files before checking `HAS_INFERENCE`
- `Inference.preprocess()` validates the input path before checking `HAS_ACL`

Impact:

- Tests that intend to verify import/ACL failure semantics can fail earlier on unrelated path validation.
- Error precedence becomes environment-sensitive and harder to reason about.

Recommended action:

- Define explicit precedence rules for:
  - module availability
  - ACL availability
  - path existence
- Update code and tests to agree on that contract.

### 2. Logging setup is functionally usable but operationally brittle for testing

Observed in `utils/logger.py`:

- Existing named logger instances are returned early when handlers already exist
- `propagate` is set to `False`
- logger setup is not refreshed when later callers request a different level, file target, or formatter

Impact:

- `capsys` / `caplog` style tests become order-dependent
- repeated setup calls cannot reliably change logger behavior
- later branch work may inherit invisible logging state

Recommended action:

- Refactor logger setup so repeat configuration is deterministic and test-friendly.

## Low Priority Findings

### 1. Current docs show encoding noise in this Windows shell context

The repository content is mostly UTF-8 Chinese, but several `Get-Content` reads in this shell rendered mojibake.

Impact:

- Low direct runtime risk
- Medium maintenance friction for Windows-based review sessions

Recommended action:

- Keep UTF-8, but standardize terminal/editor encoding expectations in contributor docs.

## Test and Validation Conclusion

- `master` cannot currently be treated as a clean baseline.
- A stabilization pass is required before using it as a trustworthy merge target.
- Not all failures are code regressions; some are environment-specific.
- The review should preserve a hard distinction between:
  - environment noise
  - baseline defects
  - feature-branch defects

## Merge Risk

- Risk level: High as a baseline, not because `master` has a huge diff, but because it lacks a trustworthy green verification state.
- Any direct merge of `codex/ascend-yolo-system` into this baseline would produce ambiguous failure ownership.

## Recommended Actions

1. Stabilize baseline tests before feature merges.
2. Commit the updated `AGENTS.md` to the target branch.
3. Define error-precedence rules for import, ACL, and path validation.
4. Refactor logger setup to be repeatable and test-safe.
