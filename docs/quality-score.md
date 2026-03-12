# Quality Score

**Last updated**: 2026-03-09

## Module Grades

| Module | Grade | Notes |
|--------|-------|-------|
| `scripts/classification/config.py` | A | Clean dataclasses, no violations |
| `scripts/classification/attention.py` | A | Well-structured, no issues |
| `scripts/classification/encoders.py` | B | 1 unused import (`Tuple`) |
| `scripts/classification/dataset.py` | A | Source of truth for normalization constants |
| `scripts/classification/models_v2.py` | B | 2 unused imports; temporal combined force bug (functional) |
| `scripts/classification/train_v2.py` | B | 2 unused imports, 1 silent except |
| `scripts/classification/train_utils.py` | A | Clean utility module |
| `scripts/classification/wandb_logger.py` | A | Clean wrapper |
| `scripts/classification/train_sweep.py` | A | Clean sweep launcher |
| `scripts/classification/eval_test_set.py` | A | Clean eval script |
| `scripts/classification/eval_new_frames.py` | C | 3 magic numbers (ImageNet + crop size) |
| `scripts/classification/eval_test_frames.py` | C | 2 magic numbers + 1 unused import |
| `scripts/classification/run_inference.py` | C | 2 magic numbers + 1 unused import |
| `scripts/classification/analyze_failures.py` | B | 2 unused imports |
| `scripts/classification/inspect_*.py` | B | 1-2 unused imports across scripts |
| `scripts/labeling/config.py` | A | Clean constants module |
| `scripts/labeling/discover_and_index.py` | B | 1 unused import |
| `scripts/labeling/generate_gt.py` | B | 1 unused import |
| `scripts/labeling/extract_valid_windows.py` | A | Clean |
| `scripts/labeling/gt_labeler.py` | A | Clean |
| `tests/` | A | Clean test files |
| `docs/` | B | Some stale markers in BACKLOG.md |

## Top 5 Fixes by Leverage

1. **Fix temporal combined force bug** (`models_v2.py`, `train_v2.py`) — Unlocks correct temporal_combined experiments. Currently the best temporal model trains with a single-frame force input bug.

2. **Centralize ImageNet constants** (`eval_new_frames.py`, `eval_test_frames.py`, `run_inference.py`) — Export from `dataset.py`, import everywhere. Prevents silent drift if normalization changes.

3. **Remove unused imports** (14 files) — Batch cleanup. Run `autoflake --remove-all-unused-imports` or fix manually.

4. **Fix silent except** (`train_v2.py:84`) — Add logging so failures are visible.

5. **Resolve Sparsh subtraction mismatch** (`dataset.py`) — Ensure `sparsh` subtraction type correctly applies `(img - ref) + 0.5` scaling to match Meta's convention.
