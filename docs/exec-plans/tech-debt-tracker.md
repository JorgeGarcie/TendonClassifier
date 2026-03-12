# Tech Debt Tracker

**Last scan**: 2026-03-09 | **Total violations**: 22 (lint) + 15 (docs)

---

## GP1: One Fact, One Place (7 violations)

ImageNet normalization constants are duplicated in 3 eval/inference scripts instead of importing from `dataset.py`.

- [ ] `scripts/classification/eval_new_frames.py:50-51` — Hardcoded ImageNet mean/std
- [ ] `scripts/classification/eval_new_frames.py:85` — Hardcoded crop size 1080
- [ ] `scripts/classification/eval_test_frames.py:39-40` — Hardcoded ImageNet mean/std
- [ ] `scripts/classification/run_inference.py:28-29` — Hardcoded ImageNet mean/std

**FIX**: Export `IMAGENET_MEAN` and `IMAGENET_STD` from `dataset.py` (or a shared constants module) and import in all eval/inference scripts. Export `CROP_SIZE` from `labeling/config.py`.

---

## GP2: Centralize, Don't Duplicate (14 violations — unused imports)

- [ ] `scripts/classification/analyze_failures.py:2` — `random`
- [ ] `scripts/classification/analyze_failures.py:5` — `pd`
- [ ] `scripts/classification/dump_run_frames.py:35` — `F`
- [ ] `scripts/classification/encoders.py:11` — `Tuple`
- [ ] `scripts/classification/eval_test_frames.py:22` — `nn`
- [ ] `scripts/classification/inspect_run_spatial.py:20` — `mpatches`
- [ ] `scripts/classification/models_v2.py:13` — `get_encoder_dim`
- [ ] `scripts/classification/models_v2.py:14` — `SimpleFusion`
- [ ] `scripts/classification/run_inference.py:20` — `np`
- [ ] `scripts/classification/sweep_status.py:10` — `json`
- [ ] `scripts/classification/train_v2.py:26` — `nn`
- [ ] `scripts/classification/train_v2.py:31` — `load_config`
- [ ] `scripts/labeling/discover_and_index.py:12` — `os`
- [ ] `scripts/labeling/generate_gt.py:19` — `shutil`

**FIX**: Remove unused imports from each file. For `models_v2.py`, check if `get_encoder_dim` and `SimpleFusion` were replaced — if so, remove from `attention.py`/`encoders.py` exports too.

---

## GP4: Fail Loud, Not Silent (1 violation)

- [ ] `scripts/classification/train_v2.py:84` — Silent except with `pass` body

**FIX**: Add `print()` or `logging.warning()` in the except block, or remove the try/except if the exception should propagate.

---

## Known Bugs (from BACKLOG.md) — VERIFIED 2026-03-10

- [x] **Temporal combined force bug** — RESOLVED. `train_v2.py:619-621` now correctly sets `return_force_sequence=True` for `temporal` combined with `use_force`. `TemporalModel` uses `TemporalForceBranch` and accepts `(B, T, 6)`. BACKLOG.md is outdated on this.
- [x] **Sparsh subtraction type mismatch** — RESOLVED. `dataset.py:358-359` implements `sparsh` subtraction as `(img - ref) + 0.5, clipped [0,1]`. All Sparsh configs correctly use `type: "sparsh"`.

---

## Priority Order

1. **Unused imports cleanup** — 14 violations, easy batch fix, reduces noise
2. **ImageNet constants centralization** — 7 violations, prevents drift
3. **Silent except** — 1 violation, quick fix
4. **Update BACKLOG.md** — remove resolved bug entries (temporal force, Sparsh subtraction)
