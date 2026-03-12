# Golden Principles

Timeless philosophies that guide all code changes. Specific violations go in `docs/exec-plans/tech-debt-tracker.md`.

New candidates enter through `lessons-learned.md` and are promoted here after validation across 3+ tasks.

---

## GP1: One Fact, One Place

Constants, configuration values, and shared logic have a single source of truth. If a value appears in 2+ files, it must live in a config file or shared module.

**Examples**: Force threshold in `labeling/config.py`, all hyperparams in YAML configs, ImageNet normalization stats in `dataset.py`.

## GP2: Centralize, Don't Duplicate

Extract repeated patterns into shared utilities. Three similar code blocks signal the need for a function. But don't abstract prematurely — wait for the third occurrence.

**Examples**: `train_utils.py` for device detection and checkpoint logic, `encoders.py` as a factory for all vision encoders.

## GP3: Validate at Boundaries

Assert shapes, types, and ranges where data enters a module. Internal code can trust its own invariants, but boundaries (user input, config loading, data ingestion) must be checked.

**Examples**: Config dataclass type checking, dataset manifest column validation, model input shape assertions.

## GP4: Fail Loud, Not Silent

Log warnings and raise errors instead of silently proceeding with bad state. A failed training run is better than a silently wrong result.

**Examples**: Missing config fields should raise `KeyError`, not default to `None`. Checkpoint loading mismatches should warn, not silently skip.

## GP5: Config Drives Experiments

All hyperparameters and experiment variations live in YAML configs. Source code should never contain hardcoded values that change between experiments. The checkpoint embeds its config for reproducibility.

**Examples**: Learning rate, batch size, fusion type, encoder choice — all in YAML. `train_v2.py` reads config, never hardcodes these.

## GP6: Frozen Encoder, Learned Head

Never fine-tune the vision backbone on this dataset size (~4.7K training samples). The encoder is always frozen; only the classification head and fusion layers are trained. This prevents overfitting and keeps the parameter count manageable.

**Examples**: `encoder.freeze = True` in all configs, no gradient flow through ResNet/DinoV2/CLIP/Sparsh.
