# Lessons Learned

Pipeline for new golden principles. Status progression: **observation** → **candidate** → **promoted to GP#N**.

Promotion criteria: validated across 3+ tasks.

---

## Observations

### Double-correction trap
**Status**: observation (validated 1x — sweep debugging)

Using both `balanced_sampling` and `class_weights: "balanced"` causes val accuracy collapse. The model over-predicts minority classes. Only use one imbalance correction mechanism at a time.

### Subtraction always helps
**Status**: observation (validated 1x — sweep results)

All top sweep runs used frame subtraction. Non-subtraction runs were consistently worse. Subtraction removes static sensor artifacts and isolates the deformation signal.

### Force provides trajectory-invariant signal
**Status**: observation (validated 1x — generalization eval)

On traverse data (same distribution as training), image alone saturates at 100%. On unseen trajectories, image drops to 55.5% while combined holds at 75.6%. Force data generalizes better than visual features across different probe trajectories.

### High-force visual ambiguity
**Status**: observation (validated 1x — failure analysis)

At forces >30N, single-tendon deformation visually resembles double. The combined model's 66 "double" false positives on p1 single-tendon frames all cluster at high force regions. Manual inspection confirms the ambiguity is genuine — not a model failure.

---

## Candidates

(None yet — observations need 2 more validations to become candidates)

---

## Promoted

(Candidates that have been promoted to golden principles are removed from this file and added to `golden-principles.md`)
