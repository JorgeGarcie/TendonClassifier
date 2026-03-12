# Experiment Template

Copy this template to `docs/experiments/active/<experiment-name>.md` when starting a new hypothesis test.

---

```markdown
# Experiment: <Title>

**Status**: Planning | Running | Analyzing | Completed
**Created**: YYYY-MM-DD
**Blocked by**: <link to exec-plan, if any>
**Enabled by**: <link to completed exec-plan, if any>

## Hypothesis

State the hypothesis clearly: "We expect X because Y."

## Method

### Setup
- Config file: `configs/<name>.yaml`
- Wandb project: `TendonClassifier`
- Key parameters: ...

### Baseline
- What are we comparing against?
- Baseline metric: ...

### Variables
- Independent: what we're changing
- Dependent: what we're measuring (macro-F1, accuracy, per-class F1)
- Controlled: what stays the same

## Results (append-only)

> Never overwrite recorded data. Add new entries below.

| Date | Run | Config | Macro-F1 | Accuracy | Notes |
|------|-----|--------|----------|----------|-------|
| YYYY-MM-DD | wandb-run-id | config.yaml | 0.XXX | XX.X% | ... |

## Analysis

What do the results mean? Was the hypothesis supported?

## Conclusions

- Key takeaway
- Next steps / follow-up experiments
```
