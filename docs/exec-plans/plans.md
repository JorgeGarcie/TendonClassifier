# ExecPlan Template

Copy this template to `docs/exec-plans/active/<plan-name>.md` when starting a new architecture or code change.

---

```markdown
# ExecPlan: <Title>

**Status**: Draft | In Progress | Blocked | Completed
**Created**: YYYY-MM-DD
**Motivated by**: <link to experiment or issue>

## Goal

One sentence: what does this change accomplish?

## Context

Why is this needed? What problem does it solve?

## Plan

### Step 1: <description>
- [ ] Task A
- [ ] Task B

### Step 2: <description>
- [ ] Task C

## Files to Change

| File | Change |
|------|--------|
| `path/to/file.py` | Description of change |

## Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| YYYY-MM-DD | Chose X over Y | Because... |

## Surprises

- (unexpected things discovered during implementation)

## Progress

- [ ] Step 1 complete
- [ ] Step 2 complete
- [ ] Tests pass
- [ ] Docs updated

## Outcomes

**Result**: (filled when completed)
**Lessons**: (what would you do differently?)
```
