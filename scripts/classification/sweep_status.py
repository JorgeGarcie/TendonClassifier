"""Quick sweep status summary — prints a compact table of sweep runs.

Usage:
    python sweep_status.py                          # latest sweep
    python sweep_status.py --sweep-id <id>          # specific sweep
    python sweep_status.py --project TendonClassifier  # different project
"""

import argparse
import json

import wandb


def get_hyperband_brackets(sweep_config):
    """Return sorted list of Hyperband termination steps, or None if not configured."""
    et = sweep_config.get("early_terminate")
    if not et or et.get("type") != "hyperband":
        return None
    min_iter = et.get("min_iter", 1)
    eta = et.get("eta", 3)
    max_iter = et.get("max_iter", 300)
    brackets = []
    s = min_iter
    while s <= max_iter:
        brackets.append(s)
        s *= eta
    return brackets


def is_hyperband_kill(run, brackets, tolerance=3):
    """Check if a crashed run's last step is near a Hyperband bracket boundary."""
    if not brackets:
        return False
    last_step = run.summary.get("_step", 0)
    return any(abs(last_step - b) <= tolerance for b in brackets)


def get_sweep_runs(entity, project, sweep_id=None, max_runs=50):
    """Fetch sweep runs and Hyperband config via wandb API."""
    api = wandb.Api()

    if sweep_id:
        sweep = api.sweep(f"{entity}/{project}/{sweep_id}")
        print(f"Sweep: {sweep_id} ({sweep.state})")
        return list(sweep.runs), sweep.config
    else:
        # Find the most recent sweep
        project_obj = api.project(name=project, entity=entity)
        sweeps = list(project_obj.sweeps())
        if not sweeps:
            print("No sweeps found.")
            return [], {}
        sweep = sweeps[0]
        print(f"Latest sweep: {sweep.id} ({sweep.state})")
        return list(sweep.runs), sweep.config


def main():
    parser = argparse.ArgumentParser(description="Sweep status summary")
    parser.add_argument("--entity", default="jalgarci-stanford-university")
    parser.add_argument("--project", default="TendonClassifier-scripts_classification")
    parser.add_argument("--sweep-id", default=None)
    args = parser.parse_args()

    runs, sweep_config = get_sweep_runs(args.entity, args.project, args.sweep_id)
    brackets = get_hyperband_brackets(sweep_config)

    # Collect data
    rows = []
    for run in runs:
        sm = run.summary._json_dict
        cfg = run.config

        val_f1 = sm.get("val/macro_f1", None)
        test_f1 = sm.get("test/macro_f1", None)
        final_f1 = sm.get("final/macro_f1", None)

        # Detect Hyperband early termination
        hb_killed = (run.state in ("crashed", "failed")
                     and is_hyperband_kill(run, brackets))

        rows.append({
            "name": run.name,
            "display": run.display_name,
            "state": run.state,
            "hb_killed": hb_killed,
            "val_f1": val_f1,
            "test_f1": test_f1,
            "final_f1": final_f1,
            "lr": cfg.get("lr"),
            "bs": cfg.get("batch_size"),
            "opt": cfg.get("optimizer"),
            "hd": cfg.get("fusion_hidden_dim"),
            "nh": cfg.get("fusion_num_heads"),
            "nl": cfg.get("fusion_num_layers"),
            "sub": cfg.get("subtraction_enabled"),
        })

    # Sort by val_f1 descending
    rows.sort(key=lambda r: r["val_f1"] or 0, reverse=True)

    # Print summary
    finished = sum(1 for r in rows if r["state"] == "finished")
    running = sum(1 for r in rows if r["state"] == "running")
    hb_count = sum(1 for r in rows if r["hb_killed"])
    real_crashed = sum(1 for r in rows if r["state"] in ("crashed", "failed") and not r["hb_killed"])
    crash_parts = []
    if hb_count:
        crash_parts.append(f"{hb_count} early-stopped [HB]")
    if real_crashed:
        crash_parts.append(f"{real_crashed} crashed")
    crash_str = " | ".join(crash_parts) if crash_parts else "0 crashed"
    print(f"Runs: {len(rows)} total | {finished} finished | {running} running | {crash_str}")
    print()

    # Print table
    header = f"{'#':<3} {'Name':<22} {'State':<9} {'Val F1':<8} {'Test F1':<8} {'LR':<11} {'BS':<4} {'Opt':<6} {'HD':<4} {'NH':<3} {'NL':<3} {'Sub':<5}"
    print(header)
    print("-" * len(header))

    for i, r in enumerate(rows, 1):
        val = f"{r['val_f1']:.4f}" if r["val_f1"] else "-"
        test = f"{r['test_f1']:.4f}" if r["test_f1"] else "-"
        lr = f"{r['lr']:.1e}" if r["lr"] else "-"
        bs = str(r["bs"]) if r["bs"] else "-"
        opt = str(r["opt"] or "-")
        hd = str(r["hd"] or "-")
        nh = str(r["nh"] or "-")
        nl = str(r["nl"] or "-")
        sub = "Y" if r["sub"] else "N" if r["sub"] is not None else "-"
        state = r["state"][:8]
        if r["hb_killed"]:
            state = "HB-stop"

        print(f"{i:<3} {r['display']:<22} {state:<9} {val:<8} {test:<8} {lr:<11} {bs:<4} {opt:<6} {hd:<4} {nh:<3} {nl:<3} {sub:<5}")


if __name__ == "__main__":
    main()
