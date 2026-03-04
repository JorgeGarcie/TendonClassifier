---
name: sweep-status
description: Get a compact summary of wandb sweep runs
---

Run the sweep status script to get a compact summary of sweep runs. Use Bash to run:

```
cd /home/aquabot/VisionFT/TendonClassifier/TendonClassifier/scripts/classification && /home/aquabot/miniforge3/envs/VISIONFT/bin/python sweep_status.py --sweep-id csx5cmvk
```

Show the output to the user. If they ask about a specific sweep, pass a different `--sweep-id <id>`. Do not use the wandb MCP tools for this — the script is more context-efficient.

Note: Runs marked "crashed" with a "[HB]" tag were killed by Hyperband early termination — this is normal sweep behavior, not an error. The script detects this automatically based on the sweep's `early_terminate` config and the run's last logged step.
