#!/usr/bin/env python
"""Backfill an existing training run into Weights & Biases from SLURM .out logs.

This does NOT retrain anything. It parses the per-step log lines that
train_prior.py printed, e.g.

    epoch=2699 step=350950 loss=6.3895e+03 lr=1.000e-04 elapsed=3.16h

and replays them into a single W&B run so the loss/lr curves are browsable.

Multiple .out files (e.g. an initial run + a resumed run) are merged on the
`step` key: later files overwrite earlier ones for any overlapping steps, so
the resulting series is strictly increasing in `step` (W&B requires that).

Usage (online, after `wandb login`):
    python backfill_wandb.py \
        --files slurm_logs/stage3_probes_29513721.out \
                slurm_logs/stage3_probes_29643092.out \
        --project lensed-galaxy-prior --run-name probes_final \
        --args-json outputs/probes_final/args.json
"""
import argparse
import json
import re
from pathlib import Path

# Matches the line format emitted by train_prior.py's logging block.
LINE_RE = re.compile(
    r"epoch=(\d+)\s+step=(\d+)\s+loss=([\d.eE+-]+)\s+"
    r"lr=([\d.eE+-]+)\s+elapsed=([\d.]+)h"
)


def parse_files(files):
    """Merge step->metrics across files; later files win on overlapping steps."""
    merged = {}
    per_file = {}
    for path in files:
        count = 0
        for line in Path(path).read_text().splitlines():
            m = LINE_RE.search(line)
            if not m:
                continue
            epoch, step = int(m.group(1)), int(m.group(2))
            loss, lr, elapsed = float(m.group(3)), float(m.group(4)), float(m.group(5))
            merged[step] = {
                "epoch": epoch,
                "train/loss": loss,
                "train/lr": lr,
                "sys/elapsed_hours": elapsed,
                "_source": Path(path).name,
            }
            count += 1
        per_file[path] = count
    return merged, per_file


def main():
    """Backfill a Weights & Biases run from saved SLURM log files."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--files", nargs="+", required=True,
                    help="SLURM .out files in chronological order.")
    ap.add_argument("--project", default="lensed-galaxy-prior")
    ap.add_argument("--entity", default=None, help="W&B team/username (optional).")
    ap.add_argument("--run-name", default="probes_final")
    ap.add_argument("--args-json", default=None,
                    help="Optional training args.json to attach as W&B config.")
    ap.add_argument("--mode", default="online", choices=["online", "offline"])
    ap.add_argument("--dry-run", action="store_true",
                    help="Parse and report only; do not touch W&B.")
    args = ap.parse_args()

    merged, per_file = parse_files(args.files)
    if not merged:
        raise SystemExit("No matching log lines found - check the file format.")

    steps = sorted(merged)
    print("Parsed:")
    for path, n in per_file.items():
        print(f"  {path}: {n} lines")
    print(f"Merged: {len(steps)} unique steps, range {steps[0]}..{steps[-1]}")
    print(f"  first loss={merged[steps[0]]['train/loss']:.4e}  "
          f"last loss={merged[steps[-1]]['train/loss']:.4e}")

    if args.dry_run:
        print("[dry-run] not contacting W&B.")
        return

    import wandb

    config = {"backfilled_from": [Path(f).name for f in args.files],
              "n_logged_steps": len(steps)}
    if args.args_json and Path(args.args_json).is_file():
        config.update(json.loads(Path(args.args_json).read_text()))

    run = wandb.init(project=args.project, entity=args.entity,
                     name=args.run_name, config=config, mode=args.mode)
    # x-axis = training step for every train/* and sys/* metric.
    wandb.define_metric("train/step")
    wandb.define_metric("train/*", step_metric="train/step")
    wandb.define_metric("sys/*", step_metric="train/step")

    for step in steps:
        row = merged[step]
        wandb.log(
            {
                "train/step": step,
                "epoch": row["epoch"],
                "train/loss": row["train/loss"],
                "train/lr": row["train/lr"],
                "sys/elapsed_hours": row["sys/elapsed_hours"],
            },
            step=step,
        )

    run.summary["final/loss"] = merged[steps[-1]]["train/loss"]
    run.summary["final/step"] = steps[-1]
    run.summary["final/epoch"] = merged[steps[-1]]["epoch"]
    run.summary["min/loss"] = min(r["train/loss"] for r in merged.values())
    wandb.finish()
    print(f"Done: logged {len(steps)} steps to W&B project '{args.project}'.")


if __name__ == "__main__":
    main()
