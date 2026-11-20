"""
Compare the runs of scripts/lr_sweep.sh and list them from best to worst.

Usage:  python scripts/pick_best_lr.py <sweep_root>
"""
import sys
import pandas as pd
from pathlib import Path
import re

SWEEP_ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "./results/lr_sweep")
METRIC = "balanced_accuracy"   # change to acc1, macro_f1, etc. if you prefer
NOISE_WINDOW = 5               # average over last N epochs instead of trusting one epoch

rows = []
skipped = []
for run_dir in sorted(SWEEP_ROOT.glob("*")):
    csv_path = run_dir / "metrics" / "val_log.csv"
    if not csv_path.exists():
        continue
    df = pd.read_csv(csv_path)
    if df.empty:
        skipped.append((run_dir.name, "val_log.csv has no rows yet -- run still in progress "
                                       "or crashed before its first eval epoch"))
        continue

    train_csv_path = run_dir / "metrics" / "train_log.csv"
    train_df = pd.read_csv(train_csv_path) if train_csv_path.exists() else pd.DataFrame()
    if train_df.empty:
        final_train_loss = float("nan")
    else:
        final_train_loss = train_df["loss"].iloc[-1]

    tail = df.tail(NOISE_WINDOW)
    rows.append({
        "run": run_dir.name,
        f"{METRIC}_last{NOISE_WINDOW}_avg": tail[METRIC].mean(),
        f"{METRIC}_last{NOISE_WINDOW}_std": tail[METRIC].std(),
        f"{METRIC}_final_epoch": df[METRIC].iloc[-1],
        "final_train_loss": final_train_loss,
        "n_epochs_logged": len(df),
    })

if skipped:
    print("Skipped (no usable data yet):")
    for name, reason in skipped:
        print(f"  - {name}: {reason}")
    print()

if not rows:
    print("No completed runs found under", SWEEP_ROOT)
    print("Check that lr_sweep.sh has actually finished at least one epoch of at least one run.")
else:
    summary = pd.DataFrame(rows).sort_values(f"{METRIC}_last{NOISE_WINDOW}_avg", ascending=False)
    print(summary.to_string(index=False))

    print("\nWinner (highest, low-noise):", summary.iloc[0]["run"])
    print("Watch for: rows with very low final_train_loss but weak val metric -> already overfitting even at 30 epochs, deprioritize that LR even if its avg looks OK.")
    print("Watch for: rows with low n_epochs_logged -> that run may still be in progress; its ranking could change once it finishes.")
