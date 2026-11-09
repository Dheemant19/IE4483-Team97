"""Print a table of all runs (mean and SD over seeds) and write runs/summary.csv.

    python scripts/summarize.py
"""
import argparse
import csv
import json
import re
import statistics as st
from pathlib import Path


def fmt(xs):
    xs = [100 * x for x in xs]
    if len(xs) == 1:
        return f"{xs[0]:.2f}"
    return f"{st.mean(xs):.2f} ± {st.stdev(xs):.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    args = ap.parse_args()

    groups = {}
    for f in sorted(Path(args.runs).glob("*/metrics.json")):
        m = json.loads(f.read_text())
        key = re.sub(r"_s\d+$", "", f.parent.name)
        groups.setdefault(key, []).append(m)

    rows = []
    for key, ms in sorted(groups.items()):
        rows.append({
            "config": key,
            "seeds": len(ms),
            "val_acc": fmt([m["val_acc"] for m in ms]),
            "val_acc_tta": fmt([m["val_acc_tta"] for m in ms]) if "val_acc_tta" in ms[0] else "",
            "best_val_acc": fmt([m["best_val_acc"] for m in ms]) if all("best_val_acc" in m for m in ms) else "",
            "lp_val_acc": fmt([m["lp_val_acc"] for m in ms]) if all(m.get("lp_val_acc") is not None for m in ms) else "",
            # runs written before 17 Oct 13:40 have no "select" key and report the best-val epoch
            "select": ",".join(sorted({m.get("select", "best" if "history" in m else "-") for m in ms})),
            "n_train": ms[0].get("n_train", ""),
            "params_m": ms[0].get("params_m", ""),
            "train_min": f"{st.mean(m.get('total_train_min', 0) for m in ms):.1f}",
        })

    cols = ["config", "seeds", "select", "val_acc", "val_acc_tta", "best_val_acc", "lp_val_acc",
            "n_train", "params_m", "train_min"]
    print("| " + " | ".join(cols) + " |")
    print("|" + "---|" * len(cols))
    for r in rows:
        print("| " + " | ".join(str(r[c]) for c in cols) + " |")
    with open(Path(args.runs) / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, cols)
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
