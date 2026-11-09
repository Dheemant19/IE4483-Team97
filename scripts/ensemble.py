"""Average the flip-TTA predictions of several finished runs and write the ensemble's submission.csv.

    python scripts/ensemble.py runs/dinov2_b_probe_s0 runs/dinov2_b_probe_s1 runs/dinov2_b_probe_s2 \
        --out runs/ens_dinov2_b_probe
"""
import argparse
import csv
import json
from pathlib import Path


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    runs = [Path(r) for r in args.runs]
    val = [read_csv(r / "val_predictions.csv") for r in runs]
    test = [read_csv(r / "test_probs.csv") for r in runs]
    for r, v, t in zip(runs, val, test):
        assert [row["path"] for row in v] == [row["path"] for row in val[0]], f"val images differ: {r}"
        assert [row["id"] for row in t] == [row["id"] for row in test[0]], f"test ids differ: {r}"

    n = len(runs)
    labels = [int(row["label"]) for row in val[0]]
    p_val = [sum(float(v[i]["prob_dog_tta"]) for v in val) / n for i in range(len(labels))]
    pred = [int(p > 0.5) for p in p_val]
    acc = sum(p == y for p, y in zip(pred, labels)) / len(labels)
    cm = [[sum(1 for p, y in zip(pred, labels) if y == t and p == q) for q in (0, 1)] for t in (0, 1)]
    singles = {}
    for r, v in zip(runs, val):
        singles[r.name] = sum(int(float(row["prob_dog_tta"]) > 0.5) == int(row["label"]) for row in v) / len(labels)

    ids = [int(row["id"]) for row in test[0]]
    p_test = [sum(float(t[i]["prob_dog"]) for t in test) / n for i in range(len(ids))]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "val_predictions.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "label", "prob_dog_tta", "pred_tta"])
        w.writerows([row["path"], y, f"{p:.6f}", q] for row, y, p, q in zip(val[0], labels, p_val, pred))
    with open(out / "test_probs.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "prob_dog"])
        w.writerows([i, f"{p:.6f}"] for i, p in zip(ids, p_test))
    with open(out / "submission.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "label"])
        w.writerows([i, int(p > 0.5)] for i, p in sorted(zip(ids, p_test)))
    metrics = dict(name=out.name, members=[r.name for r in runs], val_acc=acc, val_acc_tta=acc,
                   confusion_matrix_rows_true_cols_pred=cm, member_val_acc_tta=singles,
                   n_val=len(labels), n_test=len(ids))
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))

    for name, a in singles.items():
        print(f"  {name:32s} val acc (TTA) {a:.4f}")
    print(f"ensemble of {n}: val acc (TTA) {acc:.4f}, confusion {cm}")
    print(f"wrote {out / 'submission.csv'} ({len(ids)} rows)")


if __name__ == "__main__":
    main()
