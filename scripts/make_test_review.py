"""Contact sheets of all 500 test images, grouped by predicted class, for checking the predictions by eye.

    python scripts/make_test_review.py    # -> report/test_review/predicted_{cat,dog}_*.png, test_review.csv
"""
import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402


def sheet(items, title, path, ncols=10, nrows=6):
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.6, nrows * 1.85 + 0.5))
    for ax in axes.flat:
        ax.axis("off")
    for ax, (i, p, img) in zip(axes.flat, items):
        with Image.open(img) as im:
            im = im.convert("RGB")
            im.thumbnail((256, 256))
            ax.imshow(im)
        ax.set_title(f"id {i}   p(dog) {p:.2f}", fontsize=7.5, pad=2,
                     color="#B54A00" if abs(p - 0.5) < 0.25 else "#222222")
    fig.suptitle(title, x=0.01, ha="left", fontsize=10, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97), h_pad=0.6, w_pad=0.3)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probs", default="runs/ens_dinov2_b_probe/test_probs.csv")
    ap.add_argument("--data", default="data_288")
    ap.add_argument("--out", default="report/test_review")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = [(int(r["id"]), float(r["prob_dog"])) for r in csv.DictReader(open(args.probs))]
    for cls, name in ((0, "cat"), (1, "dog")):
        items = sorted([(i, p) for i, p in rows if int(p > 0.5) == cls], key=lambda t: abs(t[1] - 0.5))
        items = [(i, p, Path(args.data) / "test" / f"{i}.jpg") for i, p in items]
        for k in range(0, len(items), 60):
            n = k // 60 + 1
            sheet(items[k:k + 60], f"Predicted {name.upper()} ({len(items)} images) - sheet {n}, least confident first. "
                  f"Look for any {'dog' if cls == 0 else 'cat'}.", out / f"predicted_{name}_{n}.png")
    with open(out / "test_review.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "predicted", "p_dog", "wrong (1 = model wrong)", "note"])
        for i, p in sorted(rows):
            w.writerow([i, "dog" if p > 0.5 else "cat", f"{p:.4f}", "", ""])
    print(f"wrote {out / 'test_review.csv'}")


if __name__ == "__main__":
    main()
