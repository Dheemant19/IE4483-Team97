"""Part (e): test-set cases of the final model, after a visual check of all 500 test predictions.

    python scripts/make_test_cases.py        # -> report/figures/test_cases.png|pdf, test_confidence.png|pdf

test_cases       (a) the test images with the least confident predictions (all correct by visual check),
                 (b) the most confident cats and dogs, for contrast
test_confidence  distribution of the final model's p(dog) over the 500 test images
A visual check of all 500 test images (contact sheets from make_test_review.py, 18 Oct 2026) found no wrong
prediction; --wrong lets you pass ids that turn out to be wrong, which are then drawn in vermillion.
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import figstyle as fs  # noqa: E402
from figstyle import INK, INK2, MUTED, plt  # noqa: E402

NAMES = {0: "cat", 1: "dog"}


def thumb(path, side=320):
    with Image.open(path) as im:
        im = im.convert("RGB")
        im.thumbnail((side, side))
        sq = Image.new("RGB", (side, side), (255, 255, 255))
        sq.paste(im, ((side - im.width) // 2, (side - im.height) // 2))
        return np.asarray(sq)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probs", default="runs/ens_dinov2_b_probe/test_probs.csv")
    ap.add_argument("--data", default="data_288")
    ap.add_argument("--out", default="report/figures")
    ap.add_argument("--n", type=int, default=6, help="images per row")
    ap.add_argument("--wrong", type=int, nargs="*", default=[], help="test ids found wrong by visual check")
    args = ap.parse_args()
    rows = [(int(r["id"]), float(r["prob_dog"])) for r in csv.DictReader(open(args.probs))]
    p = np.array([q for _, q in rows])

    hard = sorted(rows, key=lambda t: abs(t[1] - 0.5))[:args.n]
    k = args.n // 2
    easy = (sorted([t for t in rows if t[1] < 0.5], key=lambda t: t[1])[:k]
            + sorted([t for t in rows if t[1] > 0.5], key=lambda t: -t[1])[:args.n - k])

    fig, axes = plt.subplots(2, args.n, figsize=(fs.FULL, 2 * fs.FULL / args.n + 12 * fs.MM), layout="constrained")
    for r, (group, letter) in enumerate(((hard, "a"), (easy, "b"))):
        for j, (i, q) in enumerate(group):
            ax = axes[r, j]
            ax.imshow(thumb(Path(args.data) / "test" / f"{i}.jpg"), aspect="equal")
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            wrong = i in args.wrong
            ax.set_title(f"test {i}", fontsize=6.5, pad=2, color=fs.VERMILLION if wrong else INK)
            ax.text(0.5, -0.03, f"{NAMES[int(q > 0.5)]}, p(dog) = {q:.2f}", transform=ax.transAxes, ha="center",
                    va="top", fontsize=6.2, color=fs.VERMILLION if wrong else INK2)
        fs.panel_letter(axes[r, 0], letter)
    fs.save(fig, args.out, "test_cases")

    fig, ax = plt.subplots(figsize=(fs.MID, 50 * fs.MM), layout="constrained")
    bins = np.linspace(0, 1, 41)
    ax.hist(p[p < 0.5], bins=bins, color=fs.BLUE, alpha=0.85, label=f"predicted cat (n = {int((p < 0.5).sum())})")
    ax.hist(p[p >= 0.5], bins=bins, color=fs.VERMILLION, alpha=0.85, hatch="////", edgecolor="white", lw=0,
            label=f"predicted dog (n = {int((p >= 0.5).sum())})")
    ax.axvline(0.5, color=MUTED, lw=0.6, ls=(0, (3, 2)))
    ax.set_yscale("log")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Predicted p(dog), final model (3-seed average with flip TTA)")
    ax.set_ylabel("Test images (log scale)")
    ax.legend(loc="upper center", frameon=False, ncols=2)
    fs.save(fig, args.out, "test_confidence")

    between = int(((p > 0.2) & (p < 0.8)).sum())
    cap = {
        "test_cases": (
            f"Test-set examples of the final model. (a) The {args.n} test images with the least confident "
            "predictions; a visual check of all 500 test images found every prediction correct, including these. "
            "(b) The most confident cats and dogs. p(dog) is the final model's probability (3-seed average with "
            "flip TTA); because training used label smoothing 0.1, confident predictions sit near 0.05 (cat) and "
            "0.95 (dog) rather than 0 and 1."),
        "test_confidence": (
            f"Distribution of the final model's p(dog) over the 500 test images (log-scale counts). Only {between} "
            "images fall between 0.2 and 0.8; the rest are confident. Label smoothing (0.1) keeps confident "
            "predictions near 0.05 and 0.95."),
    }
    with open(Path(args.out) / "captions_test_cases.md", "w", encoding="utf-8") as fh:
        fh.write("# Part (e) test-set figure captions (draft)\n\nAdd what each hard image shows (see CONTEXT.md §4e).\n")
        for name, c in cap.items():
            fh.write(f"\n## {name}\n\n**Caption.** {c}\n")


if __name__ == "__main__":
    main()
