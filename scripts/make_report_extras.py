"""Part (a) image-size histogram and parts (b, c) training curves of the DINOv2-B probe vs. fine-tuning.

    python scripts/make_report_extras.py --data datasets --out report/figures
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from figstyle import FULL, INK, MUTED, RULE, STRATEGY, panel_letter, plt, save

ROOT = Path(__file__).resolve().parents[1]
EXCLUDE = "val/cat/cat.12408.jpg"


def image_sizes(data, out):
    from PIL import Image
    short = []
    for split in ("train", "val", "test"):
        for p in Path(data, split).rglob("*.jpg"):
            if p.as_posix().endswith(EXCLUDE):
                continue
            with Image.open(p) as im:
                short.append(min(im.size))
    short = np.array(short)
    fig, ax = plt.subplots(figsize=(FULL * 0.7, FULL * 0.3))
    bins = np.arange(0, 540, 20)
    # the 2 images above 520 px go in the last bin
    ax.hist(np.minimum(short, 510), bins=bins, color=MUTED, edgecolor="white", linewidth=0.3)
    top = ax.get_ylim()[1]
    marks = ((128, "128 px\nscratch CNN input", 0.65), (224, "224 px\npretrained input", 0.80),
             (288, "288 px\npre-resize", 0.95))
    for x, label, h in marks:
        ax.axvline(x, color=INK, lw=0.8, ls="--" if x == 288 else ":")
        ax.text(x - 6, top * h, label, fontsize=6.5, va="top", ha="right", color=INK,
                bbox=dict(fc="white", ec="none", pad=0.6))
    ax.set_xlabel("Shorter image side (pixels)")
    ax.set_ylabel("Images")
    ax.set_xlim(0, 520)
    ax.grid(axis="y", color=RULE)
    fig.tight_layout()
    save(fig, out, "image_sizes")
    print(f"{len(short)} images; shorter side min {short.min()}, median {np.median(short):.0f}, "
          f"< 224: {(short < 224).sum()}, > 288: {(short > 288).sum()}")


def history(run):
    return json.loads((ROOT / "runs" / run / "metrics.json").read_text())["history"]


def probe_curves(out):
    groups = {"Linear probe": "dinov2_b_probe_s{}", "Fine-tuned": "dinov2_b_s{}"}
    fig, (a, b) = plt.subplots(1, 2, figsize=(FULL, FULL * 0.3))
    for label, pattern in groups.items():
        st = STRATEGY[label]
        hs = [history(pattern.format(s)) for s in (0, 1, 2)]
        ep = np.array([h["epoch"] for h in hs[0]])
        for ax, key, scale in ((a, "train_loss", 1), (b, "val_acc", 100)):
            y = np.array([[h[key] * scale for h in run] for run in hs])
            ax.fill_between(ep, y.min(0), y.max(0), color=st["color"], alpha=0.18, lw=0)
            ax.plot(ep, y.mean(0), color=st["color"], marker=st["marker"], ms=3.5, lw=1.2,
                    label=f"DINOv2-B, {label.lower()}")
    # cross-entropy of a perfect prediction against label-smoothed targets (0.95 / 0.05)
    floor = -(0.95 * math.log(0.95) + 0.05 * math.log(0.05))
    a.axhline(floor, color=MUTED, ls="--", lw=0.8)
    a.annotate("lowest loss possible with\nlabel smoothing 0.1", xy=(6.5, floor), xytext=(6.5, 0.265), fontsize=6.5,
               ha="center", va="bottom", color=MUTED, arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.7))
    a.set_ylabel("Training loss")
    b.set_ylabel("Validation accuracy (%)")
    for ax, letter in ((a, "a"), (b, "b")):
        ax.axvspan(-0.5, 0.5, color="#F0F0F0", lw=0, zorder=0)
        ax.set_xlim(-0.5, 9.5)
        ax.set_xticks(range(10))
        ax.set_xlabel("Epoch")
        ax.grid(axis="y", color=RULE)
        panel_letter(ax, letter)
    b.legend(loc="lower right", frameon=False, fontsize=6.5)
    fig.tight_layout()
    save(fig, out, "probe_curves")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "datasets"))
    ap.add_argument("--out", default=str(ROOT / "report" / "figures"))
    args = ap.parse_args()
    image_sizes(args.data, args.out)
    probe_curves(args.out)


if __name__ == "__main__":
    main()
