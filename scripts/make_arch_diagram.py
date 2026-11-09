"""Part (b): architecture diagrams of the final model and the scratch CNN.

Shapes are read from report/model_summaries/stages.csv (written by model_summary.py).

    python scripts/make_arch_diagram.py    # -> report/figures/arch_final_model.*, arch_scratch_cnn.*
"""
import argparse
import csv
from pathlib import Path

from matplotlib.colors import to_rgba
from matplotlib.patches import FancyBboxPatch

import figstyle as fs
from figstyle import INK, INK2, MUTED, plt

FROZEN_FILL, FROZEN_EDGE = "#EFEFEF", "#9A9A9A"
PLAIN_FILL, PLAIN_EDGE = "#FFFFFF", "#9A9A9A"


def shapes(path):
    out = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            dims = [int(x) for x in r["output_shape"].strip("()").split(",") if x.strip()][1:]   # drop batch dim
            out[(r["model"], r["stage"])] = " × ".join(str(d) for d in dims)
    return out


def box(ax, x, y, w, h, title, body, shape, fill, edge, bold_edge=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.0", fc=fill, ec=edge,
                                lw=1.0 if bold_edge else 0.6, zorder=2))
    has_body = bool(body)
    ax.text(x + w / 2, y + h - 2.3 if has_body else y + h / 2, title, ha="center", va="top" if has_body else "center",
            fontsize=6.3, fontweight="bold", color=INK, zorder=3, linespacing=1.15)
    if has_body:
        ax.text(x + w / 2, y + h - 2.3 - 3.4 * (title.count("\n") + 1), body, ha="center", va="top", fontsize=5.9,
                color=INK2, zorder=3, linespacing=1.25)
    if shape:
        ax.text(x + w / 2, y - 1.6, shape, ha="center", va="top", fontsize=6.4, color=INK, zorder=3)


def arrow(ax, x0, x1, y):
    ax.annotate("", xy=(x1, y), xytext=(x0, y), arrowprops=dict(arrowstyle="-|>", lw=0.6, color=INK2,
                                                               mutation_scale=6, shrinkA=0, shrinkB=0), zorder=1)


def chain(ax, items, y, h, gap):
    """items: (width, title, body, shape, fill, edge, bold). Returns the x extents of each box."""
    x, spans = 0.0, []
    for i, (w, title, body, shp, fill, edge, bold) in enumerate(items):
        box(ax, x, y, w, h, title, body, shp, fill, edge, bold)
        spans.append((x, x + w))
        if i < len(items) - 1:
            arrow(ax, x + w + 0.4, x + w + gap - 0.4, y + h / 2)
        x += w + gap
    return spans


def bracket(ax, x0, x1, y, text, color):
    ax.plot([x0, x0, x1, x1], [y - 1.2, y, y, y - 1.2], color=color, lw=0.7, solid_capstyle="butt",
            solid_joinstyle="miter", zorder=1)
    ax.text((x0 + x1) / 2, y + 1.0, text, ha="center", va="bottom", fontsize=6.2, color=INK)


def legend(ax, x, y, entries):
    for i, (fill, edge, label) in enumerate(entries):
        ax.add_patch(FancyBboxPatch((x, y - i * 4.2), 4.0, 2.6, boxstyle="round,pad=0,rounding_size=0.5", fc=fill,
                                    ec=edge, lw=0.7))
        ax.text(x + 5.4, y - i * 4.2 + 1.3, label, va="center", fontsize=6.2, color=INK)


def new_axes(width_mm, height_mm):
    fig = plt.figure(figsize=(width_mm * fs.MM, height_mm * fs.MM))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(-1.5, width_mm - 1.5)
    ax.set_ylim(0, height_mm)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def final_model(S, out):
    m = "dinov2_b_probe"
    head_fill = to_rgba(fs.GREEN, 0.16)
    items = [
        (17, "Input image", "RGB, resized,\ncentre-cropped", S[(m, "input")], PLAIN_FILL, PLAIN_EDGE, False),
        (20, "Patch embedding", "14 × 14 conv,\nstride 14 → 16 × 16\npatches of 768", S[(m, "patch_embed")],
         FROZEN_FILL, FROZEN_EDGE, False),
        (18, "+ [CLS] token", "+ learned position\nembeddings", S[(m, "block 1")], FROZEN_FILL, FROZEN_EDGE, False),
        (29, "Transformer block × 12", "LayerNorm → 12-head\nself-attention → LayerNorm\n→ MLP 768→3072→768\n"
         "(residual + LayerScale)", S[(m, "block 12")], FROZEN_FILL, FROZEN_EDGE, False),
        (17, "LayerNorm,\nkeep [CLS]", "image feature\nvector", "768", FROZEN_FILL, FROZEN_EDGE, False),
        (18, "Linear head", "768 → 2\n1,538 weights\n(only trained part)", S[(m, "head (linear)")], head_fill,
         fs.GREEN, True),
        (16, "Softmax", "p(cat), p(dog)\nlabel = argmax", "2", PLAIN_FILL, PLAIN_EDGE, False),
    ]
    gap = 4.2
    width = sum(w for w, *_ in items) + gap * (len(items) - 1) + 3
    fig, ax = new_axes(width, 35)
    spans = chain(ax, items, y=6.5, h=20, gap=gap)
    bracket(ax, spans[1][0], spans[4][1], 29.5, "DINOv2-B backbone (ViT-B/14): pretrained, frozen, 85.7 M parameters",
            FROZEN_EDGE)
    bracket(ax, spans[5][0], spans[5][1], 29.5, "trained", fs.GREEN)
    fs.save(fig, out, "arch_final_model")
    return ("arch_final_model",
            "Architecture of the final model: a DINOv2-B vision transformer (ViT-B/14) used as a frozen feature "
            "extractor with a trained linear classifier (linear probe). A 224 × 224 RGB image is split into 16 × 16 "
            "patches of 14 × 14 pixels, embedded to 768 dimensions, prefixed with a learnable [CLS] token and passed "
            "through 12 transformer blocks; the normalised [CLS] feature (768) feeds a linear layer with 2 outputs and "
            "a softmax. Only the linear layer (1,538 weights) is trained, with cross-entropy (label smoothing 0.1), "
            "AdamW and a cosine schedule for 10 epochs; the 85.7 M backbone parameters stay frozen. Shapes under each "
            "block are the stage outputs (batch dimension omitted).",
            "Flow diagram of seven boxes from input image to softmax output, with the four backbone boxes in grey "
            "(frozen) and the linear head in green (trained); tensor shapes 3×224×224, 256×768, 257×768, 257×768, 768, "
            "2 and 2 are written under the boxes.")


def scratch_cnn(S, out):
    m = "scratch_cnn"
    fill = to_rgba(fs.VERMILLION, 0.13)
    edge = fs.VERMILLION
    chans = [32, 64, 128, 256, 256]
    items = [(15, "Input", "RGB, resized\nto 128 × 128", S[(m, "input")], PLAIN_FILL, PLAIN_EDGE, False)]
    for i, c in enumerate(chans, start=1):
        items.append((16, f"Conv block {i}", f"2 × [3×3 conv\n({c} ch), BN,\nReLU]\n2×2 max-pool",
                      S[(m, f"conv block {i} (after max-pool)")], fill, edge, False))
    items += [(16.5, "Global\naverage pool", "", "256", fill, edge, False),
              (15, "Head", "dropout 0.3\nlinear 256 → 2", S[(m, "head (dropout + linear)")], fill, edge, False),
              (15, "Softmax", "p(cat), p(dog)", "2", PLAIN_FILL, PLAIN_EDGE, False)]
    gap = 3.0
    width = sum(w for w, *_ in items) + gap * (len(items) - 1) + 3
    fig, ax = new_axes(width, 35)
    spans = chain(ax, items, y=6.5, h=20, gap=gap)
    bracket(ax, spans[1][0], spans[7][1], 29.5, "trained from random initialisation, 2.35 M parameters", edge)
    fs.save(fig, out, "arch_scratch_cnn")
    return ("arch_scratch_cnn",
            "Architecture of the scratch CNN (model M1). Five convolutional blocks, each two 3 × 3 convolutions with "
            "batch normalisation and ReLU followed by 2 × 2 max-pooling, halve the 128 × 128 input to 4 × 4 while the "
            "channels grow 32 → 64 → 128 → 256 → 256; global average pooling gives a 256-dimensional feature, "
            "followed by dropout (0.3), a linear layer to 2 logits and a softmax. All 2.35 M parameters are trained "
            "from He initialisation with SGD (Nesterov momentum 0.9), a one-cycle schedule (peak LR 0.05), weight "
            "decay 5e-4, batch 128, 50 epochs and cross-entropy with label smoothing 0.1.",
            "Flow diagram of nine boxes from a 3×128×128 input through five convolutional blocks (32×64×64 down to "
            "256×4×4), global average pooling (256), a dropout-plus-linear head and a softmax.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default="report/model_summaries/stages.csv")
    ap.add_argument("--out", default="report/figures")
    args = ap.parse_args()
    S = shapes(args.stages)
    caps = [final_model(S, args.out), scratch_cnn(S, args.out)]
    with open(Path(args.out) / "captions_arch.md", "w", encoding="utf-8") as fh:
        fh.write("# Architecture figure captions and alt text (draft)\n\nGenerated by scripts/make_arch_diagram.py; "
                 "layer tables with every layer's input/output shape are in report/model_summaries/.\n")
        for name, cap, alt in caps:
            fh.write(f"\n## {name}\n\n**Caption.** {cap}\n\n**Alt text.** {alt}\n")


if __name__ == "__main__":
    main()
