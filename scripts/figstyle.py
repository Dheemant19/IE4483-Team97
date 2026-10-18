"""Shared look for every report figure (make_plots.py and analyze_errors.py).

Only needs matplotlib, numpy and Pillow, so it also runs on the MLDA server. The style file and the
Okabe-Ito palette live in scripts/style/ (copied from the scientific-visualization skill).
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

STYLE_DIR = Path(__file__).resolve().parent / "style"
sys.path.insert(0, str(STYLE_DIR))
from color_palettes import OKABE_ITO  # noqa: E402

plt.style.use(STYLE_DIR / "report.mplstyle")

MM = 1 / 25.4
FULL = 160 * MM          # full text width of an A4 report
WIDE = 120 * MM
MID = 110 * MM

INK, INK2, MUTED, RULE = "#222222", "#444444", "#767676", "#E4E4E4"
BLUE, VERMILLION, GREEN = OKABE_ITO["blue"], OKABE_ITO["vermillion"], OKABE_ITO["bluish_green"]
PURPLE, BLACK = OKABE_ITO["reddish_purple"], OKABE_ITO["black"]

# One meaning per colour across Dogs vs Cats and CIFAR-10 part (g): how the backbone was trained.
STRATEGY = {
    "No pretraining": dict(color=VERMILLION, marker="^"),
    "Fine-tuned": dict(color=BLUE, marker="o"),
    "Linear probe": dict(color=GREEN, marker="s"),
}


def panel_letter(ax, letter, extra="", x=-0.01, y=1.0):
    """Bold panel letter at the top-left corner of the axes, optionally followed by a short tag."""
    t = ax.annotate(letter, (0, 1), xycoords="axes fraction", xytext=(x * 100, 6), textcoords="offset points",
                    fontsize=8.5, fontweight="bold", ha="right", va="bottom", color=INK)
    if extra:
        ax.annotate(extra, (0, 1), xycoords="axes fraction", xytext=(x * 100 + 3, 6.4), textcoords="offset points",
                    fontsize=7, ha="left", va="bottom", color=INK)
    return t


def flatten_png(path):
    """Matplotlib writes RGBA PNGs; the background is opaque white, so store plain RGB (no alpha)."""
    from PIL import Image
    with Image.open(path) as im:
        rgb = im.convert("RGB")
        dpi = im.info.get("dpi", (600, 600))
    rgb.save(path, dpi=tuple(round(d) for d in dpi), optimize=True)


def save(fig, out, name, dpi=600, pdf=True):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.png", dpi=dpi)
    flatten_png(out / f"{name}.png")
    if pdf:
        fig.savefig(out / f"{name}.pdf")
    plt.close(fig)
    print(f"wrote {out / name}.png" + (" + .pdf" if pdf else ""))
