"""Report figures from runs/*/metrics.json -> report/figures/<name>.png (600 dpi) and .pdf.

Also writes captions.md and data/<name>.csv. Individual seeds are drawn next to their mean.

    python scripts/make_plots.py --runs runs --out report/figures
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter, NullLocator

import figstyle as fs
from figstyle import MM, INK, INK2, MUTED, RULE, STRATEGY, plt

CAPTIONS = []
pct0 = FuncFormatter(lambda v, _: f"{v:g}")


# ---------------------------------------------------------------- data

def load_runs(root):
    runs = {}
    for f in Path(root).glob("*/metrics.json"):
        try:
            runs[f.parent.name] = json.loads(f.read_text())
        except json.JSONDecodeError:
            pass
    return runs


def seeds(runs, base):
    pat = re.compile(re.escape(base) + r"_s(\d+)$")
    hits = sorted((int(m.group(1)), r) for n, r in runs.items() if (m := pat.match(n)))
    if not hits and base in runs:                       # deterministic runs (HOG) have no seed suffix
        return [runs[base]]
    return [r for _, r in hits]


def stats(values):
    v = np.asarray(list(values), dtype=float)
    return None if v.size == 0 else dict(values=v, mean=v.mean(), lo=v.min(), hi=v.max(), n=v.size)


def write_data(out, name, df):
    (Path(out) / "data").mkdir(parents=True, exist_ok=True)
    df.to_csv(Path(out) / "data" / f"{name}.csv", index=False, float_format="%.4f")


def finish(fig, out, name, caption, alt, df):
    fs.save(fig, out, name)
    write_data(out, name, df)
    CAPTIONS.append((name, caption, alt))


# ---------------------------------------------------------------- drawing helpers

def tie_offsets(xs, step=0.11):
    """0 for every seed, except tied values, which are spread across the row so none is hidden."""
    off, seen = np.zeros(len(xs)), {}
    for i, v in enumerate(np.round(xs, 6)):
        k = seen.get(v, 0)
        off[i] = (0, -step, step, -2 * step)[min(k, 3)]
        seen[v] = k + 1
    return off


def seed_marks(ax, xs, y, color, marker, horizontal=True, ms=3.0, z=3):
    """Individual seeds as small open markers on the row line (tied values spread across the row)."""
    if len(xs) < 2:
        return
    off = tie_offsets(xs)
    if horizontal:
        ax.plot(xs, y + off, ls="none", marker=marker, ms=ms, mfc="white", mec=color, mew=0.7, zorder=z)
    else:
        ax.plot(y + off, xs, ls="none", marker=marker, ms=ms, mfc="white", mec=color, mew=0.7, zorder=z)


def mean_mark(ax, x, y, color, marker, ms=5.2, z=4):
    ax.plot(x, y, ls="none", marker=marker, ms=ms, color=color, mec="white", mew=0.7, zorder=z)


def range_bar(ax, s, y, color):
    if s["n"] > 1:
        ax.plot([s["lo"], s["hi"]], [y, y], color=color, lw=2.2, alpha=0.28, solid_capstyle="round", zorder=2)


def row_axes(ax, labels, gap_after=()):
    """Categorical rows, top to bottom; returns y positions. gap_after adds space after those row indices."""
    ys, y = [], 0.0
    for i in range(len(labels)):
        ys.append(-y)
        y += 1 + (0.55 if i in gap_after else 0)
    ax.set_yticks(ys, labels)
    ax.tick_params(axis="y", length=0, pad=4)
    ax.spines["left"].set_visible(False)
    ax.set_ylim(min(ys) - 0.65, 0.65)
    ax.grid(axis="x", color=RULE, lw=0.5)
    return ys


def value_text(ax, x, y, text, side="right", pad=5):
    ax.annotate(text, (x, y), xytext=(pad if side == "right" else -pad, 0), textcoords="offset points",
                ha="left" if side == "right" else "right", va="center", fontsize=6.3, color=INK2)


def log_ticks(axis, ticks, fmt=pct0):
    axis.set_major_locator(FixedLocator(ticks))
    axis.set_major_formatter(fmt)
    axis.set_minor_formatter(NullFormatter())


def strategy_legend(ax, keys, ncol=3, y=1.0):
    hs = [Line2D([], [], ls="none", ms=5, mec="white", mew=0.6, label=k, **STRATEGY[k]) for k in keys]
    ax.legend(handles=hs, loc="lower left", bbox_to_anchor=(0, y), ncol=ncol, borderaxespad=0.3,
              handletextpad=0.2, columnspacing=1.4)


def spread_labels(ys, min_gap):
    """Push label y positions apart (keeping order) so they are at least min_gap apart."""
    order = np.argsort(ys)
    pos = np.asarray(ys, dtype=float)[order].copy()
    for _ in range(200):
        moved = False
        for i in range(1, len(pos)):
            d = pos[i] - pos[i - 1]
            if d < min_gap:
                shift = (min_gap - d) / 2
                pos[i - 1] -= shift
                pos[i] += shift
                moved = True
        if not moved:
            break
    out = np.empty_like(pos)
    out[order] = pos
    return out


def err(acc):
    return 100 * (1 - acc)


# ---------------------------------------------------------------- figures

def fig_datasize(runs, out):
    series = [("Scratch CNN", "scratch_cnn", "No pretraining", "-"),
              ("ResNet-50 (FT)", "resnet50", "Fine-tuned", "-"),
              ("ConvNeXt-T (FT)", "convnext_tiny", "Fine-tuned", (0, (4, 1.6))),
              ("DINOv2-B (probe)", "dinov2_b_probe", "Linear probe", "-")]
    markers = {"Scratch CNN": "^", "ResNet-50 (FT)": "o", "ConvNeXt-T (FT)": "D", "DINOv2-B (probe)": "s"}
    ns = [2000, 5000, 10000, 20000]
    rows = []
    for label, base, strat, _ in series:
        for n in ns:
            for i, r in enumerate(seeds(runs, base if n == 20000 else f"{base}_n{n}")):
                rows.append(dict(model=label, n_train=n, seed=i, val_error_pct=err(r["val_acc"]),
                                 val_acc_pct=100 * r["val_acc"]))
    df = pd.DataFrame(rows)
    if df.empty:
        return print("skip datasize_curve")
    fig, ax = plt.subplots(figsize=(fs.MID, 66 * MM), layout="constrained")
    ends = []
    for label, base, strat, ls in series:
        d = df[df.model == label]
        if d.empty:
            continue
        col, mk = STRATEGY[strat]["color"], markers[label]
        m = d.groupby("n_train").val_error_pct.mean()
        # seeds: small open markers, nudged sideways (multiplicatively, log axis) so ties stay visible
        for n, g in d.groupby("n_train"):
            xs = n * np.array([1.0, 0.955, 1.047])[: len(g)]
            ax.plot(xs, g.val_error_pct, ls="none", marker=mk, ms=2.6, mfc="white", mec=col, mew=0.6, zorder=3,
                    alpha=0.9)
        ax.plot(m.index, m.values, color=col, ls=ls, lw=1.3, marker=mk, ms=4.6, mec="white", mew=0.6, zorder=4)
        acc = 100 - m.values[-1]
        ends.append((m.values[-1], f"{label}  {acc:.2f}%", col))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(ns, ["2,000", "5,000", "10,000", "20,000"])
    ax.xaxis.set_minor_locator(NullLocator())
    log_ticks(ax.yaxis, [0.1, 0.2, 0.5, 1, 2, 5, 10, 20])
    ax.set_xlim(1750, 22000)
    ax.set_ylim(0.068, 18)
    ax.grid(axis="y", which="major", color=RULE, lw=0.5)
    ax.set(xlabel="Training images", ylabel="Validation error (%)")
    # direct labels at the right end instead of a legend
    ly = np.exp(spread_labels(np.log([e[0] for e in ends]), 0.22))
    for (y0, txt, col), y1 in zip(ends, ly):
        ax.annotate(txt, (20000, y0), xytext=(22600, y1), textcoords="data", va="center", ha="left",
                    fontsize=6.5, color=INK, annotation_clip=False)
    caption = ("Validation error of four Dogs vs. Cats models trained on class-balanced subsets of the training set "
               "(2,000 to 20,000 images; evaluated on the same 4,999 validation images). Large markers and lines: mean "
               "over 3 seeds; small open markers: individual seeds. Both axes are logarithmic. The value after each "
               "model name is its mean validation accuracy with 20,000 training images. FT: full fine-tuning; probe: "
               "frozen backbone with a trained linear head. Colour gives the training strategy as in the model "
               "comparison figure.")
    alt = ("Log-log line chart of validation error against training-set size. The scratch CNN's error falls from "
           "about 13% at 2,000 images to 2.4% at 20,000; ResNet-50 falls from 1.2% to 0.55%; ConvNeXt-T and the "
           "DINOv2-B probe stay roughly flat near 0.3% and 0.15%.")
    finish(fig, out, "datasize_curve", caption, alt, df)


def fig_model_ladder(runs, out):
    spec = [("HOG + linear SVM", [("hog_svm_linear", "No pretraining")]),
            ("HOG + RBF SVM", [("hog_svm_rbf", "No pretraining")]),
            ("Scratch CNN", [("scratch_cnn", "No pretraining")]),
            ("ResNet-50", [("resnet50", "Fine-tuned"), ("resnet50_probe", "Linear probe")]),
            ("EfficientNetV2-S", [("effnetv2_s", "Fine-tuned")]),
            ("ConvNeXt-T", [("convnext_tiny", "Fine-tuned"), ("convnext_tiny_probe", "Linear probe")]),
            ("DINOv2-S", [("dinov2_s", "Fine-tuned"), ("dinov2_s_probe", "Linear probe")]),
            ("DINOv2-B", [("dinov2_b", "Fine-tuned"), ("dinov2_b_probe", "Linear probe")])]
    table, data = [], []
    for label, entries in spec:
        got = []
        for base, strat in entries:
            s = stats(err(r["val_acc"]) for r in seeds(runs, base))
            if s:
                got.append((strat, s))
                data += [dict(backbone=label, strategy=strat, seed=i, val_error_pct=v, val_acc_pct=100 - v)
                         for i, v in enumerate(s["values"])]
        if got:
            table.append((label, got))
    if not table:
        return print("skip model_ladder")
    table.sort(key=lambda t: -min(s["mean"] for _, s in t[1]))
    fig, ax = plt.subplots(figsize=(fs.WIDE, 80 * MM), layout="constrained")
    ys = row_axes(ax, [t[0] for t in table])
    ax.set_xscale("log")
    sub = {"Fine-tuned": 0.19, "Linear probe": -0.19}
    for y, (label, got) in zip(ys, table):
        for strat, s in got:
            st_ = STRATEGY[strat]
            yy = y + (sub[strat] if len(got) == 2 else 0)
            range_bar(ax, s, yy, st_["color"])
            seed_marks(ax, s["values"], yy, st_["color"], st_["marker"], ms=2.6)
            mean_mark(ax, s["mean"], yy, st_["color"], st_["marker"], ms=4.8)
            value_text(ax, s["hi"], yy, f"{100 - s['mean']:.2f}")
    log_ticks(ax.xaxis, [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50])
    ax.set_xlim(0.075, 60)
    ax.set_xlabel("Validation error (%)")
    strategy_legend(ax, ["Fine-tuned", "Linear probe", "No pretraining"])
    caption = ("Every Dogs vs. Cats model on one logarithmic error axis (4,999 validation images, last training epoch). "
               "One row per backbone, sorted by its best error; blue circles: full fine-tuning, green squares: frozen "
               "backbone with a trained linear head (linear probe), vermillion triangles: no pretraining. Large "
               "markers: mean over 3 seeds; small open markers: individual seeds; pale bar: seed range (HOG + SVM is "
               "deterministic, n = 1). Numbers are mean validation accuracy (%).")
    acc = {(b, st_): 100 - s["mean"] for b, got in table for st_, s in got}
    alt = ("Dot plot with one row per backbone on a log error axis, fine-tuned and linear-probe results on two "
           f"sub-rows: HOG with a linear SVM at {acc[('HOG + linear SVM', 'No pretraining')]:.2f}% accuracy, with an "
           f"RBF SVM at {acc[('HOG + RBF SVM', 'No pretraining')]:.2f}%, the scratch CNN at "
           f"{acc[('Scratch CNN', 'No pretraining')]:.2f}%, and every pretrained backbone above "
           f"{min(v for (b, st_), v in acc.items() if st_ != 'No pretraining'):.2f}%, the best being the DINOv2-B "
           f"linear probe at {acc[('DINOv2-B', 'Linear probe')]:.2f}%.")
    finish(fig, out, "model_ladder", caption, alt, pd.DataFrame(data))


def fig_aug(runs, out):
    spec = [("None", "scratch_cnn_aug-none"), ("Crop + flip", "scratch_cnn_aug-basic"),
            ("Default", "scratch_cnn"), ("Default + TrivialAugment", "scratch_cnn_aug-trivial"),
            ("Default, 100 epochs", "scratch_cnn_100ep")]
    rows, data = [], []
    for label, base in spec:
        s = stats(100 * r["val_acc"] for r in seeds(runs, base))
        if s:
            rows.append((label, s))
            data += [dict(setting=label, seed=i, val_acc_pct=v) for i, v in enumerate(s["values"])]
    if not rows:
        return print("skip aug_ablation")
    fig, ax = plt.subplots(figsize=(fs.MID, 44 * MM), layout="constrained")
    gap = (len(rows) - 2,) if rows[-1][0].endswith("epochs") else ()
    ys = row_axes(ax, [r[0] for r in rows], gap_after=gap)
    col, mk = STRATEGY["No pretraining"]["color"], STRATEGY["No pretraining"]["marker"]
    base = rows[0][1]["mean"]
    ax.axvline(base, color=MUTED, lw=0.6, ls=(0, (2, 2)), zorder=1)
    for y, (label, s) in zip(ys, rows):
        ax.plot([base, s["mean"]], [y, y], color="#D6D6D6", lw=0.8, zorder=1)
        range_bar(ax, s, y, col)
        seed_marks(ax, s["values"], y, col, mk)
        mean_mark(ax, s["mean"], y, col, mk)
        value_text(ax, s["hi"], y, f"{s['mean']:.2f}")
    ax.xaxis.set_major_formatter(pct0)
    ax.set_xlim(94.2, 98.6)
    ax.set_xlabel("Validation accuracy (%)")
    caption = ("Training-time augmentation for the scratch CNN (128 px input, 50 epochs unless stated; 4,999 "
               "validation images). None: resize and centre crop only; crop + flip: random resized crop and "
               "horizontal flip; default: crop + flip plus rotation, colour jitter and random erasing. The dotted line "
               "marks the no-augmentation run. Large marker: mean; small open markers: seeds and pale bar: seed range "
               "(n = 3 for the default; n = 1 otherwise). Numbers are validation accuracy (%).")
    alt = ("Dot plot of scratch-CNN validation accuracy for five augmentation settings: 94.56% without augmentation, "
           "97.72% with crop and flip, 97.65% for the default, 97.50% with TrivialAugment and 97.90% for the default "
           "trained 100 epochs.")
    finish(fig, out, "aug_ablation", caption, alt, pd.DataFrame(data))


LR_COLORS = ["#348FA7", "#37659E", "#413D7B", "#2E1E3B"]       # mako, light to dark = small to large LR
LR_TEX = {"1e-5": r"$10^{-5}$", "3e-5": r"$3{\times}10^{-5}$", "1e-4": r"$10^{-4}$", "3e-4": r"$3{\times}10^{-4}$"}


def fig_lr_sweep(runs, out):
    lrs = [("1e-5", 1e-5), ("3e-5", 3e-5), ("1e-4", 1e-4), ("3e-4", 3e-4)]
    found = [(t, v, runs[f"resnet50_lr{t}_s0"]) for t, v in lrs if f"resnet50_lr{t}_s0" in runs]
    if not found:
        return print("skip lr_sweep")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(fs.FULL, 58 * MM), layout="constrained", sharey=True,
                                 gridspec_kw={"width_ratios": [2.3, 1]})
    mk = ["v", "s", "D", "o"]
    data = []
    for i, (t, v, r) in enumerate(found):
        ep = [h["epoch"] for h in r["history"]]
        acc = [100 * h["val_acc"] for h in r["history"]]
        a1.plot(ep, acc, color=LR_COLORS[i], marker=mk[i], ms=3.4, mec="white", mew=0.5, lw=1.2,
                label=LR_TEX[t])
        data += [dict(lr=t, epoch=a, val_acc_pct=b) for a, b in zip(ep, acc)]
    n_ep = max(len(r["history"]) for _, _, r in found)
    a1.axvspan(-0.5, 0.5, color="#F1F1F1", lw=0, zorder=0)
    a1.set_xlim(-0.5, n_ep - 0.5)
    a1.set_xticks(range(n_ep))
    a1.set(xlabel="Epoch", ylabel="Validation accuracy (%)")
    a1.grid(axis="y", color=RULE, lw=0.5)
    a1.legend(title="Backbone LR", title_fontsize=6.5, loc="lower right", ncol=2, alignment="left")
    fs.panel_letter(a1, "a")

    x = np.array([v for _, v, _ in found])
    y = np.array([100 * r["val_acc"] for _, _, r in found])
    lp = found[0][2].get("lp_val_acc")
    if lp:
        a2.axhline(100 * lp, color=MUTED, lw=0.7, ls=(0, (3, 2)), zorder=1)
        a2.annotate("head only", (x.max() * 1.9, 100 * lp), xytext=(0, 3), textcoords="offset points",
                    ha="right", va="bottom", fontsize=6.3, color=MUTED)
    a2.plot(x, y, color="#9E9E9E", lw=0.9, zorder=2)
    for i, (xi, yi) in enumerate(zip(x, y)):
        a2.plot(xi, yi, ls="none", marker=mk[i], ms=5, color=LR_COLORS[i], mec="white", mew=0.6, zorder=3)
        a2.annotate(f"{yi:.2f}", (xi, yi), xytext=(0, -9), textcoords="offset points", ha="center", va="top",
                    fontsize=6.3, color=INK2)
    a2.set_xscale("log")
    a2.set_xticks(x, [LR_TEX[t] for t, _, _ in found])
    a2.xaxis.set_minor_locator(NullLocator())
    a2.set_xlim(x.min() / 1.8, x.max() * 2.0)
    a2.set_xlabel("Backbone learning rate")
    a2.grid(axis="y", color=RULE, lw=0.5)
    a2.tick_params(axis="y", labelleft=True)
    a1.yaxis.set_major_locator(FixedLocator(np.arange(98.0, 99.81, 0.2)))
    a1.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.1f}"))
    a1.set_ylim(97.95, 99.8)
    fs.panel_letter(a2, "b")
    caption = ("Backbone learning-rate sweep for fine-tuning ResNet-50 on Dogs vs. Cats (AdamW, head LR 1e-3; epoch 0, "
               "shaded, trains only the new linear head, then 9 epochs fine-tune the whole network with warm-up and "
               "cosine decay; seed 0; 4,999 validation images). (a) Validation accuracy per epoch for each backbone "
               "LR. (b) Last-epoch accuracy against the backbone LR (log axis); the dashed line is the accuracy after "
               "the head-only epoch, which no fine-tuning run regains. Single seed: differences below about 0.1 "
               "point (5 images) are within seed-to-seed variation.")
    alt = ("Two panels sharing a validation-accuracy axis. Left: all four learning rates start at 99.58% after the "
           "head-only epoch, drop at epoch 1 when the backbone is unfrozen, and recover, the smallest LR least. "
           "Right: final accuracy rises with LR from 98.98% at 1e-5 to 99.52% at 3e-4, below the 99.58% head-only "
           "line.")
    finish(fig, out, "lr_sweep", caption, alt, pd.DataFrame(data))


def fig_training_curves(runs, out):
    sc = seeds(runs, "scratch_cnn")
    pre = [("ResNet-50", "resnet50_s0", "o"), ("EfficientNetV2-S", "effnetv2_s_s0", "v"),
           ("ConvNeXt-T", "convnext_tiny_s0", "D"), ("DINOv2-S", "dinov2_s_s0", "^"), ("DINOv2-B", "dinov2_b_s0", "s")]
    pre = [(l, runs[n], m) for l, n, m in pre if n in runs]
    if not sc and not pre:
        return print("skip training_curves")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(fs.FULL, 60 * MM), layout="constrained")
    rows = []
    for i, r in enumerate(sc):
        for h in r["history"]:
            rows.append(dict(model="Scratch CNN", seed=i, epoch=h["epoch"], curve="Training (augmented)",
                             error_pct=err(h["train_acc"])))
            rows.append(dict(model="Scratch CNN", seed=i, epoch=h["epoch"], curve="Validation",
                             error_pct=err(h["val_acc"])))
    d = pd.DataFrame(rows)
    if not d.empty:
        sns.lineplot(data=d, x="epoch", y="error_pct", hue="curve", style="curve",
                     hue_order=["Training (augmented)", "Validation"], style_order=["Training (augmented)", "Validation"],
                     palette=["#8C8C8C", fs.VERMILLION], dashes=[(3.5, 1.6), ""], errorbar=("pi", 100),
                     err_kws=dict(alpha=0.2, lw=0), lw=1.2, ax=a1)
        a1.set_yscale("log")
        log_ticks(a1.yaxis, [1, 2, 5, 10, 20, 50])
        a1.set_ylim(1.2, 50)
        a1.set_xlim(0, d.epoch.max())
        a1.grid(axis="y", color=RULE, lw=0.5)
        a1.set(xlabel="Epoch", ylabel="Error (%)")
        a1.legend(title=None, loc="upper right")
        fs.panel_letter(a1, "a")
    colors = [fs.BLUE, fs.VERMILLION, fs.GREEN, fs.PURPLE, fs.BLACK]
    for i, (label, r, mk) in enumerate(pre):
        ep = [h["epoch"] for h in r["history"]]
        e = [err(h["val_acc"]) for h in r["history"]]
        a2.plot(ep, e, color=colors[i], marker=mk, ms=3.3, mec="white", mew=0.5, lw=1.1, label=label)
        rows += [dict(model=label, seed=0, epoch=a, curve="Validation", error_pct=b) for a, b in zip(ep, e)]
    if pre:
        n_ep = max(len(r["history"]) for _, r, _ in pre)
        a2.axvspan(-0.5, 0.5, color="#F1F1F1", lw=0, zorder=0)
        a2.set_xlim(-0.5, n_ep - 0.5)
        a2.set_xticks(range(n_ep))
        a2.set_ylim(0, None)
        a2.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.1f}"))
        a2.grid(axis="y", color=RULE, lw=0.5)
        a2.set(xlabel="Epoch", ylabel="Validation error (%)")
        a2.legend(loc="upper right", ncol=2)
        fs.panel_letter(a2, "b")
    caption = ("Training curves on Dogs vs. Cats. (a) Scratch CNN, 50 epochs with a one-cycle LR schedule: training error "
               "is measured on the augmented training images, validation error on the 4,999 clean validation images; "
               "line: mean over 3 seeds, band: range across seeds; log axis. (b) Validation error of the pretrained "
               "models fine-tuned with linear probing then fine-tuning (seed 0): epoch 0 (shaded) trains only the new "
               "linear head; unfreezing the backbone at epoch 1 temporarily raises the error before it recovers.")
    alt = ("Two panels. Left: scratch CNN training and validation error falling from about 35% to 1.7% and 2.4% over 50 "
           "epochs. Right: validation error per epoch for five pretrained models, lowest after the head-only epoch 0, "
           "rising at epoch 1 and recovering to 0.2-0.6% by epoch 9.")
    finish(fig, out, "training_curves", caption, alt, pd.DataFrame(rows))


def fig_cifar_g(runs, out):
    spec = [("ResNet-18 (scratch)", "cifar_r18", "No pretraining"),
            ("DINOv2-B (probe)", "cifar_dinov2_b_probe", "Linear probe"),
            ("ConvNeXt-T (FT)", "cifar_convnext_tiny", "Fine-tuned"),
            ("DINOv2-S (FT)", "cifar_dinov2_s", "Fine-tuned")]
    rows, data = [], []
    for label, base, strat in spec:
        s = stats(100 * r["test_acc"] for r in seeds(runs, base))
        if s:
            rows.append((label, s, strat))
            data += [dict(model=label, strategy=strat, seed=i, test_acc_pct=v) for i, v in enumerate(s["values"])]
    if not rows:
        return print("skip cifar_g")
    fig, ax = plt.subplots(figsize=(fs.MID, 42 * MM), layout="constrained")
    ys = row_axes(ax, [r[0] for r in rows])
    for y, (label, s, strat) in zip(ys, rows):
        st_ = STRATEGY[strat]
        range_bar(ax, s, y, st_["color"])
        seed_marks(ax, s["values"], y, st_["color"], st_["marker"])
        mean_mark(ax, s["mean"], y, st_["color"], st_["marker"])
        value_text(ax, s["hi"], y, f"{s['mean']:.2f}")
    lo = min(r[1]["lo"] for r in rows)
    ax.set_xlim(np.floor(lo * 2) / 2 - 0.1, 99.7)
    ax.xaxis.set_major_formatter(pct0)
    ax.set_xlabel("Test accuracy (%)")
    strategy_legend(ax, [k for k in ["Fine-tuned", "Linear probe", "No pretraining"] if any(r[2] == k for r in rows)])
    caption = ("CIFAR-10 test accuracy (official 10,000-image test set, evaluated once after the last epoch). ResNet-18 "
               "trained from scratch for 200 epochs on 32 × 32 images versus pretrained backbones fine-tuned (FT) or "
               "linearly probed for 10 epochs on images upsampled to 224 × 224. Colours as in the Dogs vs. Cats model "
               "comparison. Large marker: mean; small open markers: seeds and pale bar: seed range (n = 3 for "
               "ResNet-18, n = 1 otherwise). Numbers are test accuracy (%).")
    alt = "Dot plot of CIFAR-10 test accuracy: " + ", ".join(f"{r[0]} {r[1]['mean']:.2f}%" for r in rows) + "."
    finish(fig, out, "cifar_g", caption, alt, pd.DataFrame(data))


METHODS = [("Cross-entropy", "ce", fs.BLACK, "s"), ("Focal loss", "focal", fs.PURPLE, "D"),
           ("Re-sampling", "resample", fs.VERMILLION, "v"), ("Class-balanced + DRW", "cbdrw", fs.GREEN, "^"),
           ("Logit adjustment", "la", fs.BLUE, "o")]


def fig_cifar_h(runs, out):
    panels, data = [], []
    for ratio in (100, 10):
        rows = []
        for label, tag, col, mk in METHODS:
            s = stats(100 * r["test_acc"] for r in seeds(runs, f"cifar_r18_lt{ratio}_{tag}"))
            if s:
                rows.append((label, s, col, mk))
                data += [dict(ratio=ratio, method=label, seed=i, test_acc_pct=v) for i, v in enumerate(s["values"])]
        if rows:
            panels.append((ratio, rows))
    if not panels:
        return print("skip cifar_h")
    fig, axes = plt.subplots(1, len(panels), figsize=(fs.FULL, 46 * MM), layout="constrained", sharey=True,
                             squeeze=False)
    for k, (ax, (ratio, rows)) in enumerate(zip(axes[0], panels)):
        ys = row_axes(ax, [r[0] for r in rows])
        base = next(r[1]["mean"] for r in rows if r[0] == "Cross-entropy")
        ax.axvline(base, color=MUTED, lw=0.6, ls=(0, (2, 2)), zorder=1)
        for y, (label, s, col, mk) in zip(ys, rows):
            range_bar(ax, s, y, col)
            seed_marks(ax, s["values"], y, col, mk)
            mean_mark(ax, s["mean"], y, col, mk)
            value_text(ax, s["hi"], y, f"{s['mean']:.1f}")
        lo = min(min(r[1]["values"]) for r in rows)
        hi = max(max(r[1]["values"]) for r in rows)
        span = hi - lo
        ax.set_xlim(lo - 0.08 * span, hi + 0.2 * span)
        ax.xaxis.set_major_formatter(pct0)
        ax.set_xlabel("Test accuracy (%)")
        fs.panel_letter(ax, "ab"[k], f"imbalance ratio {ratio}")
    caption = ("Long-tailed CIFAR-10 with an exponential class profile (largest/smallest class = 100 in (a), 10 in (b)); "
               "ResNet-18 trained from scratch for 200 epochs, tested on the balanced 10,000-image test set. Each panel "
               "has its own x range. Dotted line: plain cross-entropy. DRW: deferred re-weighting. Large marker: mean; "
               "small open markers: seeds and pale bar: seed range (n = 3 for cross-entropy, class-balanced + DRW and "
               "logit adjustment; n = 1 for focal loss and re-sampling). Numbers are test accuracy (%).")
    alt = "Two dot plots of mean test accuracy per imbalance method. " + " ".join(
        f"Ratio {ratio}: " + ", ".join(f"{r[0]} {r[1]['mean']:.1f}%" for r in rows) + "." for ratio, rows in panels)
    finish(fig, out, "cifar_h", caption, alt, pd.DataFrame(data))


def fig_cifar_h_groups(runs, out, ratio=100):
    groups = [("head", "Head"), ("medium", "Medium"), ("tail", "Tail")]
    data = []
    for label, tag, col, mk in METHODS:
        for i, r in enumerate(x for x in seeds(runs, f"cifar_r18_lt{ratio}_{tag}") if x.get("test_group_acc")):
            for g, gl in groups:
                data.append(dict(method=label, seed=i, group=gl, test_acc_pct=100 * r["test_group_acc"][g]))
    df = pd.DataFrame(data)
    if df.empty:
        return print("skip cifar_h_groups")
    fig, ax = plt.subplots(figsize=(fs.MID, 70 * MM), layout="constrained")
    dodge = np.linspace(-0.12, 0.12, len(METHODS))
    ends = []
    for k, (label, tag, col, mk) in enumerate(METHODS):
        d = df[df.method == label]
        if d.empty:
            continue
        x = np.arange(3) + dodge[k]
        g = d.groupby("group", sort=False).test_acc_pct
        m, lo, hi = g.mean().values, g.min().values, g.max().values
        ax.plot(x, m, color=col, lw=1.1, zorder=3, ls="-" if tag != "ce" else (0, (3.5, 1.5)))
        if d.seed.nunique() > 1:
            ax.vlines(x, lo, hi, color=col, lw=2.2, alpha=0.28, zorder=2)
            for j, (gl_, gg) in enumerate(d.groupby("group", sort=False)):
                ax.plot(np.full(len(gg), x[j]), gg.test_acc_pct, ls="none", marker=mk, ms=2.4, mfc="white",
                        mec=col, mew=0.6, zorder=3)
        ax.plot(x, m, ls="none", marker=mk, ms=4.8, color=col, mec="white", mew=0.6, zorder=4)
        ends.append((m[-1], label, col, x[-1]))
    ly = spread_labels([e[0] for e in ends], 3.3)
    for (y0, label, col, x0), y1 in zip(ends, ly):
        ax.annotate(f"{label}  {y0:.1f}", (x0, y0), xytext=(2.32, y1), textcoords="data", ha="left", va="center",
                    fontsize=6.5, color=INK, arrowprops=dict(arrowstyle="-", color=col, lw=0.6, shrinkA=0, shrinkB=3))
    ax.set_xticks(range(3), [g[1] for g in groups])
    ax.set_xlim(-0.3, 3.35)
    ax.spines["bottom"].set_bounds(-0.3, 2.3)
    ax.set_ylim(35, 100)
    ax.yaxis.set_major_formatter(pct0)
    ax.grid(axis="y", color=RULE, lw=0.5)
    ax.set(xlabel="Class group", ylabel="Test accuracy (%)")
    caption = (f"Where the gains come from on long-tailed CIFAR-10 (imbalance ratio {ratio}, ResNet-18 from scratch): "
               "test accuracy averaged over the 3 largest classes (head, 4,500 to 1,617 training images), the 4 middle "
               "classes (medium, 969 to 208) and the 3 smallest classes (tail, 125 to 45). Large markers: mean; small "
               "open markers and pale bars: seeds and their range (n = 3 for cross-entropy, class-balanced + DRW and "
               "logit adjustment; n = 1 for focal loss and re-sampling). Lines only connect the groups of one method. "
               "Numbers are tail accuracy (%).")
    alt = ("Slope chart of head, medium and tail class accuracy for five methods. All methods reach 94-95% on head "
           "classes; on tail classes logit adjustment reaches about 65%, class-balanced weighting 57%, cross-entropy "
           "52%, focal loss 50% and re-sampling 41%.")
    finish(fig, out, "cifar_h_groups", caption, alt, df)


FIGS = {"datasize_curve": fig_datasize, "model_ladder": fig_model_ladder, "aug_ablation": fig_aug,
        "lr_sweep": fig_lr_sweep, "training_curves": fig_training_curves, "cifar_g": fig_cifar_g,
        "cifar_h": fig_cifar_h, "cifar_h_groups": fig_cifar_h_groups}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="report/figures")
    ap.add_argument("--only", nargs="*", choices=list(FIGS), help="draw only these figures")
    args = ap.parse_args()
    runs = load_runs(args.runs)
    print(f"{len(runs)} runs with metrics.json in {args.runs}")
    for name in args.only or FIGS:
        FIGS[name](runs, args.out)
    if not args.only:
        with open(Path(args.out) / "captions.md", "w", encoding="utf-8") as fh:
            fh.write("# Figure captions and alt text (draft)\n\nGenerated by scripts/make_plots.py. The figures carry "
                     "no titles or explanations, so each caption must explain the encoding. Exact plotted values are "
                     "in data/<figure>.csv.\n")
            for name, cap, alt in CAPTIONS:
                fh.write(f"\n## {name}\n\n**Caption.** {cap}\n\n**Alt text.** {alt}\n")


if __name__ == "__main__":
    main()
