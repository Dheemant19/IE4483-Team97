"""Report tables from runs/*/metrics.json -> report/tables/*.csv and RESULTS.md.

Dogs vs. Cats uses last-epoch validation accuracy and CIFAR-10 uses test accuracy, as mean ± SD over seeds.

    python scripts/make_tables.py
"""
import argparse
import csv
import json
import re
import statistics as st
from pathlib import Path


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
    if not hits and base in runs:
        return [runs[base]]
    return [r for _, r in hits]


def ms(values, scale=100, nd=2):
    v = [scale * x for x in values if x is not None]
    if not v:
        return "", 0
    if len(v) == 1:
        return f"{v[0]:.{nd}f}", 1
    return f"{st.mean(v):.{nd}f} ± {st.stdev(v):.{nd}f}", len(v)


class Table:
    def __init__(self, name, title, cols, note=""):
        self.name, self.title, self.cols, self.note, self.rows = name, title, cols, note, []

    def add(self, *row):
        self.rows.append([str(x) for x in row])

    def md(self):
        out = [f"### {self.title}", ""]
        if self.note:
            out += [self.note, ""]
        out += ["| " + " | ".join(self.cols) + " |", "|" + "---|" * len(self.cols)]
        out += ["| " + " | ".join(r) + " |" for r in self.rows]
        return "\n".join(out) + "\n"

    def write_csv(self, out):
        with open(out / f"{self.name}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(self.cols)
            w.writerows(self.rows)


def dataset_table(summary_path):
    t = Table("t0_dataset", "Dataset (Dogs vs. Cats)", ["Split", "Images", "Cats", "Dogs", "Used for"],
              "One validation image (val/cat/cat.12408.jpg) is an exact copy of train/cat/cat.2339.jpg and was "
              "removed, so validation has 4,999 images. Test images are unlabelled.")
    try:
        g = json.loads(Path(summary_path).read_text())["groups"]
        c = {k: v["count"] for k, v in g.items()}
        t.add("train", c["train/cat"] + c["train/dog"], c["train/cat"], c["train/dog"], "training only")
        t.add("val", c["val/cat"] + c["val/dog"] - 1, c["val/cat"] - 1, c["val/dog"],
              "model selection, all reported Dogs-vs-Cats accuracies")
        t.add("test", c["test"], "-", "-", "submission.csv only")
    except (OSError, KeyError):
        t.add("train", 20000, 10000, 10000, "training only")
        t.add("val", 4999, 2499, 2500, "model selection, all reported Dogs-vs-Cats accuracies")
        t.add("test", 500, "-", "-", "submission.csv only")
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="report/tables")
    ap.add_argument("--dataset-summary", default="scripts/dataset_summary.json")
    args = ap.parse_args()
    R = load_runs(args.runs)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tables = [dataset_table(args.dataset_summary)]

    # T1: every Dogs-vs-Cats model
    t = Table("t1_dogs_vs_cats_models", "Dogs vs. Cats: all models (validation accuracy, %)",
              ["Model", "Training", "Params (M)", "Trainable", "n", "Val acc", "Val acc + flip TTA", "Train min / run"],
              "Last-epoch validation accuracy on 4,999 images, mean ± SD over n seeds.")
    spec = [("HOG + linear SVM", "hog_svm_linear", "classical, no pretraining", "-"),
            ("HOG + RBF SVM", "hog_svm_rbf", "classical, no pretraining", "-"),
            ("Scratch CNN", "scratch_cnn", "from scratch", "all"),
            ("ResNet-50", "resnet50", "fine-tuned (LP-FT)", "all"), ("ResNet-50", "resnet50_probe", "linear probe", "head"),
            ("EfficientNetV2-S", "effnetv2_s", "fine-tuned (LP-FT)", "all"),
            ("ConvNeXt-T", "convnext_tiny", "fine-tuned (LP-FT)", "all"),
            ("ConvNeXt-T", "convnext_tiny_probe", "linear probe", "head"),
            ("DINOv2-S", "dinov2_s", "fine-tuned (LP-FT)", "all"), ("DINOv2-S", "dinov2_s_probe", "linear probe", "head"),
            ("DINOv2-B", "dinov2_b", "fine-tuned (LP-FT)", "all"), ("DINOv2-B", "dinov2_b_probe", "linear probe", "head")]
    for label, base, how, train in spec:
        rs = seeds(R, base)
        if not rs:
            continue
        acc, n = ms(r["val_acc"] for r in rs)
        tta, _ = ms(r.get("val_acc_tta") for r in rs)
        tmin = st.mean(r.get("total_train_min", 0) for r in rs)
        t.add(label, how, rs[0].get("params_m", "-"), train, n, acc, tta or "-", f"{tmin:.1f}" if tmin else "-")
    ens = R.get("ens_dinov2_b_probe")
    if ens:
        t.add("**Final: DINOv2-B probe, 3-seed ensemble**", "average of the 3 seeds' probabilities", 85.73, "head", 1,
              f"{100 * ens['val_acc']:.2f}", f"{100 * ens['val_acc_tta']:.2f}", "-")
    tables.append(t)

    # T2: linear probe vs fine-tuning
    t = Table("t2_probe_vs_finetune", "Linear probe vs. full fine-tuning (validation accuracy, %)",
              ["Backbone", "Pretraining data", "Linear probe", "Fine-tuned (LP-FT)", "Difference (probe − FT)"],
              "Mean ± SD over 3 seeds each.")
    for label, pre, b in [("ResNet-50", "ImageNet-1k", "resnet50"), ("ConvNeXt-T", "ImageNet-22k", "convnext_tiny"),
                          ("DINOv2-S", "LVD-142M (self-supervised)", "dinov2_s"),
                          ("DINOv2-B", "LVD-142M (self-supervised)", "dinov2_b")]:
        p, f = seeds(R, b + "_probe"), seeds(R, b)
        if p and f:
            d = 100 * (st.mean(r["val_acc"] for r in p) - st.mean(r["val_acc"] for r in f))
            t.add(label, pre, ms(r["val_acc"] for r in p)[0], ms(r["val_acc"] for r in f)[0], f"{d:+.2f}")
    tables.append(t)

    # T3: training-set size
    ns = [2000, 5000, 10000, 20000]
    t = Table("t3_training_set_size", "Validation accuracy (%) vs. number of training images",
              ["Model"] + [f"{n:,}" for n in ns], "Class-balanced subsets (fixed subset seed); mean ± SD over 3 seeds.")
    for label, b in [("Scratch CNN", "scratch_cnn"), ("ResNet-50 (fine-tuned)", "resnet50"),
                     ("ConvNeXt-T (fine-tuned)", "convnext_tiny"), ("DINOv2-B (linear probe)", "dinov2_b_probe")]:
        t.add(label, *[ms(r["val_acc"] for r in seeds(R, b if n == 20000 else f"{b}_n{n}"))[0] or "-" for n in ns])
    tables.append(t)

    # T4: augmentation ablation
    t = Table("t4_augmentation", "Scratch CNN: training-time augmentation (validation accuracy, %)",
              ["Augmentation", "Epochs", "n", "Val acc", "Val acc + flip TTA"])
    for label, b, ep in [("none (resize + centre crop)", "scratch_cnn_aug-none", 50),
                         ("random resized crop + flip", "scratch_cnn_aug-basic", 50),
                         ("+ rotation ±15°, colour jitter, random erasing (default)", "scratch_cnn", 50),
                         ("default + TrivialAugmentWide", "scratch_cnn_aug-trivial", 50),
                         ("default", "scratch_cnn_100ep", 100)]:
        rs = seeds(R, b)
        if rs:
            acc, n = ms(r["val_acc"] for r in rs)
            t.add(label, ep, n, acc, ms(r["val_acc_tta"] for r in rs)[0])
    tables.append(t)

    # T5: learning-rate sweep
    t = Table("t5_lr_sweep", "ResNet-50 fine-tuning: backbone learning rate (validation accuracy, %, seed 0)",
              ["Backbone LR", "After head-only epoch", "Best epoch", "Last epoch (reported)"],
              "Head LR 1e-3, AdamW, 1 head-only epoch + 9 fine-tuning epochs (warm-up + cosine).")
    for lr in ["1e-5", "3e-5", "1e-4", "3e-4"]:
        r = R.get(f"resnet50_lr{lr}_s0")
        if r:
            t.add(lr, f"{100 * r['lp_val_acc']:.2f}", f"{100 * r['best_val_acc']:.2f} (epoch {r['best_epoch']})",
                  f"{100 * r['val_acc']:.2f}")
    tables.append(t)

    # T6: fine-tuning diagnostics
    t = Table("t6_finetune_diagnostics", "Why fine-tuning did not beat the linear probe: diagnostic runs (validation accuracy, %)",
              ["Run", "Setting", "n", "Val acc"])
    for label, b, change in [("DINOv2-B FT default", "dinov2_b", "backbone LR 5e-5, default augmentation, label smoothing 0.1"),
                             ("DINOv2-B FT gentle", "dinov2_b_gentle", "backbone LR 1e-5"),
                             ("DINOv2-B FT light aug", "dinov2_b_lightaug", "crop + flip only, no label smoothing"),
                             ("DINOv2-B linear probe", "dinov2_b_probe", "backbone frozen"),
                             ("ResNet-50 FT default", "resnet50", "backbone LR 1e-4, default augmentation"),
                             ("ResNet-50 FT light aug", "resnet50_lightaug", "crop + flip only, no label smoothing"),
                             ("ResNet-50 linear probe", "resnet50_probe", "backbone frozen")]:
        rs = seeds(R, b)
        if rs:
            acc, n = ms(r["val_acc"] for r in rs)
            t.add(label, change, n, acc)
    tables.append(t)

    # T7: CIFAR-10 (part g)
    t = Table("t7_cifar10", "CIFAR-10 (part g): test accuracy (%)",
              ["Model", "Training", "Input", "Epochs", "n", "Test acc", "Test acc + flip TTA", "Macro-F1"],
              "Official 10,000-image test set, evaluated once after the last epoch; 45,000 training images "
              "(500 per class held out for validation).")
    for label, b, how, size, ep in [("ResNet-18 (CIFAR variant)", "cifar_r18", "from scratch", "32 × 32", 200),
                                    ("DINOv2-B", "cifar_dinov2_b_probe", "linear probe", "224 × 224 (upsampled)", 10),
                                    ("ConvNeXt-T", "cifar_convnext_tiny", "fine-tuned (LP-FT)", "224 × 224 (upsampled)", 10),
                                    ("DINOv2-S", "cifar_dinov2_s", "fine-tuned (LP-FT)", "224 × 224 (upsampled)", 10)]:
        rs = seeds(R, b)
        if rs:
            acc, n = ms(r["test_acc"] for r in rs)
            t.add(label, how, size, ep, n, acc, ms(r["test_acc_tta"] for r in rs)[0],
                  ms((r["test_macro_f1"] for r in rs), scale=1, nd=4)[0])
    tables.append(t)

    # T8: long-tailed CIFAR-10 (part h)
    t = Table("t8_cifar10_longtail", "Long-tailed CIFAR-10 (part h): test accuracy (%) by imbalance method",
              ["Model", "Imbalance ratio", "Method", "n", "Test acc", "Macro-F1", "Head (3 largest)", "Medium (4)", "Tail (3 smallest)"],
              "ResNet-18 from scratch (200 epochs) and fine-tuned DINOv2-S (10 epochs), exponential class profile "
              "(4,500 down to 45 images per class at ratio 100, down to 450 at ratio 10); balanced 10,000-image test set.")
    methods = [("cross-entropy (baseline)", "ce"), ("focal loss (γ = 1)", "focal"), ("class-balanced re-sampling", "resample"),
               ("class-balanced weights + DRW", "cbdrw"), ("logit adjustment (τ = 1)", "la")]
    for model, prefix in (("ResNet-18 (scratch)", "cifar_r18"), ("DINOv2-S (fine-tuned)", "cifar_dinov2_s")):
        for ratio in (100, 10):
            for label, tag in methods:
                rs = seeds(R, f"{prefix}_lt{ratio}_{tag}")
                if not rs:
                    continue
                acc, n = ms(r["test_acc"] for r in rs)
                grp = [ms((r["test_group_acc"][g] for r in rs), nd=1)[0] for g in ("head", "medium", "tail")]
                t.add(model, ratio, label, n, acc, ms((r["test_macro_f1"] for r in rs), scale=1, nd=4)[0], *grp)
    tables.append(t)

    # T9: final submission
    t = Table("t9_final_submission", "Final model and submission", ["Item", "Value"])
    if ens:
        cm = ens["confusion_matrix_rows_true_cols_pred"]
        t.add("Model", "DINOv2-B (ViT-B/14) frozen + linear head; average of 3 seeds; flip TTA")
        t.add("Validation accuracy", f"{100 * ens['val_acc']:.2f}% ({cm[0][1] + cm[1][0]} errors / {ens['n_val']})")
        t.add("Validation confusion (rows true cat/dog, cols predicted)", f"cat: {cm[0]}, dog: {cm[1]}")
    sub = Path("submission.csv")
    if sub.exists():
        rows = list(csv.DictReader(open(sub)))
        t.add("submission.csv", f"{len(rows)} rows (id 1-500), {sum(int(r['label']) for r in rows)} predicted dogs")
    tables.append(t)

    for tb in tables:
        tb.write_csv(out)
    with open(out / "RESULTS.md", "w", encoding="utf-8") as fh:
        fh.write("# Results tables\n\nGenerated by `scripts/make_tables.py` from `runs/*/metrics.json`. "
                 "Accuracies in %, mean ± SD over n seeds. Dogs vs. Cats numbers are validation accuracy "
                 "(4,999 images; the test set has no labels); CIFAR-10 numbers are test accuracy.\n\n")
        for tb in tables:
            fh.write(tb.md() + "\n")
    print(f"wrote {len(tables)} tables to {out} (RESULTS.md + CSVs)")


if __name__ == "__main__":
    main()
