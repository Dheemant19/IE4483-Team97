"""Part (b): torchinfo layer tables of the report's models (random weights, CPU).

    python scripts/model_summary.py    # -> report/model_summaries/<model>.txt, stages.csv
"""
import argparse
import csv
import sys
from pathlib import Path

import torch
from torchinfo import summary

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from models import build_model  # noqa: E402

MODELS = [
    # (file name, model name, input size, freeze backbone like the final linear probe, description)
    ("dinov2_b_probe", "dinov2_b", 224, True, "Final model: DINOv2-B (ViT-B/14) frozen backbone + linear head"),
    ("scratch_cnn", "scratch_cnn", 128, False, "Scratch CNN (5 conv blocks), trained from random init"),
    ("resnet50", "resnet50", 224, False, "ResNet-50 fine-tuned (baseline)"),
]


def stage_rows(name, model, size, is_vit):
    """A few hand-picked stages with their output shapes, via forward hooks."""
    stages = []
    if is_vit:
        picks = [("patch_embed", model.patch_embed), ("block 1", model.blocks[0]),
                 (f"block {len(model.blocks)}", model.blocks[-1]), ("norm", model.norm), ("head (linear)", model.head)]
    elif name == "scratch_cnn":
        pools = [m for m in model.features if isinstance(m, torch.nn.MaxPool2d)]
        picks = [(f"conv block {i + 1} (after max-pool)", m) for i, m in enumerate(pools)]
        picks += [("global average pool", model.pool), ("head (dropout + linear)", model.head)]
    else:
        picks = [("stem", model.maxpool), ("layer1", model.layer1), ("layer2", model.layer2), ("layer3", model.layer3),
                 ("layer4", model.layer4), ("global pool", model.global_pool), ("fc", model.fc)]
    shapes = {}
    hooks = [m.register_forward_hook(lambda mod, inp, out, k=k: shapes.__setitem__(k, tuple(out.shape)))
             for k, m in picks]
    with torch.no_grad():
        model(torch.zeros(1, 3, size, size))
    for h in hooks:
        h.remove()
    stages.append([name, "input", str((1, 3, size, size))])
    stages += [[name, k, str(shapes[k])] for k, _ in picks]
    return stages


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="report/model_summaries")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    all_stages = []
    for fname, mname, size, freeze, desc in MODELS:
        model, cfg, is_vit = build_model(mname, pretrained=False, img_size=size)
        model.eval()
        if freeze:
            head = {id(p) for p in model.get_classifier().parameters()}
            for p in model.parameters():
                p.requires_grad_(id(p) in head)
        s = summary(model, input_size=(1, 3, size, size), depth=3, verbose=0,
                    col_names=("input_size", "output_size", "num_params", "trainable"), row_settings=("var_names",))
        total = sum(p.numel() for p in model.parameters())
        train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        header = (f"{desc}\nInput 1x3x{size}x{size}; output 1x2 logits (cat=0, dog=1) -> softmax.\n"
                  f"Parameters: {total:,} total, {train:,} trainable.\n\n")
        (out / f"{fname}.txt").write_text(header + str(s), encoding="utf-8")
        all_stages += stage_rows(fname, model, size, is_vit)
        print(f"{fname}: {total:,} params ({train:,} trainable) -> {out / (fname + '.txt')}")
    with open(out / "stages.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "stage", "output_shape"])
        w.writerows(all_stages)
    for r in all_stages:
        print(" ", *r)


if __name__ == "__main__":
    main()
