"""Part (b): per-layer architecture tables for the final model and the scratch CNN, as CSV.

Builds both models with the project's own code (src/models.py), runs one dummy batch through each and
writes, for every layer, its input shape, output shape and parameter count:

    report/tables/arch_dinov2_b_probe.csv     final model: DINOv2-B backbone frozen + linear head
    report/tables/arch_scratch_cnn.csv        scratch CNN

    python scripts/arch_table.py                      # dummy batch 1x3x224x224 for both models (as requested)
    python scripts/arch_table.py --scratch-size 128   # the scratch CNN at its training resolution (128 px)

Uses torchinfo if it is installed, otherwise forward hooks (same columns either way; --no-torchinfo forces
hooks). Weights are random (pretrained=False): layer shapes and parameter counts do not depend on them.
Columns: order, layer (module path), type, depth, input_shape, output_shape, params (this layer incl.
sub-layers), trainable_params, leaf (True for layers without sub-layers). Shapes include the batch dim (1).
"""
import argparse
import csv
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from models import build_model  # noqa: E402

COLS = ["order", "layer", "type", "depth", "input_shape", "output_shape", "params", "trainable_params", "leaf"]


def shape_of(x):
    if isinstance(x, torch.Tensor):
        return list(x.shape)
    if isinstance(x, (list, tuple)):
        for v in x:
            s = shape_of(v)
            if s is not None:
                return s
    if isinstance(x, dict):
        for v in x.values():
            s = shape_of(v)
            if s is not None:
                return s
    return None


def rows_with_hooks(model, x):
    names = {m: n for n, m in model.named_modules()}
    seen, rows, hooks = set(), [], []

    def hook(m, inp, out):
        if m in seen:                      # count a module once even if it is called repeatedly
            return
        seen.add(m)
        n = names[m] or type(m).__name__
        params = sum(p.numel() for p in m.parameters())
        rows.append(dict(layer=n, type=type(m).__name__, depth=0 if not names[m] else names[m].count(".") + 1,
                         input_shape=shape_of(inp), output_shape=shape_of(out), params=params,
                         trainable_params=sum(p.numel() for p in m.parameters() if p.requires_grad),
                         leaf=len(list(m.children())) == 0))

    for m in model.modules():
        hooks.append(m.register_forward_hook(hook))
    with torch.no_grad():
        model(x)
    for h in hooks:
        h.remove()
    # forward hooks fire child-first; order the table like the module tree (parents before children)
    order = {n: i for i, (n, _) in enumerate(model.named_modules())}
    rows.sort(key=lambda r: order.get(r["layer"] if r["depth"] else "", 0))
    return rows


def rows_with_torchinfo(model, x):
    from torchinfo import summary
    stats = summary(model, input_data=x, depth=100, verbose=0,
                    col_names=("input_size", "output_size", "num_params", "trainable"))
    names = {m: n for n, m in model.named_modules()}
    rows = []
    for li in stats.summary_list:
        mod = getattr(li, "module", None)
        name = names.get(mod) if mod is not None else None
        if name is None:
            name = getattr(li, "var_name", "") or li.class_name
        params = sum(p.numel() for p in mod.parameters()) if mod is not None else getattr(li, "num_params", 0)
        train = (sum(p.numel() for p in mod.parameters() if p.requires_grad) if mod is not None
                 else getattr(li, "trainable_params", 0))
        rows.append(dict(layer=name or li.class_name, type=li.class_name, depth=li.depth,
                         input_shape=li.input_size, output_shape=li.output_size, params=params,
                         trainable_params=train, leaf=bool(getattr(li, "is_leaf_layer", False))))
    return rows


def write(rows, path, title):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(COLS)
        for i, r in enumerate(rows, start=1):
            w.writerow([i, r["layer"], r["type"], r["depth"],
                        "x".join(map(str, r["input_shape"])) if r["input_shape"] else "",
                        "x".join(map(str, r["output_shape"])) if r["output_shape"] else "",
                        r["params"], r["trainable_params"], r["leaf"]])
    print(f"{title}: {len(rows)} layers -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="report/tables")
    ap.add_argument("--size", type=int, default=224, help="dummy input size for the DINOv2-B probe")
    ap.add_argument("--scratch-size", type=int, default=224,
                    help="dummy input size for the scratch CNN (it was trained at 128)")
    ap.add_argument("--no-torchinfo", action="store_true", help="use forward hooks even if torchinfo is installed")
    args = ap.parse_args()
    try:
        if args.no_torchinfo:
            raise ImportError
        import torchinfo  # noqa: F401
        method = "torchinfo"
    except ImportError:
        method = "forward hooks"
    print(f"per-layer shapes via {method}")
    out = Path(args.out)

    for fname, mname, size, freeze in (("arch_dinov2_b_probe", "dinov2_b", args.size, True),
                                       ("arch_scratch_cnn", "scratch_cnn", args.scratch_size, False)):
        model, _, _ = build_model(mname, pretrained=False, img_size=size if mname != "scratch_cnn" else None)
        if freeze:                                  # final model: backbone frozen, only the linear head trains
            head = {id(p) for p in model.get_classifier().parameters()}
            for p in model.parameters():
                p.requires_grad_(id(p) in head)
        model.eval()
        x = torch.zeros(1, 3, size, size)
        rows = rows_with_torchinfo(model, x) if method == "torchinfo" else rows_with_hooks(model, x)
        total = sum(p.numel() for p in model.parameters())
        train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        write(rows, out / f"{fname}.csv", f"{fname} (input 1x3x{size}x{size}, {total:,} params, {train:,} trainable)")


if __name__ == "__main__":
    main()
