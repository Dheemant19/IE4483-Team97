"""Part (a): what the network actually sees - a few training images and random draws of the training
augmentation, next to the deterministic validation/test view. Uses the exact transforms from src/data.py.

    python scripts/make_aug_examples.py              # -> report/figures/aug_examples.png|pdf
Needs torch + torchvision (CPU is enough). On the laptop:
    uv run --no-project --python 3.12 --with torch --with torchvision --with matplotlib --with pillow \
        --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple \
        python scripts/make_aug_examples.py
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))
from data import eval_transform, train_transform  # noqa: E402

import figstyle as fs  # noqa: E402
from figstyle import INK, plt  # noqa: E402

# ImageNet statistics at 224 px, as used by the pretrained models (DINOv2 / ResNet / ConvNeXt timm configs)
CFG = {"input_size": (3, 224, 224), "mean": (0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225),
       "crop_pct": 1.0, "interpolation": "bicubic"}


def to_img(t):
    """Undo the normalisation for display (clipped to the displayable range)."""
    m = torch.tensor(CFG["mean"])[:, None, None]
    s = torch.tensor(CFG["std"])[:, None, None]
    return (t * s + m).clamp(0, 1).permute(1, 2, 0).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data_288")
    ap.add_argument("--images", nargs="*", default=["train/cat/cat.10.jpg", "train/dog/dog.10.jpg"])
    ap.add_argument("--n", type=int, default=5, help="random training draws per image")
    ap.add_argument("--out", default="report/figures")
    args = ap.parse_args()
    torch.manual_seed(0)
    tr, ev = train_transform(CFG, "default"), eval_transform(CFG)
    ncol = 2 + args.n
    fig, axes = plt.subplots(len(args.images), ncol, figsize=(fs.FULL, len(args.images) * fs.FULL / ncol * 1.0 + 6 * fs.MM),
                             squeeze=False, layout="constrained")
    heads = ["original", "val / test input"] + [f"training draw {i + 1}" for i in range(args.n)]
    for i, rel in enumerate(args.images):
        im = Image.open(Path(args.data) / rel).convert("RGB")
        panels = [np.asarray(im), to_img(ev(im))] + [to_img(tr(im)) for _ in range(args.n)]
        for j, p in enumerate(panels):
            ax = axes[i, j]
            ax.imshow(p, aspect="equal")
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(j > 1)
                sp.set_color("#BBBBBB")
                sp.set_linewidth(0.5)
            if i == 0:
                ax.set_title(heads[j], fontsize=6.5, color=INK, pad=3)
    fs.save(fig, args.out, "aug_examples")
    cap = ("Training-time augmentation for the pretrained models (default setting). Column 1: the stored image "
           "(shorter side 288 px); column 2: the deterministic validation/test input (resize + 224 × 224 centre "
           "crop); columns 3-7: independent random draws of the training transform - random resized crop (50-100 % "
           "of the area, aspect 3/4-4/3) to 224 × 224, horizontal flip (p = 0.5), rotation (±15°), colour jitter "
           "(brightness/contrast/saturation 0.2, hue 0.02) and random erasing (p = 0.25; the erased box takes the ImageNet mean colour); rotation leaves black corners. The "
           "scratch CNN uses the same transform at 128 × 128 with crops of 60-100 % of the area.")
    alt = ("Two rows (a cat and a dog) of seven images: the original photo, its centre crop and five randomly "
           "cropped, flipped, rotated and colour-shifted versions, some with a grey erased rectangle.")
    with open(Path(args.out) / "captions_aug.md", "w", encoding="utf-8") as fh:
        fh.write(f"# Augmentation figure caption (draft)\n\n## aug_examples\n\n**Caption.** {cap}\n\n**Alt text.** {alt}\n")


if __name__ == "__main__":
    main()
