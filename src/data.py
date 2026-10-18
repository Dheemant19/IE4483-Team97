"""Datasets and transforms (PLAN.md section 5)."""
import json
import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset, Subset
from torchvision import datasets
from torchvision import transforms as T

INTERP = {
    "bicubic": T.InterpolationMode.BICUBIC,
    "bilinear": T.InterpolationMode.BILINEAR,
    "nearest": T.InterpolationMode.NEAREST,
}


def eval_transform(cfg):
    size = cfg["input_size"][-1]
    interp = INTERP.get(cfg.get("interpolation", "bilinear"), T.InterpolationMode.BILINEAR)
    return T.Compose([
        T.Resize(int(round(size / cfg.get("crop_pct", 0.875))), interpolation=interp),
        T.CenterCrop(size),
        T.ToTensor(),
        T.Normalize(cfg["mean"], cfg["std"]),
    ])


def train_transform(cfg, aug="default", scale_min=0.5):
    """aug levels for the augmentation ablation (PLAN.md 8.4):
    none -> eval-style resize + centre crop
    basic -> RandomResizedCrop + flip
    default -> basic + rotation + colour jitter + random erasing
    trivial -> default + TrivialAugmentWide
    """
    if aug == "none":
        return eval_transform(cfg)
    size = cfg["input_size"][-1]
    interp = INTERP.get(cfg.get("interpolation", "bilinear"), T.InterpolationMode.BILINEAR)
    tf = [T.RandomResizedCrop(size, scale=(scale_min, 1.0), ratio=(3 / 4, 4 / 3), interpolation=interp),
          T.RandomHorizontalFlip()]
    if aug in ("default", "trivial"):
        tf += [T.RandomRotation(15, interpolation=T.InterpolationMode.BILINEAR),
               T.ColorJitter(0.2, 0.2, 0.2, 0.02)]
    if aug == "trivial":
        tf += [T.TrivialAugmentWide(interpolation=T.InterpolationMode.BILINEAR)]
    tf += [T.ToTensor(), T.Normalize(cfg["mean"], cfg["std"])]
    if aug in ("default", "trivial"):
        tf += [T.RandomErasing(p=0.25)]
    return T.Compose(tf)


class TestImages(Dataset):
    """Flat folder of 1.jpg ... 500.jpg, sorted numerically."""

    def __init__(self, root, transform):
        self.paths = sorted(Path(root).glob("*.jpg"), key=lambda p: int(p.stem))
        self.ids = [int(p.stem) for p in self.paths]
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as im:
            return self.transform(im.convert("RGB")), self.ids[i]


# val/cat/cat.12408.jpg is an exact copy of train/cat/cat.2339.jpg (train-to-val leak), so it is left out
EXCLUDE = ("val/cat/cat.12408.jpg",)


def _valid(path):
    path = str(path).replace("\\", "/")
    return path.lower().endswith(".jpg") and not path.endswith(EXCLUDE)


def _loader(path):
    with Image.open(path) as im:
        return im.convert("RGB")


def stratified_subset(ds, n, seed=1234):
    """n images split evenly across classes; fixed seed so every run sees the same subset."""
    rng = random.Random(seed)
    by_class = {}
    for i, y in enumerate(ds.targets):
        by_class.setdefault(y, []).append(i)
    idx = []
    for y, items in sorted(by_class.items()):
        rng.shuffle(items)
        idx += items[: n // len(by_class)]
    return Subset(ds, sorted(idx))


def build_datasets(root, train_tf, eval_tf, n_train=None, n_val=None):
    root = Path(root)
    train = datasets.ImageFolder(root / "train", train_tf, loader=_loader, is_valid_file=_valid)
    val = datasets.ImageFolder(root / "val", eval_tf, loader=_loader, is_valid_file=_valid)
    assert train.class_to_idx == {"cat": 0, "dog": 1}, train.class_to_idx
    if n_train:
        train = stratified_subset(train, n_train)
    if n_val:
        val = stratified_subset(val, n_val)
    test = TestImages(root / "test", eval_tf)
    return train, val, test


def train_stats(root, size=128, n=2000, cache_name="stats_scratch.json"):
    """Mean/std of the training set for the scratch CNN, cached next to the data."""
    cache = Path(root) / cache_name
    if cache.exists():
        return json.loads(cache.read_text())
    ds = datasets.ImageFolder(Path(root) / "train",
                              T.Compose([T.Resize(size), T.CenterCrop(size), T.ToTensor()]),
                              loader=_loader)
    idx = random.Random(0).sample(range(len(ds)), min(n, len(ds)))
    x = torch.stack([ds[i][0] for i in idx])
    stats = {"mean": x.mean((0, 2, 3)).tolist(), "std": x.std((0, 2, 3)).tolist()}
    cache.write_text(json.dumps(stats))
    return stats
