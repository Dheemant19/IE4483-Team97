"""Download every pretrained backbone and CIFAR-10 once (PLAN.md section 7).

Set HF_HOME before running so the cache lands in the project folder:

    export HF_HOME=~/ie4483/hf_cache
    python scripts/prefetch_weights.py --cifar-root ~/ie4483/cifar
"""
import argparse
import os

import timm
import torchvision

MODELS = [
    "resnet18.a1_in1k",
    "resnet50.a1_in1k",
    "vgg16_bn.tv_in1k",
    "tf_efficientnetv2_s.in21k_ft_in1k",
    "convnext_tiny.fb_in22k_ft_in1k",
    "vit_small_patch14_dinov2.lvd142m",
    "vit_base_patch14_dinov2.lvd142m",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cifar-root", default=os.path.expanduser("~/ie4483/cifar"))
    args = ap.parse_args()

    print("HF_HOME =", os.environ.get("HF_HOME", "(not set, using ~/.cache/huggingface)"))
    for name in MODELS:
        m = timm.create_model(name, pretrained=True)
        print(f"ok {name}  {sum(p.numel() for p in m.parameters()) / 1e6:.1f}M params", flush=True)
        del m

    for train in (True, False):
        torchvision.datasets.CIFAR10(args.cifar_root, train=train, download=True)
    print("ok CIFAR-10 ->", args.cifar_root)


if __name__ == "__main__":
    main()
