"""CIFAR-10 (part g) and imbalanced CIFAR-10 (part h), PLAN.md section 9.

The whole dataset lives on the GPU as uint8 and is augmented there (random crop with
4 px padding, horizontal flip, optional cutout), so there is no DataLoader bottleneck.

Splits: 500 images per class are held out from the official 50k train set as a
balanced validation set (fixed seed 1234), so training uses 4,500 per class. The
official 10k test set is only evaluated once, after the last epoch.

Models:
    resnet18_cifar   CIFAR-style ResNet-18 from scratch (3x3 stem, no max-pool), 32x32 input
    any train.py name (convnext_tiny, dinov2_s, dinov2_b, resnet50, ...): pretrained timm model,
                     images upsampled to 224 on the GPU; --freeze backbone for a linear probe

Imbalance (part h): --imb-type exp|step --imb-ratio R builds a long-tailed / step training set
from the 4,500-per-class pool (class 0 keeps 4,500; validation and test stay balanced).
Methods (--method):
    ce        plain cross-entropy (baseline)
    resample  class-balanced re-sampling (every class drawn equally often)
    cb        class-balanced loss weights, effective number of samples (Cui et al. 2019)
    cb_drw    the same weights, switched on only for the last 20% of epochs (DRW, Cao et al. 2019)
    la        logit-adjusted loss, logits + tau * log(prior) during training (Menon et al. 2021)
    focal     focal loss, gamma 1 (Lin et al. 2017)

Examples:
    python src/cifar.py --model resnet18_cifar --seed 0                              # part g, from scratch
    python src/cifar.py --model dinov2_s --seed 0                                    # part g, transfer
    python src/cifar.py --model resnet18_cifar --imb-ratio 100 --method la --seed 0  # part h
    python src/cifar.py --model resnet18_cifar --epochs 1 --limit 2000 --name cifar_smoke   # smoke test
Writes runs/<name>/: last.pt, tb/, metrics.json (val_acc, test_acc, per-class accuracy, macro-F1, ...).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models import ALIASES  # noqa: E402
from train import make_scheduler, param_groups, seed_everything  # noqa: E402

CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
CIFAR_MEAN, CIFAR_STD = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)


# ---------------------------------------------------------------- model

class BasicBlock(nn.Module):
    def __init__(self, c_in, c_out, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        self.shortcut = nn.Sequential()
        if stride != 1 or c_in != c_out:
            self.shortcut = nn.Sequential(nn.Conv2d(c_in, c_out, 1, stride, bias=False), nn.BatchNorm2d(c_out))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class CifarResNet18(nn.Module):
    """ResNet-18 for 32x32 inputs: 3x3 stem with stride 1 and no max-pool, stages 64-128-256-512."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        layers, c_in = [], 64
        for c_out, stride in [(64, 1), (128, 2), (256, 2), (512, 2)]:
            layers += [BasicBlock(c_in, c_out, stride), BasicBlock(c_out, c_out, 1)]
            c_in = c_out
        self.layers = nn.Sequential(*layers)
        self.fc = nn.Linear(512, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        return self.fc(torch.flatten(F.adaptive_avg_pool2d(self.layers(self.stem(x)), 1), 1))

    def get_classifier(self):
        return self.fc


def build(name, img_size):
    """Returns (model, size, mean, std, is_vit)."""
    if name == "resnet18_cifar":
        return CifarResNet18(), 32, CIFAR_MEAN, CIFAR_STD, False
    timm_name = ALIASES.get(name, name)
    is_vit = timm_name.startswith("vit_")
    kw = {"num_classes": 10}
    if is_vit:
        kw["img_size"] = img_size
    model = timm.create_model(timm_name, pretrained=True, **kw)
    head = model.get_classifier()
    std = head.weight.std().item()
    if std > 0.05:   # same narrow head fix as models.py (EfficientNet)
        nn.init.trunc_normal_(head.weight, std=0.02)
        nn.init.zeros_(head.bias)
        print(f"re-initialised head (std {std:.2f} -> 0.02)", flush=True)
    cfg = timm.data.resolve_data_config({}, model=model)
    return model, img_size, tuple(cfg["mean"]), tuple(cfg["std"]), is_vit


# ---------------------------------------------------------------- data

def load_cifar(root):
    tr = torchvision.datasets.CIFAR10(root, train=True, download=False)
    te = torchvision.datasets.CIFAR10(root, train=False, download=False)
    to_t = lambda d: torch.from_numpy(np.ascontiguousarray(d.data.transpose(0, 3, 1, 2)))   # N,3,32,32 uint8
    return to_t(tr), torch.tensor(tr.targets), to_t(te), torch.tensor(te.targets)


def class_counts(imb_type, ratio, n_max=4500, C=10):
    if ratio <= 1:
        return [n_max] * C
    if imb_type == "exp":
        return [int(n_max * ratio ** (-c / (C - 1))) for c in range(C)]
    return [n_max if c < C // 2 else int(n_max / ratio) for c in range(C)]   # step


def make_splits(y, counts, n_val=500, seed=1234):
    """Balanced val (n_val per class) + training indices with the requested per-class counts."""
    g = torch.Generator().manual_seed(seed)
    tr_idx, val_idx = [], []
    for c, n in enumerate(counts):
        idx = torch.nonzero(y == c).flatten()
        idx = idx[torch.randperm(len(idx), generator=g)]
        val_idx.append(idx[:n_val])
        tr_idx.append(idx[n_val:n_val + n])
    return torch.cat(tr_idx), torch.cat(val_idx)


def stratified_limit(y, n, seed=1234):
    g = torch.Generator().manual_seed(seed)
    out = []
    for c in range(10):
        idx = torch.nonzero(y == c).flatten()
        out.append(idx[torch.randperm(len(idx), generator=g)][: max(1, n // 10)])
    return torch.cat(out)


def augment(x_u8, gen, pad=4, cutout=0):
    """Random crop (zero padding) + horizontal flip (+ cutout) on a uint8 GPU batch -> float in [0, 1]."""
    x = x_u8.float() / 255
    B, _, H, W = x.shape
    xp = F.pad(x, (pad, pad, pad, pad))
    oy = torch.randint(0, 2 * pad + 1, (B,), device=x.device, generator=gen)
    ox = torch.randint(0, 2 * pad + 1, (B,), device=x.device, generator=gen)
    rows = (oy[:, None] + torch.arange(H, device=x.device))[:, :, None]
    cols = (ox[:, None] + torch.arange(W, device=x.device))[:, None, :]
    x = xp[torch.arange(B, device=x.device)[:, None, None], :, rows, cols].permute(0, 3, 1, 2)
    flip = torch.rand(B, device=x.device, generator=gen) < 0.5
    x = torch.where(flip[:, None, None, None], x.flip(3), x)
    if cutout:
        cy = torch.randint(0, H, (B, 1, 1), device=x.device, generator=gen)
        cx = torch.randint(0, W, (B, 1, 1), device=x.device, generator=gen)
        ys = torch.arange(H, device=x.device)[None, :, None]
        xs = torch.arange(W, device=x.device)[None, None, :]
        hole = ((ys - cy).abs() < cutout // 2 + cutout % 2) & ((xs - cx).abs() < cutout // 2 + cutout % 2)
        x = x * (~hole)[:, None].float()
    return x


def prep(x, size, mean, std, cl):
    """Float [0, 1] batch -> resized (if needed) and normalised network input."""
    if x.shape[-1] != size:
        x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
    m = torch.tensor(mean, device=x.device)[None, :, None, None]
    s = torch.tensor(std, device=x.device)[None, :, None, None]
    return ((x - m) / s).to(memory_format=cl)   # .contiguous() rejects preserve_format (ViTs)


@torch.no_grad()
def predict(model, X, size, mean, std, cl, tta=False, bs=500):
    model.eval()
    probs = []
    for i in range(0, len(X), bs):
        x = prep(X[i:i + bs].float() / 255, size, mean, std, cl)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            p = model(x).float().softmax(1)
            if tta:
                p = (p + model(torch.flip(x, dims=[3])).float().softmax(1)) / 2
        probs.append(p)
    return torch.cat(probs)


def report(probs, y, counts, imb_type, ratio):
    pred = probs.argmax(1)
    cm = torch.zeros(10, 10, dtype=torch.long)
    for t, p in zip(y.tolist(), pred.tolist()):
        cm[t, p] += 1
    per_class = (cm.diag().float() / cm.sum(1).clamp_min(1)).tolist()
    prec = cm.diag().float() / cm.sum(0).clamp_min(1)
    rec = cm.diag().float() / cm.sum(1).clamp_min(1)
    f1 = (2 * prec * rec / (prec + rec).clamp_min(1e-12)).tolist()
    out = dict(acc=(pred.cpu() == y.cpu()).float().mean().item(), macro_f1=float(np.mean(f1)),
               per_class_acc=dict(zip(CLASSES, [round(a, 4) for a in per_class])),
               confusion_matrix_rows_true_cols_pred=cm.tolist())
    if ratio > 1:
        order = sorted(range(10), key=lambda c: -counts[c])
        if imb_type == "exp":
            groups = {"head": order[:3], "medium": order[3:7], "tail": order[7:]}
        else:
            groups = {"majority": [c for c in range(10) if counts[c] == max(counts)],
                      "minority": [c for c in range(10) if counts[c] < max(counts)]}
        out["group_acc"] = {k: float(np.mean([per_class[c] for c in v])) for k, v in groups.items()}
    return out


# ---------------------------------------------------------------- training

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="resnet18_cifar")
    ap.add_argument("--cifar-root", default=os.path.expanduser("~/ie4483/cifar"))
    ap.add_argument("--out", default="runs")
    ap.add_argument("--name", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--bs", type=int, default=None)
    ap.add_argument("--opt", choices=["sgd", "adamw"], default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--head-lr", type=float, default=None)
    ap.add_argument("--wd", type=float, default=None)
    ap.add_argument("--warmup-epochs", type=float, default=None)
    ap.add_argument("--lp-epochs", type=int, default=None)
    ap.add_argument("--freeze", choices=["none", "backbone"], default="none")
    ap.add_argument("--layer-decay", type=float, default=None)
    ap.add_argument("--label-smoothing", type=float, default=0.0)
    ap.add_argument("--img-size", type=int, default=224, help="input size for pretrained models")
    ap.add_argument("--cutout", type=int, default=0, help="cutout square size in px (0 = off)")
    ap.add_argument("--imb-type", choices=["exp", "step"], default="exp")
    ap.add_argument("--imb-ratio", type=float, default=1.0, help="largest / smallest class size (1 = balanced)")
    ap.add_argument("--method", choices=["ce", "resample", "cb", "cb_drw", "la", "focal"], default="ce")
    ap.add_argument("--cb-beta", type=float, default=0.9999)
    ap.add_argument("--drw-frac", type=float, default=0.8, help="cb_drw: weights switch on after this fraction of epochs")
    ap.add_argument("--la-tau", type=float, default=1.0)
    ap.add_argument("--focal-gamma", type=float, default=1.0)
    ap.add_argument("--limit", type=int, default=None, help="smoke test: this many train / val / test images")
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    scratch = args.model == "resnet18_cifar"
    vit = "dinov2" in args.model or args.model.startswith("vit_")
    vit_b = "dinov2_b" in args.model or "vit_base" in args.model
    defaults = dict(
        epochs=200 if scratch else 10,
        bs=128 if not vit_b else 64,
        opt="sgd" if scratch else "adamw",
        lr=0.1 if scratch else (5e-5 if vit else 1e-4),
        head_lr=None if scratch else 1e-3,
        wd=5e-4 if scratch else 0.05,
        warmup_epochs=5 if scratch else 1,
        lp_epochs=0 if scratch else 1,
        layer_decay=0.75 if vit else 0.0,
    )
    for k, v in defaults.items():
        if getattr(args, k) is None:
            setattr(args, k, v)
    if args.head_lr is None:
        args.head_lr = args.lr
    if args.freeze == "backbone":
        args.lp_epochs = args.epochs
    args.sched = "cosine"
    if args.name is None:
        imb = "bal" if args.imb_ratio <= 1 else f"{args.imb_type}{args.imb_ratio:g}"
        probe = "_probe" if args.freeze == "backbone" else ""
        args.name = f"cifar_{args.model}{probe}_{imb}_{args.method}_s{args.seed}"
    return args


def main():
    args = parse_args()
    out = Path(args.out) / args.name
    out.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    torch.backends.cudnn.benchmark = True
    if not torch.cuda.is_available():
        sys.exit("CUDA not available: check the conda env and CUDA_VISIBLE_DEVICES")
    device = torch.device("cuda")
    print(f"device: cuda:0 = {torch.cuda.get_device_name(0)} "
          f"(CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'all')}), torch {torch.__version__}", flush=True)

    # data
    Xtr_all, ytr_all, Xte, yte = load_cifar(args.cifar_root)
    counts = class_counts(args.imb_type, args.imb_ratio)
    tr_idx, val_idx = make_splits(ytr_all, counts)
    Xtr, ytr, Xval, yval = Xtr_all[tr_idx], ytr_all[tr_idx], Xtr_all[val_idx], ytr_all[val_idx]
    if args.limit:
        keep = stratified_limit(ytr, args.limit)
        Xtr, ytr = Xtr[keep], ytr[keep]
        keep = stratified_limit(yval, args.limit)
        Xval, yval = Xval[keep], yval[keep]
        keep = stratified_limit(yte, args.limit)
        Xte, yte = Xte[keep], yte[keep]
    n_per_class = torch.bincount(ytr, minlength=10).tolist()
    Xtr, ytr, Xval, yval, Xte, yte = (t.to(device) for t in (Xtr, ytr, Xval, yval, Xte, yte))

    # model and optimiser
    model, size, mean, std, is_vit = build(args.model, args.img_size)
    cl = torch.preserve_format if is_vit else torch.channels_last
    model.to(device, memory_format=cl)
    head = model.get_classifier()
    head_ids = {id(p) for p in head.parameters()}
    groups = param_groups(model, args, is_vit)
    if args.opt == "sgd":
        opt = torch.optim.SGD(groups, lr=args.lr, momentum=0.9, nesterov=True)
    else:
        opt = torch.optim.AdamW(groups, lr=args.lr)
    steps = len(ytr) // args.bs
    sched = make_scheduler(opt, args, steps)

    # imbalance handling
    prior = torch.tensor(n_per_class, dtype=torch.float, device=device)
    prior = prior / prior.sum()
    eff = 1.0 - torch.pow(torch.tensor(args.cb_beta, device=device), prior * len(ytr))
    cb_w = (1.0 - args.cb_beta) / eff
    cb_w = cb_w / cb_w.sum() * 10
    sample_w = (1.0 / prior)[ytr]
    drw_start = int(args.drw_frac * args.epochs)

    def loss_fn(logits, y, epoch):
        logits = logits.float()
        if args.method == "la":
            logits = logits + args.la_tau * prior.log()
        w = cb_w if args.method == "cb" or (args.method == "cb_drw" and epoch >= drw_start) else None
        if args.method == "focal":
            logp = F.log_softmax(logits, 1).gather(1, y[:, None]).squeeze(1)
            return (-(1 - logp.exp()) ** args.focal_gamma * logp).mean()
        return F.cross_entropy(logits, y, weight=w, label_smoothing=args.label_smoothing)

    gen_cpu = torch.Generator().manual_seed(args.seed)
    gen_gpu = torch.Generator(device=device).manual_seed(args.seed)
    start_epoch, history = 0, []
    if (out / "last.pt").exists() and not args.no_resume:
        ck = torch.load(out / "last.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        start_epoch, history = ck["epoch"] + 1, ck["history"]
        gen_cpu.set_state(ck["gen_cpu"])
        gen_gpu.set_state(ck["gen_gpu"])
        print(f"resumed from epoch {ck['epoch']}", flush=True)

    (out / "args.json").write_text(json.dumps(vars(args) | {"n_per_class": n_per_class, "input_size": size}, indent=2))
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{args.name}: {n_params / 1e6:.1f}M params, train {len(ytr)} {n_per_class}, val {len(yval)}, "
          f"test {len(yte)}, input {size}px, method {args.method}", flush=True)
    tb = SummaryWriter(out / "tb")

    for epoch in range(start_epoch, args.epochs):
        frozen = epoch < args.lp_epochs
        for p in model.parameters():
            p.requires_grad_(not frozen or id(p) in head_ids)
        model.train()
        if frozen:
            model.eval()
            head.train()
        if args.method == "resample":
            order = torch.multinomial(sample_w.cpu(), len(ytr), replacement=True, generator=gen_cpu)
        else:
            order = torch.randperm(len(ytr), generator=gen_cpu)
        order = order.to(device)

        t0, seen, loss_sum, correct = time.time(), 0, 0.0, 0
        for i in range(steps):
            idx = order[i * args.bs:(i + 1) * args.bs]
            x = prep(augment(Xtr[idx], gen_gpu, cutout=args.cutout), size, mean, std, cl)
            y = ytr[idx]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x)
            loss = loss_fn(logits, y, epoch)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            seen += len(y)
            loss_sum += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
        train_time = time.time() - t0

        pv = predict(model, Xval, size, mean, std, cl)
        val_acc = (pv.argmax(1) == yval).float().mean().item()
        rec = dict(epoch=epoch, frozen=frozen, train_loss=loss_sum / seen, train_acc=correct / seen, val_acc=val_acc,
                   lr=opt.param_groups[0]["lr"], epoch_sec=round(train_time, 1), img_per_sec=round(seen / train_time, 1))
        history.append(rec)
        for k in ("train_loss", "train_acc", "val_acc"):
            tb.add_scalar(k.replace("_", "/", 1), rec[k], epoch)
        print(f"ep {epoch:3d}{' LP' if frozen else '   '} train loss {rec['train_loss']:.4f} acc {rec['train_acc']:.4f} | "
              f"val acc {val_acc:.4f} | {train_time:.0f}s {rec['img_per_sec']:.0f} img/s", flush=True)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "epoch": epoch, "history": history, "gen_cpu": gen_cpu.get_state(),
                    "gen_gpu": gen_gpu.get_state()}, out / "last.pt")
    tb.close()

    # final evaluation of the last-epoch model; the test set is used only here
    val = report(predict(model, Xval, size, mean, std, cl, tta=True), yval, counts, args.imb_type, args.imb_ratio)
    test = report(predict(model, Xte, size, mean, std, cl), yte, counts, args.imb_type, args.imb_ratio)
    test_tta = report(predict(model, Xte, size, mean, std, cl, tta=True), yte, counts, args.imb_type, args.imb_ratio)
    summary = dict(name=args.name, model=args.model, seed=args.seed, method=args.method,
                   imb_type=args.imb_type, imb_ratio=args.imb_ratio, n_train=len(ytr), n_per_class=n_per_class,
                   params_m=round(n_params / 1e6, 2), select="last", selected_epoch=history[-1]["epoch"],
                   val_acc=history[-1]["val_acc"], val_acc_tta=val["acc"],
                   test_acc=test["acc"], test_acc_tta=test_tta["acc"], test_macro_f1=test["macro_f1"],
                   test_group_acc=test.get("group_acc"), test_per_class_acc=test["per_class_acc"],
                   test_confusion=test["confusion_matrix_rows_true_cols_pred"], test_tta=test_tta,
                   total_train_min=round(sum(h["epoch_sec"] for h in history) / 60, 1), history=history)
    (out / "metrics.json").write_text(json.dumps(summary, indent=2))
    groups_txt = "" if "group_acc" not in test else " | " + ", ".join(f"{k} {v:.4f}" for k, v in test["group_acc"].items())
    print(f"last epoch {history[-1]['epoch']}: val acc {history[-1]['val_acc']:.4f} | TEST acc {test['acc']:.4f} "
          f"(flip TTA {test_tta['acc']:.4f}), macro-F1 {test['macro_f1']:.4f}{groups_txt}", flush=True)
    print(f"wrote {out / 'metrics.json'}", flush=True)


if __name__ == "__main__":
    main()
