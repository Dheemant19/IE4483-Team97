"""Train, validate and predict for Dogs vs. Cats: one model and one seed per run.

Writes runs/<name>/: args.json, metrics.json, val_predictions.csv, test_probs.csv,
submission.csv, last.pt (resumable), best.pt and tb/ (TensorBoard).

    python src/train.py --model dinov2_b --freeze backbone --seed 0 --name dinov2_b_probe_s0
"""
import argparse
import csv
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import build_datasets, eval_transform, train_stats, train_transform  # noqa: E402
from models import build_model, head_parameters  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="scratch_cnn, resnet50, convnext_tiny, dinov2_b, ... or a timm name")
    ap.add_argument("--data", default="data_288")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--name", default=None, help="run folder name (default <model>_s<seed>)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--bs", type=int, default=None)
    ap.add_argument("--opt", choices=["adamw", "sgd"], default=None)
    ap.add_argument("--lr", type=float, default=None, help="backbone LR (top layer LR for ViTs with layer decay)")
    ap.add_argument("--head-lr", type=float, default=None)
    ap.add_argument("--wd", type=float, default=None)
    ap.add_argument("--sched", choices=["cosine", "onecycle"], default=None)
    ap.add_argument("--warmup-epochs", type=float, default=1.0)
    ap.add_argument("--lp-epochs", type=int, default=None, help="epochs with frozen backbone before full fine-tune (LP-FT)")
    ap.add_argument("--freeze", choices=["none", "backbone"], default="none",
                    help="backbone = linear probe: head only for every epoch (no fine-tune phase)")
    ap.add_argument("--layer-decay", type=float, default=None, help="layer-wise LR decay for ViTs (0 = off)")
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--aug", choices=["none", "basic", "default", "trivial"], default="default")
    ap.add_argument("--img-size", type=int, default=None)
    ap.add_argument("--init", choices=["kaiming", "default"], default="kaiming", help="scratch CNN only")
    ap.add_argument("--no-pretrained", action="store_true", help="random init for a timm model (pretraining ablation)")
    ap.add_argument("--n-train", type=int, default=None, help="stratified train subset size (data-size ablation)")
    ap.add_argument("--limit", type=int, default=None, help="smoke test: use this many train and val images")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--clip", type=float, default=None, help="grad-norm clipping")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--select", choices=["last", "best"], default="last",
                    help="checkpoint used for the final metrics and submission: last epoch (default) or best val acc")
    args = ap.parse_args()

    scratch = args.model == "scratch_cnn"
    vit_b = "dinov2_b" in args.model or "vit_base" in args.model
    vit = "dinov2" in args.model or args.model.startswith("vit_")
    defaults = dict(
        epochs=50 if scratch else 10,
        bs=64 if vit_b else 128,
        opt="sgd" if scratch else "adamw",
        lr=0.05 if scratch else (5e-5 if vit else 1e-4),
        head_lr=None if scratch else 1e-3,
        wd=5e-4 if scratch else 0.05,
        sched="onecycle" if scratch else "cosine",
        lp_epochs=0 if (scratch or args.no_pretrained) else 1,
        layer_decay=0.75 if vit else 0.0,
    )
    for k, v in defaults.items():
        if getattr(args, k) is None:
            setattr(args, k, v)
    if args.head_lr is None:
        args.head_lr = args.lr
    if args.freeze == "backbone":
        args.lp_epochs = args.epochs
    args.name = args.name or f"{args.model}_s{args.seed}"
    return args


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def worker_init(_):
    s = torch.initial_seed() % 2**32
    np.random.seed(s)
    random.seed(s)


def param_groups(model, args, is_vit):
    """Head at head_lr, backbone at lr; no weight decay on biases/norms/tokens.
    ViTs with layer decay: block i gets lr * decay^(L + 1 - i)."""
    head_ids = {id(p) for p in head_parameters(model)}
    n_blocks = len(model.blocks) if is_vit and hasattr(model, "blocks") else 0
    groups = {}
    for name, p in model.named_parameters():
        no_wd = p.ndim <= 1 or name.endswith(("pos_embed", "cls_token", "reg_token"))
        if id(p) in head_ids:
            key, lr = "head", args.head_lr
        elif args.layer_decay and n_blocks:
            if name.startswith(("cls_token", "pos_embed", "reg_token", "patch_embed", "mask_token")):
                layer = 0
            elif name.startswith("blocks."):
                layer = int(name.split(".")[1]) + 1
            else:
                layer = n_blocks + 1
            key, lr = f"layer{layer}", args.lr * args.layer_decay ** (n_blocks + 1 - layer)
        else:
            key, lr = "backbone", args.lr
        key += "_nowd" if no_wd else ""
        g = groups.setdefault(key, {"params": [], "lr": lr, "weight_decay": 0.0 if no_wd else args.wd})
        g["params"].append(p)
    return list(groups.values())


def make_scheduler(opt, args, steps_per_epoch):
    total = args.epochs * steps_per_epoch
    if args.sched == "onecycle":
        return torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=[g["lr"] for g in opt.param_groups], total_steps=total, pct_start=0.3)
    # LP phase: constant LR (only the head moves). Then linear warm-up + cosine to 0.
    # Pure linear probe (--freeze backbone): warm-up + cosine over the whole run.
    start = 0 if args.freeze == "backbone" else args.lp_epochs * steps_per_epoch
    warm = max(1, int(args.warmup_epochs * steps_per_epoch))

    def f(step):
        if step < start:
            return 1.0
        s = step - start
        if s < warm:
            return (s + 1) / warm
        return 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - start - warm)))

    return torch.optim.lr_scheduler.LambdaLR(opt, f)


@torch.no_grad()
def predict(model, loader, device, cl, tta=False):
    """Returns (prob_dog, targets_or_ids, nll per sample)."""
    model.eval()
    probs, ys = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True, memory_format=cl)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            p = model(x).float().softmax(1)
            if tta:
                p = (p + model(torch.flip(x, dims=[3])).float().softmax(1)) / 2
        probs.append(p.cpu())
        ys.append(torch.as_tensor(y))
    return torch.cat(probs), torch.cat(ys)


def metrics_from(probs, y):
    acc = (probs.argmax(1) == y).float().mean().item()
    nll = F.nll_loss(probs.clamp_min(1e-8).log(), y).item()
    return acc, nll


def sample_paths(ds):
    if isinstance(ds, Subset):
        return [ds.dataset.samples[i][0] for i in ds.indices]
    return [s[0] for s in ds.samples]


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

    model, cfg, is_vit = build_model(args.model, pretrained=not args.no_pretrained,
                                     img_size=args.img_size, init=args.init)
    if args.model == "scratch_cnn":
        cfg.update(train_stats(args.data, size=cfg["input_size"][-1]))
    train_tf = train_transform(cfg, args.aug, scale_min=0.6 if args.model == "scratch_cnn" else 0.5)
    eval_tf = eval_transform(cfg)
    n_train = args.limit or args.n_train
    train_ds, val_ds, test_ds = build_datasets(args.data, train_tf, eval_tf, n_train=n_train, n_val=args.limit)

    g = torch.Generator().manual_seed(args.seed)
    kw = dict(num_workers=args.workers, pin_memory=True, worker_init_fn=worker_init,
              persistent_workers=args.workers > 0)
    train_dl = DataLoader(train_ds, args.bs, shuffle=True, drop_last=True, generator=g, **kw)
    val_dl = DataLoader(val_ds, 256, shuffle=False, **kw)
    test_dl = DataLoader(test_ds, 256, shuffle=False, **kw)

    cl = torch.preserve_format if is_vit else torch.channels_last
    model.to(device, memory_format=cl)
    head = model.get_classifier()
    head_ids = {id(p) for p in head.parameters()}

    if args.opt == "sgd":
        opt = torch.optim.SGD(param_groups(model, args, is_vit), lr=args.lr, momentum=0.9, nesterov=True)
    else:
        opt = torch.optim.AdamW(param_groups(model, args, is_vit), lr=args.lr)
    sched = make_scheduler(opt, args, len(train_dl))
    crit = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    start_epoch, best_acc, history = 0, -1.0, []
    if (out / "last.pt").exists() and not args.no_resume:
        ck = torch.load(out / "last.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        start_epoch, best_acc, history = ck["epoch"] + 1, ck["best_acc"], ck["history"]
        torch.set_rng_state(ck["rng"])
        g.set_state(ck["loader_rng"])
        print(f"resumed from epoch {ck['epoch']} (best val acc {best_acc:.4f})", flush=True)

    (out / "args.json").write_text(json.dumps(vars(args) | {"data_cfg": {k: list(v) if isinstance(v, tuple) else v
                                                                        for k, v in cfg.items()}}, indent=2))
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{args.name}: {n_params / 1e6:.1f}M params, train {len(train_ds)}, val {len(val_ds)}, "
          f"test {len(test_ds)}, input {cfg['input_size']}", flush=True)
    tb = SummaryWriter(out / "tb")

    for epoch in range(start_epoch, args.epochs):
        frozen = epoch < args.lp_epochs
        for p in model.parameters():
            p.requires_grad_(not frozen or id(p) in head_ids)
        model.train()
        if frozen:
            model.eval()       # keep backbone BatchNorm statistics fixed during linear probing
            head.train()

        t0, seen, loss_sum, correct = time.time(), 0, 0.0, 0
        for x, y in train_dl:
            x = x.to(device, non_blocking=True, memory_format=cl)
            y = y.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x)
                loss = crit(logits.float(), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if args.clip:
                nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step()
            sched.step()
            seen += len(y)
            loss_sum += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
        train_time = time.time() - t0

        probs, yv = predict(model, val_dl, device, cl)
        val_acc, val_loss = metrics_from(probs, yv)
        rec = dict(epoch=epoch, frozen=frozen, train_loss=loss_sum / seen, train_acc=correct / seen,
                   val_loss=val_loss, val_acc=val_acc, lr=opt.param_groups[0]["lr"],
                   epoch_sec=round(train_time, 1), img_per_sec=round(seen / train_time, 1))
        history.append(rec)
        for k in ("train_loss", "train_acc", "val_loss", "val_acc"):
            tb.add_scalar(k.replace("_", "/", 1), rec[k], epoch)
        tb.add_scalar("lr/head", opt.param_groups[-1]["lr"], epoch)
        print(f"ep {epoch:3d}{' LP' if frozen else '   '} train loss {rec['train_loss']:.4f} acc {rec['train_acc']:.4f} | "
              f"val loss {val_loss:.4f} acc {val_acc:.4f} | {train_time:.0f}s {rec['img_per_sec']:.0f} img/s", flush=True)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_acc": val_acc}, out / "best.pt")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "epoch": epoch, "best_acc": best_acc, "history": history,
                    "rng": torch.get_rng_state(), "loader_rng": g.get_state()}, out / "last.pt")
    tb.close()

    # Final evaluation: last epoch by default (--select best for the best-val-acc checkpoint)
    accs = [h["val_acc"] for h in history]
    best_epoch = history[max(range(len(accs)), key=lambda i: (accs[i], -i))]["epoch"]   # ties -> earliest
    lp_val_acc = history[args.lp_epochs - 1]["val_acc"] if args.freeze == "none" and args.lp_epochs > 0 else None
    ck = torch.load(out / ("best.pt" if args.select == "best" else "last.pt"), map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    sel_epoch = ck["epoch"]
    p_val, yv = predict(model, val_dl, device, cl)
    p_val_tta, _ = predict(model, val_dl, device, cl, tta=True)
    acc, nll = metrics_from(p_val, yv)
    acc_tta, nll_tta = metrics_from(p_val_tta, yv)
    pred = p_val.argmax(1)
    cm = [[int(((yv == t) & (pred == q)).sum()) for q in (0, 1)] for t in (0, 1)]

    with open(out / "val_predictions.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "label", "prob_dog", "prob_dog_tta", "pred", "pred_tta"])
        for path, t, p, pt in zip(sample_paths(val_ds), yv.tolist(), p_val[:, 1].tolist(), p_val_tta[:, 1].tolist()):
            w.writerow([path, t, f"{p:.6f}", f"{pt:.6f}", int(p > 0.5), int(pt > 0.5)])

    p_test, ids = predict(model, test_dl, device, cl, tta=True)
    with open(out / "test_probs.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "prob_dog"])
        w.writerows([i, f"{p:.6f}"] for i, p in zip(ids.tolist(), p_test[:, 1].tolist()))
    with open(out / "submission.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "label"])
        w.writerows([i, int(p > 0.5)] for i, p in sorted(zip(ids.tolist(), p_test[:, 1].tolist())))

    summary = dict(name=args.name, model=args.model, seed=args.seed, params_m=round(n_params / 1e6, 2),
                   select=args.select, selected_epoch=sel_epoch, val_acc=acc, val_loss=nll,
                   val_acc_tta=acc_tta, val_loss_tta=nll_tta,
                   last_val_acc=history[-1]["val_acc"], best_val_acc=max(accs), best_epoch=best_epoch,
                   lp_val_acc=lp_val_acc,
                   confusion_matrix_rows_true_cols_pred=cm, n_train=len(train_ds), n_val=len(val_ds),
                   total_train_min=round(sum(h["epoch_sec"] for h in history) / 60, 1), history=history)
    (out / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(f"{args.select} epoch {sel_epoch}: val acc {acc:.4f} (flip TTA {acc_tta:.4f}), confusion {cm} | "
          f"best {max(accs):.4f} @ ep {best_epoch}" + (f", LP {lp_val_acc:.4f}" if lp_val_acc is not None else ""),
          flush=True)
    print(f"wrote {out / 'submission.csv'}", flush=True)


if __name__ == "__main__":
    main()
