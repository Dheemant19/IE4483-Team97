"""Part (e): validation error galleries, uncertain test images and Grad-CAM figures.

    python scripts/analyze_errors.py                                              # galleries (CPU)
    python scripts/analyze_errors.py --gradcam --out report/gradcam               # Grad-CAM maps (GPU)
    python scripts/analyze_errors.py --cams-from report/gradcam/gradcam_maps.npz  # redraw Grad-CAM (CPU)
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "src"))
import figstyle as fs  # noqa: E402
from figstyle import MM, INK, INK2, MUTED, VERMILLION, plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

# torch, torchvision and the project modules are imported inside the Grad-CAM functions, so everything
# else also works on a machine without PyTorch.
NAMES = {0: "cat", 1: "dog"}
TRACK, OK_INK = "#D9D9D9", "#4D4D4D"
CAPTIONS = []


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, cols):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def load_rgb(path):
    """The whole image (nothing cropped), centred on a white square so grid cells line up.
    Returns the array and the extent of the photo inside the square (for the error frame)."""
    with Image.open(path) as im:
        im = im.convert("RGB")
        side = max(im.size)
        sq = Image.new("RGB", (side, side), (255, 255, 255))
        x0, y0 = (side - im.width) // 2, (side - im.height) // 2
        sq.paste(im, (x0, y0))
        return np.asarray(sq), (x0, y0, im.width, im.height)


def show_photo(ax, path, wrong=False):
    img, (x0, y0, w, h) = load_rgb(path)
    ax.imshow(img, interpolation="lanczos")
    if wrong:
        ax.add_patch(Rectangle((x0 - 0.5, y0 - 0.5), w, h, fill=False, ec=VERMILLION, lw=1.6))
    ax.set_axis_off()


def prob_bar(ax, p, wrong):
    """p(dog) on [0, 1]: grey track, tick at 0.5, coloured segment from 0.5 to p, dot at p."""
    col = VERMILLION if wrong else OK_INK
    ax.plot([0, 1], [0, 0], color=TRACK, lw=2.6, solid_capstyle="butt", zorder=1)
    ax.plot([0.5, 0.5], [-0.55, 0.55], color=MUTED, lw=0.6, zorder=2)
    ax.plot([0.5, p], [0, 0], color=col, lw=2.6, solid_capstyle="butt", zorder=3)
    ax.plot(p, 0, "o", ms=3.6, color=col, mec="white", mew=0.5, zorder=4)
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-1, 1)
    ax.set_axis_off()


def gallery(items, out, name, caption, alt, ncols=6, groups=None):
    """items: dicts with path, name, p, wrong. groups: optional [(letter, n_items), ...] -> one block each."""
    if not items:
        return print(f"skip {name} (nothing to show)")
    groups = groups or [("", len(items))]
    blocks = [(letter, (n + ncols - 1) // ncols) for letter, n in groups if n]
    cell = fs.FULL / ncols
    row_h = cell + 7.5 * MM
    height = sum(r for _, r in blocks) * row_h + (5 * MM if any(l for l, _ in blocks) else 0) * len(blocks)
    fig = plt.figure(figsize=(fs.FULL, height), layout="constrained")
    fig.get_layout_engine().set(w_pad=0.6 * MM, h_pad=0.8 * MM, wspace=0.03, hspace=0.04)
    subfigs = fig.subfigures(len(blocks), 1, height_ratios=[r for _, r in blocks], hspace=0.04) \
        if len(blocks) > 1 else [fig]
    k = 0
    for sf, (letter, nrows) in zip(subfigs, blocks):
        n_here = dict(groups)[letter]
        gs = sf.add_gridspec(nrows * 3, ncols, height_ratios=[cell, 2.2 * MM, 4.2 * MM] * nrows)
        first = None
        for j in range(n_here):
            it = items[k + j]
            r, c = divmod(j, ncols)
            ax = sf.add_subplot(gs[3 * r, c])
            first = first or ax
            show_photo(ax, it["path"], it.get("wrong"))
            prob_bar(sf.add_subplot(gs[3 * r + 1, c]), it["p"], it.get("wrong"))
            tx = sf.add_subplot(gs[3 * r + 2, c])
            tx.set_axis_off()
            col = VERMILLION if it.get("wrong") else INK2
            tx.text(0.02, 0.85, it["name"], ha="left", va="top", fontsize=6, color=INK, transform=tx.transAxes)
            tx.text(0.98, 0.85, f"{it['p']:.2f}", ha="right", va="top", fontsize=6, color=col,
                    fontweight="bold" if it.get("wrong") else "normal", transform=tx.transAxes)
        for j in range(n_here, nrows * ncols):              # keep empty cells empty
            r, c = divmod(j, ncols)
            for q in range(3):
                sf.add_subplot(gs[3 * r + q, c]).set_axis_off()
        if letter:
            first.annotate(letter, (0, 1), xycoords="axes fraction", xytext=(-1, 3), textcoords="offset points",
                           fontsize=8.5, fontweight="bold", ha="left", va="bottom", color=INK)
        k += n_here
    fs.save(fig, out, name)
    CAPTIONS.append((name, caption, alt))


def short(name):
    return name[:-4] if name.endswith(".jpg") else name


# ---------------------------------------------------------------- Grad-CAM (server)

def load_run_model(run_dir, device):
    import torch
    import torch.nn as nn
    from models import build_model

    a = json.loads((Path(run_dir) / "args.json").read_text())
    cfg = a["data_cfg"]
    model, _, is_vit = build_model(a["model"], pretrained=False, img_size=a.get("img_size"), init="default")
    ck = torch.load(Path(run_dir) / "last.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    for m in model.modules():           # in-place ReLUs interfere with Grad-CAM's gradient hooks
        if isinstance(m, nn.ReLU):
            m.inplace = False
    model.to(device).eval()
    if is_vit:
        n_prefix = getattr(model, "num_prefix_tokens", 1)

        def reshape(t):
            t = t[:, n_prefix:, :]
            side = int(round(t.shape[1] ** 0.5))
            return t.reshape(t.shape[0], side, side, t.shape[2]).permute(0, 3, 1, 2)
        return model, cfg, model.blocks[-1].norm1, reshape
    return model, cfg, model.features[-2], None       # scratch CNN: last ReLU before the final max-pool


def view_transform(cfg):
    from torchvision import transforms as T
    from data import INTERP

    size = cfg["input_size"][-1]
    interp = INTERP.get(cfg.get("interpolation", "bilinear"), T.InterpolationMode.BILINEAR)
    return T.Compose([T.Resize(int(round(size / cfg.get("crop_pct", 0.875))), interpolation=interp), T.CenterCrop(size)])


def cams_for(run_dir, paths, device):
    """Per image: the centre crop the model saw (uint8), the Grad-CAM map in [0, 1], and p(dog)."""
    import torch
    from data import eval_transform
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    model, cfg, layer, reshape = load_run_model(run_dir, device)
    tf, vt = eval_transform(cfg), view_transform(cfg)
    imgs = [Image.open(p).convert("RGB") for p in paths]
    x = torch.stack([tf(im) for im in imgs]).to(device)
    with torch.no_grad():
        p_dog = model(x).softmax(1)[:, 1].cpu().numpy()
    targets = [ClassifierOutputTarget(int(p > 0.5)) for p in p_dog]
    with GradCAM(model=model, target_layers=[layer], reshape_transform=reshape) as cam:
        maps = cam(input_tensor=x, targets=targets)        # (N, H, W), each min-max scaled to [0, 1]
    views = np.stack([np.asarray(vt(im), dtype=np.uint8) for im in imgs])
    del model
    torch.cuda.empty_cache()
    return views, maps.astype(np.float16), p_dog.astype(np.float32)


MODELS = [("probe", "DINOv2-B\nprobe"), ("scratch", "Scratch\nCNN")]


def compute_cams(sel, runs, out):
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paths = [s["path"] for s in sel]
    arrays = {}
    for key, _ in MODELS:
        run = runs[key]
        if run and Path(run, "last.pt").exists():
            arrays[f"{key}_views"], arrays[f"{key}_cams"], arrays[f"{key}_p"] = cams_for(run, paths, device)
        else:
            print(f"Grad-CAM: no checkpoint in {run}, row skipped")
    meta = dict(images=sel, runs=runs)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "gradcam_maps.npz", meta=np.array(json.dumps(meta)), **arrays)
    print(f"wrote {out / 'gradcam_maps.npz'}")
    return out / "gradcam_maps.npz"


# ---------------------------------------------------------------- Grad-CAM figure (GPU or CPU)

def heat_rgba(view, cam, cmap):
    """Photo in light greyscale under the heat map; map opacity grows with relevance."""
    g = view.astype(np.float32) @ np.array([0.299, 0.587, 0.114], dtype=np.float32) / 255
    return blend(0.35 + 0.65 * g, cam, cmap)


def blend(grey, cam, cmap):
    base = np.repeat(np.asarray(grey, dtype=np.float32)[..., None], 3, axis=2)
    c = np.clip(cam.astype(np.float32), 0, 1)
    col = cmap(c)[..., :3]
    a = (0.9 * c ** 1.6)[..., None]    # low relevance stays close to the plain photo
    return np.clip(base * (1 - a) + col * a, 0, 1)


def draw_gradcam(npz, out):
    d = np.load(npz, allow_pickle=False)
    meta = json.loads(str(d["meta"]))
    sel = meta["images"]
    rows = [(key, label) for key, label in MODELS if f"{key}_cams" in d.files]
    n = len(sel)
    cmap = plt.get_cmap("inferno")
    label_w = 11 * MM
    cell = (fs.FULL - label_w) / n
    fig = plt.figure(figsize=(fs.FULL, (1 + len(rows)) * (cell + 4.2 * MM) + 12 * MM), layout="constrained")
    fig.get_layout_engine().set(w_pad=0.5 * MM, h_pad=0.5 * MM, wspace=0.02, hspace=0.02)
    gs = fig.add_gridspec(2 * (1 + len(rows)) + 1, n,
                          height_ratios=[cell, 4.2 * MM] * (1 + len(rows)) + [2.4 * MM])
    for j, s in enumerate(sel):
        ax = fig.add_subplot(gs[0, j])
        show_photo(ax, s["path"], s.get("true") is not None and s.get("missed"))
        tx = fig.add_subplot(gs[1, j])
        tx.set_axis_off()
        tx.text(0.5, 0.9, s["name"], transform=tx.transAxes, ha="center", va="top", fontsize=6, color=INK)
        if j == 0:
            ax.text(-0.08, 0.5, "Input", transform=ax.transAxes, ha="right", va="center", fontsize=6.8, color=INK)
        for i, (key, label) in enumerate(rows, start=1):
            ax = fig.add_subplot(gs[2 * i, j])
            view, cam, p = d[f"{key}_views"][j], d[f"{key}_cams"][j], float(d[f"{key}_p"][j])
            ax.imshow(heat_rgba(view, cam, cmap), interpolation="lanczos")
            ax.set_axis_off()
            pred = int(p > 0.5)
            wrong = s.get("true") is not None and pred != s["true"]
            tx = fig.add_subplot(gs[2 * i + 1, j])
            tx.set_axis_off()
            tx.text(0.5, 0.9, f"{NAMES[pred]}, p(dog) {p:.2f}", transform=tx.transAxes, ha="center", va="top",
                    fontsize=6, color=VERMILLION if wrong else INK2, fontweight="bold" if wrong else "normal")
            if j == 0:
                ax.text(-0.08, 0.5, label, transform=ax.transAxes, ha="right", va="center", fontsize=6.8,
                        color=INK, linespacing=1.2)
    cax = fig.add_subplot(gs[-1, max(0, n // 2 - 2):min(n, n // 2 + 2)])
    # colour key drawn the way the maps are drawn: blended over a mid-grey photo
    ramp = np.linspace(0, 1, 256)[None, :].repeat(8, axis=0)
    cax.imshow(blend(np.full(ramp.shape, 0.67), ramp, cmap), extent=(0, 1, 0, 1), aspect="auto")
    cax.set_yticks([])
    cax.set_xticks([0, 0.5, 1], ["0", "0.5", "1"])
    cax.tick_params(axis="x", labelsize=6, length=2, width=0.5)
    for sp in cax.spines.values():
        sp.set_linewidth(0.4)
        sp.set_visible(True)
    cax.set_xlabel("Grad-CAM relevance (scaled to 0–1 per image)", fontsize=6.5, labelpad=2)
    fs.save(fig, out, "gradcam")
    caption = ("Grad-CAM for the final model's backbone (DINOv2-B linear probe, seed 0; last-block attention-norm "
               "features on the 16 × 16 patch grid) and the scratch CNN (last convolutional block; seed 0). Columns: "
               "the four validation images the final model is most confidently wrong about (framed), one easy cat, "
               "one easy dog, and the two test images it is least sure about (no labels). Top row: the whole image; "
               "lower rows: the centre crop each model receives, in greyscale, under its Grad-CAM map for the "
               "predicted class. Each map is rescaled to 0–1 within its image (colour and opacity both rise with "
               "relevance), so it compares locations within an image, not strength across images. Labels give each "
               "single model's prediction and p(dog) without flip TTA; vermillion = wrong.")
    alt = ("Grid of eight photographs (top row) with Grad-CAM heat maps for the DINOv2-B linear probe (middle row) and "
           "the scratch CNN (bottom row), each labelled with the model's prediction and p(dog).")
    CAPTIONS.append(("gradcam", caption, alt))


# ---------------------------------------------------------------- main

def prep_rows(rows, key):
    for r in rows:
        r["p"], r["y"] = float(r[key]), int(r["label"])
        r["pred"] = int(r["p"] > 0.5)
        r["conf"] = abs(r["p"] - 0.5) * 2
        r["name"] = Path(r["path"]).name
    return rows


def prob_key(rows):
    return "prob_dog_tta" if "prob_dog_tta" in rows[0] else "prob_dog"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ens", default="runs/ens_dinov2_b_probe", help="final model (ensemble or single run folder)")
    ap.add_argument("--probe-run", default="runs/dinov2_b_probe_s0", help="checkpoint for the DINOv2-B Grad-CAM")
    ap.add_argument("--scratch-run", default="runs/scratch_cnn_s0")
    ap.add_argument("--data", default="data_288", help="folder with test/ (for the test gallery)")
    ap.add_argument("--out", default="report/figures")
    ap.add_argument("--n", type=int, default=12, help="images per gallery")
    ap.add_argument("--gradcam", action="store_true", help="compute Grad-CAM maps (needs torch + checkpoints)")
    ap.add_argument("--cams-from", help="redraw the Grad-CAM figure from a saved gradcam_maps.npz")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    val = read_csv(Path(args.ens) / "val_predictions.csv")
    val = prep_rows(val, prob_key(val))
    wrong = sorted([r for r in val if r["pred"] != r["y"]], key=lambda r: -r["conf"])
    tp = read_csv(Path(args.ens) / "test_probs.csv")
    for r in tp:
        r["p"] = float(r["prob_dog"])
        r["conf"] = abs(r["p"] - 0.5) * 2
        r["path"] = str(Path(args.data) / "test" / f"{r['id']}.jpg")
    tu = sorted(tp, key=lambda r: r["conf"])[:args.n]

    if args.gradcam or args.cams_from:
        npz = Path(args.cams_from) if args.cams_from else None
        if args.gradcam:
            sel = [dict(path=r["path"], true=r["y"], missed=True, name=short(r["name"])) for r in wrong[:4]]
            for cls in (0, 1):
                c = max((r for r in val if r["y"] == cls and r["pred"] == cls), key=lambda r: r["conf"])
                sel.append(dict(path=c["path"], true=cls, missed=False, name=short(c["name"])))
            sel += [dict(path=r["path"], true=None, missed=False, name=f"test {r['id']}") for r in tu[:2]]
            npz = compute_cams(sel, {"probe": args.probe_run, "scratch": args.scratch_run}, out)
            try:                        # the CPU redraw from the .npz makes the final figure anyway
                draw_gradcam(npz, out)
            except Exception as e:      # noqa: BLE001
                print(f"maps saved; figure not drawn here ({type(e).__name__}: {e})")
        else:
            draw_gradcam(npz, out)
    else:
        borderline = sorted([r for r in val if r["pred"] == r["y"]], key=lambda r: r["conf"])[:max(0, args.n - len(wrong))]
        items = [dict(path=r["path"], name=short(r["name"]), p=r["p"], wrong=True) for r in wrong]
        items += [dict(path=r["path"], name=short(r["name"]), p=r["p"]) for r in borderline]
        gallery(items, out, "errors_final_val",
                f"(a) All {len(wrong)} validation images misclassified by the final model (3-seed ensemble of DINOv2-B "
                f"linear probes; framed in vermillion), most confident first; (b) its {len(borderline)} least confident "
                "correct predictions. The file name gives the dataset label. Under each image, the bar shows p(dog) "
                "from 0 (cat, left) to 1 (dog, right): the coloured segment runs from the 0.5 threshold to the "
                "prediction, and the number is p(dog) (flip test-time augmentation). Images are shown whole.",
                f"Two rows of photographs: {len(wrong)} the final model got wrong and {len(borderline)} it got right "
                "with low confidence, each with its file name and predicted probability of dog.",
                groups=[("a", len(wrong)), ("b", len(borderline))])
        write_csv(out / "data" / "errors_final_val.csv", wrong + borderline, ["path", "label", "p", "pred", "conf"])

        sv_path = Path(args.scratch_run) / "val_predictions.csv"
        if sv_path.exists():
            sv = read_csv(sv_path)
            sv = prep_rows(sv, prob_key(sv))
            sw = sorted([r for r in sv if r["pred"] != r["y"]], key=lambda r: -r["conf"])
            gallery([dict(path=r["path"], name=short(r["name"]), p=r["p"], wrong=True) for r in sw[:args.n]],
                    out, "errors_scratch_val",
                    f"The {min(args.n, len(sw))} most confident of the scratch CNN's {len(sw)} validation mistakes "
                    f"({Path(args.scratch_run).name}). File names give the dataset label; bars and numbers show p(dog) "
                    "with flip test-time augmentation, as in the final-model gallery.",
                    "Two rows of photographs the scratch CNN misclassified with high confidence.")
            write_csv(out / "data" / "errors_scratch_val.csv", sw, ["path", "label", "p", "pred", "conf"])

        gallery([dict(path=r["path"], name=f"test {r['id']}", p=r["p"]) for r in tu], out, "test_uncertain",
                f"The {len(tu)} test images on which the final model is least confident (p(dog) closest to 0.5), most "
                "uncertain first. Test labels are not available, so the bar only shows the prediction: left of the "
                "centre tick = cat, right = dog.",
                "Two rows of test photographs with the final model's predicted probability of dog, all near 0.5.")
        write_csv(out / "data" / "test_uncertain.csv", tu, ["id", "prob_dog", "conf", "path"])

    name = "captions_gradcam.md" if (args.gradcam or args.cams_from) else "captions_errors.md"
    with open(out / name, "w", encoding="utf-8") as fh:
        fh.write("# Part (e) figure captions and alt text (draft)\n\nGenerated by scripts/analyze_errors.py. "
                 "Add what the images actually show (e.g. mislabelled or ambiguous photos) to the caption.\n")
        for n_, cap, alt in CAPTIONS:
            fh.write(f"\n## {n_}\n\n**Caption.** {cap}\n\n**Alt text.** {alt}\n")


if __name__ == "__main__":
    main()
