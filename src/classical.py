"""Classical baseline: HOG + colour histogram features with a linear and an RBF SVM (CPU only).

    python src/classical.py --data data_288 --jobs 8    # -> runs/hog_svm_linear/, runs/hog_svm_rbf/
"""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from PIL import Image
from skimage.feature import hog
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC

EXCLUDE = ("val/cat/cat.12408.jpg",)   # exact copy of train/cat/cat.2339.jpg

SIZE = 128


def features(path):
    with Image.open(path) as im:
        im = im.convert("RGB")
        w, h = im.size
        s = SIZE / min(w, h)                       # resize shorter side, then centre crop
        im = im.resize((max(SIZE, round(w * s)), max(SIZE, round(h * s))), Image.BILINEAR)
        w, h = im.size
        l, t = (w - SIZE) // 2, (h - SIZE) // 2
        im = im.crop((l, t, l + SIZE, t + SIZE))
    rgb = np.asarray(im)
    gray = np.asarray(im.convert("L"), dtype=np.float32) / 255
    f_hog = hog(gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm="L2-Hys")
    f_col = np.concatenate([np.histogram(rgb[..., c], bins=16, range=(0, 256), density=True)[0] for c in range(3)])
    return np.concatenate([f_hog, f_col * 256]).astype(np.float32)


def load_split(root, jobs):
    paths, labels = [], []
    for y, cls in enumerate(["cat", "dog"]):
        ps = [p for p in sorted((root / cls).glob("*.jpg")) if not p.as_posix().endswith(EXCLUDE)]
        paths += ps
        labels += [y] * len(ps)
    X = np.stack(Parallel(n_jobs=jobs)(delayed(features)(p) for p in paths))
    return X, np.array(labels), paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data_288")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--n-train", type=int, default=None, help="balanced train subset (fixed seed)")
    ap.add_argument("--skip-rbf", action="store_true")
    args = ap.parse_args()
    root = Path(args.data)

    t0 = time.time()
    Xtr, ytr, _ = load_split(root / "train", args.jobs)
    Xva, yva, pva = load_split(root / "val", args.jobs)
    test_paths = sorted((root / "test").glob("*.jpg"), key=lambda p: int(p.stem))
    Xte = np.stack(Parallel(n_jobs=args.jobs)(delayed(features)(p) for p in test_paths))
    ids = [int(p.stem) for p in test_paths]
    if args.n_train:
        rng = np.random.default_rng(1234)
        idx = np.concatenate([rng.permutation(np.where(ytr == c)[0])[: args.n_train // 2] for c in (0, 1)])
        Xtr, ytr = Xtr[idx], ytr[idx]
    print(f"features {Xtr.shape[1]}-d: train {len(ytr)}, val {len(yva)}, test {len(ids)} "
          f"({time.time() - t0:.0f}s)", flush=True)

    clfs = {"hog_svm_linear": make_pipeline(StandardScaler(), LinearSVC(C=0.01, max_iter=5000))}
    if not args.skip_rbf:
        clfs["hog_svm_rbf"] = make_pipeline(StandardScaler(), PCA(256, random_state=0), SVC(C=10, gamma="scale"))

    for name, clf in clfs.items():
        t1 = time.time()
        clf.fit(Xtr, ytr)
        pred = clf.predict(Xva)
        acc = float((pred == yva).mean())
        cm = [[int(((yva == t) & (pred == q)).sum()) for q in (0, 1)] for t in (0, 1)]
        out = Path(args.out) / (name + (f"_n{args.n_train}" if args.n_train else ""))
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "val_predictions.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["path", "label", "pred"])
            w.writerows([str(p), t, int(q)] for p, t, q in zip(pva, yva, pred))
        with open(out / "submission.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "label"])
            w.writerows(sorted(zip(ids, map(int, clf.predict(Xte)))))
        metrics = dict(name=out.name, model=name, val_acc=acc, confusion_matrix_rows_true_cols_pred=cm,
                       n_train=int(len(ytr)), n_val=int(len(yva)), feature_dim=int(Xtr.shape[1]),
                       total_train_min=round((time.time() - t1) / 60, 1))
        (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
        print(f"{name}: val acc {acc:.4f}, confusion {cm} ({time.time() - t1:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
