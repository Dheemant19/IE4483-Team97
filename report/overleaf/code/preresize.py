"""One-time copy of the Dogs vs. Cats data with the shorter side resized to 288 px.

Aspect ratio is kept, smaller images are not upscaled, and the validation copy of a training image is dropped.

    python scripts/preresize.py --src datasets --dst data_288
"""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image

DROP = {Path("val/cat/cat.12408.jpg")}


def resize_one(job):
    src, dst, short = job
    with Image.open(src) as im:
        im = im.convert("RGB")
        w, h = im.size
        s = min(w, h)
        if s > short:
            scale = short / s
            im = im.resize((round(w * scale), round(h * scale)), Image.BICUBIC)
        dst.parent.mkdir(parents=True, exist_ok=True)
        im.save(dst, "JPEG", quality=95)
    return 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="datasets")
    ap.add_argument("--dst", default="data_288")
    ap.add_argument("--short", type=int, default=288)
    ap.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 4))
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    jobs, dropped = [], 0
    for p in sorted(src.rglob("*.jpg")):
        rel = p.relative_to(src)
        if rel in DROP:
            dropped += 1
            continue
        out = dst / rel
        if not out.exists():
            jobs.append((p, out, args.short))

    print(f"{len(jobs)} images to resize, {dropped} dropped (train/val duplicate)", flush=True)
    with ProcessPoolExecutor(args.workers) as ex:
        done = 0
        for _ in ex.map(resize_one, jobs, chunksize=64):
            done += 1
            if done % 5000 == 0:
                print(f"  {done}/{len(jobs)}", flush=True)

    for split in ["train/cat", "train/dog", "val/cat", "val/dog", "test"]:
        print(f"{split:10s} {len(list((dst / split).glob('*.jpg')))}")


if __name__ == "__main__":
    main()
