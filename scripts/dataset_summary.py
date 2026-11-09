"""Summarise the dataset folder: file counts, image sizes and modes, corrupt files and exact duplicates.

    python scripts/dataset_summary.py datasets --out dataset_summary.json
"""
import argparse, hashlib, json, os, statistics, sys, threading, time
from io import BytesIO
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from PIL import Image

def inspect(path):
    with open(path, "rb") as f:
        data = f.read()
    rec = {"path": path, "bytes": len(data)}
    rec["md5"] = hashlib.md5(data).hexdigest()
    try:
        im = Image.open(BytesIO(data))
        rec["format"], rec["mode"], (rec["w"], rec["h"]) = im.format, im.mode, im.size
        im.load()  # full decode catches truncated files that verify() misses
    except Exception as e:
        rec["error"] = f"{type(e).__name__}: {e}"
    return rec

def stats(xs):
    if not xs: return {}
    xs = sorted(xs)
    q = lambda p: xs[min(len(xs) - 1, int(p * len(xs)))]
    return {"min": xs[0], "p5": q(.05), "median": q(.5), "mean": round(statistics.mean(xs), 1), "p95": q(.95), "max": xs[-1]}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", default=os.path.join(os.path.dirname(__file__), "..", "datasets"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--cache", default=None, help="JSONL cache so an interrupted run can resume")
    a = ap.parse_args()
    root = os.path.abspath(a.root)

    groups = {}
    for split in ("train", "val", "test"):
        sp = os.path.join(root, split)
        if not os.path.isdir(sp): continue
        # scandir's is_dir()/is_file() use cached dirent types, avoiding a slow stat() per file on network/VM mounts
        subs = sorted(e.name for e in os.scandir(sp) if e.is_dir())
        for c in subs or [None]:
            d = os.path.join(sp, c) if c else sp
            groups[f"{split}/{c}" if c else split] = sorted(e.path for e in os.scandir(d) if e.is_file())

    cache = {}
    if a.cache and os.path.exists(a.cache):
        with open(a.cache) as f:
            for line in f:
                try: r = json.loads(line); cache[r["path"]] = r
                except ValueError: pass
    cache_f = open(a.cache, "a") if a.cache else None
    lock = threading.Lock()

    def cached_inspect(path):
        if path in cache: return cache[path]
        r = inspect(path)
        if cache_f:
            with lock: cache_f.write(json.dumps(r) + "\n"); cache_f.flush()
        return r

    t0 = time.time()
    report, by_md5 = {"root": root, "groups": {}}, defaultdict(list)
    with ThreadPoolExecutor(a.workers) as ex:
        for g, files in groups.items():
            recs = list(ex.map(cached_inspect, files))
            ok = [r for r in recs if "error" not in r]
            for r in recs: by_md5[r["md5"]].append(os.path.relpath(r["path"], root))
            report["groups"][g] = {
                "count": len(recs),
                "extensions": dict(Counter(os.path.splitext(r["path"])[1].lower() for r in recs)),
                "formats": dict(Counter(r.get("format") for r in ok)),
                "modes": dict(Counter(r.get("mode") for r in ok)),
                "width": stats([r["w"] for r in ok]), "height": stats([r["h"] for r in ok]),
                "aspect_w_over_h": stats([round(r["w"] / r["h"], 2) for r in ok]),
                "short_side_lt_100": sum(min(r["w"], r["h"]) < 100 for r in ok),
                "kbytes": stats([round(r["bytes"] / 1024, 1) for r in recs]),
                "corrupt": [{"file": os.path.relpath(r["path"], root), "error": r["error"]} for r in recs if "error" in r],
                "sample_names": [os.path.basename(p) for p in files[:3]],
            }
            print(f"{g}: {len(recs)} files, {len(recs) - len(ok)} unreadable ({time.time() - t0:.0f}s)", file=sys.stderr)

    dups = [v for v in by_md5.values() if len(v) > 1]
    cross = [v for v in dups if len({p.replace(os.sep, "/").split("/")[0] for p in v}) > 1]
    label_conflict = [v for v in dups if len({os.path.basename(os.path.dirname(p)) for p in v if not p.startswith("test")}) > 1]
    report["duplicates"] = {"exact_duplicate_sets": len(dups), "files_in_duplicate_sets": sum(map(len, dups)),
                            "sets_spanning_splits": len(cross), "sets_with_conflicting_labels": len(label_conflict),
                            "examples": dups[:15]}
    report["seconds"] = round(time.time() - t0, 1)
    out = json.dumps(report, indent=2)
    print(out)
    if a.out:
        with open(a.out, "w") as f: f.write(out)

if __name__ == "__main__":
    main()
