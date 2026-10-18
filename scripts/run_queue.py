"""Run a list of train.py jobs across several GPUs, one job per GPU at a time.

Each non-empty line of the job file is the argument list for src/train.py, e.g.
    --model resnet50 --seed 1
    --model convnext_tiny --seed 0 --n-train 2000 --name convnext_tiny_n2000_s0
Lines starting with # are skipped. Runs that already have runs/<name>/metrics.json
are skipped, so the same file can be re-run after adding lines or after a crash
(unfinished runs resume from their last.pt).

    nohup python scripts/run_queue.py jobs/week1.txt --gpus 0,1,2,3 > runs/queue.log 2>&1 &
    nohup python scripts/run_queue.py jobs/cifar_h.txt --script src/cifar.py > runs/queue_cifar_h.log 2>&1 &
For src/cifar.py give every line an explicit --name (its default names differ from model_s<seed>).

--skip-busy lets a new queue share GPUs and a job file with work that is already running
(e.g. after stopping an old queue process while its runs carry on): a GPU is only used when
less than 500 MB of its memory is in use, and a run is skipped at launch time if it has finished
meanwhile or its last.pt was written in the last 5 minutes (someone else is training it).
"""
import argparse
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path


def run_name(args):
    def get(flag, default=None):
        return args[args.index(flag) + 1] if flag in args else default
    return get("--name") or f"{get('--model')}_s{get('--seed', '0')}"


def gpu_busy(gpu):
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", gpu],
                             capture_output=True, text=True, timeout=30).stdout
        return int(out.strip().splitlines()[0]) > 500   # idle 3090s show 15-40 MiB (Xorg)
    except Exception:
        return True


def taken(out, name):
    """Finished, or last.pt written in the last 5 minutes by another process."""
    run = Path(out) / name
    if (run / "metrics.json").exists():
        return "done"
    last = run / "last.pt"
    if last.exists() and time.time() - last.stat().st_mtime < 300:
        return "running elsewhere"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs")
    ap.add_argument("--gpus", default="0,1,2,3")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--script", default="src/train.py", help="training script, e.g. src/cifar.py")
    ap.add_argument("--skip-busy", action="store_true",
                    help="only use GPUs with < 500 MB memory in use; skip runs that are being trained elsewhere")
    args = ap.parse_args()

    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    lines = [l.strip() for l in Path(args.jobs).read_text().splitlines()]
    jobs = []
    for l in lines:
        if not l or l.startswith("#"):
            continue
        a = shlex.split(l)
        name = run_name(a)
        if (Path(args.out) / name / "metrics.json").exists():
            print(f"skip {name} (done)", flush=True)
            continue
        jobs.append((name, a))
    print(f"{len(jobs)} jobs on GPUs {gpus}", flush=True)

    running = {}   # gpu -> (name, Popen, logfile)
    failed = []
    while jobs or running:
        for gpu, (name, p, log) in list(running.items()):
            if p.poll() is not None:
                log.close()
                status = "done" if p.returncode == 0 else f"FAILED (exit {p.returncode})"
                if p.returncode != 0:
                    failed.append(name)
                print(f"{time.strftime('%H:%M:%S')} gpu {gpu} {name} {status}", flush=True)
                del running[gpu]
        for gpu in gpus:
            if gpu not in running and jobs:
                if args.skip_busy and gpu_busy(gpu):
                    continue
                name, a = jobs.pop(0)
                why = taken(args.out, name) if args.skip_busy else None
                if why:
                    print(f"{time.strftime('%H:%M:%S')} skip {name} ({why})", flush=True)
                    continue
                log = open(Path(args.out) / f"{name}.log", "a")
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
                p = subprocess.Popen([sys.executable, args.script, *a, "--out", args.out],
                                     stdout=log, stderr=subprocess.STDOUT, env=env)
                running[gpu] = (name, p, log)
                print(f"{time.strftime('%H:%M:%S')} gpu {gpu} start {name}", flush=True)
        time.sleep(10)
    print("all done" + (f"; failed: {failed}" if failed else ""), flush=True)


if __name__ == "__main__":
    main()
