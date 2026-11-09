# IE4483 Mini Project (Option 2), Team 97: Dogs vs. Cats, CIFAR-10 and long-tailed CIFAR-10

This repository holds all code, settings and results behind our IE4483 report. It covers:

- **Dogs vs. Cats.** The final model is a DINOv2-B linear probe: a frozen DINOv2-B backbone with a trained linear head. Three seeds are ensembled with flip test-time augmentation. It reaches **99.88% validation accuracy**.
- **CIFAR-10 (part g).** Fine-tuned DINOv2-S reaches **98.87% test accuracy**.
- **Long-tailed CIFAR-10 (part h).** Logit adjustment reaches **78.53% test accuracy** at imbalance ratio 100.

The submission file is [`submission.csv`](submission.csv). The report's LaTeX source is in [`report/overleaf/`](report/overleaf/).

There are two ways to reproduce the results:

| | What it reproduces | Needs | Time |
|---|---|---|---|
| **A. Quick check** | `submission.csv` and every result table and figure, rebuilt from the saved outputs of all training runs | Any computer with Python 3.9+, no GPU | about 5 minutes |
| **B. Retrain the best models** | The best model of each part, trained from scratch: the Dogs vs. Cats final model, the best CIFAR-10 model and the best long-tailed CIFAR-10 method | An NVIDIA GPU with CUDA, the course dataset | about 1 hour on one GPU |

---

## Repository layout

| Path | Contents |
|---|---|
| `src/train.py` | Dogs vs. Cats training: one model and one seed per run (linear probe or LP-FT, ablation flags, flip TTA, writes a `submission.csv` per run) |
| `src/models.py` | Model factory: the scratch CNN and the pretrained timm backbones (`resnet50`, `effnetv2_s`, `convnext_tiny`, `dinov2_s`, `dinov2_b`) |
| `src/data.py` | Datasets and transforms (augmentation levels, class-balanced subsets, removal of the train/val duplicate) |
| `src/classical.py` | HOG + colour histogram features with a linear and an RBF SVM (CPU only) |
| `src/cifar.py` | CIFAR-10 (part g) and long-tailed CIFAR-10 (part h) |
| `scripts/preresize.py` | One-time copy of the images with the shorter side resized to 288 px (`data_288/`) |
| `scripts/prefetch_weights.py` | Downloads all pretrained weights and CIFAR-10 once |
| `scripts/run_queue.py` | Runs a job file (one run per line) on several GPUs, one run per GPU at a time |
| `scripts/ensemble.py` | Averages the predictions of several runs and writes the ensemble's `submission.csv` |
| `scripts/make_tables.py`, `scripts/make_plots.py` | Result tables (`report/tables/`) and figures (`report/figures/`) |
| other `scripts/` | Figures for single report parts: `make_report_extras.py` (a, b), `make_aug_examples.py` (a), `model_summary.py`, `arch_table.py`, `make_arch_diagram.py` (b), `analyze_errors.py`, `make_test_cases.py`, `make_test_review.py` (e), `summarize.py` (quick overview of all runs) |
| `jobs/*.txt` | Job files: the command-line arguments of all 106 neural-network runs, one run per line |
| `runs/` | Saved outputs of all 108 runs and 2 ensembles: `args.json` (exact settings), `metrics.json` (metrics and per-epoch history), `val_predictions.csv`, `test_probs.csv` and the log file. Model checkpoints are not included because of their size |
| `report/` | Report figures (PNG and PDF), tables, layer tables, Grad-CAM maps and the LaTeX source (`report/overleaf/`) |
| `requirements.txt` | Python packages apart from PyTorch |
| `submission.csv` | Final test predictions: 500 rows, `id,label` (1 = dog, 0 = cat) |
| `CONTEXT.md`, `PLAN.md` | Project notes. `PLAN.md` is the original plan, and parts of it are out of date |

The course dataset is **not** included, because it is not ours to redistribute. Step B3 explains where to put it.

---

## A. Quick check (no GPU, about 5 minutes)

This rebuilds the final submission and all result tables and figures from the saved run outputs in `runs/`.

```bash
git clone https://github.com/Dheemant19/IE4483-Team97.git
cd IE4483-Team97
pip install numpy pandas matplotlib seaborn pillow          # only make_plots.py needs these

python scripts/ensemble.py runs/dinov2_b_probe_s0 runs/dinov2_b_probe_s1 runs/dinov2_b_probe_s2 --out check/ens_dinov2_b_probe
python scripts/make_tables.py --out check/tables
python scripts/make_plots.py --out check/figures
```

What you should see:

1. `ensemble.py` prints `ensemble of 3: val acc (TTA) 0.9988, confusion [[2495, 4], [2, 2498]]`.
2. `check/ens_dinov2_b_probe/submission.csv` is identical to `submission.csv` (259 dogs, 241 cats).
3. `check/tables/` is identical to `report/tables/` (`RESULTS.md` collects every table in one file).
4. `check/figures/` holds the result figures of the report.

`ensemble.py` and `make_tables.py` use only the Python standard library.

---

## B. Retrain the best models (NVIDIA GPU, about 1 hour)

This retrains the three results the report is built on. Retraining all 108 runs is not needed, but every other run can be repeated in the same way (see [Other experiments](#other-experiments)).

### B1. What you need

- **A Linux or Windows machine with an NVIDIA GPU and a recent NVIDIA driver.** The training scripts stop with a message if no GPU is found. Training uses bfloat16 mixed precision, so an Ampere GPU (RTX 30xx, A100) or newer is recommended. We used RTX 3090s with 24 GB of memory.
- **Python 3.9 or newer.** We used 3.11.
- **A few GB of free disk space** for the pre-resized data, the pretrained weights and CIFAR-10.

### B2. Install the software

**Step 1. Create an environment.** Either option works.

```bash
# Option 1: conda
conda create -y -n ie4483 python=3.11
conda activate ie4483

# Option 2: venv
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
```

**Step 2. Install PyTorch for your GPU.** PyTorch has a separate build for each CUDA version, so it is installed on its own. Go to <https://pytorch.org/get-started/locally/>, choose your operating system, `Pip` and a CUDA version that your driver supports (`nvidia-smi` shows the highest supported version at the top right), and run the command it gives. For example:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128   # CUDA 12.8 (what we used)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126   # CUDA 12.6
pip install torch torchvision                                                       # default build (CUDA on Linux, but CPU-only on Windows)
```

Any recent PyTorch 2.x release works. The report was produced with PyTorch 2.11.0 and CUDA 12.8. The code was also tested with PyTorch 2.15 and timm 1.0.31.

**Step 3. Install the remaining packages.**

```bash
pip install -r requirements.txt
```

**Step 4. Check the installation.**

```bash
python -c "import torch, timm; print(torch.__version__, timm.__version__, 'CUDA:', torch.cuda.is_available(), 'bf16:', torch.cuda.is_available() and torch.cuda.is_bf16_supported())"
```

This must print `CUDA: True` and `bf16: True`. If it prints `CUDA: False`, PyTorch was installed without GPU support. Repeat step 2 with a CUDA build.

**Step 5 (optional). Choose where the pretrained weights are stored.** By default they go to `~/.cache/huggingface`. To keep them elsewhere, set `HF_HOME` in every new terminal:

```bash
export HF_HOME=~/ie4483/hf_cache            # Windows PowerShell: $env:HF_HOME = "$HOME\ie4483\hf_cache"
```

### B3. Prepare the data

Place the course data in a folder `datasets/` at the top of the repository:

```
datasets/train/cat/*.jpg   (10,000 images)
datasets/train/dog/*.jpg   (10,000 images)
datasets/val/cat/*.jpg     (2,500 images)
datasets/val/dog/*.jpg     (2,500 images)
datasets/test/1.jpg ... datasets/test/500.jpg
```

Then run these two commands once:

```bash
python scripts/preresize.py --src datasets --dst data_288
python scripts/prefetch_weights.py --cifar-root ~/ie4483/cifar
```

- `preresize.py` writes a copy with the shorter side resized to 288 pixels to `data_288/`. It leaves out `val/cat/cat.12408.jpg`, an exact copy of a training image. It ends by printing the image counts, which must be 10,000, 10,000, 2,499, 2,500 and 500.
- `prefetch_weights.py` downloads the pretrained weights of every backbone and CIFAR-10. This is optional for Dogs vs. Cats, since the weights are also downloaded on first use. It is needed for the CIFAR runs (B5, B6), because `src/cifar.py` does not download CIFAR-10 itself and expects it in `~/ie4483/cifar` (change this with `--cifar-root`).

### B4. Dogs vs. Cats: final model (about 25 minutes)

The final model is an ensemble of three DINOv2-B linear probes, one per seed. **Write new runs to `runs_repro/`**, so that the saved results in `runs/` stay untouched.

```bash
python src/train.py --model dinov2_b --freeze backbone --seed 0 --name dinov2_b_probe_s0 --out runs_repro
python src/train.py --model dinov2_b --freeze backbone --seed 1 --name dinov2_b_probe_s1 --out runs_repro
python src/train.py --model dinov2_b --freeze backbone --seed 2 --name dinov2_b_probe_s2 --out runs_repro
python scripts/ensemble.py runs_repro/dinov2_b_probe_s0 runs_repro/dinov2_b_probe_s1 runs_repro/dinov2_b_probe_s2 --out runs_repro/ens_dinov2_b_probe
```

- Each run takes about 8 minutes on an RTX 3090. It writes `args.json`, `metrics.json`, `val_predictions.csv`, `test_probs.csv`, `submission.csv`, the checkpoints `last.pt` and `best.pt`, and TensorBoard logs to `runs_repro/<name>/`.
- `ensemble.py` averages the flip-TTA probabilities of the three runs and writes `runs_repro/ens_dinov2_b_probe/submission.csv`, in the format of the course's sample submission (`id,label`, 1 = dog, 0 = cat).

### B5. CIFAR-10 (part g): fine-tuned DINOv2-S (about 17 minutes)

```bash
python src/cifar.py --model dinov2_s --seed 0 --name cifar_dinov2_s_s0 --out runs_repro
```

### B6. Long-tailed CIFAR-10 (part h): ResNet-18 with logit adjustment (about 20 minutes)

```bash
python src/cifar.py --model resnet18_cifar --imb-ratio 100 --method la --seed 0 --name cifar_r18_lt100_la_s0 --out runs_repro
python src/cifar.py --model resnet18_cifar --imb-ratio 100 --method la --seed 1 --name cifar_r18_lt100_la_s1 --out runs_repro
python src/cifar.py --model resnet18_cifar --imb-ratio 100 --method la --seed 2 --name cifar_r18_lt100_la_s2 --out runs_repro
```

### B7. Compare with the report

Each run stores its results in `runs_repro/<name>/metrics.json`. The same file for the original run is in `runs/<name>/metrics.json`.

| Run | Metric in `metrics.json` | Expected value |
|---|---|---|
| `dinov2_b_probe_s0`, `_s1`, `_s2` | `val_acc_tta` (validation accuracy with flip TTA) | 0.9990, 0.9988, 0.9986 |
| `ens_dinov2_b_probe` | `val_acc` (printed by `ensemble.py`) | 0.9988, with 259 dogs in `submission.csv` |
| `cifar_dinov2_s_s0` | `test_acc` and `test_acc_tta` | 0.9887 and 0.9893 |
| `cifar_r18_lt100_la_s0`, `_s1`, `_s2` | `test_acc` | 0.7890, 0.7795, 0.7875 (mean 78.53%) |

GPU computations are not fully deterministic, so a retrained run will not match to the last digit. The Dogs vs. Cats probes (B4) differ by at most about 0.05 percentage points. The CIFAR runs (B5, B6) update the whole network and can vary more. For the ResNet-18 runs, expect results within the spread of the three seeds above rather than exactly the same values.

On a machine with several GPUs, the runs can be spread over them. Prefix each command with `CUDA_VISIBLE_DEVICES=0`, `=1`, ... (Linux and macOS shells), or use `scripts/run_queue.py` as described below.

---

## Reference

### Training options (`src/train.py`)

| Flag | What it does |
|---|---|
| `--model` | `scratch_cnn`, `resnet50`, `effnetv2_s`, `convnext_tiny`, `dinov2_s`, `dinov2_b` or any timm model name |
| `--freeze backbone` | Linear probe: only the new head is trained. Without it, the model is trained with LP-FT |
| `--lp-epochs` | Head-only epochs before fine-tuning (default 1) |
| `--lr`, `--head-lr` | Backbone and head learning rates |
| `--epochs` | Number of training epochs |
| `--aug {none,basic,default,trivial}` | Augmentation level |
| `--n-train N` | Class-balanced training subset of N images |
| `--label-smoothing` | Label smoothing amount (default 0.1) |
| `--no-pretrained` | Train a timm backbone from random weights |
| `--seed`, `--name`, `--out` | Random seed, run folder name and output folder |
| `--limit N --epochs 1` | Quick smoke test on N images |

Default settings:

- **Scratch CNN:** SGD with Nesterov momentum 0.9, one-cycle learning rate 0.05, weight decay 5e-4, batch 128, 50 epochs, 128 px.
- **Pretrained models:** AdamW, weight decay 0.05, head learning rate 1e-3, backbone learning rate 1e-4 (CNNs) or 5e-5 with layer-wise decay 0.75 (ViTs), 1 head-only epoch and 9 fine-tuning epochs, 1-epoch warm-up then cosine decay, batch 128 (64 for DINOv2-B), bfloat16. The input is 224 px (300 px for EfficientNetV2-S).
- **All models:** cross-entropy with label smoothing 0.1. The reported result is always the **last** epoch.

### CIFAR-10 options (`src/cifar.py`)

- `--model resnet18_cifar` trains a CIFAR-style ResNet-18 from scratch (200 epochs). Any `train.py` model name fine-tunes a pretrained model on images upsampled to 224 px (10 epochs).
- `--imb-ratio R` (with `--imb-type exp|step`) makes the training set long-tailed.
- `--method` picks the imbalance handling: `ce` (plain cross-entropy), `resample`, `cb` (class-balanced weights), `cb_drw` (the same weights for the last 20% of epochs), `la` (logit adjustment, τ = 1) or `focal` (focal loss, γ = 1).
- 500 images per class are held out as a balanced validation set. The official test set is evaluated once, after the last epoch.

### Other experiments

Every run in the report can be repeated with the same two training scripts. Its exact arguments are listed in the job files (one run per line) and stored in `runs/<name>/args.json`. `scripts/run_queue.py` runs a job file on the GPUs you name, one run per GPU at a time, and skips runs that already have a `metrics.json` in the output folder:

```bash
python scripts/run_queue.py jobs/week1.txt --gpus 0,1,2,3 --out runs_repro
python scripts/run_queue.py jobs/cifar_g.txt --script src/cifar.py --gpus 0,1,2,3 --out runs_repro
python src/classical.py --data data_288 --out runs_repro --jobs 8       # HOG + SVM baselines, CPU only
```

| Job file | Runs | Report part |
|---|---|---|
| `week1.txt`, `week1b.txt` | Model comparison, linear probe and LP-FT | (b), (c), (f) |
| `week2b.txt` | Seeds 1 and 2 of the ConvNeXt-T and DINOv2 probes, EfficientNetV2-S and the training-set size runs, and seed 0 of ResNet-50 and the scratch CNN | (b), (c), (f) |
| `lr_sweep.txt` | ResNet-50 learning-rate sweep | (c) |
| `week2a.txt` | Lower fine-tuning learning rate, augmentation and training-set size (seed 0) | (c), (f) |
| `cifar_g.txt` | CIFAR-10 from scratch and with pretrained models (`--script src/cifar.py`) | (g) |
| `cifar_h.txt`, `cifar_h_seeds.txt` | Long-tailed CIFAR-10 (`--script src/cifar.py`) | (h) |

The tables, figures and Grad-CAM maps can then be rebuilt from the new runs. Grad-CAM needs the checkpoints of the seed-0 DINOv2-B probe and scratch CNN, which are not included in `runs/`.

```bash
python scripts/make_tables.py --runs runs_repro --out runs_repro/tables
python scripts/make_plots.py --runs runs_repro --out runs_repro/figures
python scripts/analyze_errors.py --gradcam --probe-run runs_repro/dinov2_b_probe_s0 --scratch-run runs_repro/scratch_cnn_s0 --ens runs_repro/ens_dinov2_b_probe --out runs_repro/gradcam
```

### Reproducibility notes

- **Seeds:** every run fixes the seeds of Python, NumPy and PyTorch, which also fixes the order of the training images. The main configurations were run with seeds 0, 1 and 2 and are reported as mean ± SD. The number of seeds of every result is given as `n` in the report's tables. Class-balanced subsets always use a fixed seed (1234), so every training seed sees the same images.
- **Non-determinism:** GPU kernels are not fully deterministic, so repeating a run with the same seed is not bit-identical. Repeats differ by up to about 0.05 percentage points for the pretrained models and about 0.3 for the scratch CNN.
- **Exact settings:** `runs/<name>/args.json` stores every setting of every run, including the defaults.
- **Test sets:** the Dogs vs. Cats test images are unlabelled and were never used for any decision. The CIFAR-10 test set was evaluated once per run, after training.
- **Compute:** 108 training runs took about 15 GPU-hours on RTX 3090 GPUs.
