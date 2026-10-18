# IE4483 Mini Project (Option 2): Dogs vs. Cats, CIFAR-10 and long-tailed CIFAR-10

This README covers the code: how to set it up and run every experiment, the final submission, and every table and figure in the report. For what was done, what came out and where each result lives, read `CONTEXT.md`.

## Contents of this folder

| Path | What it is |
|---|---|
| `src/train.py` | Dogs vs. Cats training: one model and one seed per run. It handles LP-FT, the linear probe, ablation flags, last-epoch reporting, flip TTA, and writes `submission.csv` per run |
| `src/models.py` | Model factory: the scratch CNN, plus timm backbones `resnet50`, `effnetv2_s`, `convnext_tiny`, `dinov2_s`, `dinov2_b`, ... |
| `src/data.py` | Datasets and transforms (augmentation levels, class-balanced subsets, train/val duplicate exclusion) |
| `src/classical.py` | HOG + colour histogram → linear / RBF SVM baseline (CPU) |
| `src/cifar.py` | CIFAR-10 (part g) and long-tailed CIFAR-10 (part h): ResNet-18 from scratch or a pretrained backbone; imbalance methods |
| `scripts/preresize.py` | One-time copy of the images with shorter side 288 px (`data_288/`) |
| `scripts/prefetch_weights.py` | Downloads all pretrained weights and CIFAR-10 once |
| `scripts/run_queue.py` | Runs a job file (one run per line) across several GPUs, one run per GPU at a time |
| `scripts/ensemble.py` | Averages the predictions of several runs and writes the final `submission.csv` |
| `scripts/summarize.py` | Quick table of all runs (validation accuracy, mean ± SD over seeds) |
| `scripts/make_tables.py` | Final report tables → `report/tables/RESULTS.md` + CSVs |
| `scripts/make_plots.py` | Report charts → `report/figures/` |
| `scripts/analyze_errors.py` | Part (e) galleries and Grad-CAM |
| `scripts/make_arch_diagram.py` | Part (b) architecture diagrams |
| `scripts/model_summary.py` | Part (b) layer tables (input/output shape and parameters per layer) → `report/model_summaries/` |
| `scripts/make_aug_examples.py` | Part (a) figure of augmented training images |
| `scripts/figstyle.py`, `scripts/style/` | Shared figure style and colour-blind-safe palette |
| `jobs/*.txt` | Job lists used with `run_queue.py` (a few later batches were written on the server only; every run's `runs/<name>/args.json` holds its exact settings) |
| `runs/` | Results of all 108 training runs and 2 ensembles: `metrics.json`, `args.json`, `val_predictions.csv`, `test_probs.csv`, logs. Checkpoints are not included |
| `report/` | Figures (PNG at 600 dpi + PDF), draft captions, tables, layer tables, Grad-CAM maps |
| `submission_final.csv` | The submission: 500 rows, `id,label` (1 = dog, 0 = cat) |
| `PLAN.md` | The original project plan. Some parts are out of date (compute moved from EEE to MLDA); `CONTEXT.md` is the up-to-date account |

## 1. Environment

Training ran on a Linux server with 4× RTX 3090, Python 3.11 and PyTorch 2.11 (CUDA 12.8).

```bash
conda create -y -n ie4483 python=3.11 && conda activate ie4483
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install timm scikit-learn scikit-image pandas matplotlib seaborn tensorboard torchinfo grad-cam pillow
export HF_HOME=~/ie4483/hf_cache          # where pretrained weights are cached; set it in every shell
```

The figure and table scripts (`make_plots.py`, `make_tables.py`, `analyze_errors.py` without `--gradcam`, `make_arch_diagram.py`) only need numpy, pandas, matplotlib, seaborn and Pillow, so they also run on a laptop without a GPU.

## 2. Data

Put the course data in `datasets/`:
- `datasets/train/{cat,dog}/*.jpg` (10,000 each);
- `datasets/val/{cat,dog}/*.jpg` (2,500 each);
- `datasets/test/1.jpg … 500.jpg`.

Then run these once:

```bash
python scripts/preresize.py --src datasets --dst data_288    # shorter side 288 px; drops the train/val duplicate
python scripts/prefetch_weights.py --cifar-root ~/ie4483/cifar
```

`val/cat/cat.12408.jpg` is an exact copy of `train/cat/cat.2339.jpg`. It is excluded, so validation has 4,999 images.

## 3. Dogs vs. Cats

One run means one model and one seed. Output goes to `runs/<name>/`.

```bash
CUDA_VISIBLE_DEVICES=0 python src/train.py --model resnet50 --seed 0                          # ResNet-50, LP-FT
CUDA_VISIBLE_DEVICES=0 python src/train.py --model dinov2_b --freeze backbone --seed 0 --name dinov2_b_probe_s0   # final model type
CUDA_VISIBLE_DEVICES=0 python src/train.py --model scratch_cnn --seed 0                       # scratch CNN, 50 epochs
CUDA_VISIBLE_DEVICES=0 python src/train.py --model resnet50 --limit 512 --epochs 1 --name smoke   # 1-minute smoke test
```

**Useful flags:**

| Flag | What it does |
|---|---|
| `--freeze backbone` | Linear probe: only the head is trained |
| `--lp-epochs` | Number of head-only epochs before fine-tuning (default 1) |
| `--lr`, `--head-lr` | Backbone and head learning rates |
| `--epochs` | Number of training epochs |
| `--aug {none,basic,default,trivial}` | Augmentation level |
| `--n-train N` | Class-balanced training subset of N images |
| `--img-size` | Input size |
| `--label-smoothing` | Label smoothing amount |
| `--no-pretrained` | Train a timm backbone from random weights |
| `--select {last,best}` | Which epoch is reported (default `last`) |
| `--seed` | Random seed |
| `--name` | Run folder name |

Every run writes:
- `metrics.json`, with validation accuracy (last epoch), flip-TTA accuracy, best/head-only-epoch accuracy, confusion matrix and per-epoch history;
- `val_predictions.csv`, `test_probs.csv`, `submission.csv`;
- `last.pt` (resumable), `best.pt`, and TensorBoard logs.

**Default recipes (from `PLAN.md` §8.2):**
- **Scratch CNN:** SGD (Nesterov 0.9), one-cycle LR 0.05, weight decay 5e-4, batch 128, 50 epochs, 128 px.
- **Pretrained CNNs:** AdamW, weight decay 0.05, head LR 1e-3, backbone LR 1e-4, 1 head-only epoch + 9 fine-tuning epochs, 1-epoch warm-up then cosine decay, batch 128, 224 px, bf16.
- **ViTs:** backbone LR 5e-5 with layer-wise decay 0.75, batch 64 for ViT-B.
- **All models:** cross-entropy with label smoothing 0.1.

**Classical baseline (CPU):** `python src/classical.py --data data_288 --jobs 8`

**Many runs at once**, using the job files that produced the results:

```bash
nohup python scripts/run_queue.py jobs/week1.txt --gpus 0,1,2,3 > runs/queue.log 2>&1 &
```

Runs that already have `metrics.json` are skipped, so a queue can be re-run safely. `--skip-busy` shares GPUs with runs that are already going.

**Final submission:** average the three seeds of the DINOv2-B linear probe.

```bash
python scripts/ensemble.py runs/dinov2_b_probe_s0 runs/dinov2_b_probe_s1 runs/dinov2_b_probe_s2 --out runs/ens_dinov2_b_probe
cp runs/ens_dinov2_b_probe/submission.csv submission_final.csv
```

## 4. CIFAR-10 (parts g and h)

The whole dataset is held on the GPU and augmented there. 500 images per class are held out as a balanced validation set. The official 10,000-image test set is evaluated once, after the last epoch.

```bash
python src/cifar.py --model resnet18_cifar --seed 0 --name cifar_r18_s0                     # from scratch, 200 epochs
python src/cifar.py --model dinov2_s --seed 0 --name cifar_dinov2_s_s0                      # transfer, 224 px, 10 epochs
python src/cifar.py --model resnet18_cifar --imb-ratio 100 --method la --seed 0 --name cifar_r18_lt100_la_s0
nohup python scripts/run_queue.py jobs/cifar_h.txt --script src/cifar.py --gpus 0,1,2,3 > runs/queue_cifar.log 2>&1 &
```

**Imbalance options:**
- `--imb-type exp|step` and `--imb-ratio R` set the long-tailed class profile.
- `--method` chooses the handling:
  - `ce`: plain cross-entropy (baseline);
  - `resample`: class-balanced re-sampling;
  - `cb`: class-balanced loss weights;
  - `cb_drw`: the same weights, switched on for the last 20% of epochs;
  - `la`: logit-adjusted loss (τ = 1);
  - `focal`: focal loss (γ = 1).

## 5. Tables and figures

```bash
python scripts/summarize.py                     # quick overview (validation accuracy)
python scripts/make_tables.py                   # report/tables/RESULTS.md + CSVs (CIFAR uses test accuracy)
python scripts/make_plots.py                    # report/figures/*.png|pdf + captions.md + data/*.csv
python scripts/analyze_errors.py                # part (e) galleries
python scripts/analyze_errors.py --gradcam --out report/gradcam                 # on the GPU server: Grad-CAM maps
python scripts/analyze_errors.py --cams-from report/gradcam/gradcam_maps.npz    # laptop: redraw Grad-CAM figure
python scripts/make_arch_diagram.py             # part (b) diagrams (reads report/model_summaries/stages.csv)
python scripts/model_summary.py                 # part (b) layer tables (needs torch, timm, torchinfo; CPU is fine)
python scripts/make_aug_examples.py             # part (a) augmentation figure (needs torch + torchvision; CPU)
```

## 6. Reproducibility

- **Seeds:** every reported configuration was run with seeds 0, 1 and 2, and is reported as mean ± SD. The seed fixes Python, NumPy, PyTorch and the data-loader order. Class-balanced subsets use a separate fixed seed (1234), so every training seed sees the same images.
- **Non-determinism:** `cudnn.benchmark` and GPU non-determinism are on. Repeating a seed is therefore not bit-identical. Repeats differ by up to ~0.3 points for the scratch CNN and ≤ 0.05 points for pretrained models, which is why results are given as mean ± SD rather than single runs.
- **Reported epoch:** the reported number is always the **last** epoch. `best_val_acc` is kept in `metrics.json` only for reference.
- **Test sets:** the Dogs-vs-Cats test images are unlabelled and were never used for any decision. The CIFAR-10 test set was evaluated once per run, after training.
- **Compute:** 108 training runs, about 15 GPU-hours of training in total on RTX 3090s.
