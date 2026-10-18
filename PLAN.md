# IE4483 Mini Project (Option 2): Dogs vs. Cats, project plan

Prepared 17 Oct 2026. Report and `submission.csv` are due **Fri 13 Nov 2026, 11:59 pm** on NTULearn (group of three, contributions must be stated).

**Contents**

1. [TL;DR](#1-tldr)
2. [What the brief asks for](#2-what-the-brief-asks-for)
3. [What the dataset looks like](#3-what-the-dataset-looks-like)
4. [Compute choice](#4-compute-choice)
5. [Preprocessing and augmentation](#5-preprocessing-and-augmentation)
6. [Models to compare](#6-models-to-compare)
7. [Models to download and where from](#7-models-to-download-and-where-from)
8. [Training plan](#8-training-plan)
9. [CIFAR-10 and class imbalance (parts g, h)](#9-cifar-10-and-class-imbalance-parts-g-h)
10. [How to start](#10-how-to-start)
11. [Team split and timeline](#11-team-split-and-timeline)
12. [References](#12-references)

---

## 1. TL;DR

- **Compute (updated 17 Oct):** use the **MLDA server `gpu21` (4× RTX 3090, 24 GB each)**. On 17 Oct every EEE Cluster 02 GPU type was fully allocated, and the earliest start for a 1-GPU job was 21 Oct. The whole project needs roughly 20 to 40 GPU-hours. Produce every reported number on the 3090s so results come from one environment. EEE (`a40`) stays the fallback; see §4.
- **Data:** use all of it: 20,000 train images (10k cat, 10k dog) and 5,000 val (2.5k each), with 500 unlabeled test images. All files are readable RGB JPEGs, about 400×370 px on average. There is one train/val duplicate to remove.
- **Models (comparison ladder):** HOG + SVM (classical) → small CNN from scratch → frozen ResNet-50 + linear head → fine-tuned ResNet-50, EfficientNetV2-S, ConvNeXt-Tiny and DINOv2 ViT-S/B. The final submission is the best fine-tuned model (most likely ConvNeXt-Tiny or DINOv2 ViT-B), with flip test-time augmentation.
- **Where weights come from:** all pretrained weights are pulled by the `timm` library from the Hugging Face Hub (`huggingface.co/timm/...`). Download them once on the cluster login node, because compute nodes may not have internet access.
- **First three things to do:** (1) set up the Python environment, (2) copy the dataset to the cluster and run the pre-resize step, (3) get the scratch CNN and the ResNet-50 baseline training end to end, including writing `submission.csv`.

## 2. What the brief asks for

| Part | Marks | What you must deliver | Where this plan covers it |
|---|---|---|---|
| a | 10% | Amount of train/test data, preprocessing, augmentation | §3, §5 |
| b | 20% | At least one model; architecture figure, I/O dims, loss, training strategy, code + run instructions, reproducibility (seeds / multiple runs) | §6, §8, §10 |
| c | 10% | How you chose hyperparameters (learning rate etc.) and why | §8.3 |
| d | 20% | Validation accuracy + `submission.csv` for the 500 test images (`id,label`, 1 = dog, 0 = cat) | §8.5 |
| e | 10% | Analyse correct and incorrect test samples; 1 to 2 cases on strengths and weaknesses | §8.6 |
| f | 10% | How model choice and data processing affect validation accuracy | §8.4 (ablation table) |
| g | 10% | Apply and improve the method on CIFAR-10; report test accuracy | §9.1 |
| h | 10% | CIFAR-10 with some classes heavily under-sampled; at least 2 imbalance methods, justified | §9.2 |

Rules from the brief that affect the plan: optimise only on `train/`, use `val/` only for validation and model selection, conventional classifiers are welcome as a comparison, and neural networks (scratch CNNs or pretrained backbones like VGG and ResNet) are encouraged.

## 3. What the dataset looks like

Summarised with [`scripts/dataset_summary.py`](scripts/dataset_summary.py) (full output in `scripts/dataset_summary.json`). It reads every file once, decodes it fully with Pillow and hashes it, so no images were opened by hand.

| Split | Count | Format / mode | Width (median, p5–p95) | Height (median, p5–p95) | Short side < 100 px | Unreadable |
|---|---|---|---|---|---|---|
| train/cat | 10,000 | JPEG / RGB | 473 (198–500) | 374 (179–500) | 77 | 0 |
| train/dog | 10,000 | JPEG / RGB | 423 (186–500) | 374 (187–500) | 83 | 0 |
| val/cat | 2,500 | JPEG / RGB | 466 (192–500) | 374 (179–500) | 22 | 0 |
| val/dog | 2,500 | JPEG / RGB | 423 (186–500) | 374 (185–500) | 21 | 0 |
| test | 500 | JPEG / RGB | 437 (176–500) | 374 (169–500) | 5 | 0 |

What this means for the work:

- **Classes are perfectly balanced**, so accuracy is a fair headline metric. Still report a confusion matrix.
- **No corrupt files and no grayscale/CMYK images**, so no cleaning step is needed. Keep `.convert("RGB")` in the loader anyway for safety.
- **Sizes vary a lot** (32 px up to 1050 px, aspect ratio 0.31 to 5.9). Use resize + crop rather than a plain squash to a square, so the animals are not distorted.
- **Duplicates (exact MD5 matches):** `train/cat/cat.2339.jpg` is identical to `val/cat/cat.12408.jpg`. That is a small train-to-val leak, so drop the val copy and mention it in part (a). There are also 4 duplicate sets inside train (for example `cat.10360` = `cat.3345`); they are harmless and can stay.
- **Tiny images:** about 200 images have a short side under 100 px. Keep them, since upscaling is fine, but they are good candidates for the failure analysis in part (e).
- **Test file names are `1.jpg` … `500.jpg`.** Sort them **numerically** when writing `submission.csv`. A plain string sort gives 1, 10, 100, … and scrambles the rows.
- **Label mapping:** `torchvision.datasets.ImageFolder` sorts class folders alphabetically, which gives `cat = 0, dog = 1`. That already matches the required submission encoding.
- The Kaggle source data is known to contain a few mislabeled or non-pet images. That caps achievable accuracy slightly below 100% and gives material for part (e).

## 4. Compute choice

**Decision (updated 17 Oct): MLDA `gpu21`, with EEE Cluster 02 (`a40`, account `ug-proj`, QOS `ug`) as the fallback.**

What the live check on 17 Oct found:

- **MLDA `gpu21`** (`ssh s126mdg21_01@gpu21.dynip.ntu.edu.sg`, NTU network or VPN required):
  - Hardware: 4× RTX 3090 24 GB, all idle; 36 CPU cores; 251 GB RAM.
  - Software: Ubuntu 22.04, driver CUDA 12.8.
  - Storage: home on `/usr1` with 1.8 TB free.
  - No scheduler. Pick a free GPU with `nvidia-smi` and `CUDA_VISIBLE_DEVICES`, and run inside `screen` or `tmux`.
  - Environment: Miniforge in `~/miniforge3`, env `ie4483`, PyTorch cu128 wheels.
  - Project folder: `~/ie4483`.
- **EEE Cluster 02**:
  - Every `a40`, `l40`, `6000ada`, `pro6000` and `rtx5090` GPU was allocated. Only one `a6000` slot and the ARM `gh200` node were free.
  - `sbatch --test-only` gave the earliest 1-GPU start as 21 Oct.
  - Project storage: `/projects/dheemantmlprojs` (50 GB SSD shared with other projects, plus 300 GB HDD).

The original EEE reasoning is kept below for reference.

Why this option:

| | Laptop RTX 5060 (8 GB) | MLDA 4× RTX 3090 (24 GB) | EEE Cluster 02 (`a40`, 48 GB) |
|---|---|---|---|
| Enough VRAM for every model here | Yes for CNNs with mixed precision; tight for ViT-B with large batches | Yes | Yes, with lots of headroom |
| Budget / rules known | Free, but ties up your laptop and throttles thermally on long runs | No stated quota; shared with other MLDA users | 178,771 SU left this month (resets 1 Nov), ~993 A40-hours |
| Parallel runs | 1 | Depends on availability | 2 jobs at once, up to 2 GPUs each; 0-SU "killable" QOS for up to 8 GPUs |
| Long runs | Laptop must stay on | Unknown | Batch jobs up to 3 days |
| Software risk | Blackwell GPU (sm_120) needs a CUDA 12.8+ PyTorch build | Ampere, works with any build | Ampere, works with any build |
| Shared with teammates | No | Possibly | Results and checkpoints live in one place |

Why A40 rather than the other EEE GPUs: these models are small, so a faster GPU mostly waits on data loading (only 4 CPU cores per GPU). The A40 is the cheapest per hour at 180 SU and has the same 48 GB as the L40 and 6000 Ada. If the A40 queue is long, `a6000` is the same price and `l40` / `6000ada` (240 SU) are the next choice. The RTX 5090 (360 SU) and Pro 6000 (480 SU) are not worth the cost here.

**Budget estimate** (A40, mixed precision, images pre-resized; rough figures):

| Experiment block | Runs | GPU-hours |
|---|---|---|
| Classical HOG + SVM | 1 to 2 (CPU only) | 0 |
| Scratch CNN, 3 seeds + small sweep | ~8 | ~3 |
| Linear probes (frozen backbones) | ~5 | ~1 |
| Fine-tuning 5 backbones × 3 seeds | 15 | ~6 |
| LR / augmentation / resolution sweeps (one backbone) | ~20 | ~6 |
| CIFAR-10 (scratch ResNet-18 + fine-tuned backbone, 3 seeds) | ~6 | ~8 |
| CIFAR-10 imbalance experiments | ~10 | ~6 |
| Slack for reruns | | ~10 |
| **Total** | | **~40 h ≈ 7,200 SU** |

Use the normal `ug` QOS for anything you will report. Use `--qos=override-limits-but-killable` (0 SU) for large sweeps only, and only if the training script saves a checkpoint every epoch and resumes from it, because those jobs can be requeued.

The laptop's job is writing code and running quick smoke tests on a few hundred images. If you do want it to run PyTorch on the GPU, install a build compiled for CUDA 12.8 or newer; older builds do not support RTX 50-series (sm_120) cards.

## 5. Preprocessing and augmentation

### 5.1 Pre-resize: measured, kept as the default

On 17 Oct we timed decoding plus resizing to 224×224 for 2,000 train/cat images on one MLDA core:

| | Original `datasets/` | Pre-resized `data_288/` | Speed-up |
|---|---|---|---|
| Cold (first read from disk) | 110 img/s | 127 img/s | 1.15× |
| Warm (files cached in RAM) | 426 img/s | 586 img/s | 1.38× |

After the first epoch every read is warm (the dataset is 1.2 GB; gpu21 has 251 GB RAM), so the relevant figure is **1.38× faster decoding**.

**Decision: keep the pre-resized copy `data_288/` as the default** (`--data data_288` in `train.py` and `classical.py`). It mainly helps the scratch CNN and four runs at once. The originals stay usable with `--data datasets`.

How `data_288/` is made (`scripts/preresize.py`, already done on gpu21):

- Shorter side resized to 288 px, aspect ratio kept, tiny images never upscaled, saved as JPEG quality 95.
- 288 leaves room for random crops of 224 (and 256 for some ViTs). The scratch CNN resizes again to 128 on the fly.
- For a 288 px input-resolution ablation, use `--data datasets` (or a `--short 320` copy) so crops are not upscaled.

The train/val duplicate `val/cat/cat.12408.jpg` is dropped by `preresize.py`. It is also excluded in the data loader (`src/data.py`, `EXCLUDE`), so `--data datasets` gives the same 4,999 val images.

### 5.2 Per-model normalisation

Always take input size, mean and std from the model itself, because they differ between models (for example EfficientNetV2-S uses mean/std 0.5, while ResNet, ConvNeXt and DINOv2 use the ImageNet values):

```python
cfg = timm.data.resolve_data_config({}, model=model)   # input_size, mean, std, crop_pct, interpolation
```

### 5.3 Train-time augmentation (pretrained models)

| Transform | Setting | Why |
|---|---|---|
| `RandomResizedCrop(224)` | `scale=(0.5, 1.0)`, `ratio=(3/4, 4/3)` | Scale/translation invariance. A milder scale range than the ImageNet default (0.08) avoids crops that cut the animal out entirely |
| `RandomHorizontalFlip` | p = 0.5 | Cats and dogs are left/right symmetric |
| `RandomRotation` | ±15° | Covered in class; handles tilted photos. Avoid vertical flips, since upside-down pets are rare |
| `ColorJitter` | brightness/contrast/saturation 0.2, hue 0.02 | Lighting and camera differences |
| `TrivialAugmentWide` *or* `RandAugment(n=2, m=9)` | Ablation only | Stronger augmentation, tested in §8.4 |
| `RandomErasing` | p = 0.25 | Occlusion robustness (applied after normalisation) |
| Mixup / CutMix | α = 0.2 / 1.0 | Ablation only; regularises the large models |
| Label smoothing | 0.1 | Cheap regulariser that also makes probabilities better calibrated |

### 5.4 Eval-time preprocessing

`Resize(int(224 / crop_pct))` (≈ 256 for crop_pct 0.875) → `CenterCrop(224)` → normalise. For the final submission, add **test-time augmentation**: average the softmax of the original and the horizontally flipped image.

### 5.5 Scratch CNN and classical baseline

- Scratch CNN: 128×128 input, the same augmentations but a wider crop scale (0.6 to 1.0), and statistics computed from the training set.
- HOG + SVM: grayscale, resize to 128×128, HOG (9 orientations, 8×8 cells, 2×2 blocks) concatenated with a 3×16-bin colour histogram, then `StandardScaler` → `LinearSVC` or an RBF SVM. You can also fit `PCA(256)` first to speed up the RBF SVM.

## 6. Models to compare

The goal is a ladder that shows how much each ingredient (learned features, pretraining, pretraining data scale, architecture) adds. That ladder directly answers parts (b), (c) and (f).

| # | Model | Type | Params | Input | Role in the report | Rough expected val acc.\* |
|---|---|---|---|---|---|---|
| M0 | HOG + colour histogram → SVM | Conventional | n/a | 128² | "Conventional classifier" comparison the brief mentions | 70–80% |
| M1 | Small CNN from scratch (5 conv blocks: Conv-BN-ReLU ×2 + MaxPool, channels 32→256, GAP, dropout 0.3, FC→2) | CNN, random init | ~1–2 M | 128² | Shows what you get with no pretraining | 88–94% |
| M2 | ResNet-50, frozen, + logistic regression on 2048-d features | Linear probe | 25.6 M (frozen) | 224² | "Pretrained feature extractor + classifier layer", as suggested in the brief | 97–98.5% |
| M3 | ResNet-50 fine-tuned | Transfer learning | 25.6 M | 224² | Brief's reference architecture [3] | 98.5–99% |
| M4 | VGG16-BN fine-tuned *(optional)* | Transfer learning | 138 M | 224² | Brief's other reference [2]; shows that parameter count is not accuracy | ~98% |
| M5 | EfficientNetV2-S fine-tuned (ImageNet-21k → 1k) | Transfer learning | 21.5 M | 300² | Efficient modern CNN | ~99% |
| M6 | **ConvNeXt-Tiny fine-tuned (ImageNet-22k → 1k)** | Transfer learning | 28.6 M | 224² | Strong modern CNN; **main candidate** | 99%+ |
| M7 | DINOv2 ViT-S/14 and ViT-B/14: linear probe, then fine-tune | Self-supervised ViT | 22 M / 86 M | 224² | Transformer comparison; DINOv2 features are very strong even frozen | 99%+ |
| M8 | Ensemble of the best 2–3 + flip TTA | Ensemble | n/a | n/a | Final `submission.csv` if it beats the best single model on val | ≥ best single |

\* Rough ranges from the Kaggle competition and the transfer-learning literature, to set expectations only. Your measured numbers go in the report.

Point worth making in the report: ImageNet already contains many cat and dog breed classes, so ImageNet-pretrained backbones start out very close to solving this task. That is why the jump from M1 to M2 is so large, and it is a good discussion point for part (f).

## 7. Models to download and where from

Install `timm` (`pip install timm`). Every model below is downloaded automatically from the Hugging Face Hub the first time you call `timm.create_model(name, pretrained=True)`. No account or token is needed for these.

| Model | `timm` name (pass to `create_model`) | Download page |
|---|---|---|
| ResNet-50 | `resnet50.a1_in1k` | https://huggingface.co/timm/resnet50.a1_in1k |
| ResNet-18 *(CIFAR-10 fine-tune, fast experiments)* | `resnet18.a1_in1k` | https://huggingface.co/timm/resnet18.a1_in1k |
| VGG16-BN *(optional)* | `vgg16_bn.tv_in1k` | https://huggingface.co/timm/vgg16_bn.tv_in1k |
| EfficientNetV2-S | `tf_efficientnetv2_s.in21k_ft_in1k` | https://huggingface.co/timm/tf_efficientnetv2_s.in21k_ft_in1k |
| ConvNeXt-Tiny | `convnext_tiny.fb_in22k_ft_in1k` | https://huggingface.co/timm/convnext_tiny.fb_in22k_ft_in1k |
| DINOv2 ViT-S/14 | `vit_small_patch14_dinov2.lvd142m` | https://huggingface.co/timm/vit_small_patch14_dinov2.lvd142m |
| DINOv2 ViT-B/14 | `vit_base_patch14_dinov2.lvd142m` | https://huggingface.co/timm/vit_base_patch14_dinov2.lvd142m |
| DINOv3 ViT-S/16 *(optional, newer)* | `vit_small_patch16_dinov3.lvd1689m` | https://huggingface.co/timm/vit_small_patch16_dinov3.lvd1689m (may ask you to accept Meta's licence on the page first) |

Notes:

- DINOv2 models default to 518 px input. Create them at 224 with `timm.create_model(name, pretrained=True, num_classes=2, img_size=224)`; timm interpolates the position embeddings for you.
- `num_classes=2` replaces the 1000-class ImageNet head with a fresh 2-way linear layer. Use `num_classes=0` to get a pure feature extractor for the linear probes (M2, M7-probe).
- **CIFAR-10** (for parts g, h): `torchvision.datasets.CIFAR10(root, download=True)`, or the tarball from https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz.
- **Software:** PyTorch + torchvision from https://pytorch.org/get-started/locally/ (pick a CUDA 12.x build for the cluster; CUDA 12.8 or newer if you ever train on the RTX 5060). Also `timm`, `scikit-learn`, `scikit-image` (HOG), `pandas`, `matplotlib`, `tensorboard`, `torchinfo` (layer table and I/O shapes for part b) and `grad-cam` (heat maps for part e).

**Pre-download on the login node** (compute nodes may have no internet):

```bash
export HF_HOME=$HOME/IE4483/hf_cache          # also set this in every job script
python - <<'EOF'
import timm
for n in ["resnet18.a1_in1k", "resnet50.a1_in1k", "vgg16_bn.tv_in1k",
          "tf_efficientnetv2_s.in21k_ft_in1k", "convnext_tiny.fb_in22k_ft_in1k",
          "vit_small_patch14_dinov2.lvd142m", "vit_base_patch14_dinov2.lvd142m"]:
    timm.create_model(n, pretrained=True); print("ok", n)
EOF
python -c "import torchvision; torchvision.datasets.CIFAR10('$HOME/IE4483/cifar', download=True)"
```

In job scripts, set `HF_HUB_OFFLINE=1` so nothing tries to reach the internet.

## 8. Training plan

### 8.1 Data usage

- **Training:** all 20,000 images in `train/` (10k per class).
- **Validation:** all 5,000 in `val/` minus the one duplicate, so 4,999. Use it for early stopping, model selection and every number in the report.
- **Test:** 500 images, predicted once by the final model.
- Do not tune on test images. Since the test set is unlabeled, the val set is your only honest estimate.

### 8.2 Default recipe (fine-tuning M3, M5–M7)

| Setting | Value | Reason |
|---|---|---|
| Optimiser | AdamW, weight decay 0.05 (CNNs) / 0.05–0.1 (ViTs) | Standard for fine-tuning modern backbones |
| Learning rate | Head 1e-3; backbone 1e-4 (CNNs) or 3e-5 to 5e-5 with layer-wise decay 0.75 (ViTs) | Pretrained layers need small updates; a new head needs larger ones |
| Schedule | 1 epoch linear warm-up, then cosine decay to 0 | Avoids wrecking pretrained features in the first steps |
| Epochs | 10 to 15 (fine-tune); 40 to 60 (scratch CNN, with one-cycle or cosine) | Val accuracy usually plateaus within ~5 epochs for pretrained models |
| Batch size | 64 (ViT-B) to 128 (CNNs) | Fits on 48 GB with room to spare |
| Precision | `torch.autocast(dtype=torch.bfloat16)`, `channels_last` for CNNs | About 2× faster on the A40 with no accuracy loss |
| Loss | Cross-entropy with label smoothing 0.1 (2 logits + softmax) | Same as binary cross-entropy in practice, but extends directly to CIFAR-10 |
| Strategy | **LP-FT**: 1 epoch with the backbone frozen (train the head only), then unfreeze everything | Reduces feature distortion and is usually more stable [Kumar et al. 2022] |
| Checkpointing | Keep the best-val-accuracy epoch; save a resumable checkpoint every epoch | Model selection, plus survival of requeues |
| Seeds | 0, 1, 2 for every reported configuration; report mean ± std | The brief asks for reproducibility. Fix `torch`, `numpy` and `random` seeds and the DataLoader `generator` |
| Logging | TensorBoard (train/val loss, accuracy, LR) | Training curves are useful figures for parts b and c |

Scratch CNN (M1): SGD with momentum 0.9 and Nesterov, LR 0.05 with one-cycle schedule, weight decay 5e-4, batch 128, 50 epochs, He (Kaiming) initialisation. Compare He vs default init once, since the brief lists "initialization" as a parameter to try.

### 8.3 Choosing hyperparameters (part c)

Run these on ConvNeXt-Tiny (M6) and the scratch CNN (M1), 5 epochs each, seed 0, and pick by val accuracy. Then confirm the winner with 3 seeds.

1. **Learning-rate sweep:** backbone LR ∈ {1e-5, 3e-5, 1e-4, 3e-4, 1e-3}. Plot final val accuracy against LR on a log axis; it is usually a clear U-shape and makes a good figure. Optionally add a quick LR-range test (LR increased exponentially over one epoch).
2. **Batch size:** {32, 64, 128}, scaling LR linearly with batch size.
3. **Freeze depth:** head only vs last stage vs full fine-tune.
4. **Input resolution:** 128 vs 224 vs 288.
5. **Epochs / early stopping:** read off from the val curve of the best config.

### 8.4 Ablations (part f)

One table, each row changing one thing from the best configuration, all at 3 seeds:

| Ablation | Variants |
|---|---|
| Model family | M0 → M1 → M2 → M3 → M5 → M6 → M7 |
| Pretraining | Random init vs ImageNet-1k vs ImageNet-22k vs self-supervised (DINOv2), same architecture where possible |
| Augmentation | None → flip + crop → + rotation + colour jitter → + RandAugment/TrivialAugment → + Mixup/CutMix |
| Resolution | 128 / 224 / 288 |
| Training data size | 2k / 5k / 10k / 20k training images (shows the value of the full dataset; the brief's minimum hint is 2k) |
| Test-time augmentation | Off vs horizontal-flip TTA |
| Fine-tuning strategy | Linear probe (`--freeze backbone`) vs LP-FT (default) vs full fine-tune from the start (`--lp-epochs 0`) |
| Regularisation (cheap extras) | Label smoothing 0 vs 0.1; Mixup/CutMix on vs off |
| Training-label cleanup | Find the hardest and most confidently wrong train images, inspect them, drop the ones that are mislabeled or not pets, and retrain. This changes `train/` only, never val, and the inspected images also feed part (e) |

**Where each ablation shows a real effect (added 17 Oct).** Pretrained models will all score around 99%+ on 4,999 val images. One standard error at that level is about ±0.15%, roughly 7 images. Augmentation or label-smoothing changes on ConvNeXt or DINOv2 will therefore mostly fall within seed noise. Run each ablation where it can show a difference:

- **Augmentation and initialisation:** mainly on the scratch CNN (M1). There they change accuracy by whole percentage points. Run them on one pretrained model only as a confirmation.
- **Data size (headline figure for part f):** a learning curve of the scratch CNN vs ResNet-50 vs ConvNeXt-Tiny at `--n-train 2000/5000/10000/20000`. It shows that pretraining matters most with little data.
- **Model family, pretraining, fine-tuning strategy and resolution:** compare on the pretrained models as listed.
- **Reporting:** for every row, report mean ± std over 3 seeds. Call a difference real only if it is clearly larger than the seed spread.

### 8.5 Final model and submission (part d)

1. Choose the best configuration by mean val accuracy over 3 seeds.
2. Optionally average the softmax outputs of the 2–3 best models (M8) and keep the ensemble only if it beats the best single model on val.
3. Predict the 500 test images with flip TTA, sort by numeric id, and write:

```python
import pandas as pd
df = pd.DataFrame({"id": ids, "label": preds})        # label: 1 = dog, 0 = cat
df.sort_values("id").to_csv("submission.csv", index=False)
```

4. Check the file against `sampleSubmission.csv`: 500 rows, header `id,label`, ids 1 to 500.

### 8.6 Error analysis (part e)

- Confusion matrix and per-class accuracy on val.
- List the most confident wrong predictions and the least confident right predictions on val and on test (by eye for test).
- Grad-CAM heat maps (`pip install grad-cam`) on 2–4 test images to show what the model looks at. Typical failure cases are tiny or low-resolution images, both animals in one photo, heavy occlusion, cartoons/drawings and mislabeled images.

### 8.7 Architecture figure (part b)

`torchinfo.summary(model, input_size=(1, 3, 224, 224))` gives the layer table and input/output shapes. Draw a block diagram (draw.io or PowerPoint is fine) of: input 3×224×224 → backbone stages → global average pool → 768-d (ConvNeXt-T) / 2048-d (ResNet-50) feature → linear → 2 logits → softmax.

## 9. CIFAR-10 and class imbalance (parts g, h)

### 9.1 CIFAR-10 (part g)

- 60,000 32×32 colour images in 10 classes (50k train, 10k test). The whole dataset fits in GPU memory.
- What changes from Dogs vs. Cats: a 10-way softmax instead of 2; much smaller images; train with `RandomCrop(32, padding=4)` + horizontal flip (+ Cutout/RandomErasing) instead of RandomResizedCrop; and the official test set is labeled, so you report real test accuracy. Hold out 5k of the training set as validation for tuning.
- **Approach A, from scratch:** a CIFAR-style ResNet-18 (3×3 stem, no max-pool), SGD with momentum 0.9, LR 0.1, cosine, weight decay 5e-4, batch 128, 100 to 200 epochs. Expect roughly 94–95%.
- **Approach B, transfer:** the Dogs vs. Cats best backbone (ConvNeXt-Tiny or DINOv2 ViT-S), inputs upsampled to 224, fine-tuned for 10 epochs. Expect roughly 97–98%+. This is the "improve your algorithm" story: same pipeline, new head, adjusted resolution and augmentation.

### 9.2 Imbalanced CIFAR-10 (part h)

Build the imbalanced training set yourself and keep the balanced 10k test set. Two common setups; pick one and state it:

- **Long-tailed CIFAR-10 (standard in the literature):** class sizes decay exponentially from 5,000 down to 50 (imbalance ratio 100), or to 500 (ratio 10).
- **Step imbalance:** keep 5 classes at 5,000 images and cut the other 5 to 500.

Methods to compare (the brief asks for at least two; run all against a plain cross-entropy baseline):

| Method | Idea | Implementation effort |
|---|---|---|
| Re-sampling | `WeightedRandomSampler` so every class is drawn equally often | A few lines |
| Class-balanced re-weighting | Loss weight ∝ (1 − β)/(1 − βⁿ) per class, "effective number of samples" [Cui et al. 2019], or focal loss [Lin et al. 2017] | A few lines |
| **Logit adjustment** | Add τ·log(class prior) to the logits during training [Menon et al. 2021]; simple and strong | A few lines |
| LDAM + deferred re-weighting (DRW) | Larger margins for rare classes; re-weight only in the last part of training [Cao et al. 2019] | Moderate |
| Augmentation / transfer | Stronger augmentation or Mixup for minority classes; a pretrained backbone is far less hurt by imbalance | Reuses existing code |

**Suggested pick:** logit-adjusted loss + class-balanced re-weighting with DRW, with re-sampling as a third if time allows. Report overall accuracy, per-class accuracy (minority vs majority) and macro-F1, because overall accuracy hides minority-class failures.

## 10. How to start

### 10.1 Folder layout (suggested)

```
IE4483/
├── datasets/                 # raw data (already here)
├── data_288/                 # pre-resized copy, default --data (created by scripts/preresize.py)
├── scripts/
│   ├── dataset_summary.py    # already here
│   ├── preresize.py
│   └── prefetch_weights.py
├── src/
│   ├── data.py               # datasets, transforms
│   ├── models.py             # model factory (timm names, scratch CNN)
│   ├── train.py              # train/val loop, checkpoints, seeds, TensorBoard
│   ├── predict.py            # test predictions + TTA → submission.csv
│   ├── classical.py          # HOG + SVM baseline
│   └── cifar.py              # parts g and h
├── configs/                  # one YAML per experiment
├── slurm/                    # job scripts
├── runs/                     # logs, checkpoints, metrics.json per run
└── report/
```

Use a shared Git repo (GitHub, private) so all three of you work on the same code; keep `datasets/`, `data_288/` and `runs/` out of Git.

### 10.2 Day-one checklist

1. **Laptop environment** (code editing and smoke tests):
   ```bash
   python -m venv .venv && .venv\Scripts\activate          # Windows
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
   pip install timm scikit-learn scikit-image pandas matplotlib tensorboard torchinfo grad-cam pyyaml
   python -c "import torch; print(torch.cuda.get_device_name(0), torch.cuda.get_arch_list())"   # should list sm_120
   ```
   Check https://pytorch.org/get-started/locally/ for the current CUDA 12.8+ wheel index if `cu128` has moved on. On Windows, keep DataLoader code under `if __name__ == "__main__":`.
2. **Cluster environment:** log in, create a conda or venv environment with the same packages (any CUDA 12.x build works on the A40), then run the pre-download block from §7.
3. **Copy the data:** zip `datasets/` on the laptop and `scp datasets.zip dheemant001@<cluster-login-host>:~/IE4483/`, then unzip there. Run `preresize.py` once on the cluster to produce `data_288/` (the default data folder).
4. **Write `train.py` and smoke-test it** on the laptop with 200 images and 1 epoch.
5. **First cluster job:** ResNet-50 fine-tune, 10 epochs, seed 0. Check val accuracy is ~98%+ and write a first `submission.csv`. From then on, everything else is a config change.

### 10.3 Minimal model/data code

```python
import timm, torch
from torchvision import datasets, transforms as T

model = timm.create_model("convnext_tiny.fb_in22k_ft_in1k", pretrained=True, num_classes=2)
cfg = timm.data.resolve_data_config({}, model=model)
size, mean, std = cfg["input_size"][-1], cfg["mean"], cfg["std"]

train_tf = T.Compose([
    T.RandomResizedCrop(size, scale=(0.5, 1.0)), T.RandomHorizontalFlip(),
    T.RandomRotation(15), T.ColorJitter(0.2, 0.2, 0.2, 0.02),
    T.ToTensor(), T.Normalize(mean, std), T.RandomErasing(p=0.25)])
eval_tf = T.Compose([T.Resize(int(size / cfg["crop_pct"])), T.CenterCrop(size),
                     T.ToTensor(), T.Normalize(mean, std)])

train_ds = datasets.ImageFolder("data_288/train", train_tf)    # class_to_idx == {'cat': 0, 'dog': 1}
val_ds   = datasets.ImageFolder("data_288/val",   eval_tf)
```

### 10.4 Running on MLDA gpu21 (current setup)

Code lives in `src/` (`data.py`, `models.py`, `train.py`) and `scripts/` (`preresize.py`, `prefetch_weights.py`). Copy it to `~/ie4483` on gpu21. Each run writes the following to `runs/<model>_s<seed>/`:

- `last.pt`, a resumable checkpoint
- `best.pt`, the best-val-accuracy checkpoint
- TensorBoard logs
- `metrics.json`, with val accuracy with and without flip TTA, plus the confusion matrix
- `val_predictions.csv`
- `test_probs.csv`
- `submission.csv`

```bash
conda activate ie4483 && cd ~/ie4483
export HF_HOME=~/ie4483/hf_cache
python scripts/preresize.py --src datasets --dst data_288          # once (already done on gpu21)
python scripts/prefetch_weights.py --cifar-root ~/ie4483/cifar     # once
nvidia-smi                                                          # pick an idle GPU
CUDA_VISIBLE_DEVICES=0 python src/train.py --model resnet50 --limit 512 --epochs 1 --name smoke   # smoke test
screen -S r50
CUDA_VISIBLE_DEVICES=0 python src/train.py --model resnet50 --seed 0 2>&1 | tee runs/resnet50_s0.log
```

Model names: `scratch_cnn`, `resnet18`, `resnet50`, `vgg16_bn`, `effnetv2_s`, `convnext_tiny`, `dinov2_s`, `dinov2_b`, or any timm name. Defaults follow §8.2:

- **Scratch CNN:** SGD, LR 0.05, one-cycle schedule, 50 epochs.
- **Pretrained CNNs:** AdamW, backbone LR 1e-4, head LR 1e-3, 1 LP epoch, 10 epochs in total.
- **ViTs:** backbone LR 5e-5 with layer decay 0.75.

Ablation flags:

| Flag | Values |
|---|---|
| `--aug` | `none`, `basic`, `default`, `trivial` |
| `--img-size` | any input resolution |
| `--n-train` | training-set size |
| `--no-pretrained` | random init for a timm model |
| `--init` | `kaiming` or `default` (scratch CNN) |
| `--lr`, `--bs`, `--lp-epochs` | override the defaults |

### 10.5 Slurm job template (A40, fallback)

Check the EEE cluster docs for the exact partition and GPU-type names (`sinfo -o "%P %G"` lists them).

```bash
#!/bin/bash
#SBATCH --job-name=ie4483-convnext
#SBATCH --account=ug-proj
#SBATCH --qos=ug
#SBATCH --gres=gpu:a40:1          # CPUs and RAM come with the GPU; --mem / --cpus-per-task are banned on GPU jobs
#SBATCH --time=04:00:00
#SBATCH --output=runs/slurm-%j.out

source ~/IE4483/.venv/bin/activate      # or: conda activate ie4483
export HF_HOME=$HOME/IE4483/hf_cache HF_HUB_OFFLINE=1
cd ~/IE4483
for seed in 0 1 2; do
  python src/train.py --config configs/convnext_tiny.yaml --seed $seed
done
```

For a sweep on the 0-SU QOS, swap in `--qos=override-limits-but-killable` and add `#SBATCH --requeue`. Make sure `train.py` resumes from the last checkpoint in its run folder.

## 11. Team split and timeline

Suggested split (adjust to the people; the report must state who did what):

| Member | Owns | Report parts |
|---|---|---|
| A | Data pipeline (pre-resize, loaders, augmentation), HOG + SVM, scratch CNN, augmentation ablations | a, f (data side) |
| B | Transfer-learning models, hyperparameter sweeps, final model, `submission.csv`, error analysis | b, c, d, e |
| C | CIFAR-10 and the imbalance study; cluster job scripts | g, h |

Everyone writes their own sections, and one person edits the whole report for consistency.

| Week | Dates | Goals |
|---|---|---|
| 1 | 17–23 Oct | Environments on laptop + cluster, data copied and pre-resized, `train.py` working, M0 and M1 results, first ResNet-50 submission |
| 2 | 24–31 Oct | All transfer models (M2–M7) × 3 seeds, LR and augmentation sweeps (uses October's SU) |
| 3 | 1–7 Nov | SU budget resets 1 Nov. Final model + TTA/ensemble, `submission.csv`, Grad-CAM error analysis; CIFAR-10 and imbalance runs |
| 4 | 8–13 Nov | Write the report: figures, tables, contributions, references. Submit by **13 Nov, 11:59 pm** |

## 12. References

1. Kaggle, *Dogs vs. Cats*. https://www.kaggle.com/c/dogs-vs-cats
2. Simonyan & Zisserman, *Very Deep Convolutional Networks for Large-Scale Image Recognition* (VGG), arXiv:1409.1556, 2014.
3. He et al., *Deep Residual Learning for Image Recognition* (ResNet), CVPR 2016.
4. Liu et al., *A ConvNet for the 2020s* (ConvNeXt), CVPR 2022. Woo et al., *ConvNeXt V2*, CVPR 2023. https://github.com/facebookresearch/convnext-v2
5. Tan & Le, *EfficientNetV2: Smaller Models and Faster Training*, ICML 2021.
6. Oquab et al., *DINOv2: Learning Robust Visual Features without Supervision*, TMLR 2024.
7. Kumar et al., *Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution* (LP-FT), ICLR 2022.
8. Cubuk et al., *RandAugment*, 2020; Müller & Hutter, *TrivialAugment*, ICCV 2021; Zhang et al., *mixup*, ICLR 2018; Yun et al., *CutMix*, ICCV 2019.
9. Cui et al., *Class-Balanced Loss Based on Effective Number of Samples*, CVPR 2019. https://arxiv.org/abs/1901.05555
10. Cao et al., *Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss* (LDAM-DRW), NeurIPS 2019. https://arxiv.org/abs/1906.07413
11. Menon et al., *Long-Tail Learning via Logit Adjustment*, ICLR 2021. https://arxiv.org/abs/2007.07314
12. Lin et al., *Focal Loss for Dense Object Detection*, ICCV 2017.
13. Selvaraju et al., *Grad-CAM*, ICCV 2017.
14. timm (PyTorch Image Models) documentation. https://huggingface.co/docs/timm
15. Krizhevsky, *Learning Multiple Layers of Features from Tiny Images* (CIFAR-10), 2009. https://www.cs.toronto.edu/~kriz/cifar.html
16. PyTorch forum, RTX 50-series (sm_120) support. https://discuss.pytorch.org/t/pytorch-support-for-sm120/216099
