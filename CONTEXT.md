# Context for the report writers: everything that was done, why, and where the results are

Written 18 Oct 2026 by the person who ran the experiments. It assumes you have the brief (`IE4483-Project2.pdf`) open.

- **Numbers:** every number below comes from `report/tables/RESULTS.md`, which is regenerated from `runs/` by `scripts/make_tables.py`. If a number here and in RESULTS.md ever disagree, trust RESULTS.md.
- **How to run things:** see `README.md`.

## 1. The short version

- **Final model:** the DINOv2-B vision transformer, used as a **frozen feature extractor with a trained linear classifier** (a "linear probe"). Three seeds are averaged, and each image is also scored mirrored (flip test-time augmentation, "TTA").
  - **Validation accuracy: 99.88%** (6 errors out of 4,999).
  - The submission file is `submission_final.csv`: 500 rows, 259 dogs and 241 cats.
- **Pretraining matters far more than anything else.** The model ladder (all validation accuracy):

  | Model | Val acc |
  |---|---|
  | HOG + SVM | 68–80% |
  | CNN trained from scratch | 97.65% |
  | ImageNet ResNet-50 | 99.45–99.52% |
  | ImageNet-21k/22k CNNs | 99.64–99.70% |
  | DINOv2 (142 M images, self-supervised) | 99.71–99.86% |

  With only 2,000 training images, DINOv2-B already reaches 99.83%. That beats the scratch CNN trained on all 20,000 images.
- **Linear probe ≥ fine-tuning here.** For ResNet-50 and DINOv2-B, training only the head beats fine-tuning the whole network. For ConvNeXt-T and DINOv2-S they tie. Diagnostic runs explain it (§4c).
- **CIFAR-10 (part g):** ResNet-18 from scratch reaches 94.78% test accuracy. A fine-tuned pretrained DINOv2-S reaches **98.87%**.
- **Long-tailed CIFAR-10 (part h):** **logit adjustment** is the best of five methods. At imbalance ratio 100 it lifts test accuracy from 74.1% to 78.5%, and tail-class accuracy from 52% to 65%. Class-balanced re-weighting with deferred re-weighting is second. Naive re-sampling is worse than doing nothing.

## 2. Where each part of the brief is covered

| Part | Evidence (figures in `report/figures/`, tables in `report/tables/RESULTS.md`) |
|---|---|
| a. Data, preprocessing, augmentation | Table "Dataset"; `aug_examples` (what the network sees); `aug_ablation`; §3 below |
| b. Model, architecture, I/O dims, loss, training strategy, code, reproducibility | `arch_final_model`, `arch_scratch_cnn`; layer-by-layer shapes in `report/model_summaries/*.txt`; `training_curves`; `README.md` (code and run instructions, seeds) |
| c. Choice of hyperparameters | `lr_sweep` + table "learning rate"; table "diagnostic runs"; `aug_ablation`; default recipes in §4a |
| d. Validation accuracy and submission | Table "Final model and submission"; `submission_final.csv` |
| e. Correct/incorrect samples, strengths/weaknesses | `test_cases`, `test_confidence`, `test_uncertain`, `errors_final_val`, `errors_scratch_val`, `gradcam`; contact sheets of all 500 test images in `report/test_review/`; §4e |
| f. How model choice and data processing affect accuracy | `model_ladder`, `datasize_curve`, `aug_ablation`; tables "all models", "linear probe vs fine-tuning", "training-set size" |
| g. CIFAR-10 | `cifar_g`; table "CIFAR-10" |
| h. Imbalanced CIFAR-10, ≥ 2 methods | `cifar_h`, `cifar_h_groups`; table "Long-tailed CIFAR-10" |

**Draft captions and alt text** for every figure are in `report/figures/captions*.md`. The figures deliberately contain no titles, so the caption must say what is plotted. In the plots, the colour shows how the backbone was trained:
- blue = fine-tuned;
- green = linear probe;
- vermillion = no pretraining.

Filled markers are means; small open markers are individual seeds. The exact plotted numbers are in `report/figures/data/*.csv`.

## 3. Data and preprocessing (part a)

- **Splits:**
  - 20,000 training images (10,000 cats, 10,000 dogs);
  - 5,000 validation images, minus 1, so **4,999** are used;
  - 500 unlabelled test images.
- **Duplicate removed:** one validation image is an exact copy of a training image (MD5 hash): `val/cat/cat.12408.jpg` = `train/cat/cat.2339.jpg`. It was removed from validation to avoid a train-to-val leak. There are also 4 duplicate pairs inside the training set; they are harmless and kept.
- **Image sizes:** they vary a lot, from 32 to 1,050 px, with a median of about 470 × 374. About 200 images have a shorter side under 100 px. All files are readable RGB JPEGs. The details are in `scripts/dataset_summary.json` and `PLAN.md` §3.
- **One-time pre-resize:** images were resized once so the shorter side is 288 px, saved as JPEG quality 95, to speed up loading. We measured image decoding at 1.38× faster on the server. Aspect ratio is kept and nothing is upscaled.
- **Validation/test input:** resize, centre crop to 224 × 224 (128 × 128 for the scratch CNN), then normalise with each model's own mean/std.
- **Training augmentation** (the "default" setting):
  - random resized crop, 50–100% of the area (60–100% for the scratch CNN);
  - horizontal flip, p = 0.5;
  - rotation ±15°;
  - colour jitter: brightness/contrast/saturation 0.2, hue 0.02;
  - random erasing, p = 0.25.

  `aug_examples` shows real examples.
- **Test-time augmentation:** the reported numbers average the prediction on the image and its mirror image.
- **Labels:** cat = 0, dog = 1, matching the submission format.

## 4. Models, training and findings

### a. Models compared and default recipes (parts b, c)

| Model | Recipe |
|---|---|
| HOG + SVM (classical) | 128 × 128 greyscale HOG (9 orientations, 8 × 8 cells, 2 × 2 blocks) + 48-bin colour histogram → standardise → linear SVM (C = 0.01), or PCA(256) + RBF SVM (C = 10) |
| Scratch CNN (2.35 M params) | 5 blocks of [3×3 conv-BN-ReLU] × 2 + max-pool (channels 32 → 256), global average pool, dropout 0.3, linear → 2. He initialisation. SGD with Nesterov momentum 0.9, one-cycle LR (peak 0.05), weight decay 5e-4, batch 128, 50 epochs |
| Pretrained backbones (timm): ResNet-50 (ImageNet-1k), EfficientNetV2-S (ImageNet-21k), ConvNeXt-T (ImageNet-22k), DINOv2-S/B (self-supervised, LVD-142M) | **Fine-tuning ("LP-FT")**: 1 epoch training only the new linear head, then 9 epochs training everything. AdamW, weight decay 0.05 (none on biases/norms), head LR 1e-3, backbone LR 1e-4 (CNNs) or 5e-5 with layer-wise decay 0.75 (ViTs), 1-epoch warm-up then cosine decay, batch 128 (64 for DINOv2-B), bf16 mixed precision. **Linear probe**: backbone frozen, only the head trained, 10 epochs, LR 1e-3 |
| All models | Cross-entropy loss with label smoothing 0.1 |

Further details:
- **Reported epoch:** the reported number is the **last** epoch, not the best one. Picking the best epoch on validation would be optimistic, and it turned out the best epoch was often the head-only epoch 0.
- **Seeds:** each configuration was run with seeds 0, 1 and 2. Tables give mean ± SD.

### b. Model comparison (part f)

The full table is "Dogs vs. Cats: all models" in RESULTS.md; the figure is `model_ladder`.
- **Pretraining source** is the biggest factor: scratch 97.65 → ImageNet-1k 99.45 → 21k/22k 99.64–99.70 → DINOv2 99.73–99.86.
- **Architecture size matters less:** DINOv2-S (22 M params) beats ConvNeXt-T (28 M) and EfficientNetV2-S.
- **Training-set size** (`datasize_curve`, table "training-set size"):
  - the scratch CNN climbs from 86.6% (2k images) to 97.7% (20k) and is still improving;
  - ResNet-50 gains only 0.6 points;
  - ConvNeXt-T and the DINOv2-B probe are flat at ≈ 99.6–99.9% from 2,000 images on.

  Strong pretraining replaces at least 10× the labelled data.
- **Augmentation** (scratch CNN, `aug_ablation`):
  - none 94.6% → random crop + flip 97.7%: this does almost all the work;
  - adding rotation, colour jitter and erasing gives 97.65%; TrivialAugment gives 97.5%. No gain at 50 epochs;
  - 100 epochs gives 97.9% (98.2% with TTA), so the scratch CNN is still under-trained at 50 epochs.

  Flip TTA helps most when training had no augmentation (+0.8).

### c. Hyperparameters, and why fine-tuning did not win (part c)

- **Learning-rate sweep** (ResNet-50, `lr_sweep`): backbone LR 1e-5, 3e-5, 1e-4 and 3e-4 give 98.98, 99.30, 99.48 and 99.52% at the last epoch.
  - Accuracy rises with LR; there is no U-shape.
  - Every run is best right after the head-only epoch (99.58%). Unfreezing the backbone first *costs* accuracy at epoch 1, and the higher LRs recover best.
- **Linear probe vs fine-tuning** (3 seeds each):

  | Backbone | Probe vs fine-tuned | Verdict |
  |---|---|---|
  | ResNet-50 | 99.52 vs 99.45 | probe better |
  | DINOv2-B | 99.86 vs 99.77 | probe better |
  | ConvNeXt-T | 99.66 vs 99.70 | tie |
  | DINOv2-S | 99.71 vs 99.73 | tie |

- **Diagnostic runs** (table "diagnostic runs"):
  - Gentler fine-tuning of DINOv2-B (backbone LR 1e-5) reaches **99.87 ± 0.04**, equal to the probe.
  - Lighter augmentation reaches 99.82.
  - For ResNet-50, lighter augmentation does not help (99.36).
- **Interpretation for the report:**
  - ImageNet contains many cat and dog breeds, and DINOv2 saw 142 M images. Their frozen features already separate the classes almost perfectly, so fine-tuning has little to add and can slightly distort them; this is the "feature distortion" effect from LP-FT (Kumar et al. 2022).
  - Our one-size fine-tuning recipe was a bit aggressive for DINOv2.
  - At 99.5–99.9% the remaining errors are mostly label noise (§4e), so differences under ~0.1 point (5 images) are within seed noise.

### d. Final model and submission (part d)

- **Model choice:** the DINOv2-B linear probe was chosen as the best model on validation (99.86 ± 0.02% per seed). The submission averages the probabilities of its 3 seeds, which gives 99.88% and avoids cherry-picking a seed.
- **Why the ensemble is not more accurate:** the seeds are nearly identical, so the 3-seed average adds nothing over a single seed. The 3 single-seed submissions agree on all 500 test images.
- **Ensembling different models does not help:** averaging DINOv2-B probe + DINOv2-S + ConvNeXt-T gave 99.84%, *worse* than DINOv2-B alone, because the weaker models dilute the strongest one.
- **Validation confusion:**
  - true cats: 2,495 correct, 4 called dog;
  - true dogs: 2,498 correct, 2 called cat.

### e. Error analysis (part e)

Figures: `errors_final_val`, `errors_scratch_val`, `test_uncertain`, `test_cases`, `test_confidence`, `gradcam`.
- **Most of the final model's 6 validation "errors" are dataset problems**, judged by eye (check them yourself in the figure):
  - `cat.7920` and `cat.9882` show **a dog and a cat together**;
  - `cat.7377` is a portrait of **a man**;
  - `cat.3672` is **a group of people** with a small cat;
  - `dog.11248` is a **tiny, wide strip of several animals**;
  - `dog.7164` is a small, unclear animal photo.

  So 99.88% is close to the ceiling the labels allow. This is worth a paragraph: the Kaggle Dogs vs. Cats data is known to contain such images.
- **The scratch CNN's mistakes** are mostly ordinary, clear single-animal photos. It has not learned robust features; the pretrained model has.
- **Test set, visual check:** Dheemant inspected all 500 test predictions by eye, using the contact sheets in `report/test_review/` (made by `scripts/make_test_review.py`). He found **no wrong prediction**, so the estimated test accuracy is 500/500. Say clearly that this comes from visual inspection, since the test labels are not public.
  - **Confidence** (`test_confidence`): only 3 of 500 test images have p(dog) between 0.2 and 0.8.
  - **Hardest correct test cases** (`test_cases` panel a, all predicted correctly):
    - **469** (dog, p = 0.65): a tiny, blurry photo of fluffy merle puppies;
    - **216** (cat, 0.25): cats behind cage bars with large yellow text overlaid;
    - **47** (dog, 0.80): puppies far away in a basket, seen through a chain-link fence;
    - **179** (dog, 0.83): a greyhound, a slim, unusual dog shape;
    - **293** (cat, 0.15): a child with a toy pram; the cat is small, in the pram;
    - **342** (cat, 0.14): a small, dark, low-contrast cat photo.
  - **Strengths:** confident and correct on ordinary photos, whatever the pose, breed or background (`test_cases` panel b).
  - **Weaknesses:** confidence drops when the animal is small in the frame, occluded by bars or a fence, low-resolution or blurred, overlaid with text, or an unusual breed.
  - **Reading p(dog):** training used label smoothing (0.1), so confident predictions sit near 0.05 and 0.95, not 0 and 1. Values of 0.09–0.15 are still confident.
- **Grad-CAM:** the heat maps show where each model looks. Describe what they show after looking at `gradcam.png`; the caption draft is in `captions_gradcam.md`.

### f. CIFAR-10 (part g)

Figure `cifar_g`, table "CIFAR-10".
- **Setup:**
  - 45,000 training images, plus 500 per class held out for validation;
  - test accuracy is on the official 10,000-image test set, evaluated once.
- **From scratch:** a CIFAR-style ResNet-18 (3×3 stem, no max-pool) trained for 200 epochs. SGD LR 0.1 with 5-epoch warm-up and cosine decay, weight decay 5e-4, random crop with 4 px padding + flip. **94.78 ± 0.25%**.
- **"Improving the method"** means reusing the Dogs-vs-Cats pipeline: a pretrained backbone, images upsampled 32 → 224, 10 epochs.
  - DINOv2-S fine-tuned: **98.87%**.
  - ConvNeXt-T fine-tuned: 98.69%.
  - DINOv2-B linear probe: 97.99%.
- **Contrast with Dogs vs. Cats:** here the frozen linear probe is the *weakest* transfer model. Upsampled 32 px images look very different from the photos DINOv2 was trained on, so its frozen features fit worse and fine-tuning helps. On Dogs vs. Cats the data is close to the pretraining data, and the frozen probe won. This is a useful discussion point.

### g. Long-tailed CIFAR-10 (part h)

Figures `cifar_h`, `cifar_h_groups`; table "Long-tailed CIFAR-10".
- **Setup:**
  - training set made long-tailed with an exponential profile: 4,500 images for class 0 down to 45 (ratio 100) or 450 (ratio 10);
  - validation and test stay balanced;
  - ResNet-18 from scratch, 200 epochs.
- **Methods compared** against plain cross-entropy:
  - focal loss (γ = 1);
  - class-balanced re-sampling;
  - class-balanced loss weights using the "effective number of samples" (Cui et al. 2019), switched on only in the last 20% of epochs (DRW, Cao et al. 2019);
  - **logit adjustment** (Menon et al. 2021, τ = 1).
- **Results at ratio 100:**

  | Method | Test acc | Tail-class acc |
  |---|---|---|
  | Cross-entropy | 74.12 ± 0.15 | 52.1 |
  | Focal loss | 73.53 | 50.0 |
  | Re-sampling | 69.24 | 40.6 |
  | CB weights + DRW | 76.05 ± 0.28 | 56.9 |
  | **Logit adjustment** | **78.53 ± 0.51** | **65.1** |

  Head-class accuracy stays at ≈ 94% for every method, so the gains come from the rare classes.
- **Results at ratio 10:** the same order but smaller gaps. Logit adjustment 89.86 ± 0.13 vs CE 88.89 ± 0.12; CB+DRW 89.33 ± 0.45 is within noise of CE.
- **Why re-sampling hurts:** the 45 images of the rarest class are repeated ~100× per epoch and get memorised. This is why deferred schedules like DRW exist.
- **Seeds:** CE, CB+DRW and LA have 3 seeds; focal and re-sampling have 1 seed each, so say so.

## 5. Problems found along the way (worth knowing; mention briefly where relevant)

- **Last-epoch reporting:** early runs reported the best-validation epoch. From 17 Oct 13:40 every run reports the last epoch, and earlier runs were re-run or replaced, so all numbers in RESULTS.md use last-epoch reporting.
- **EfficientNetV2-S head bug:** timm initialises a new 2-class EfficientNet head with weights ~30× larger than other models, so its head-only epoch scored only 90%. Re-initialising the head like the other models fixed it (99.12 → 99.64%). Only EfficientNet was affected.
- **CIFAR transformer crash:** the first CIFAR runs of the transformer models crashed on a memory-format bug. It was fixed and they were re-run. All CIFAR numbers come from the fixed runs.
- **Reproducibility:** repeating the same seed is not bit-identical (GPU non-determinism), which is why everything is reported as mean ± SD over 3 seeds.

## 6. Compute

- **Hardware:** all training ran on the MLDA GPU server `gpu21` (4× NVIDIA RTX 3090 24 GB, PyTorch 2.11, CUDA 12.8), with 4 runs in parallel. The EEE GPU cluster was fully booked when the project started.
- **Volume:** 108 training runs (80 Dogs-vs-Cats, 28 CIFAR) and about 15 GPU-hours of training.
- **Timing:** a ResNet-50 fine-tune takes ~8 min, a DINOv2-B probe ~8 min, a DINOv2-B fine-tune ~21 min, and a CIFAR ResNet-18 ~26 min.

## 7. File index (what is in this zip)

| Path | Contents |
|---|---|
| `README.md` | Code: setup, how to run every experiment, the figures and tables |
| `CONTEXT.md` | This file |
| `report/tables/RESULTS.md` (+ `*.csv`) | **All final numbers**, in 10 tables |
| `report/figures/*.png` / `*.pdf` | All report figures (PNG 600 dpi, PDF vector) |
| `report/figures/captions*.md` | Draft captions and alt text for every figure |
| `report/figures/data/*.csv` | The exact values plotted in each figure |
| `report/model_summaries/` | Layer tables (input/output shapes, parameters) for DINOv2-B, the scratch CNN and ResNet-50 |
| `report/gradcam/` | Grad-CAM maps computed on the server (`gradcam_maps.npz`) |
| `runs/<run>/` | Per run: `metrics.json` (all numbers + per-epoch history), `args.json` (exact settings), `val_predictions.csv`, `test_probs.csv`; logs in `runs/<run>.log` |
| `jobs/*.txt` | Run lists for most batches. A few later batches (extra seeds, data-size seeds, fine-tuning diagnostics) were written directly on the server and are not here; each run's `args.json` records its exact settings |
| `submission_final.csv` | The file to submit |
| `src/`, `scripts/` | All code |
| `PLAN.md` | The original plan (§3 dataset analysis is still accurate; the compute section is outdated) |

**Not included:** the datasets, the resized copies, and model checkpoints (`last.pt`, up to ~1 GB each). They are still on the MLDA server under `~/ie4483` until after submission.

## 8. Still to do (report side)

- **Writing:** the report itself, using the figures, `RESULTS.md` and the caption drafts.
- **Contributions:** the brief requires a statement of who did what. Training, experiments, results, figures and code: Dheemant.
- **Part (e):** describe the mislabelled/ambiguous validation images (§4e) and what the Grad-CAM maps show.
- **References:** Kumar et al. 2022 (LP-FT); Oquab et al. 2024 (DINOv2); Cui et al. 2019; Cao et al. 2019; Menon et al. 2021; Lin et al. 2017; He et al. 2016; Liu et al. 2022 (ConvNeXt); Tan & Le 2021; Selvaraju et al. 2017 (Grad-CAM). Full list in `PLAN.md` §12.
- **Submission:** upload `submission_final.csv` with the report.
