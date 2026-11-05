---
title: Homework 2 -- Multiple Object Video Detection on VisDrone-VID
author: Дарина Ничипорук, Таня Мороз
course: Artificial Intelligence · UCU
date: 2026-11-05
---

<!-- Target: ~1500 words. Replace every TODO with numbers from the notebooks. Each table names the notebook it comes from. -->

## 1. Data and preprocessing  *(notebook 01)*

- **Dataset.** 63 labelled clips (27k frames), 48 test clips; 8 classes. TODO: class shares, % of COCO-small boxes, median box side, objects per frame (median / p99), variation across clips.
- **Observations → decisions.**

| Observation (notebook 01) | Decision |
|---|---|
| TODO % boxes are small; at imgsz 640, TODO % fall below 8 px | input long side 1280; P2 head and sliced inference as experiments |
| TODO objects per frame at p99 | 500 detections per image; ≥ 300 DETR queries |
| same-track IoU at lag 10 = TODO | train on every 10th frame |
| ignored regions in TODO % of frames | grey-filled in training images; crowd boxes (all classes) in evaluation |

- **Annotations.** COCO json used as the single source; YOLO txt verified against it (0 mismatches). Non-target VisDrone classes are part of the ignored regions provided by the organisers.
- **Validation.** `GroupKFold(5)` over clips (train + val pooled). Fold 0 is the fixed holdout for all comparisons; evaluation = COCO mAP@[.5:.95] with 500 detections, ignored regions as crowd. Sanity checks: GT as predictions → AP 1.0; detections inside ignored regions do not change AP.
- **Preprocessing.** Long side resized to 1280, ImageNet normalisation (torchvision) or /255 (Ultralytics), horizontal flip only for all model comparisons; seed 42.

## 2. Two-stage: Faster R-CNN  *(notebook 02)*

Setup: TODO (one sentence). 

| Run | Change | AP | AP50 | AP_small | ms/img (T4) | Public LB |
|---|---|---|---|---|---|---|
| E0 | baseline | | | | | |
| E1 | 3000 RPN proposals | | | | | |
| E2 | GIoU box loss | | | | | |

Stage contributions: TODO (RPN recall by size vs final recall). Failure cases (figure `images/frcnn_failures.png`): 1) TODO 2) TODO 3) TODO.

## 3. One-stage anchor-based: RetinaNet  *(notebook 03)*

Anchors from training boxes: TODO (coverage default vs chosen). Table R0–R3 as above + best Faster R-CNN row. Proposal-stage removal: TODO.

## 4. Anchor-free: YOLOv8  *(notebook 04)*

Assignment without anchors (TAL) and grid points per box: TODO. Search range: architecture {P3–P5, P2–P5}, imgsz {1024, 1280, 1536}, TAL top-k {10, 13}. Table by size vs tuned anchor-based models.

## 5. Query-based: RT-DETR  *(notebook 05)*

Bipartite matching / no NMS: TODO (2–3 sentences). Curve `images/detr_vs_cnn_curves.png`; converged: yes/no. Dense vs sparse frames: TODO.

## 6. Final pipeline  *(notebook 06)*

| Part A: inference on fixed model | AP | AP_small |
|---|---|---|
| plain | | |
| best NMS IoU | | |
| TTA | | |
| sliced 640 | | |
| WBF ensemble | | |

| Part B: training data / model | AP | AP_small |
|---|---|---|
| base | | |
| step 5 | | |
| full augmentation | | |
| 24 epochs | | |

Summary of all detectors: `images/summary_all_models.png`. Unsuccessful attempts and why: TODO.

## 7. Final model and how to reproduce

Selected submission: TODO (model, settings, inference strategy; holdout AP; public LB).

1. Kaggle notebook with the competition data, GPU T4, Internet on.
2. Run `notebooks/01_eda_preprocessing.ipynb` (CPU) → `folds.csv`.
3. Run `notebooks/06_final_pipeline.ipynb` Section 4: trains `final_all_clips` and writes `submission_final.csv`.
4. Dependencies: `requirements.txt` (Ultralytics 8.3.x). Seed 42 everywhere (`vdet.SEED`); `vdet.PROTOCOL` holds every shared setting.

## References

Faster R-CNN, FPN, RetinaNet/Focal loss, YOLOv2 (k-means anchors), TOOD, YOLOv8 (Ultralytics), RT-DETR, DETR, GIoU, WBF, SAHI, COCO metrics — full links TODO. AI assistance: TODO (per course policy).
