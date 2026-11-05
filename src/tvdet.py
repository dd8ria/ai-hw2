"""Training / inference for torchvision detectors (Faster R-CNN, RetinaNet, FCOS).

One code path for every torchvision model keeps the protocol identical:
same frames, same input size, same augmentation, same optimiser schedule,
same evaluator. Only the model config changes between experiments.
"""
from __future__ import annotations

import json
import math
import time
from functools import partial
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import roi_heads as tv_roi_heads
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.ops import complete_box_iou_loss, distance_box_iou_loss, generalized_box_iou_loss

import vdet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class FrameDataset(Dataset):
    """Frames from a DataFrame (columns: key, img_path, width, height).

    Training mode: grey out ignored regions (same rule as the YOLO export) and
    random horizontal flip. Eval mode: the original image, no targets needed.
    Labels are 1-based (0 = background in torchvision).
    """

    def __init__(self, frames: pd.DataFrame, anns: pd.DataFrame | None = None, train: bool = False,
                 hflip: float = 0.5, mask_ignore: bool = True):
        self.frames = frames.reset_index(drop=True)
        self.by_key = dict(tuple(anns.groupby("key"))) if anns is not None else {}
        self.empty = anns.iloc[:0] if anns is not None else None
        self.train, self.hflip, self.mask_ignore = train, hflip, mask_ignore

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, i):
        r = self.frames.iloc[i]
        img = cv2.imread(r.img_path)
        a = self.by_key.get(r.key, self.empty)
        target = {}
        if a is not None:
            gt, ign = a[a.iscrowd == 0], a[a.iscrowd == 1]
            if self.train and self.mask_ignore and len(ign):
                img = vdet._mask_image(img, ign[["x", "y", "w", "h"]].values, gt[["x", "y", "w", "h"]].values)
            H, W = img.shape[:2]
            b = gt[["x", "y", "w", "h"]].values.astype(np.float32).reshape(-1, 4)
            b = np.c_[b[:, 0], b[:, 1], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]]
            b[:, [0, 2]] = b[:, [0, 2]].clip(0, W); b[:, [1, 3]] = b[:, [1, 3]].clip(0, H)
            keep = ((b[:, 2] - b[:, 0]) > 1) & ((b[:, 3] - b[:, 1]) > 1)
            b, labels = b[keep], gt.cls.values[keep] + 1
            if self.train and np.random.rand() < self.hflip:
                img = img[:, ::-1]
                b = np.c_[W - b[:, 2], b[:, 1], W - b[:, 0], b[:, 3]]
            target = {"boxes": torch.as_tensor(b, dtype=torch.float32).reshape(-1, 4),
                      "labels": torch.as_tensor(labels, dtype=torch.int64)}
        img = torch.from_numpy(np.ascontiguousarray(img[:, :, ::-1])).permute(2, 0, 1).float() / 255
        return img, target, r.key


def collate(batch):
    return tuple(zip(*batch))


# --------------------------------------------------------------------------- #
# Box-regression loss for the Faster R-CNN second stage
# --------------------------------------------------------------------------- #
_ORIG_FASTRCNN_LOSS = tv_roi_heads.fastrcnn_loss
_BOX_WEIGHTS = (10.0, 10.0, 5.0, 5.0)  # torchvision RoIHeads BoxCoder weights
_CLIP = math.log(1000.0 / 16)


def _decode_unit(d: torch.Tensor) -> torch.Tensor:
    """Decode deltas against a unit reference box. IoU-type losses are invariant to the
    per-axis scaling and translation that maps a proposal onto this box, so the loss
    equals the loss in image coordinates."""
    d = d.float()
    dx, dy = d[:, 0] / _BOX_WEIGHTS[0], d[:, 1] / _BOX_WEIGHTS[1]
    dw = (d[:, 2] / _BOX_WEIGHTS[2]).clamp(max=_CLIP); dh = (d[:, 3] / _BOX_WEIGHTS[3]).clamp(max=_CLIP)
    w, h = dw.exp(), dh.exp()
    return torch.stack([dx - w / 2, dy - h / 2, dx + w / 2, dy + h / 2], 1)


def make_iou_fastrcnn_loss(kind: str = "giou", weight: float = 10.0):
    """Replacement for torchvision's fastrcnn_loss with an IoU-based box loss.
    weight=10 follows MMDetection's Faster R-CNN GIoU config."""
    fn = {"giou": generalized_box_iou_loss, "diou": distance_box_iou_loss, "ciou": complete_box_iou_loss}[kind]

    def loss(class_logits, box_regression, labels, regression_targets):
        labels = torch.cat(labels, 0); regression_targets = torch.cat(regression_targets, 0)
        cls_loss = F.cross_entropy(class_logits, labels)
        pos = torch.where(labels > 0)[0]
        n = class_logits.shape[0]
        box_regression = box_regression.reshape(n, box_regression.size(-1) // 4, 4)
        if len(pos) == 0:
            return cls_loss, box_regression.sum() * 0
        pred = _decode_unit(box_regression[pos, labels[pos]])
        tgt = _decode_unit(regression_targets[pos])
        box_loss = weight * fn(pred, tgt, reduction="sum") / labels.numel()
        return cls_loss, box_loss

    return loss


def set_focal(gamma: float = 2.0, alpha: float = 0.25) -> None:
    """Focal-loss parameters for RetinaNet / FCOS classification heads (hard-coded in torchvision,
    so the name their compute_loss looks up is replaced with a partial)."""
    from torchvision.models.detection import fcos as tv_fcos, retinanet as tv_retina
    from torchvision.ops import sigmoid_focal_loss
    f = partial(sigmoid_focal_loss, alpha=alpha, gamma=gamma)
    tv_retina.sigmoid_focal_loss = f
    tv_fcos.sigmoid_focal_loss = f


def set_box_loss(kind: str = "smooth_l1", weight: float = 10.0) -> None:
    """Patch the module-level function that RoIHeads.forward calls."""
    tv_roi_heads.fastrcnn_loss = _ORIG_FASTRCNN_LOSS if kind == "smooth_l1" else make_iou_fastrcnn_loss(kind, weight)


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
def build_model(cfg: dict, pretrained: bool = True) -> torch.nn.Module:
    """cfg keys: arch, imgsz, anchor_sizes, plus arch-specific kwargs under 'kwargs'.

    imgsz is the long side after resizing (min_size = max_size = imgsz makes
    torchvision scale the long side to imgsz, like a YOLO letterbox).
    """
    from torchvision.models import detection as D

    nc = vdet.NUM_CLASSES + 1
    common = dict(min_size=cfg["imgsz"], max_size=cfg["imgsz"])
    kw = dict(cfg.get("kwargs", {}))
    w = "DEFAULT" if pretrained else None
    wb = None  # backbone weights come with the full COCO checkpoint
    arch = cfg["arch"]

    if arch == "fasterrcnn_v2":
        if cfg.get("anchor_sizes"):
            kw["rpn_anchor_generator"] = AnchorGenerator(tuple((s,) for s in cfg["anchor_sizes"]),
                                                         ((0.5, 1.0, 2.0),) * len(cfg["anchor_sizes"]))
        m = D.fasterrcnn_resnet50_fpn_v2(weights=w, weights_backbone=wb, box_detections_per_img=vdet.MAX_DETS,
                                         box_score_thresh=0.01, **common, **kw)
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
        m.roi_heads.box_predictor = FastRCNNPredictor(m.roi_heads.box_predictor.cls_score.in_features, nc)

    elif arch == "retinanet_v2":
        from torchvision.models.detection.retinanet import RetinaNetClassificationHead, RetinaNetRegressionHead
        m = D.retinanet_resnet50_fpn_v2(weights=w, weights_backbone=wb, detections_per_img=vdet.MAX_DETS,
                                        score_thresh=0.01, **common, **kw)
        if cfg.get("anchor_sizes"):  # 3 octave scales per level, like the default generator
            sizes = tuple((s, s * 2 ** (1 / 3), s * 2 ** (2 / 3)) for s in cfg["anchor_sizes"])
            ratios = (tuple(cfg.get("aspect_ratios", (0.5, 1.0, 2.0))),) * len(sizes)
            m.anchor_generator = AnchorGenerator(sizes, ratios)
        na = m.anchor_generator.num_anchors_per_location()[0]
        gn = partial(torch.nn.GroupNorm, 32)
        if na != m.head.regression_head.bbox_reg.out_channels // 4:  # new anchor count -> fresh last layers
            m.head.regression_head = RetinaNetRegressionHead(256, na, norm_layer=gn)
        m.head.regression_head._loss_type = cfg.get("box_loss", "giou")  # v2 default is GIoU
        m.head.classification_head = RetinaNetClassificationHead(256, na, nc, norm_layer=gn)
    elif arch == "fcos":
        from torchvision.models.detection.fcos import FCOSClassificationHead
        m = D.fcos_resnet50_fpn(weights=w, weights_backbone=wb, detections_per_img=vdet.MAX_DETS,
                                score_thresh=0.01, **common, **kw)
        m.head.classification_head = FCOSClassificationHead(256, m.anchor_generator.num_anchors_per_location()[0], nc)
    else:
        raise ValueError(arch)
    return m


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def _lr_lambda(total: int, warmup: int, sched: str):
    def f(it):
        if it < warmup:
            return 0.001 + (1 - 0.001) * it / warmup
        p = (it - warmup) / max(1, total - warmup)
        if sched == "cosine":
            return 0.5 * (1 + math.cos(math.pi * p)) * 0.99 + 0.01
        return 0.1 ** sum(p >= m for m in (0.67, 0.89))  # step schedule like the 1x Detectron recipe
    return f


def train_run(cfg: dict, train_frames: pd.DataFrame, val_frames: pd.DataFrame, anns: pd.DataFrame,
              val_imgs: pd.DataFrame, out_dir: str | Path, workers: int = 4, pretrained: bool = True,
              log_wandb: bool = False) -> dict:
    """Train one config, evaluate on the validation fold after every epoch.

    Resumable: if out_dir/last.pth exists, training continues from it.
    Writes history.json (per-epoch metrics), best.pth and metrics.json.
    """
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    if (out / "metrics.json").exists():
        print(f"{out.name}: already finished, loading metrics"); return json.loads((out / "metrics.json").read_text())
    (out / "config.json").write_text(json.dumps(cfg, indent=1))
    vdet.seed_everything(cfg.get("seed", vdet.SEED))
    # box_loss means the RoI-head loss for Faster R-CNN; RetinaNet reads it in build_model
    is_rcnn = cfg["arch"].startswith("fasterrcnn")
    set_box_loss(cfg.get("box_loss", "smooth_l1") if is_rcnn else "smooth_l1", cfg.get("box_loss_weight", 10.0))
    set_focal(cfg.get("focal_gamma", 2.0), cfg.get("focal_alpha", 0.25))

    model = build_model(cfg, pretrained=pretrained).to(DEVICE)
    g = torch.Generator(); g.manual_seed(cfg.get("seed", vdet.SEED))
    dl = DataLoader(FrameDataset(train_frames, anns, train=True), batch_size=cfg["batch"], shuffle=True,
                    num_workers=workers, collate_fn=collate, generator=g, drop_last=True,
                    worker_init_fn=lambda i: np.random.seed(cfg.get("seed", vdet.SEED) + i))
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=cfg["lr"], momentum=0.9, weight_decay=cfg.get("wd", 1e-4))
    total = cfg["epochs"] * len(dl)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda(total, min(500, total // 10), cfg.get("sched", "step")))
    scaler = torch.amp.GradScaler(enabled=DEVICE.type == "cuda")

    start, history, best = 0, [], -1.0
    if (out / "last.pth").exists():
        ck = torch.load(out / "last.pth", map_location=DEVICE, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"]); scaler.load_state_dict(ck["scaler"])
        start, history, best = ck["epoch"] + 1, ck["history"], ck["best"]
        print(f"resumed from epoch {start}")

    if log_wandb:
        import wandb
        wandb.init(project="visdrone-hw2", name=out.name, config=cfg, resume="allow", id=out.name)

    for ep in range(start, cfg["epochs"]):
        model.train(); t0 = time.time(); run = {}
        for it, (imgs, tgts, _) in enumerate(dl):
            imgs = [i.to(DEVICE, non_blocking=True) for i in imgs]
            tgts = [{k: v.to(DEVICE) for k, v in t.items()} for t in tgts]
            with torch.autocast(DEVICE.type, enabled=DEVICE.type == "cuda"):
                losses = model(imgs, tgts)
            loss = sum(losses.values())
            if not torch.isfinite(loss):
                print("non-finite loss, skipping batch", {k: float(v) for k, v in losses.items()}); continue
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(params, 10.0)
            scaler.step(opt); scaler.update(); sched.step()
            for k, v in losses.items():
                run[k] = run.get(k, 0) + v.item()
            if it % 100 == 0:
                print(f"ep {ep} it {it}/{len(dl)} loss {float(loss):.3f} lr {opt.param_groups[0]['lr']:.5f}")
        train_time = time.time() - t0
        preds = predict(model, val_frames, batch=cfg["batch"], workers=workers)
        m = vdet.coco_evaluate(val_imgs, anns, preds, per_class=False)
        rec = {"epoch": ep + 1, "train_min": train_time / 60, **{k: v / len(dl) for k, v in run.items()}, **m}
        history.append(rec)
        print(f"epoch {ep + 1}: AP {m['AP']:.4f} AP50 {m['AP50']:.4f} AP_s {m['AP_small']:.4f} ({train_time / 60:.1f} min)")
        if log_wandb:
            wandb.log(rec)
        if m["AP"] > best:
            best = m["AP"]; torch.save(model.state_dict(), out / "best.pth")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(), "epoch": ep, "history": history, "best": best}, out / "last.pth")
        (out / "history.json").write_text(json.dumps(history, indent=1))

    # Final metrics use the LAST epoch: picking the best epoch on the same holdout would
    # leak validation information into the reported number. best.pth is kept for reference.
    torch.save(model.state_dict(), out / "final.pth")
    preds = predict(model, val_frames, batch=cfg["batch"], workers=workers)
    preds.to_csv(out / "val_preds.csv.gz", index=False)
    m = vdet.coco_evaluate(val_imgs, anns, preds)
    m.update(measure_speed(model, val_frames))
    m["best_epoch"] = int(np.argmax([h["AP"] for h in history]) + 1)
    m["epochs"] = len(history)
    (out / "metrics.json").write_text(json.dumps(m, indent=1))
    if log_wandb:
        wandb.summary.update(m); wandb.finish()
    set_box_loss("smooth_l1"); set_focal()
    return m


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
@torch.no_grad()
def predict(model, frames: pd.DataFrame, batch: int = 4, workers: int = 4) -> pd.DataFrame:
    """Run the model on frames; returns key, cls (0-based), score, x, y, w, h in original pixels."""
    model.eval()
    dl = DataLoader(FrameDataset(frames), batch_size=batch, num_workers=workers, collate_fn=collate)
    rows = []
    for imgs, _, keys in dl:
        with torch.autocast(DEVICE.type, enabled=DEVICE.type == "cuda"):
            outs = model([i.to(DEVICE) for i in imgs])
        for k, o in zip(keys, outs):
            b = o["boxes"].float().cpu().numpy()
            rows.append(pd.DataFrame({"key": k, "cls": o["labels"].cpu().numpy() - 1, "score": o["scores"].float().cpu().numpy(),
                                      "x": b[:, 0], "y": b[:, 1], "w": b[:, 2] - b[:, 0], "h": b[:, 3] - b[:, 1]}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["key", "cls", "score", "x", "y", "w", "h"])


@torch.no_grad()
def measure_speed(model, frames: pd.DataFrame, n: int = 100, warmup: int = 10) -> dict:
    """Batch-1 latency including resize and postprocessing (not JPEG decoding)."""
    model.eval()
    ds = FrameDataset(frames.sample(min(n + warmup, len(frames)), random_state=0))
    times = []
    warmup = min(warmup, len(ds) // 2)
    for i in range(len(ds)):
        img = ds[i][0].to(DEVICE)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        t = time.perf_counter()
        with torch.autocast(DEVICE.type, enabled=DEVICE.type == "cuda"):
            model([img])
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        if i >= warmup:
            times.append(time.perf_counter() - t)
    ms = 1000 * float(np.mean(times)) if times else float("nan")
    return {"ms_per_img": ms, "fps": 1000 / ms, "device": torch.cuda.get_device_name() if DEVICE.type == "cuda" else "cpu"}


@torch.no_grad()
def rpn_proposals(model, frames: pd.DataFrame) -> pd.DataFrame:
    """Faster R-CNN first-stage output: class-agnostic proposals in original pixels."""
    model.eval(); rows = []
    ds = FrameDataset(frames)
    for i in range(len(ds)):
        img, _, key = ds[i]
        H, W = img.shape[1:]
        il, _ = model.transform([img.to(DEVICE)])
        feats = model.backbone(il.tensors)
        props, _ = model.rpn(il, feats)
        h, w = il.image_sizes[0]
        p = props[0].float().cpu().numpy() * np.array([W / w, H / h, W / w, H / h])
        rows.append(pd.DataFrame({"key": key, "rank": np.arange(len(p)), "x": p[:, 0], "y": p[:, 1],
                                  "w": p[:, 2] - p[:, 0], "h": p[:, 3] - p[:, 1]}))
    return pd.concat(rows, ignore_index=True)


def load_run(run_dir: str | Path, weights: str = "final.pth") -> torch.nn.Module:
    """Rebuild a model from run_dir/config.json and load its weights."""
    run_dir = Path(run_dir)
    cfg = json.loads((run_dir / "config.json").read_text())
    m = build_model(cfg, pretrained=False)
    m.load_state_dict(torch.load(run_dir / weights, map_location="cpu"))
    return m.to(DEVICE).eval()


def predict_test(model, root: Path, batch: int = 4, workers: int = 4) -> pd.DataFrame:
    """Predictions on test/images in the long format write_submission expects."""
    test = pd.read_csv(root / "test_images.csv")
    frames = test.assign(key=test.image_id, img_path=[str(root / f) for f in test.file_name])
    p = predict(model, frames, batch, workers)
    return p.rename(columns={"key": "image_id"})
