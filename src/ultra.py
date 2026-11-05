"""Ultralytics helpers: YOLOv8 (anchor-free) and RT-DETR (query-based) under the shared protocol.

Training uses the Ultralytics trainer on the dataset exported by
vdet.build_yolo_dataset; the reported numbers always come from our own
evaluator (vdet.coco_evaluate) on the original, unmasked holdout frames.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import vdet

# Augmentation switched off except horizontal flip: the same recipe as the
# torchvision detectors, so model comparisons are not augmentation comparisons.
PROTOCOL_AUG = dict(fliplr=0.5, flipud=0.0, mosaic=0.0, mixup=0.0, copy_paste=0.0, degrees=0.0,
                    translate=0.0, scale=0.0, shear=0.0, perspective=0.0, hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
                    erasing=0.0, close_mosaic=0)


def _build(cfg: dict):
    from ultralytics import RTDETR, YOLO

    cls = RTDETR if cfg.get("family") == "rtdetr" else YOLO
    m = cls(cfg["model"])
    if cfg.get("weights"):  # e.g. yolov8s-p2.yaml initialised from yolov8s.pt
        m = m.load(cfg["weights"])
    return m


def _add_callbacks(model, cfg: dict) -> None:
    """Settings the Ultralytics CLI does not expose, applied right after the trainer builds its model."""

    def patch(trainer):
        targets = [trainer.model] + ([trainer.ema.ema] if getattr(trainer, "ema", None) else [])
        for mm in targets:
            if cfg.get("family") == "rtdetr":
                head = mm.model[-1]
                if "num_queries" in cfg:
                    head.num_queries = cfg["num_queries"]
                if "num_denoising" in cfg:
                    head.num_denoising = cfg["num_denoising"]
                if "cost_gain" in cfg:
                    crit = mm.init_criterion(); crit.matcher.cost_gain = dict(cfg["cost_gain"]); mm.criterion = crit
            elif "tal_topk" in cfg:
                from ultralytics.utils.loss import v8DetectionLoss
                mm.criterion = v8DetectionLoss(mm, tal_topk=cfg["tal_topk"])
        print("patched:", {k: cfg[k] for k in ("num_queries", "num_denoising", "cost_gain", "tal_topk") if k in cfg})

    model.add_callback("on_pretrain_routine_end", patch)


def train_ultra(cfg: dict, data_yaml: str | Path, val_frames: pd.DataFrame, anns: pd.DataFrame,
                out_dir: str | Path, workers: int = 4) -> dict:
    """Train (or resume) one config, then evaluate the last epoch with our evaluator."""
    out = Path(out_dir)
    if (out / "metrics.json").exists():
        print(f"{out.name}: already finished"); return json.loads((out / "metrics.json").read_text())
    vdet.seed_everything(cfg.get("seed", vdet.SEED))
    last = out / "weights" / "last.pt"
    if last.exists():  # resume an interrupted session
        from ultralytics import RTDETR, YOLO
        model = (RTDETR if cfg.get("family") == "rtdetr" else YOLO)(str(last))
        _add_callbacks(model, cfg)
        model.train(resume=True)
    else:
        model = _build(cfg)
        _add_callbacks(model, cfg)
        args = dict(data=str(data_yaml), imgsz=cfg["imgsz"], epochs=cfg["epochs"], batch=cfg["batch"],
                    seed=cfg.get("seed", vdet.SEED), deterministic=False, project=str(out.parent), name=out.name,
                    exist_ok=True, workers=workers, max_det=vdet.MAX_DETS, plots=True,
                    **{**PROTOCOL_AUG, **cfg.get("aug", {})}, **cfg.get("train_args", {}))
        model.train(**args)
    (out / "config.json").write_text(json.dumps(cfg, indent=1))

    hist = pd.read_csv(out / "results.csv"); hist.columns = [c.strip() for c in hist.columns]
    hist = hist.rename(columns={"metrics/mAP50-95(B)": "AP_ultra", "metrics/mAP50(B)": "AP50_ultra"})
    (out / "history.json").write_text(hist.to_json(orient="records", indent=1))

    model = load(out)
    preds = predict_frames(model, val_frames, cfg["imgsz"], conf=cfg.get("conf", 0.01), iou=cfg.get("nms_iou", 0.6))
    preds.to_csv(out / "val_preds.csv.gz", index=False)
    m = vdet.coco_evaluate(val_frames, anns, preds)
    m.update(measure_speed(model, val_frames, cfg["imgsz"]))
    m["epochs"] = int(len(hist))
    (out / "metrics.json").write_text(json.dumps(m, indent=1))
    return m


def load(run_dir: str | Path, weights: str = "last.pt"):
    """Load a finished run. last.pt = last epoch (same rule as the torchvision runs)."""
    from ultralytics import RTDETR, YOLO
    run_dir = Path(run_dir)
    cfg = json.loads((run_dir / "config.json").read_text()) if (run_dir / "config.json").exists() else {}
    return (RTDETR if cfg.get("family") == "rtdetr" else YOLO)(str(run_dir / "weights" / weights))


def _results_to_df(results, keys) -> list[pd.DataFrame]:
    out = []
    for k, r in zip(keys, results):
        b = r.boxes
        xyxy = b.xyxy.cpu().numpy() if len(b) else np.zeros((0, 4))
        out.append(pd.DataFrame({"key": k, "cls": b.cls.cpu().numpy().astype(int) if len(b) else [],
                                 "score": b.conf.cpu().numpy() if len(b) else [],
                                 "x": xyxy[:, 0], "y": xyxy[:, 1], "w": xyxy[:, 2] - xyxy[:, 0], "h": xyxy[:, 3] - xyxy[:, 1]}))
    return out


def predict_images(model, images: list, imgsz: int, conf: float = 0.01, iou: float = 0.6,
                   augment: bool = False) -> list[pd.DataFrame]:
    """images: list of BGR numpy arrays or paths -> one DataFrame per image (key = index)."""
    res = model.predict(images, imgsz=imgsz, conf=conf, iou=iou, max_det=vdet.MAX_DETS,
                        augment=augment, verbose=False, half=_half())
    return _results_to_df(res, range(len(images)))


def predict_frames(model, frames: pd.DataFrame, imgsz: int, conf: float = 0.01, iou: float = 0.6,
                   augment: bool = False, chunk: int = 16) -> pd.DataFrame:
    """Predict frames (key, img_path) -> long DataFrame in original pixel coordinates."""
    dfs = []
    keys, paths = frames.key.tolist(), frames.img_path.tolist()
    for i in range(0, len(paths), chunk):
        res = model.predict(paths[i:i + chunk], imgsz=imgsz, conf=conf, iou=iou, max_det=vdet.MAX_DETS,
                            augment=augment, verbose=False, half=_half())
        dfs += _results_to_df(res, keys[i:i + chunk])
    return pd.concat(dfs, ignore_index=True)


def _half() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def measure_speed(model, frames: pd.DataFrame, imgsz: int, n: int = 100, warmup: int = 10) -> dict:
    """Batch-1 latency = Ultralytics preprocess + inference + postprocess (ms)."""
    paths = frames.sample(min(n + warmup, len(frames)), random_state=0).img_path.tolist()
    warmup = min(warmup, len(paths) // 2)
    times = []
    for i, p in enumerate(paths):
        r = model.predict(p, imgsz=imgsz, conf=0.01, max_det=vdet.MAX_DETS, verbose=False, half=_half())[0]
        if i >= warmup:
            times.append(sum(r.speed.values()))
    ms = float(np.mean(times)) if times else float("nan")
    try:
        import torch
        dev = torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu"
    except ImportError:
        dev = "cpu"
    return {"ms_per_img": ms, "fps": 1000 / ms, "device": dev}


def predict_test(model, root: Path, imgsz: int, **kw) -> pd.DataFrame:
    test = pd.read_csv(root / "test_images.csv")
    frames = test.assign(key=test.image_id, img_path=[str(root / f) for f in test.file_name])
    return predict_frames(model, frames, imgsz, **kw).rename(columns={"key": "image_id"})
