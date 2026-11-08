"""Shared utilities for the VisDrone-VID multi-object detection homework.

Everything that more than one notebook needs lives here: data loading,
sequence-level folds, YOLO dataset export with ignore-region masking,
COCO-style evaluation that honours ignored regions, and submission I/O.
"""
from __future__ import annotations

import json
import os
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

CLASSES = ["person", "bicycle", "car", "van", "truck", "tricycle", "bus", "motor"]
NUM_CLASSES = len(CLASSES)
SEED = 42
MAX_DETS = 500  # the competition scores at most 500 detections per image


# --------------------------------------------------------------------------- #
# Reproducibility and paths
# --------------------------------------------------------------------------- #
def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def find_data_root(extra: list[str] | None = None) -> Path:
    """Locate the folder that contains train/, val/ and test/."""
    cands: list[Path] = []
    if os.environ.get("VDET_DATA"):
        cands.append(Path(os.environ["VDET_DATA"]))
    cands += [Path(p) for p in (extra or [])]
    kin = Path("/kaggle/input")
    if kin.exists():
        cands += sorted(kin.glob("*")) + sorted(kin.glob("*/*"))
    cands += [Path("data"), Path("../data")]
    for c in cands:
        if (c / "train" / "annotations_coco.json").exists():
            return c.resolve()
    raise FileNotFoundError("Data root not found; set the VDET_DATA env variable.")


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_split(root: Path, split: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (images, annotations) for one labelled split from its COCO json.

    images: one row per frame with a globally unique `key` = "split/clip/frame".
    annotations: one row per box, pixel xywh (top-left), `cls` 0-based,
    `iscrowd` == 1 for ignored regions (their `cls` is -1).
    """
    with open(root / split / "annotations_coco.json") as f:
        coco = json.load(f)

    imgs = pd.DataFrame(coco["images"])
    fn = imgs["file_name"].map(Path)
    imgs["clip"] = fn.map(lambda p: p.parts[-2])
    imgs["frame"] = fn.map(lambda p: p.stem)
    imgs["split"] = split
    imgs["clip_uid"] = split + "/" + imgs["clip"]
    imgs["key"] = imgs["clip_uid"] + "/" + imgs["frame"]
    imgs["img_path"] = [str(root / split / "images" / c / p.name) for c, p in zip(imgs["clip"], fn)]
    num = pd.to_numeric(imgs["frame"], errors="coerce")
    imgs["frame_idx"] = num if num.notna().all() else imgs.groupby("clip")["frame"].rank(method="first")
    imgs["frame_idx"] = imgs["frame_idx"].astype(int)
    imgs = imgs.rename(columns={"id": "coco_id"}).sort_values(["clip", "frame_idx"]).reset_index(drop=True)

    anns = pd.DataFrame(coco["annotations"])
    xywh = np.array(anns["bbox"].tolist(), dtype=float).reshape(-1, 4)
    anns[["x", "y", "w", "h"]] = xywh
    anns["iscrowd"] = anns.get("iscrowd", 0).fillna(0).astype(int)
    anns["cls"] = np.where(anns["iscrowd"] == 1, -1, anns["category_id"] - 1).astype(int)
    # The organisers' json repeats every ignored region once per class (8 copies, one per
    # category_id) so that per-category COCO matching skips it. Keep one class-agnostic copy;
    # coco_evaluate re-creates the per-class copies itself.
    crowd = anns["iscrowd"] == 1
    dup = crowd & anns.duplicated(subset=["image_id", "x", "y", "w", "h", "iscrowd"])
    anns = anns[~dup].reset_index(drop=True)
    anns = anns.drop(columns=["bbox", "segmentation"], errors="ignore")
    anns = anns.merge(
        imgs[["coco_id", "key", "clip", "clip_uid", "frame_idx", "width", "height", "split"]],
        left_on="image_id", right_on="coco_id", how="left",
    ).drop(columns=["coco_id"])
    return imgs, anns


def load_all(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Concatenate the labelled train and val splits."""
    parts = [load_split(root, s) for s in ("train", "val")]
    return (pd.concat([p[0] for p in parts], ignore_index=True),
            pd.concat([p[1] for p in parts], ignore_index=True))


def read_yolo_txt(path: str | Path, W: int, H: int, has_cls: bool = True) -> np.ndarray:
    """Read a YOLO txt into pixel top-left xywh. Rows: [cls, x, y, w, h] or [x, y, w, h]."""
    ncol = 5 if has_cls else 4
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return np.zeros((0, ncol))
    a = np.loadtxt(p, ndmin=2)
    off = 1 if has_cls else 0
    cx, cy, w, h = (a[:, off + i] for i in range(4))
    box = np.stack([(cx - w / 2) * W, (cy - h / 2) * H, w * W, h * H], 1)
    return np.concatenate([a[:, :1], box], 1) if has_cls else box


def label_path(img_path: str, kind: str = "labels") -> Path:
    """images/<clip>/<frame>.jpg -> labels|ignore/<clip>/<frame>.txt"""
    p = Path(img_path)
    return p.parents[2] / kind / p.parent.name / (p.stem + ".txt")


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #
def iou_xywh(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Element-wise IoU between two (n,4) arrays of xywh boxes."""
    x1 = np.maximum(a[:, 0], b[:, 0]); y1 = np.maximum(a[:, 1], b[:, 1])
    x2 = np.minimum(a[:, 0] + a[:, 2], b[:, 0] + b[:, 2])
    y2 = np.minimum(a[:, 1] + a[:, 3], b[:, 1] + b[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    return inter / (a[:, 2] * a[:, 3] + b[:, 2] * b[:, 3] - inter + 1e-9)


def kmeans_anchors(wh: np.ndarray, k: int = 9, iters: int = 300, seed: int = SEED) -> tuple[np.ndarray, float]:
    """YOLOv2-style k-means on box (w, h) with 1 - IoU distance. Returns (anchors, mean best IoU)."""
    rng = np.random.default_rng(seed)
    wh = wh[(wh > 0).all(1)]
    cent = wh[rng.choice(len(wh), k, replace=False)].copy()

    def iou(b, c):
        inter = np.minimum(b[:, None, 0], c[None, :, 0]) * np.minimum(b[:, None, 1], c[None, :, 1])
        return inter / (b[:, None].prod(2) + c[None].prod(2) - inter)

    for _ in range(iters):
        assign = iou(wh, cent).argmax(1)
        new = np.array([np.median(wh[assign == j], 0) if (assign == j).any() else cent[j] for j in range(k)])
        if np.allclose(new, cent):
            break
        cent = new
    cent = cent[np.argsort(cent.prod(1))]
    return cent, float(iou(wh, cent).max(1).mean())


# --------------------------------------------------------------------------- #
# Sequence-level folds
# --------------------------------------------------------------------------- #
def make_folds(imgs: pd.DataFrame, n_splits: int = 5) -> pd.DataFrame:
    """GroupKFold over clips: every frame of a clip lands in the same fold."""
    from sklearn.model_selection import GroupKFold

    folds = pd.Series(-1, index=imgs.index)
    for f, (_, va) in enumerate(GroupKFold(n_splits=n_splits).split(imgs, groups=imgs["clip_uid"])):
        folds.iloc[va] = f
    out = imgs.assign(fold=folds).groupby("clip_uid")["fold"].first().reset_index()
    assert out.groupby("clip_uid")["fold"].nunique().max() == 1
    return out


def subsample(imgs: pd.DataFrame, step: int) -> pd.DataFrame:
    """Keep every `step`-th frame of each clip (by position, so gaps in numbering are fine)."""
    pos = imgs.sort_values(["clip_uid", "frame_idx"]).groupby("clip_uid").cumcount()
    return imgs.loc[pos[pos % step == 0].index]


# --------------------------------------------------------------------------- #
# YOLO export with ignore-region masking
# --------------------------------------------------------------------------- #
def _mask_image(img: np.ndarray, ign: np.ndarray, gt: np.ndarray, fill: int = 114) -> np.ndarray:
    """Grey out ignored regions, then restore pixels of annotated boxes that overlap them."""
    out = img.copy()
    H, W = img.shape[:2]

    def rect(b):
        x, y, w, h = b
        return (max(int(x), 0), max(int(y), 0), min(int(np.ceil(x + w)), W), min(int(np.ceil(y + h)), H))

    for b in ign:
        x1, y1, x2, y2 = rect(b)
        out[y1:y2, x1:x2] = fill
    for b in gt:
        x1, y1, x2, y2 = rect(b)
        out[y1:y2, x1:x2] = img[y1:y2, x1:x2]
    return out


def build_yolo_dataset(
    imgs: pd.DataFrame, anns: pd.DataFrame, folds: pd.DataFrame, val_fold: int, out_dir: str | Path,
    step: int = 5, val_step: int | None = None, mask_ignore: bool = True, workers: int = 8,
    train_all: bool = False,
) -> Path:
    """Export one fold to Ultralytics layout: out/images/{train,val}/<clip>/x.jpg (+labels/...).

    Train frames are subsampled with `step`; masked copies are written only for
    frames that have ignored regions, all other images are symlinked.
    Returns the path of the generated data.yaml.
    """
    import cv2

    out = Path(out_dir)
    df = imgs.merge(folds, on="clip_uid")
    parts = {
        # train_all: final model on every labelled clip (val part is then only for monitoring)
        "train": subsample(df if train_all else df[df["fold"] != val_fold], step),
        "val": subsample(df[df["fold"] == val_fold], val_step or step),
    }
    by_key = {k: g for k, g in anns.groupby("key")}
    empty = anns.iloc[:0]

    def export(row, part):
        safe_clip = row.clip_uid.replace("/", "__")
        dst_img = out / "images" / part / safe_clip / Path(row.img_path).name
        dst_lbl = out / "labels" / part / safe_clip / (Path(row.img_path).stem + ".txt")
        dst_img.parent.mkdir(parents=True, exist_ok=True)
        dst_lbl.parent.mkdir(parents=True, exist_ok=True)
        a = by_key.get(row.key, empty)
        gt, ign = a[a.iscrowd == 0], a[a.iscrowd == 1]
        W, H = row.width, row.height
        # labels: clip to the image, convert to normalised centre format
        x1 = gt.x.clip(0, W); y1 = gt.y.clip(0, H)
        x2 = (gt.x + gt.w).clip(0, W); y2 = (gt.y + gt.h).clip(0, H)
        ok = (x2 - x1 > 1) & (y2 - y1 > 1)
        lines = [f"{c} {(a_ + b_) / 2 / W:.6f} {(c_ + d_) / 2 / H:.6f} {(b_ - a_) / W:.6f} {(d_ - c_) / H:.6f}"
                 for c, a_, b_, c_, d_ in zip(gt.cls[ok], x1[ok], x2[ok], y1[ok], y2[ok])]
        dst_lbl.write_text("\n".join(lines) + ("\n" if lines else ""))
        if dst_img.exists() or dst_img.is_symlink():
            dst_img.unlink()
        if mask_ignore and len(ign):
            img = cv2.imread(row.img_path)
            img = _mask_image(img, ign[["x", "y", "w", "h"]].values, gt[["x", "y", "w", "h"]].values)
            cv2.imwrite(str(dst_img), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            return 1
        try:
            os.symlink(row.img_path, dst_img)
        except OSError:
            import shutil
            shutil.copy(row.img_path, dst_img)
        return 0

    stats = {}
    for part, sub in parts.items():
        with ThreadPoolExecutor(workers) as ex:
            masked = sum(ex.map(lambda r: export(r, part), sub.itertuples()))
        stats[part] = (len(sub), masked)
    yaml = out / "data.yaml"
    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(CLASSES))
    yaml.write_text(f"path: {out.resolve()}\ntrain: images/train\nval: images/val\nnames:\n{names}\n")
    for part, (n, m) in stats.items():
        print(f"{part:5s}: {n:6d} frames, {m:5d} masked copies")
    return yaml


# --------------------------------------------------------------------------- #
# Evaluation (COCO protocol, 500 detections, ignored regions as crowd)
# --------------------------------------------------------------------------- #
def coco_evaluate(gt_imgs: pd.DataFrame, gt_anns: pd.DataFrame, preds: pd.DataFrame,
                  max_dets: int = MAX_DETS, per_class: bool = True, verbose: bool = False) -> dict:
    """COCO bbox evaluation on the frames in `gt_imgs`.

    preds columns: key, cls (0-based), score, x, y, w, h (pixels, top-left).
    Ignored regions are class-agnostic, but COCO matches crowd boxes per
    category, so each region is duplicated once for every class.
    """
    try:
        from faster_coco_eval import COCO, COCOeval_faster as COCOeval
    except ImportError:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval

    kid = {k: i + 1 for i, k in enumerate(gt_imgs["key"])}
    anns = gt_anns[gt_anns["key"].isin(kid)]
    real, crowd = anns[anns.iscrowd == 0], anns[anns.iscrowd == 1]
    crowd = pd.concat([crowd.assign(cls=c) for c in range(NUM_CLASSES)], ignore_index=True)
    allg = pd.concat([real, crowd], ignore_index=True)
    gt = {
        "images": [{"id": kid[k], "width": int(w), "height": int(h)}
                   for k, w, h in zip(gt_imgs["key"], gt_imgs["width"], gt_imgs["height"])],
        "categories": [{"id": i + 1, "name": n} for i, n in enumerate(CLASSES)],
        "annotations": [
            {"id": i + 1, "image_id": kid[r.key], "category_id": int(r.cls) + 1,
             "bbox": [r.x, r.y, r.w, r.h], "area": r.w * r.h, "iscrowd": int(r.iscrowd)}
            for i, r in enumerate(allg.itertuples())
        ],
    }
    p = preds[preds["key"].isin(kid)]
    p = p.sort_values("score", ascending=False).groupby("key").head(max_dets)
    keys = ["AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large"]
    if len(p) == 0:
        return {k: 0.0 for k in keys}
    dt = [{"image_id": kid[r.key], "category_id": int(r.cls) + 1, "bbox": [r.x, r.y, r.w, r.h],
           "score": float(r.score)} for r in p.itertuples()]

    import contextlib, io
    sink = contextlib.nullcontext() if verbose else contextlib.redirect_stdout(io.StringIO())
    with sink:
        cg = COCO(); cg.dataset = gt; cg.createIndex()
        cd = cg.loadRes(dt)
        ev = COCOeval(cg, cd, "bbox")
        ev.params.maxDets = [1, 100, max_dets]  # summarize() reports AP at maxDets[2]
        ev.evaluate(); ev.accumulate(); ev.summarize()
    res = dict(zip(keys, map(float, ev.stats[:6])))
    res["AR_max"] = float(ev.stats[8])
    if per_class:
        prec = ev.eval["precision"][:, :, :, 0, -1]  # [iou, recall, class]
        for c, n in enumerate(CLASSES):
            v = prec[:, :, c]
            res[f"AP_{n}"] = float(v[v > -1].mean()) if (v > -1).any() else float("nan")
    return res


# --------------------------------------------------------------------------- #
# Submission I/O
# --------------------------------------------------------------------------- #
def to_prediction_string(df: pd.DataFrame, max_dets: int = MAX_DETS) -> str:
    df = df.sort_values("score", ascending=False).head(max_dets)
    return " ".join(f"{int(c)} {s:.4f} {x:.1f} {y:.1f} {w:.1f} {h:.1f}"
                    for c, s, x, y, w, h in df[["cls", "score", "x", "y", "w", "h"]].itertuples(index=False))


def write_submission(preds: pd.DataFrame, test_csv: str | Path, path: str | Path) -> pd.DataFrame:
    """preds columns: image_id, cls, score, x, y, w, h. Missing images get an empty string."""
    test = pd.read_csv(test_csv)
    # clip boxes to the image and drop degenerate ones (a box at the frame edge can have
    # sub-pixel width, which becomes 0.0 after rounding and is rejected by the checker)
    p = preds.merge(test[["image_id", "width", "height"]], on="image_id", how="inner")
    x1 = p.x.clip(0, p.width - 1); y1 = p.y.clip(0, p.height - 1)
    x2 = (p.x + p.w).clip(0, p.width); y2 = (p.y + p.h).clip(0, p.height)
    p = p.assign(x=x1, y=y1, w=x2 - x1, h=y2 - y1)
    p = p[(p.w >= 1) & (p.h >= 1)]
    strings = {k: to_prediction_string(g) for k, g in p.groupby("image_id")}
    sub = pd.DataFrame({"image_id": test["image_id"],
                        "PredictionString": [strings.get(k, "") for k in test["image_id"]]})
    sub.to_csv(path, index=False)
    check_submission(path, test_csv)
    return sub


def check_submission(path: str | Path, test_csv: str | Path) -> None:
    test = pd.read_csv(test_csv).set_index("image_id")
    sub = pd.read_csv(path, keep_default_na=False)
    assert list(sub.columns) == ["image_id", "PredictionString"], sub.columns
    assert set(sub["image_id"]) == set(test.index) and sub["image_id"].is_unique
    for iid, s in zip(sub["image_id"], sub["PredictionString"]):
        if not s:
            continue
        v = np.array(s.split(), dtype=float).reshape(-1, 6)
        assert len(v) <= MAX_DETS, iid
        assert set(np.unique(v[:, 0])) <= set(range(NUM_CLASSES)), iid
        W, H = test.loc[iid, ["width", "height"]]
        assert (v[:, 4] > 0).all() and (v[:, 5] > 0).all(), iid
        assert (v[:, 2] < W).all() and (v[:, 3] < H).all(), iid
    print(f"submission OK: {len(sub)} rows, {(sub.PredictionString != '').sum()} non-empty")


def parse_submission(path: str | Path) -> pd.DataFrame:
    """Inverse of write_submission: long-format DataFrame of detections."""
    sub = pd.read_csv(path, keep_default_na=False)
    rows = []
    for iid, s in zip(sub["image_id"], sub["PredictionString"]):
        if s:
            for c, sc, x, y, w, h in np.array(s.split(), dtype=float).reshape(-1, 6):
                rows.append((iid, int(c), sc, x, y, w, h))
    return pd.DataFrame(rows, columns=["image_id", "cls", "score", "x", "y", "w", "h"])


# --------------------------------------------------------------------------- #
# Fixed experimental protocol shared by every detector notebook
# --------------------------------------------------------------------------- #
PROTOCOL = {
    "val_fold": 0,     # development holdout (sequence-level GroupKFold, see notebook 01)
    "n_folds": 5,
    "step": 10,        # train on every 10th frame of each clip (notebook 01, Section 8)
    "val_step": 10,    # evaluate on every 10th frame of the holdout clips
    "imgsz": 1280,     # long side after resize (notebook 01, Section 4)
    "epochs": 12,
    "seed": SEED,
}


def protocol_frames(imgs: pd.DataFrame, folds: pd.DataFrame, p: dict = PROTOCOL):
    """(train_frames, val_frames) under the fixed protocol."""
    df = imgs.merge(folds, on="clip_uid")
    return (subsample(df[df.fold != p["val_fold"]], p["step"]),
            subsample(df[df.fold == p["val_fold"]], p["val_step"]))


# --------------------------------------------------------------------------- #
# Error analysis (TIDE-like categories, counted at a fixed score threshold)
# --------------------------------------------------------------------------- #
def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 0] + a[:, None, 2], b[None, :, 0] + b[None, :, 2])
    y2 = np.minimum(a[:, None, 1] + a[:, None, 3], b[None, :, 1] + b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    return inter / (a[:, None, 2] * a[:, None, 3] + b[None, :, 2] * b[None, :, 3] - inter + 1e-9)


def _iof(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Intersection over the area of each box in a (fraction of a inside b)."""
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 0] + a[:, None, 2], b[None, :, 0] + b[None, :, 2])
    y2 = np.minimum(a[:, None, 1] + a[:, None, 3], b[None, :, 1] + b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    return inter / (a[:, None, 2] * a[:, None, 3] + 1e-9)


def error_breakdown(gt_anns: pd.DataFrame, preds: pd.DataFrame, keys, score_thr: float = 0.3, iou_thr: float = 0.5):
    """Label every detection above score_thr and every GT box of the given frames.

    Detection types (checked in order): tp, duplicate (right class, GT already taken),
    class (IoU>=thr with a GT of another class), localization (right class, 0.1<=IoU<thr),
    ignored (>50% inside an ignored region), background.
    Returns (dets, gts) DataFrames; gts has `matched` and `best_iou`.
    """
    keys = set(keys)
    p = preds[preds.key.isin(keys) & (preds.score >= score_thr)].sort_values("score", ascending=False)
    a = gt_anns[gt_anns.key.isin(keys)]
    pd_by, gt_by = dict(tuple(p.groupby("key"))), dict(tuple(a.groupby("key")))
    det_rows, gt_rows = [], []
    for k in keys:
        d = pd_by.get(k, p.iloc[:0]); g_all = gt_by.get(k, a.iloc[:0])
        g, ig = g_all[g_all.iscrowd == 0], g_all[g_all.iscrowd == 1]
        db, gb = d[["x", "y", "w", "h"]].values, g[["x", "y", "w", "h"]].values
        dc, gc = d.cls.values, g.cls.values
        iou = _iou_matrix(db, gb) if len(db) and len(gb) else np.zeros((len(db), len(gb)))
        inside = (_iof(db, ig[["x", "y", "w", "h"]].values).max(1) > 0.5) if len(db) and len(ig) else np.zeros(len(db), bool)
        taken = np.zeros(len(gb), bool); types, gcls = [], []
        for i in range(len(db)):
            same = np.where(gc == dc[i], iou[i], 0)
            j = int(same.argmax()) if len(gb) else -1
            if j >= 0 and same[j] >= iou_thr:
                # prefer an untaken GT of the same class
                free = np.where(~taken, same, 0); jf = int(free.argmax())
                if free[jf] >= iou_thr:
                    taken[jf] = True; types.append("tp")
                else:
                    types.append("duplicate")
            elif len(gb) and iou[i].max() >= iou_thr:
                types.append("class"); gcls.append(int(gc[iou[i].argmax()])); continue
            elif j >= 0 and same[j] >= 0.1:
                types.append("localization")
            elif inside[i]:
                types.append("ignored")
            else:
                types.append("background")
            gcls.append(-1)
        det_rows.append(d.assign(type=types, gt_cls=gcls))
        best = iou.max(0) if len(db) and len(gb) else np.zeros(len(gb))
        gt_rows.append(g.assign(matched=taken, best_iou=best))
    dets = pd.concat(det_rows, ignore_index=True) if det_rows else p.assign(type=[])
    gts = pd.concat(gt_rows, ignore_index=True)
    gts["bucket"] = pd.cut(gts.w * gts.h, [0, 32**2, 96**2, np.inf], labels=["small", "medium", "large"])
    return dets, gts


# --------------------------------------------------------------------------- #
# Collecting results across notebooks / teammates
# --------------------------------------------------------------------------- #
RESULT_FILES = ("config.json", "metrics.json", "history.json")


def export_results(runs_dir: str | Path, dst: str | Path, preds_for: tuple = ()) -> Path:
    """Copy the small JSON files of every finished run to dst/<run>/ and zip dst.

    preds_for: runs whose val_preds.csv.gz is also copied (only the best run of each
    detector family, needed for cross-model comparisons and the WBF ensemble).
    Unzip the archive into the repo's results/ folder and commit it.
    """
    import shutil
    dst = Path(dst)
    for d in Path(runs_dir).iterdir():
        if (d / "metrics.json").exists():
            (dst / d.name).mkdir(parents=True, exist_ok=True)
            files = RESULT_FILES + (("val_preds.csv.gz",) if d.name in preds_for else ())
            for f in files:
                if (d / f).exists():
                    shutil.copy(d / f, dst / d.name / f)
    archive = shutil.make_archive(str(dst), "zip", dst)
    print("results archive:", archive)
    return Path(archive)


def collect_results(dirs, prefix: str = "") -> pd.DataFrame:
    """metrics.json of every run found in any of dirs (later dirs do not override earlier ones)."""
    rows = {}
    for base in dirs:
        base = Path(base)
        if not base.exists():
            continue
        for d in sorted(base.glob(f"{prefix}*")):
            if (d / "metrics.json").exists() and d.name not in rows:
                rows[d.name] = {"run": d.name, **json.loads((d / "metrics.json").read_text())}
    return pd.DataFrame(rows.values()).set_index("run") if rows else pd.DataFrame()


def load_history(dirs, run: str) -> pd.DataFrame | None:
    for base in dirs:
        f = Path(base) / run / "history.json"
        if f.exists():
            return pd.DataFrame(json.loads(f.read_text()))
    return None
