"""Inference strategies that work with any detector: sliced inference (SAHI-style),
class-wise NMS merging and Weighted Boxes Fusion (implemented from the paper,
Solovyev et al. 2019, arXiv:1910.13302).

A detector is passed as `predict_fn(list_of_bgr_images) -> list_of_DataFrames`
with columns cls, score, x, y, w, h in the pixel coordinates of each input image.
"""
from __future__ import annotations

import cv2
import numpy as np
import pandas as pd

import vdet

COLS = ["key", "cls", "score", "x", "y", "w", "h"]


# --------------------------------------------------------------------------- #
# NMS
# --------------------------------------------------------------------------- #
def nms_merge(df: pd.DataFrame, iou: float = 0.6, max_dets: int = vdet.MAX_DETS) -> pd.DataFrame:
    """Class-wise NMS inside one image."""
    if len(df) == 0:
        return df
    import torch
    from torchvision.ops import batched_nms

    b = torch.as_tensor(df[["x", "y", "w", "h"]].values, dtype=torch.float32)
    b[:, 2:] += b[:, :2]
    keep = batched_nms(b, torch.tensor(df.score.to_numpy(copy=True), dtype=torch.float32),
                       torch.as_tensor(df.cls.values), iou).numpy()[:max_dets]
    return df.iloc[keep]


# --------------------------------------------------------------------------- #
# Sliced inference
# --------------------------------------------------------------------------- #
def tiles(W: int, H: int, tile: int, overlap: float = 0.2) -> list[tuple[int, int, int, int]]:
    """Overlapping square-ish crops covering the image (last row/column aligned to the border)."""
    tw, th = min(tile, W), min(tile, H)
    step_x, step_y = max(1, int(tw * (1 - overlap))), max(1, int(th * (1 - overlap)))
    xs = list(range(0, max(W - tw, 0) + 1, step_x)); ys = list(range(0, max(H - th, 0) + 1, step_y))
    if xs[-1] + tw < W: xs.append(W - tw)
    if ys[-1] + th < H: ys.append(H - th)
    return [(x, y, x + tw, y + th) for y in ys for x in xs]


def sliced_predict(predict_fn, frames: pd.DataFrame, tile: int, overlap: float = 0.2, full: bool = True,
                   merge_iou: float = 0.6, edge_margin: int = 2) -> pd.DataFrame:
    """Run predict_fn on crops (+ the full frame) and merge with class-wise NMS.

    Boxes touching an inner crop border are dropped: they are usually truncated
    objects that a neighbouring crop sees completely.
    """
    out = []
    for r in frames.itertuples():
        img = cv2.imread(r.img_path); H, W = img.shape[:2]
        boxes = tiles(W, H, tile, overlap)
        crops = [img[y0:y1, x0:x1] for x0, y0, x1, y1 in boxes]
        preds = predict_fn(crops + ([img] if full else []))
        parts = []
        for (x0, y0, x1, y1), p in zip(boxes, preds[:len(boxes)]):
            if not len(p): continue
            inner = np.ones(len(p), bool)
            if x0 > 0: inner &= p.x.values > edge_margin
            if y0 > 0: inner &= p.y.values > edge_margin
            if x1 < W: inner &= (p.x + p.w).values < (x1 - x0) - edge_margin
            if y1 < H: inner &= (p.y + p.h).values < (y1 - y0) - edge_margin
            parts.append(p[inner].assign(x=p.x[inner] + x0, y=p.y[inner] + y0))
        if full and len(preds[-1]):
            parts.append(preds[-1])
        merged = nms_merge(pd.concat(parts, ignore_index=True), merge_iou) if parts else pd.DataFrame(columns=COLS[1:])
        out.append(merged.assign(key=r.key))
    return pd.concat(out, ignore_index=True)[COLS]


# --------------------------------------------------------------------------- #
# Weighted Boxes Fusion
# --------------------------------------------------------------------------- #
def _iou_one(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    x1 = np.maximum(box[0], boxes[:, 0]); y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2]); y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    a = (box[2] - box[0]) * (box[3] - box[1]); b = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / (a + b - inter + 1e-9)


def wbf_image(dfs: list[pd.DataFrame], weights: list[float], iou_thr: float = 0.55,
              skip_thr: float = 0.0) -> pd.DataFrame:
    """WBF for one image. dfs: one DataFrame per model (cls, score, x, y, w, h)."""
    rows = []
    n_models_w = float(sum(weights))
    for c in sorted(set().union(*[set(d.cls.unique()) for d in dfs if len(d)])):
        items = []
        for d, w in zip(dfs, weights):
            d = d[(d.cls == c) & (d.score >= skip_thr)]
            if len(d):
                xyxy = np.c_[d.x, d.y, d.x + d.w, d.y + d.h]
                items.append(np.c_[d.score.values * w, np.full(len(d), w), xyxy])
        if not items: continue
        a = np.concatenate(items); a = a[np.argsort(-a[:, 0])]
        fused = np.zeros((0, 4)); members: list[list[np.ndarray]] = []
        for row in a:
            j = -1
            if len(fused):
                ious = _iou_one(row[2:], fused); j = int(ious.argmax())
                if ious[j] <= iou_thr: j = -1
            if j < 0:
                members.append([row]); fused = np.vstack([fused, row[2:]])
            else:
                members[j].append(row)
                m = np.array(members[j]); fused[j] = (m[:, 0:1] * m[:, 2:]).sum(0) / m[:, 0].sum()
        for m, box in zip(members, fused):
            m = np.array(m)
            # mean weighted score, down-weighted when fewer models agree
            score = m[:, 0].sum() / len(m) * min(m[:, 1].sum(), n_models_w) / n_models_w
            rows.append((c, score, box[0], box[1], box[2] - box[0], box[3] - box[1]))
    out = pd.DataFrame(rows, columns=["cls", "score", "x", "y", "w", "h"])
    return out.sort_values("score", ascending=False).head(vdet.MAX_DETS)


def wbf(preds: list[pd.DataFrame], weights: list[float] | None = None, iou_thr: float = 0.55,
        skip_thr: float = 0.01) -> pd.DataFrame:
    """Fuse long-format predictions of several models (each with a key column)."""
    weights = weights or [1.0] * len(preds)
    groups = [dict(tuple(p.groupby("key"))) for p in preds]
    keys = sorted(set().union(*[g.keys() for g in groups]))
    empty = pd.DataFrame(columns=COLS)
    out = [wbf_image([g.get(k, empty) for g in groups], weights, iou_thr, skip_thr).assign(key=k) for k in keys]
    return pd.concat(out, ignore_index=True)[COLS] if out else empty
