"""Run several experiments at once, one per GPU (Kaggle "GPU T4 x2" = 2 runs in parallel).

Kaggle charges GPU quota per session hour, not per GPU, so two runs in parallel
halve the quota and wall time. Each run is a separate process with its own
CUDA_VISIBLE_DEVICES; a queue starts the next job as soon as a GPU is free.

From a notebook:
    parallel.run_parallel([(name, cfg), ...], RUNS, folds_csv=OUT / "folds.csv", data_yaml=...)
Command line (used internally):
    python parallel.py --name NAME --cfg '{...}' --runs DIR --folds folds.csv [--data-yaml data.yaml]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
TV_ARCHS = {"fasterrcnn_v2", "retinanet_v2", "fcos"}


def _tail(path: Path, n: int = 1) -> str:
    try:
        lines = [l for l in path.read_text(errors="ignore").replace("\r", "\n").splitlines() if l.strip()]
        return "\n".join(lines[-n:])
    except FileNotFoundError:
        return ""


def run_parallel(jobs: list[tuple], runs_dir: str | Path, folds_csv: str | Path,
                 data_yaml: str | Path | None = None, gpus: list | None = None, workers: int = 2,
                 pretrained: bool = True, poll: int = 30, report_every: int = 600) -> dict:
    """Run jobs (name, cfg) or (name, cfg, data_yaml), at most one per GPU at a time. Blocks until all are done; returns {name: returncode}.
    Logs: runs_dir/<name>.log (the notebook prints the latest line of each every `report_every` s)."""
    runs_dir = Path(runs_dir); runs_dir.mkdir(parents=True, exist_ok=True)
    if gpus is None:
        try:
            import torch
            gpus = list(range(torch.cuda.device_count())) or [None]
        except ImportError:
            gpus = [None]
    queue = [j for j in jobs if not (runs_dir / j[0] / "metrics.json").exists()]
    for name in [j[0] for j in jobs]:
        if (runs_dir / name / "metrics.json").exists():
            print(f"{name}: already finished, skipped")
    print(f"{len(queue)} jobs on GPUs {gpus}")
    active, codes, last_report, t0 = {}, {}, 0.0, time.time()
    while queue or active:
        for g in gpus:
            if g not in active and queue:
                job = queue.pop(0); name, cfg = job[0], job[1]
                yaml = job[2] if len(job) > 2 else data_yaml
                log = runs_dir / f"{name}.log"
                cmd = [sys.executable, str(HERE / "parallel.py"), "--name", name, "--cfg", json.dumps(cfg),
                       "--runs", str(runs_dir), "--folds", str(folds_csv), "--workers", str(workers)]
                if yaml:
                    cmd += ["--data-yaml", str(yaml)]
                if not pretrained:
                    cmd += ["--no-pretrained"]
                env = dict(os.environ)
                if g is not None:
                    env["CUDA_VISIBLE_DEVICES"] = str(g)
                fh = open(log, "a")
                active[g] = (name, subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env), fh, log)
                print(f"[{(time.time() - t0) / 60:6.1f} min] started {name} on GPU {g}")
        time.sleep(poll)
        for g, (name, p, fh, log) in list(active.items()):
            if p.poll() is not None:
                fh.close(); codes[name] = p.returncode; del active[g]
                status = "finished" if p.returncode == 0 else f"FAILED (code {p.returncode}), last lines:\n{_tail(log, 15)}"
                print(f"[{(time.time() - t0) / 60:6.1f} min] {name} {status}")
        if active and time.time() - last_report > report_every:
            last_report = time.time()
            for g, (name, _, _, log) in active.items():
                print(f"[{(time.time() - t0) / 60:6.1f} min] {name} (GPU {g}): {_tail(log)[:200]}")
    return codes


def _main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True); ap.add_argument("--cfg", required=True)
    ap.add_argument("--runs", required=True); ap.add_argument("--folds", required=True)
    ap.add_argument("--data-yaml"); ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--no-pretrained", action="store_true")
    a = ap.parse_args()
    sys.path.insert(0, str(HERE))
    import pandas as pd
    import vdet

    cfg = json.loads(a.cfg)
    imgs, anns = vdet.load_all(vdet.find_data_root())
    folds = pd.read_csv(a.folds)
    _, val_frames = vdet.protocol_frames(imgs, folds, vdet.PROTOCOL)
    out = Path(a.runs) / a.name
    if cfg.get("arch") in TV_ARCHS:
        import tvdet
        train_frames, _ = vdet.protocol_frames(imgs, folds, vdet.PROTOCOL)
        m = tvdet.train_run(cfg, train_frames, val_frames, anns, val_frames, out, workers=a.workers,
                            pretrained=not a.no_pretrained)
    else:
        import ultra
        m = ultra.train_ultra(cfg, a.data_yaml, val_frames, anns, out, workers=a.workers)
    print("FINAL", json.dumps({k: v for k, v in m.items() if isinstance(v, (int, float))}))


if __name__ == "__main__":
    _main()
