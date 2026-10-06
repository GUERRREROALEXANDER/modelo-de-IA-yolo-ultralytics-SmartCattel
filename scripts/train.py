"""Fine-tune YOLO on cattle and record training and test metrics."""

import argparse
import csv
import json
import os
import platform
import shutil
import time
from pathlib import Path

import torch
from ultralytics import YOLO
from ultralytics.utils import YAML


ROOT = Path(__file__).resolve().parents[1]
METRIC_KEYS = {
    "precision": "metrics/precision(B)",
    "recall": "metrics/recall(B)",
    "mAP50": "metrics/mAP50(B)",
    "mAP50-95": "metrics/mAP50-95(B)",
}


def hardware_summary(device):
    try:
        import psutil
        ram_gb = round(psutil.virtual_memory().total / 1024**3, 2)
    except ImportError:
        ram_gb = None
    return {
        "cpu": platform.processor(),
        "logical_cores": os.cpu_count(),
        "ram_gb": ram_gb,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": device,
    }


def detection_metrics(results_dict):
    return {name: float(results_dict[key]) for name, key in METRIC_KEYS.items()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/train_cattle.yaml"))
    for key in ("epochs", "batch", "imgsz"):
        parser.add_argument(f"--{key}", type=int)
    parser.add_argument("--name")
    parser.add_argument("--weights-out", type=Path, default=Path("models/cattle_yolo11n_best.pt"))
    parser.add_argument("--fraction", type=float, help="Fraction of training data for a debug run")
    args = parser.parse_args(argv)

    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    cfg = YAML.load(config_path)
    for key in ("epochs", "batch", "imgsz", "name", "fraction"):
        value = getattr(args, key)
        if value is not None:
            cfg[key] = value
    if args.fraction is not None and not 0 < args.fraction <= 1:
        parser.error("--fraction must be greater than 0 and at most 1")

    hardware = hardware_summary(cfg.get("device", "cpu"))
    print("Hardware:", json.dumps(hardware, indent=2))

    data = Path(cfg["data"])
    data = data if data.is_absolute() else ROOT / data
    if not data.is_file():
        parser.error(f"Dataset configuration missing: {data}; run python scripts/build_dataset.py first")
    cfg["data"] = str(data)
    project = Path(cfg["project"])
    cfg["project"] = str(project if project.is_absolute() else ROOT / project)
    local_model = ROOT / cfg["model"]
    if local_model.is_file():
        cfg["model"] = str(local_model)

    model_path = cfg.pop("model")
    run_dir = Path(cfg["project"]) / cfg["name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "hardware.json").write_text(json.dumps(hardware, indent=2) + "\n", encoding="utf-8")
    model = YOLO(model_path)
    start = time.perf_counter()
    train_metrics = model.train(**cfg)
    duration = time.perf_counter() - start
    run_dir = Path(model.trainer.save_dir)
    (run_dir / "hardware.json").write_text(json.dumps(hardware, indent=2) + "\n", encoding="utf-8")

    best = run_dir / "weights" / "best.pt"
    if not best.is_file():
        raise FileNotFoundError(f"Training finished without best weights: {best}")
    weights = args.weights_out if args.weights_out.is_absolute() else ROOT / args.weights_out
    weights.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, weights)

    with (run_dir / "results.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"No training epochs found in {run_dir / 'results.csv'}")
    best_row = max(rows, key=lambda row: float(row.get("fitness") or row["metrics/mAP50-95(B)"]))
    test_metrics = YOLO(str(weights)).val(data=str(data), split="test", imgsz=cfg["imgsz"],
                                          batch=cfg["batch"], device=cfg["device"],
                                          project=cfg["project"], name=f'{cfg["name"]}_test',
                                          exist_ok=True)

    training_reports = ROOT / "reports" / "training" / run_dir.name
    training_reports.mkdir(parents=True, exist_ok=True)
    for name in ("results.csv", "results.png"):
        shutil.copy2(run_dir / name, training_reports / name)
    report = {
        "config": {"model": model_path, **cfg},
        "hardware": hardware,
        "epochs_run": len(rows),
        "best_epoch": int(best_row["epoch"]),
        "final_val_metrics": detection_metrics(train_metrics.results_dict),
        "test_metrics": detection_metrics(test_metrics.results_dict),
        "training_duration_seconds": duration,
        "best_pt": str(best),
    }
    report_path = ROOT / "reports" / f"{run_dir.name}.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Saved weights: {weights}")
    print(f"Saved report: {report_path}")


if __name__ == "__main__":
    main()
