"""Export YOLO weights to ONNX and optionally benchmark .pt against .onnx."""

import argparse
import shutil
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from smartcattle_ai import Detector, Settings
from smartcattle_ai.media import read_image


def export(weights: Path, imgsz: int = 640, out_dir: Path = ROOT / "outputs" / "onnx") -> Path:
    weights = Path(weights)
    if not weights.is_file():
        raise ValueError(f"Weights not found: {weights}")
    if imgsz < 32 or imgsz % 32:
        raise ValueError("imgsz must be a positive multiple of 32")

    produced = YOLO(str(weights)).export(format="onnx", imgsz=imgsz, dynamic=False, simplify=True)
    source = Path(produced) if produced else weights.with_suffix(".onnx")
    if not source.is_file():
        source = weights.with_suffix(".onnx")
    if not source.is_file():
        raise ValueError(f"Export did not produce an ONNX file for {weights}")

    target = Path(out_dir) / f"{weights.stem}_{imgsz}.onnx"
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != target.resolve():
        if target.exists():
            target.unlink()
        shutil.move(str(source), str(target))
    return target


def benchmark(weights: str, image, runs: int = 20, imgsz: int = 640) -> float:
    if runs < 1:
        raise ValueError("runs must be at least 1")
    detector = Detector(Settings(weights=weights, imgsz=imgsz, device="cpu"))
    detector.detect(image)
    start = perf_counter()
    for _ in range(runs):
        detector.detect(image)
    return (perf_counter() - start) * 1000 / runs


def _benchmark_image() -> np.ndarray:
    samples = sorted((ROOT / "data" / "samples").glob("*.jpg"))
    if not samples:
        return np.zeros((480, 640, 3), dtype=np.uint8)
    return read_image(samples[0])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path,
                        default=ROOT / "models" / "cattle_coco_yolo11n_best.pt")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs" / "onnx")
    parser.add_argument("--benchmark", action="store_true",
                        help="Time .pt and .onnx inference on the exported model")
    args = parser.parse_args(argv)

    try:
        exported = export(args.weights, args.imgsz, args.out_dir)
        print(f"Exported: {exported}")
        if args.benchmark:
            image = _benchmark_image()
            pt_ms = benchmark(str(args.weights), image, imgsz=args.imgsz)
            onnx_ms = benchmark(str(exported), image, imgsz=args.imgsz)
            speedup = pt_ms / onnx_ms if onnx_ms > 0 else float("inf")
            print(f"\n| Model | ms/frame | Speedup |")
            print(f"|---|---:|---:|")
            print(f"| {Path(args.weights).name} | {pt_ms:.2f} | 1.00x |")
            print(f"| {exported.name} | {onnx_ms:.2f} | {speedup:.2f}x |")
    except (ValueError, FileNotFoundError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
