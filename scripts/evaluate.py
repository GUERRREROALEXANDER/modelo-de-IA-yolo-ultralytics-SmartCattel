"""Compare cattle detection metrics across one or more model weights."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from smartcattle_ai.evaluation import evaluate


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", action="append", required=True)
    parser.add_argument("--data", type=Path, default=ROOT / "datasets" / "cattle" / "data.yaml")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--operating-conf", type=float, default=0.35)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    results = [evaluate(weight, args.data, split=args.split, imgsz=args.imgsz,
                        operating_conf=args.operating_conf) for weight in args.weights]
    print("| Weights | P | R | mAP50 | mAP50-95 | TP | FP | FN | Negative FP | Count MAE | ms/image |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for result in results:
        op = result["operating"]
        print(f'| {result["weights"]} | {result["precision"]:.4f} | {result["recall"]:.4f} | '
              f'{result["mAP50"]:.4f} | {result["mAP50-95"]:.4f} | {op["TP"]} | '
              f'{op["FP"]} | {op["FN"]} | {op["false_positives_on_negative_images"]} | '
              f'{op["mean_absolute_count_error"]:.3f} | {result["mean_inference_ms_per_image"]:.2f} |')
    output = args.out or ROOT / "reports" / (Path(args.weights[0]).stem + ".json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"Saved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
