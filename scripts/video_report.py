"""Compare cow/person detections across weights and videos, with contact sheets."""

import argparse
import os
import sys
from pathlib import Path
from time import perf_counter
from urllib.parse import quote

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from smartcattle_ai import Detector, Settings
from smartcattle_ai.media import process_video, save_image


def contact_sheet(video: Path, frame_count: int, output: Path) -> None:
    """Stack first, middle and last processed frames; repeat frames for short clips."""
    indices = [round(index * (frame_count - 1) / 2) for index in range(3)]
    capture = cv2.VideoCapture(str(video))
    selected = {}
    try:
        if not capture.isOpened():
            raise ValueError(f"Could not open annotated video: {video}")
        for index in range(indices[-1] + 1):
            ok, frame = capture.read()
            if not ok:
                raise ValueError(f"Could not read annotated frame {index}: {video}")
            if index in indices:
                height, width = frame.shape[:2]
                selected[index] = cv2.resize(frame, (640, max(1, round(height * 640 / width))))
    finally:
        capture.release()
    save_image(output, np.vstack([selected[index] for index in indices]))


def format_table(rows: list[dict]) -> str:
    lines = [
        "| Weights | Video | Frames processed | Max cattle | Avg cattle | Max persons | Avg persons | Frames with cattle | Mean ms/frame |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        values = [row["weights"], row["video"], row["frames_processed"],
                  row["max_cattle"], f'{row["avg_cattle"]:.2f}',
                  row["max_persons"], f'{row["avg_persons"]:.2f}',
                  row["frames_with_cattle"], f'{row["mean_ms"]:.2f}']
        lines.append("| " + " | ".join(str(value).replace("|", "\\|").replace("\n", " ")
                                      for value in values) + " |")
    return "\n".join(lines)


def _link(path: Path, report_path: Path) -> str:
    try:
        relative = Path(os.path.relpath(path, report_path.parent)).as_posix()
        return quote(relative, safe="/.")
    except ValueError:  # Windows paths on different drives cannot be relative.
        return path.resolve().as_uri()


def report(videos, weights, out: Path, vid_stride: int = 2) -> list[dict]:
    if vid_stride < 1:
        raise ValueError("vid_stride must be at least 1")
    videos = [Path(video) for video in videos]
    weights = list(weights)
    # Output names use stems, so reject ambiguous inputs before overwriting artifacts.
    for label, paths in (("video", videos), ("weights", weights)):
        stems = [Path(path).stem.casefold() for path in paths]
        if len(stems) != len(set(stems)):
            raise ValueError(f"Each {label} must have a unique filename stem")
    out = Path(out).resolve()
    rows = []
    artifacts = []
    for weight in weights:
        weight = str(weight)
        for video in videos:
            detector = Detector(Settings(weights=weight, class_names=("cow", "person")))
            start = perf_counter()
            result = process_video(detector, video,
                                   ROOT / "outputs" / "videos" / Path(weight).stem,
                                   vid_stride=vid_stride)
            elapsed = perf_counter() - start
            count = result["frames_processed"]
            rows.append({
                "weights": weight, "video": str(video), "frames_processed": count,
                "max_cattle": result["max_cattle"], "avg_cattle": result["avg_cattle"],
                "max_persons": result["max_persons"],
                "avg_persons": result["total_persons"] / count if count else 0.0,
                "frames_with_cattle": result["frames_with_cattle"],
                "mean_ms": elapsed * 1000 / count if count else 0.0,
            })
            if count:
                sheet = ROOT / "reports" / "video_frames" / f"{Path(weight).stem}_{video.stem}.jpg"
                contact_sheet(Path(result["output"]), count, sheet)
                artifacts.append(f"- {Path(weight).name} / {video.name}: "
                                 f"[Contact sheet]({_link(sheet, out)}) / "
                                 f"[Annotated video]({_link(Path(result['output']), out)})")
            else:
                artifacts.append(f"- {Path(weight).name} / {video.name}: no readable frames; no contact sheet.")
    table = format_table(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("# Video report\n\n" + table + "\n\n"
                   f"Video stride: {vid_stride}. Averages use processed frames only. "
                   "Mean ms/frame includes model loading, decoding, inference, annotation and video writing; "
                   "contact-sheet generation is excluded. Empty videos report zero averages and timing.\n\n"
                   + "\n".join(artifacts) + "\n", encoding="utf-8")
    print(table)
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos", type=Path, action="append", required=True,
                        help="Input video path; repeat for each video")
    parser.add_argument("--weights", action="append", required=True,
                        help="Model weights; repeat for each checkpoint")
    parser.add_argument("--vid-stride", type=int, default=2)
    parser.add_argument("--out", type=Path, default=ROOT / "reports" / "video_report.md")
    args = parser.parse_args(argv)
    if args.vid_stride < 1:
        parser.error("--vid-stride must be at least 1")
    report(args.videos, args.weights, args.out, args.vid_stride)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
