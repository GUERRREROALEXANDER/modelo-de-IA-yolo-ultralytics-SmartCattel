"""Comparable single-class cattle detection metrics for any YOLO class index."""

from pathlib import Path

import numpy as np
import torch
from ultralytics.utils.metrics import ap_per_class, box_iou

from .config import Settings
from .detector import Detector
from .media import read_image


IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)


def match_boxes(ground_truth: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    """Match by descending IoU, then unique detection, then unique label."""
    correct = np.zeros((len(predicted), len(IOU_THRESHOLDS)), dtype=bool)
    if not len(ground_truth) or not len(predicted):
        return correct
    iou = box_iou(torch.as_tensor(ground_truth, dtype=torch.float32),
                  torch.as_tensor(predicted, dtype=torch.float32)).numpy()
    for column, threshold in enumerate(IOU_THRESHOLDS):
        matches = np.array(np.nonzero(iou >= threshold)).T
        if len(matches):
            if len(matches) > 1:
                matches = matches[iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), column] = True
    return correct


def xywh_to_xyxy(values: tuple[float, ...], width: int, height: int) -> list[float]:
    x, y, w, h = values
    return [(x - w / 2) * width, (y - h / 2) * height,
            (x + w / 2) * width, (y + h / 2) * height]


def summarize(images: list[dict], operating_conf: float = 0.35) -> dict:
    """Calculate Ultralytics AP and independent operating-point counts at IoU 0.5."""
    correctness, confidences = [], []
    tp = fp = fn = negative_fp = 0
    count_errors = []
    target_count = 0
    for item in images:
        truth = np.asarray(item["ground_truth"], dtype=np.float32).reshape(-1, 4)
        predictions = np.asarray(item["predictions"], dtype=np.float32).reshape(-1, 4)
        scores = np.asarray(item["confidences"], dtype=np.float32)
        if len(predictions) != len(scores):
            raise ValueError("Prediction and confidence counts differ")
        target_count += len(truth)
        correctness.append(match_boxes(truth, predictions))
        confidences.append(scores)
        selected = scores >= operating_conf
        operating_matches = match_boxes(truth, predictions[selected])[:, 0]
        image_tp = int(operating_matches.sum())
        tp += image_tp
        fp += int(selected.sum()) - image_tp
        fn += len(truth) - image_tp
        if not len(truth):
            negative_fp += int(selected.sum())
        count_errors.append(abs(int(selected.sum()) - len(truth)))
    all_correct = np.concatenate(correctness) if correctness else np.zeros((0, 10), dtype=bool)
    all_scores = np.concatenate(confidences) if confidences else np.zeros(0)
    if target_count:
        result = ap_per_class(all_correct, all_scores, np.zeros(len(all_scores)),
                              np.zeros(target_count), plot=False)
        precision, recall = float(result[2][0]), float(result[3][0])
        map50, map5095 = float(result[5][0, 0]), float(result[5][0].mean())
    else:
        precision = recall = map50 = map5095 = 0.0
    return {"precision": precision, "recall": recall, "mAP50": map50,
            "mAP50-95": map5095, "operating_conf": operating_conf,
            "operating": {"TP": tp, "FP": fp, "FN": fn,
                          "precision": tp / (tp + fp) if tp + fp else 0.0,
                          "recall": tp / (tp + fn) if tp + fn else 0.0,
                          "false_positives_on_negative_images": negative_fp,
                          "mean_absolute_count_error": sum(count_errors) / len(images) if images else 0.0}}


def evaluate(weights, data_yaml, split="test", conf=0.001, iou=0.6, imgsz=640,
             class_names=("cow",), operating_conf=0.35) -> dict:
    """Evaluate every image in a YOLO split with the same prediction settings."""
    if split not in ("train", "val", "test"):
        raise ValueError(f"Unknown split: {split}")
    if not 0 < conf <= operating_conf <= 1:
        raise ValueError("Require 0 < conf <= operating_conf <= 1")
    from scripts.validate_dataset import dataset_paths, label_path, parse_labels

    directory = dataset_paths(Path(data_yaml))[split]
    images = sorted(path for path in directory.iterdir()
                    if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
    if not images:
        raise ValueError(f"No images in {directory}")
    detector = Detector(Settings(weights=str(weights), confidence=conf, iou=iou,
                                 imgsz=imgsz, class_names=tuple(class_names)))
    class_ids = detector.class_ids
    records, inference_ms = [], []
    for image in images:
        frame = read_image(image)
        height, width = frame.shape[:2]
        label = label_path(image)
        if not label.exists():
            raise ValueError(f"Missing label: {label}")
        boxes, errors = parse_labels(label)
        if errors:
            raise ValueError(errors[0])
        truth = [xywh_to_xyxy(box, width, height) for box in boxes]
        result = detector._model.predict(frame, conf=conf, iou=iou, imgsz=imgsz,
                                         classes=class_ids, device=detector.settings.device,
                                         verbose=False)[0]
        predictions, scores = [], []
        for box in result.boxes:
            if int(box.cls.item()) in class_ids:
                predictions.append(box.xyxy[0].cpu().tolist())
                scores.append(float(box.conf.item()))
        records.append({"ground_truth": truth, "predictions": predictions,
                        "confidences": scores})
        inference_ms.append(float(result.speed["inference"]))
    metrics = summarize(records, operating_conf)
    metrics.update({"weights": str(weights), "split": split, "images": len(images),
                    "conf": conf, "iou": iou, "imgsz": imgsz,
                    "class_names": list(class_names), "model_class_ids": class_ids,
                    "mean_inference_ms_per_image": sum(inference_ms) / len(inference_ms)})
    return metrics
