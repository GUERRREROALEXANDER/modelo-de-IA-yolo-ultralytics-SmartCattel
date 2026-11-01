"""Occupancy heat map: a single frame says where a cow is, many frames say where the herd lives.

The tracker already hands us a box per animal every frame; dropping the bottom center of
each cow into a coarse grid turns thousands of detections into one picture of the pen.
Hot cells are where the animals stand, lie or crowd, which is what matters for shade,
water and calving spots. No new dependency: a numpy counter, a resize and a JET palette.
"""

import cv2
import numpy as np


def _is_integer(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


class HeatMap:
    def __init__(self, columns: int = 32, rows: int = 18):
        if not _is_integer(columns) or not 2 <= columns <= 256:
            raise ValueError("columns must be an integer between 2 and 256")
        if not _is_integer(rows) or not 2 <= rows <= 256:
            raise ValueError("rows must be an integer between 2 and 256")
        self.columns = columns
        self.rows = rows
        # One float64 counter per cell; normalized only when asked, so the raw counts survive.
        self._counts = np.zeros((rows, columns), dtype=np.float64)
        self.samples = 0

    def add(self, detections: list[dict], width: int, height: int) -> None:
        """Accumulate the bottom center of every cow; other classes and the boxes are untouched."""
        if width <= 0 or height <= 0:
            raise ValueError("width and height must be greater than 0")
        for detection in detections:
            if detection.get("class") != "cow":
                continue
            x1, y1, x2, y2 = detection["bbox"]
            # Clip first so a box hanging outside the frame still counts on the nearest edge.
            x = min(max((x1 + x2) / 2 / width, 0.0), 1.0)
            y = min(max(y2 / height, 0.0), 1.0)
            # A point exactly at 1.0 belongs to the last cell, not one past it.
            column = min(int(x * self.columns), self.columns - 1)
            row = min(int(y * self.rows), self.rows - 1)
            self._counts[row, column] += 1
            self.samples += 1

    def reset(self) -> None:
        self._counts[:] = 0
        self.samples = 0

    def grid(self) -> np.ndarray:
        """A copy of the counts scaled so the hottest cell is 1.0; all zeros when empty."""
        grid = self._counts.copy()
        peak = grid.max()
        if peak > 0:
            grid /= peak
        return grid

    def overlay(self, frame: np.ndarray, alpha: float = 0.45) -> np.ndarray:
        """A new frame with the heat painted only over cells that actually saw a cow."""
        if not isinstance(alpha, (int, float)) or not 0 <= alpha <= 1:
            raise ValueError("alpha must be between 0 and 1")
        normalized = np.asarray(self.grid(), dtype=np.float32)
        resized = cv2.resize(normalized, (frame.shape[1], frame.shape[0]),
                             interpolation=cv2.INTER_LINEAR)
        scaled = np.clip(np.round(resized * 255), 0, 255).astype(np.uint8)
        colored = cv2.applyColorMap(scaled, cv2.COLORMAP_JET)
        blended = cv2.addWeighted(colored, float(alpha), frame, 1 - float(alpha), 0)
        result = frame.copy()
        mask = resized > 0
        result[mask] = blended[mask]
        return result

    def describe(self) -> dict:
        return {
            "columns": self.columns,
            "rows": self.rows,
            "samples": self.samples,
            "grid": np.round(self.grid(), 3).tolist(),
        }
