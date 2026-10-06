"""Cattle detection tools built on YOLO and OpenCV."""

from .config import Settings
from .detector import Detector

__all__ = ["Detector", "Settings"]
