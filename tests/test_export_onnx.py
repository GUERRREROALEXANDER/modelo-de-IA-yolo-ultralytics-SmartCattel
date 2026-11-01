"""ONNX export CLI coverage: fast fake-model tests plus one real export."""

from pathlib import Path

import numpy as np
import pytest

from scripts import export_onnx
from smartcattle_ai import Detector, Settings


def fake_yolo(payload=b"dummy"):
    class FakeYOLO:
        def __init__(self, source):
            self.source = source

        def export(self, **kwargs):
            assert kwargs == {"format": "onnx", "imgsz": 320,
                              "dynamic": False, "simplify": True}
            produced = Path(self.source).with_suffix(".onnx")
            produced.write_bytes(payload)
            return str(produced)

    return FakeYOLO


def test_export_moves_and_renames(tmp_path, monkeypatch):
    weights = tmp_path / "model.pt"
    weights.write_bytes(b"weights")
    monkeypatch.setattr(export_onnx, "YOLO", fake_yolo(b"exported"))
    target = export_onnx.export(weights, imgsz=320, out_dir=tmp_path / "onnx")
    assert target == tmp_path / "onnx" / "model_320.onnx"
    assert target.read_bytes() == b"exported"
    assert not weights.with_suffix(".onnx").exists()


def test_export_overwrites_existing_target(tmp_path, monkeypatch):
    weights = tmp_path / "model.pt"
    weights.write_bytes(b"weights")
    out_dir = tmp_path / "onnx"
    out_dir.mkdir()
    stale = out_dir / "model_320.onnx"
    stale.write_bytes(b"stale")
    monkeypatch.setattr(export_onnx, "YOLO", fake_yolo(b"fresh"))
    target = export_onnx.export(weights, imgsz=320, out_dir=out_dir)
    assert target == stale
    assert target.read_bytes() == b"fresh"


def test_export_rejects_missing_weights(tmp_path):
    with pytest.raises(ValueError, match="not found"):
        export_onnx.export(tmp_path / "missing.pt")


@pytest.mark.parametrize("imgsz", [0, 31, 33, -32])
def test_export_rejects_bad_imgsz(tmp_path, imgsz):
    weights = tmp_path / "model.pt"
    weights.write_bytes(b"weights")
    with pytest.raises(ValueError, match="multiple of 32"):
        export_onnx.export(weights, imgsz=imgsz)


def test_main_returns_one_on_error(tmp_path, capsys):
    assert export_onnx.main(["--weights", str(tmp_path / "missing.pt")]) == 1
    assert capsys.readouterr().err.startswith("Error: ")


def test_main_prints_exported_path(tmp_path, monkeypatch, capsys):
    weights = tmp_path / "model.pt"
    weights.write_bytes(b"weights")
    exported = tmp_path / "onnx" / "model_320.onnx"

    def fake_export(path, imgsz, out_dir):
        exported.parent.mkdir(parents=True, exist_ok=True)
        exported.write_bytes(b"exported")
        return exported

    monkeypatch.setattr(export_onnx, "export", fake_export)
    assert export_onnx.main(["--weights", str(weights), "--imgsz", "320",
                             "--out-dir", str(tmp_path / "onnx")]) == 0
    assert str(exported) in capsys.readouterr().out


def test_benchmark_rejects_zero_runs(monkeypatch):
    monkeypatch.setattr(export_onnx, "Detector",
                        lambda settings: pytest.fail("runs must be validated first"))
    with pytest.raises(ValueError, match="at least 1"):
        export_onnx.benchmark("fake.pt", np.zeros((48, 48, 3), np.uint8), runs=0)


def test_benchmark_skips_warmup_and_averages(monkeypatch):
    calls = []

    class FakeDetector:
        def __init__(self, settings):
            self.settings = settings

        def detect(self, image):
            calls.append(self.settings.imgsz)
            return []

    times = iter([0.0, 0.2])
    monkeypatch.setattr(export_onnx, "Detector", FakeDetector)
    monkeypatch.setattr(export_onnx, "perf_counter", lambda: next(times))
    mean_ms = export_onnx.benchmark("fake.pt", np.zeros((48, 48, 3), np.uint8),
                                    runs=2, imgsz=320)
    assert mean_ms == pytest.approx(100.0)
    assert calls == [320, 320, 320]


@pytest.mark.real
def test_real_export_runs_with_detector(tmp_path):
    weights = export_onnx.ROOT / "models" / "cattle_coco_yolo11n_best.pt"
    if not weights.is_file():
        pytest.skip("Fine-tuned cattle weights are not available")
    pytest.importorskip("onnx", reason="pip install -r requirements-onnx.txt")
    pytest.importorskip("onnxruntime", reason="pip install -r requirements-onnx.txt")
    path = export_onnx.export(weights, imgsz=320, out_dir=tmp_path)
    assert path.is_file()
    detector = Detector(Settings(weights=str(path), imgsz=320, device="cpu"))
    detections = detector.detect(np.zeros((320, 320, 3), dtype=np.uint8))
    assert isinstance(detections, list)
    assert 19 in detector.class_ids
    assert 0 in detector.class_ids
