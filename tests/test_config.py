import pytest

from smartcattle_ai import Settings


@pytest.mark.parametrize("weights_exist", [False, True])
def test_defaults(monkeypatch, weights_exist):
    from smartcattle_ai import config
    from pathlib import Path

    expected_path = Path(config.__file__).resolve().parents[1] / "models" / "cattle_yolo11n_best.pt"
    original_is_file = Path.is_file
    monkeypatch.setattr(Path, "is_file", lambda path: weights_exist if path == expected_path
                        else original_is_file(path))
    settings = Settings()
    assert settings.weights == (str(expected_path) if weights_exist else "yolo11n.pt")
    assert settings.confidence == 0.35
    assert settings.iou == 0.5
    assert settings.imgsz == 640
    assert settings.class_names == ("cow",)
    assert settings.device in ("0", "cpu")


def test_from_env(monkeypatch):
    values = {
        "WEIGHTS": "custom.pt", "CONF": "0.7", "IOU": "0.6",
        "IMGSZ": "320", "CLASSES": "cow, person", "DEVICE": "cpu",
    }
    for key, value in values.items():
        monkeypatch.setenv(f"SMARTCATTLE_{key}", value)
    assert Settings.from_env() == Settings(
        "custom.pt", 0.7, 0.6, 320, ("cow", "person"), "cpu"
    )


@pytest.mark.parametrize("changes,message", [
    ({"confidence": 0}, "confidence"),
    ({"confidence": 1.1}, "confidence"),
    ({"iou": 0}, "iou"),
    ({"iou": 1.1}, "iou"),
    ({"imgsz": 31}, "imgsz"),
    ({"imgsz": 33}, "imgsz"),
    ({"class_names": ()}, "class_names"),
    ({"class_names": ("",)}, "class_names"),
])
def test_invalid_settings(changes, message):
    with pytest.raises(ValueError, match=message):
        Settings(**changes)
