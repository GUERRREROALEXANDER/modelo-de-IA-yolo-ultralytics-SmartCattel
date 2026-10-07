import pytest

from smartcattle_ai import Settings


@pytest.mark.parametrize("coco_exists,cattle_exists,expected", [
    (True, True, "cattle_coco_yolo11n_best.pt"),
    (True, False, "cattle_coco_yolo11n_best.pt"),
    (False, True, None),
    (False, False, None),
])
def test_defaults(monkeypatch, tmp_path, coco_exists, cattle_exists, expected):
    from smartcattle_ai import config
    from pathlib import Path

    models = Path(config.__file__).resolve().parents[1] / "models"
    available = {
        models / "cattle_coco_yolo11n_best.pt": coco_exists,
        models / "cattle_yolo11n_best.pt": cattle_exists,
    }
    original_is_file = Path.is_file
    monkeypatch.setattr(Path, "is_file", lambda path: available[path] if path in available
                        else original_is_file(path))
    monkeypatch.chdir(tmp_path)
    settings = Settings()
    assert settings.weights == (str(models / expected) if expected else "yolo11n.pt")
    assert settings.confidence == 0.35
    assert settings.iou == 0.5
    assert settings.imgsz == 640
    assert settings.class_names == ("cow", "person")
    assert settings.person_weights == "yolo11n.pt"
    assert settings.device in ("0", "cpu")


def test_from_env(monkeypatch):
    values = {
        "WEIGHTS": "custom.pt", "CONF": "0.7", "IOU": "0.6",
        "IMGSZ": "320", "CLASSES": "cow, person", "DEVICE": "cpu",
        "PERSON_WEIGHTS": "people.pt",
    }
    for key, value in values.items():
        monkeypatch.setenv(f"SMARTCATTLE_{key}", value)
    assert Settings.from_env() == Settings(
        "custom.pt", 0.7, 0.6, 320, ("cow", "person"), "cpu", "people.pt"
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
    ({"class_names": ("horse",)}, "class_names"),
    ({"person_weights": ""}, "person_weights"),
])
def test_invalid_settings(changes, message):
    with pytest.raises(ValueError, match=message):
        Settings(**changes)
