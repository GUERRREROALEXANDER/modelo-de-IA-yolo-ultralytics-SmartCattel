"""Load YOLO and map requested class names to model class IDs."""


def load_model(weights: str):
    from ultralytics import YOLO

    return YOLO(weights)


def resolve_class_ids(names: dict[int, str], wanted) -> list[int]:
    ids = [class_id for class_id, name in names.items() if name in wanted]
    if not ids:
        available = ", ".join(list(names.values())[:10])
        raise RuntimeError(f"None of the requested classes exist in the model. Available classes: {available}")
    return ids
