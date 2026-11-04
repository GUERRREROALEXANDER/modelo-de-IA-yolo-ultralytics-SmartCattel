from types import SimpleNamespace

import numpy as np

from smartcattle_ai.config import Settings
from smartcattle_ai.detector import Detector


class Value:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


class Box:
    def __init__(self, class_id):
        self.cls, self.conf = Value(class_id), Value(0.9)
        self.xyxy = [SimpleNamespace(tolist=lambda: [1.0, 2.0, 3.0, 4.0])]


class Model:
    names = {0: "cow_standing", 1: "cow_lying", 2: "person"}

    def predict(self, frame, classes, **kwargs):
        return [SimpleNamespace(boxes=[Box(class_id) for class_id in classes])]


def test_a_trained_model_says_cow_and_what_it_is_doing():
    settings = Settings(weights="x.pt", person_weights="x.pt", class_names=("cow", "person"))
    found = Detector(settings, model=Model()).detect(np.zeros((4, 4, 3), dtype=np.uint8))
    assert {(item["class"], item.get("posture")) for item in found} == {
        ("cow", "standing"), ("cow", "lying"), ("person", None),
    }
