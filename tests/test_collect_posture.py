import importlib.util
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "collect_posture_frames", Path(__file__).resolve().parents[1] / "scripts" / "collect_posture_frames.py",
)
collect = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(collect)


def test_a_picture_goes_where_it_is_rarest():
    cow = {"class": "cow"}
    assert collect.folder_for([{**cow, "posture": "standing"}, {**cow, "posture": "lying"}]) == "lying"
    assert collect.folder_for([{**cow, "posture": "walking"}]) == "standing"
    assert collect.folder_for([{"class": "person"}]) is None
