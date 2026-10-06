# Fine-tuned cattle weights

Model weights are not committed. Run the following commands from the repository root.
First build the cleaned cattle dataset and local Open Images metadata with
`python scripts/build_dataset.py` if they are not already available.

`cattle_coco_yolo11n_best.pt` is the recommended checkpoint: it retains the COCO
head and supports cow and person in one model, with one prediction per frame.
Regenerate the dataset variant and train it with:

```powershell
python scripts/build_coco_variant.py
python scripts/train.py --config configs/train_cattle_coco.yaml --weights-out models/cattle_coco_yolo11n_best.pt
```

The original best checkpoint stays at
`runs/cattle/yolo11n_coco_finetune/weights/best.pt`.

`cattle_yolo11n_best.pt` is the single-class cow experiment. Person detection
uses a separate fallback model when this checkpoint is selected. Regenerate it with:

```powershell
python scripts/build_dataset.py
python scripts/train.py --config configs/train_cattle.yaml --weights-out models/cattle_yolo11n_best.pt
```

Its original best checkpoint stays at `runs/cattle/yolo11n_finetune/weights/best.pt`.
Default selection tries the recommended checkpoint, then the single-class
checkpoint, then `yolo11n.pt`. `SMARTCATTLE_WEIGHTS` overrides this selection.
