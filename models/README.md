# Fine-tuned cattle weights

Model weights are not committed. Training copies the best checkpoint to
`models/cattle_yolo11n_best.pt`; the original stays at
`runs/cattle/yolo11n_finetune/weights/best.pt`.

Build the dataset with `python scripts/build_dataset.py`, then regenerate the
weights with `python scripts/train.py` from the repository root.
