# SmartCattle AI

SmartCattle AI detects **cows and persons** in images and videos with YOLO and OpenCV. The default model is `models/cattle_coco_yolo11n_best.pt`: pretrained COCO YOLO11n fine-tuned on a cleaned Open Images cattle dataset. It keeps the 80-class COCO head, so cow (class 19) and person (class 0) come from one model in one prediction per frame. Other classes are filtered out. An optional normalized safe-zone rule can send events for cows outside the zone to the SmartCattle FastAPI backend. Persons never create events.

```text
video -> OpenCV frames -> YOLO -> detections -> annotated boxes / optional events
```

## Current status

| Item | Value |
|---|---|
| Default model | `models/cattle_coco_yolo11n_best.pt` (copy of `runs/cattle/yolo11n_coco_finetune/weights/best.pt`) |
| Base model | `yolo11n.pt` (COCO pretrained) |
| Output classes | `cow`, `person` |
| Training | 25-epoch schedule, best epoch 16, CPU, config `configs/train_cattle_coco.yaml` |
| Test result, cow | mAP50 0.690, mAP50-95 0.549 (pretrained: 0.625, 0.523) |
| Test result, person | mAP50 0.524, mAP50-95 0.295 (pretrained: 0.486, 0.300) |
| Inference input | Image files, image folders, video files |

Details and sources for every number are in [Model and training](#model-and-training) and [Evaluation](#evaluation).

## Project structure

```text
.
├── detect.py                          # Image, folder, video, and backend CLI.
├── smartcattle_ai/
│   ├── __init__.py                    # Exports Detector and Settings.
│   ├── config.py                      # Validated inference settings and environment defaults.
│   ├── detector.py                    # Cow/person model routing and detection records.
│   ├── model_loader.py                # Ultralytics loader and class-ID helper.
│   ├── media.py                       # OpenCV reading, annotation, and output.
│   ├── rules.py                       # Safe-zone rule based on box bottom center.
│   ├── pipeline.py                    # Rule-event publishing.
│   ├── backend_client.py              # HTTP client, payload conversion, and cooldown.
│   ├── evaluation.py                  # Cattle matching and metrics.
│   └── camera.py                      # Stream-reading helper, separate from detect.py.
├── scripts/
│   ├── build_dataset.py               # Build the one-class Open Images YOLO dataset.
│   ├── build_coco_variant.py          # Build the COCO-indexed cow+person dataset variant.
│   ├── validate_dataset.py            # Check labels, images, and split leakage.
│   ├── train.py                       # Fine-tune YOLO, resume, and write reports.
│   ├── evaluate.py                    # Compare checkpoints per class on a YOLO split.
│   ├── scenario_report.py             # Predictions on licensed scenario images.
│   └── video_report.py                # Compare checkpoints on videos with contact sheets.
├── configs/
│   ├── train_cattle_coco.yaml         # Settings of the default model.
│   └── train_cattle.yaml              # Single-class experiment settings.
├── data/
│   ├── ATTRIBUTION.csv                # Per-image authors, licenses, and source links.
│   ├── label_review.csv               # Manual exclusions and reasons.
│   ├── dataset_stats.json             # Dataset selection and final split counts.
│   ├── samples/manifest.csv           # Scenario descriptions, counts, and credits.
│   ├── samples/labels/                # Scenario ground-truth boxes where present.
│   ├── samples/*.jpg                  # Scenario images.
│   └── raw/                           # Downloaded Open Images files and videos.
├── datasets/cattle_coco/data.yaml     # Generated COCO-indexed dataset (training data).
├── datasets/cattle/data.yaml          # Generated one-class dataset.
├── reports/                           # Metrics, scenario and video reports (see Evaluation).
├── models/README.md                   # Checkpoint paths and regeneration notes.
├── runs/cattle/                       # Generated training records and checkpoints.
├── outputs/                           # Annotated media.
├── tests/                             # pytest suite (see Tests).
├── requirements.txt                   # Python dependencies.
└── pytest.ini                         # Test discovery and real marker.
```

`datasets/`, `data/raw/`, `runs/`, `outputs/`, and `*.pt` are ignored by Git, so the trained model must be kept or regenerated locally. The scenario images, labels, manifest, and attribution metadata are repository data.

## Requirements and installation

Use Python 3.13 on Windows with PowerShell. `requirements.txt` lists Ultralytics, OpenCV, NumPy, and pytest; PyTorch comes through Ultralytics' dependencies. Everything runs on CPU; the recorded training and timings used CPU.

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The first use of pretrained weights may download them. If HTTPS certificate validation fails, Ultralytics can fall back to `curl` for its weight download. The dataset downloader also tries the system certificate store, `certifi` when installed, and then `curl.exe`.

## Running inference

Run from the repository root. No `--weights` flag is needed: the CLI loads `models/cattle_coco_yolo11n_best.pt` when it exists and prints the model it uses.

```powershell
python detect.py --source data/samples/0dfb0cb56b5f6f83.jpg
python detect.py --source data/samples
python detect.py --source data/raw/videos/Herding_the_cows_video.webm --vid-stride 2
```

The folder command processes supported images directly inside that folder. `--vid-stride 2` processes every second video frame. Other examples:

```powershell
python detect.py --source data/samples/0dfb0cb56b5f6f83.jpg --json
python detect.py --source data/samples/0dfb0cb56b5f6f83.jpg --no-save
python detect.py --source data/samples/0dfb0cb56b5f6f83.jpg --classes cow --conf 0.5
python detect.py --source data/samples/0dfb0cb56b5f6f83.jpg --weights yolo11n.pt
```

`--json` prints an object for one source or a list for a folder. `--no-save` disables annotated output. `--classes` accepts `cow`, `person`, or a comma-separated pair. The default confidence threshold is `0.35`. Other controls include `--weights`, `--iou`, `--imgsz`, `--device`, `--output-dir`, and video-only `--max-frames`. By default, annotated images go to `outputs/<source-stem>_annotated.<extension>` and videos to `outputs/<source-stem>_annotated.mp4`. Cow boxes are drawn in green and person boxes in orange, each labeled with its confidence.

Real output of the default model on a scenario image (two cows lying on grass):

```text
Model: D:\Escritorio\modelo de ia yolo\models\cattle_coco_yolo11n_best.pt
File: data\samples\224ad16181d9fb6b.jpg
Cattle: 2; Persons: 0
cow  conf=0.94 bbox=[500.4, 246.1, 921.7, 420.3]
cow  conf=0.91 bbox=[126.3, 295.0, 617.5, 524.9]
```

And on a video:

```text
Model: D:\Escritorio\modelo de ia yolo\models\cattle_coco_yolo11n_best.pt
File: data\raw\videos\Herding_the_cows_video.webm
Frames: 198 (read: 396)
Cattle: 2101; Persons: 173
Max cattle: 18; max persons: 4; average cattle: 10.61
Annotated video: outputs\default_check\Herding_the_cows_video_annotated.mp4
```

Video `Cattle` and `Persons` sum detections across processed frames; they are not counts of unique animals or people.

## Output format

This JSON example uses the synthetic 64 × 64 image and fake detection from `tests/test_cli.py`; it illustrates field names and types, not model performance:

```json
{
  "source": "<test-directory>/cow.jpg",
  "width": 64,
  "height": 64,
  "total_cattle": 1,
  "total_persons": 0,
  "detections": [{"class": "cow", "confidence": 0.91, "bbox": [1, 2, 30, 40]}],
  "output": null
}
```

`bbox` contains pixel-space `[x1, y1, x2, y2]`; confidence is rounded to four decimals and coordinates to one. `output` is `null` with `--no-save`. Video JSON also has `frames_read`, `frames_processed`, `fps`, `max_cattle`, `max_persons`, `avg_cattle`, `frames_with_cattle`, and a `frames` array with per-frame detections. The `Model:` line is printed only without `--json`, so JSON output stays parseable.

## Model and training

### Why fine-tune

The [pretrained scenario report](reports/scenarios_pretrained_yolo11n.md) shows pretrained YOLO11n missing small, distant, occluded, and low-light cows. A first experiment replaced the head with a single `cow` class (`configs/train_cattle.yaml`); it scored below the pretrained model on the test split (mAP50 0.546, mAP50-95 0.335, 33 false positives; `reports/singleclass_experiment_test.json`) and lost person detection. The default model instead keeps the COCO head and fine-tunes with a low learning rate and a frozen backbone, so pretrained cow and person knowledge is adjusted, not relearned.

### Training configuration

`configs/train_cattle_coco.yaml`: base `yolo11n.pt`, dataset `datasets/cattle_coco/data.yaml`, 25 epochs, patience 8, image size 640, batch 8, CPU, 4 workers, seed 0, deterministic, AdamW, `lr0` 0.0005, 1 warmup epoch, `warmup_bias_lr` 0, `freeze` 10 (backbone layers), `close_mosaic` 5.

The run was interrupted after epoch 9 and resumed from `last.pt` with `--resume`. `reports/yolo11n_coco_finetune.json` records 24 epoch rows and the best epoch, 16; the resumed segment took 1276 s (about 21 min) on an Intel Core i5-1235U (12 logical cores, 15.71 GiB RAM, PyTorch `2.14.0+cpu`, no CUDA). Per-epoch curves are in `reports/training/yolo11n_coco_finetune/`.

```powershell
python scripts/build_dataset.py
python scripts/build_coco_variant.py
python scripts/validate_dataset.py --data datasets/cattle_coco/data.yaml --out reports/dataset_coco_validation.json
python scripts/train.py --config configs/train_cattle_coco.yaml --weights-out models/cattle_coco_yolo11n_best.pt
```

`train.py` keeps the best checkpoint at `runs/cattle/yolo11n_coco_finetune/weights/best.pt`, copies it to `--weights-out`, validates it on the test split, and writes `reports/<run-name>.json`. `--weights-out` defaults to `models/cattle_yolo11n_best.pt` (the single-class experiment path), so pass it explicitly for the default model.

### Resuming an interrupted run

```powershell
python scripts/train.py --config configs/train_cattle_coco.yaml --resume --weights-out models/cattle_coco_yolo11n_best.pt
```

`--resume` loads `runs/cattle/<name>/weights/last.pt` of the run named in the config and continues with the settings stored in that checkpoint (epochs, learning rate, freeze); CLI overrides such as `--epochs` do not apply. It fails with a clear message if no `last.pt` exists. After training it writes the same weights copy and report as a fresh run.

### Validation metrics (Ultralytics)

Best epoch on the validation split (46 images), from the training log and `reports/yolo11n_coco_finetune.json`:

| Class | Images | Instances | P | R | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|---:|---:|
| all | 46 | 118 | 0.664 | 0.549 | 0.574 | 0.407 |
| cow | 39 | 93 | 0.845 | 0.538 | 0.663 | 0.492 |
| person | 8 | 25 | 0.483 | 0.560 | 0.485 | 0.322 |

Ultralytics test validation after training (94 images): all 0.769 P, 0.546 R, 0.596 mAP50, 0.426 mAP50-95; cow 0.689 mAP50, 0.564 mAP50-95; person 0.504 mAP50, 0.288 mAP50-95. For the comparison with the pretrained model, use the table below, which evaluates both models with the same script and settings.

## Dataset

`scripts/build_dataset.py` maps Open Images V7 **Cattle** and **Bull** boxes to `cow`. The Open Images `validation` split becomes the held-out `test` split; its `test` split is shuffled with seed 42 and divided approximately 85%/15% into `train`/`val`. Background-class images without selected cattle labels supply negatives with empty label files; the builder samples roughly 12% negatives relative to the positive selection before later exclusions.

`scripts/build_coco_variant.py` builds `datasets/cattle_coco` from the same 404 images and local metadata, writing cows as COCO class 19 and Open Images person-type boxes as COCO class 0. Its `data.yaml` lists all 80 COCO names so the pretrained head stays aligned. Label counts per split, counted from the label files (`reports/dataset_coco_validation.json` reports valid, no errors, warnings, or duplicates):

| Split | Images | Positive | Negative | Cow boxes | Person boxes |
|---|---:|---:|---:|---:|---:|
| Train | 264 | 232 | 32 | 454 | 520 |
| Validation | 46 | 40 | 6 | 93 | 25 |
| Test | 94 | 84 | 10 | 152 | 153 |
| **Total** | **404** | **356** | **48** | **699** | **698** |

Ultralytics reports 149 cow instances in the test split while the label files contain 152; `scripts/evaluate.py` uses all 152.

Source `IsGroupOf` and `IsDepiction` images are excluded, as are invalid or very small boxes and rotated images whose boxes would disagree with downloaded image orientation. Manual review removes other species, ambiguous cases, artwork, and one non-exhaustive herd label. `data/label_review.csv` contains **135** exclusions:

| Reason | Count | Reason | Count |
|---|---:|---|---:|
| Water buffalo | 29 | Bison | 20 |
| African/Cape buffalo | 19 | Goat | 16 |
| Yak | 14 | Sheep | 13 |
| Antelope/deer/wild ungulate | 10 | Ambiguous species | 6 |
| Gaur | 4 | Depiction (art/illustration) | 3 |
| Single box covers a herd | 1 | **Total** | **135** |

`data/dataset_stats.json` records 36 group, 7 depiction, and 0 rotated exclusions from source validation; 80 group, 25 depiction, and 2 rotated exclusions from source test. All 404 rows of `data/ATTRIBUTION.csv` identify an image author, title, CC BY 2.0 license, original URL, and landing URL. Open Images annotations are CC BY 4.0. The builder accepts `--limit` for a smaller debug build and `--workers` for download concurrency.

## Evaluation

### Test split: pretrained vs fine-tuned

`scripts/evaluate.py` collects predictions at confidence `0.001` with model NMS IoU `0.6`, then reports maximum-F1 precision/recall, mAP50, and mAP50-95 per class. TP, FP, FN, false positives on negative images, and mean absolute count error use the operating confidence `0.35` and matching IoU 0.50. Results from `reports/coco_finetune_comparison.json` (94 test images, 152 cows, 153 persons):

| Class | Weights | P | R | mAP50 | mAP50-95 | TP | FP | FN | FP on negatives | Count MAE |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cow | `yolo11n.pt` | 0.839 | 0.583 | 0.625 | 0.523 | 83 | 5 | 69 | 0 | 0.745 |
| cow | **fine-tuned** | **0.942** | **0.643** | **0.690** | **0.549** | **99** | 7 | **53** | 1 | **0.553** |
| person | `yolo11n.pt` | 0.515 | 0.556 | 0.486 | 0.300 | 85 | 79 | 68 | 21 | 0.968 |
| person | **fine-tuned** | 0.665 | 0.484 | 0.524 | 0.295 | 62 | **21** | 91 | 7 | 0.979 |

Mean inference: 66.5 ms/image (`yolo11n.pt`) and 61.0 ms/image (fine-tuned) on the CPU above.

For cows, fine-tuning finds 16 more animals at the operating point and lowers count error. For persons, it cuts false positives from 79 to 21 but misses more people. Among cow-only pretrained baselines in `reports/pretrained_test.json`, the larger `yolo11s.pt` reaches mAP50-95 0.576 (mAP50 0.657) at 140.8 ms/image.

```powershell
python scripts/evaluate.py --weights yolo11n.pt --weights models/cattle_coco_yolo11n_best.pt --data datasets/cattle_coco/data.yaml --split test --classes cow,person --out reports/coco_finetune_comparison.json
```

### Scenario images

The 16 images in `data/samples/manifest.csv` cover close-up, grouped, distant, occluded, breed, low-light, and two negative scenarios with expected cattle counts. Full tables: [pretrained](reports/scenarios_pretrained_yolo11n.md), [fine-tuned](reports/scenarios_coco_finetune.md); annotated images in `outputs/scenarios/cattle_coco_yolo11n_best/`.

| | `yolo11n.pt` | Fine-tuned |
|---|---:|---:|
| Expected cows | 36 | 36 |
| Missed | 14 | 14 |
| False positives | 2 | 3 |
| Detections on the 2 negative images | 0 | 0 |

| Scenario | `yolo11n.pt` (detected / missed) | Fine-tuned (detected / missed) |
|---|---|---|
| Night rodeo, motion blur (1) | 0 / 1 | 1 / 0 |
| Holstein behind bars (3) | 1 / 2 | 3 / 0 |
| Dairy barn behind railings (2) | 3 / 0, 1 FP | 2 / 0 |
| Dark rodeo pen (2) | 1 / 1 | 2 / 1, 1 FP |
| Tiny distant cows on a hillside (3) | 2 / 1 | 1 / 2 |
| Small close overlapping group (3) | 3 / 0 | 3 / 2, 2 FP (misplaced boxes) |
| Dense overlapping herd (8) | 3 / 5 | 3 / 5 |
| Watusi, Highland breeds (2 each) | 1 / 1 each | 1 / 1 each |

The fine-tuned model improves on occlusion and low light and loses on very small distant cows and tightly overlapping groups.

```powershell
python scripts/scenario_report.py --weights models/cattle_coco_yolo11n_best.pt --out reports/scenarios_coco_finetune.md
```

### Videos

From [`reports/video_report_coco_finetune.md`](reports/video_report_coco_finetune.md), stride 2. The videos have no labels, so these are detection statistics, not accuracy:

| Video | Weights | Frames | Max cows | Avg cows | Max persons | Avg persons | Frames with cows | ms/frame |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Herding the cows | `yolo11n.pt` | 198 | 22 | 11.44 | 4 | 1.08 | 193 | 113.9 |
| Herding the cows | fine-tuned | 198 | 18 | 10.61 | 4 | 0.87 | 198 | 99.2 |
| Moving cows | `yolo11n.pt` | 418 | 13 | 7.63 | 3 | 0.94 | 398 | 97.9 |
| Moving cows | fine-tuned | 418 | 13 | 7.01 | 2 | 0.76 | 404 | 98.5 |

The fine-tuned model keeps cows in view in more frames but counts slightly fewer per frame. Contact sheets of annotated frames are in `reports/video_frames/`.

```powershell
python scripts/video_report.py --videos data/raw/videos/Herding_the_cows_video.webm --videos "data/raw/videos/Moving_cows_to_the_summer_range__42877666722_.webm" --weights yolo11n.pt --weights models/cattle_coco_yolo11n_best.pt --out reports/video_report_coco_finetune.md
```

Evaluation videos are downloaded to `data/raw/videos/` and are not committed:

- [“Herding the cows video”](https://commons.wikimedia.org/wiki/File:Herding_the_cows_video.webm) by Marie Nouvellon, CC BY 2.0.
- [“Moving cows to the summer range”](https://commons.wikimedia.org/wiki/File:Moving_cows_to_the_summer_range_(42877666722).webm) by BLM Oregon & Washington, public domain.

## Backend integration

Set the API key **only through the environment**. With `--backend-url` or `SMARTCATTLE_BACKEND_URL`, the CLI first requires `GET /health` to return `{"status":"ok"}`. The safe zone is normalized `left,top,right,bottom` and defaults to `0.1,0.1,0.9,0.9`; a cow is outside when its box's bottom center falls outside it. Boundaries count as inside.

```powershell
$env:SMARTCATTLE_API_KEY = "<your-api-key>"
python detect.py --source data/samples/0dfb0cb56b5f6f83.jpg --backend-url http://127.0.0.1:8000 --camera-id camera-01 --zone 0.1,0.1,0.9,0.9
python detect.py --source data/raw/videos/Herding_the_cows_video.webm --backend-url http://127.0.0.1:8000 --camera-id camera-01 --zone 0.1,0.1,0.9,0.9 --event-cooldown 10 --vid-stride 2
```

Outside-zone cows create `POST /api/ai/events` requests with `Content-Type: application/json` and, when set, `X-API-Key`. The backend must return HTTP 201. Payload shape (the confidence comes from the backend test fixture; the timestamp is supplied at runtime):

```json
{
  "event_type": "cattle_out_of_zone",
  "camera_id": "camera-01",
  "detected_object": "cow",
  "confidence": 0.91,
  "timestamp": "<UTC ISO 8601 timestamp>"
}
```

For videos, `--event-cooldown` defaults to 10 seconds per camera ID, measured in video time; at most one event is attempted during each cooldown interval. Images and folders use zero cooldown. A failed POST logs a warning and processing continues, but that attempt consumes the cooldown. Events contain neither bounding boxes nor person detections. The CLI prints `Backend events sent: <count>`.

## Configuration

`smartcattle_ai/config.py` reads the inference variables below; `detect.py` reads the backend variables. CLI flags override matching settings.

| Environment variable | Default | Meaning / CLI flag |
|---|---|---|
| `SMARTCATTLE_WEIGHTS` | `models/cattle_coco_yolo11n_best.pt` if present, else `yolo11n.pt` | Primary model; `--weights`. |
| `SMARTCATTLE_PERSON_WEIGHTS` | `yolo11n.pt` | Loaded only if the primary model lacks `person`; no flag. |
| `SMARTCATTLE_CONF` | `0.35` | Inference confidence; `--conf`. |
| `SMARTCATTLE_IOU` | `0.5` | Inference NMS IoU; `--iou`. |
| `SMARTCATTLE_IMGSZ` | `640` | Input size; `--imgsz`. |
| `SMARTCATTLE_CLASSES` | `cow,person` | Output classes; `--classes`. |
| `SMARTCATTLE_DEVICE` | CUDA `0` if available, else `cpu` | Inference device; `--device`. |
| `SMARTCATTLE_BACKEND_URL` | Unset | Enables publishing; `--backend-url`. |
| `SMARTCATTLE_API_KEY` | Unset | Event `X-API-Key`; environment only. |
| `SMARTCATTLE_CAMERA_ID` | `camera-01` | Event camera ID; `--camera-id`. |
| `SMARTCATTLE_ZONE` | `0.1,0.1,0.9,0.9` | Normalized safe zone; `--zone`. |

If `models/cattle_coco_yolo11n_best.pt` is missing (for example on a fresh clone, since weights are not committed), inference falls back to pretrained `yolo11n.pt`; the `Model:` line shows which one is in use. `--event-cooldown` defaults to 10 seconds and has no environment variable. Settings permit only `cow` and `person`, require confidence and IoU in `(0, 1]`, and require `imgsz` to be at least 32 and divisible by 32.

## Tests

```powershell
python -m pytest -q
python -m pytest -q -m real
```

The first command runs the whole suite (74 tests), **including** tests marked `real`; the second selects only those. The real tests load YOLO and infer on an Ultralytics sample and, when available, local cattle samples. Other tests cover configuration and default-model selection, class routing, image/video processing, stride, zone boundaries, CLI behavior, a local test backend and its contract, event cooldown, dataset validation, and metric calculations.

## Limitations

- **Small dataset.** 404 images, 94 in test. Results do not establish performance on other farms, cameras, weather, or lighting, and nothing has been measured on the target farm's camera yet.
- **Recall.** At confidence 0.35 the model still misses 53 of 152 test cows (recall 0.651) and 91 of 153 test persons (recall 0.405).
- **Distant and crowded cows.** Very small distant cows and tightly overlapping herds are the weakest scenarios; the fine-tuned model is worse than pretrained on tiny distant cows.
- **Unusual breeds.** Watusi and Highland cattle are each detected only 1 of 2 times.
- **No tracking.** Video totals count the same animal in every frame; there is no identity, so throttling is per camera, not per animal.
- **CPU speed.** About 60–100 ms per frame at 640 px on the recorded CPU, so a live stream needs frame skipping (`--vid-stride`).
- **Inputs.** `detect.py` requires an existing file or folder; the live camera goes through `live.py` instead.

## Live camera

`live.py` reads the IMOU camera over RTSP (credentials from `.env`, see `.env.example`), runs the detector on the newest frame only (about 10 FPS on CPU), and serves on `http://localhost:8090`:

| Path | Content |
| --- | --- |
| `/video.mjpg` | MJPEG video with boxes and the safe zone drawn in; cows outside the zone are red. |
| `/status` | Camera state, frame size, FPS and the latest detections (pixel boxes, `inside_zone`). |
| `/snapshot.jpg` | Latest annotated frame. |
| `/health` | `{"status":"ok"}` |

With `SMARTCATTLE_BACKEND_URL` set it reports the camera status every 15 s (`PUT /api/ai/cameras/{id}/status`) and posts out-of-zone events, at most one per `--event-cooldown` seconds. A backend that is down does not stop the video. `/status` allows CORS only from `SMARTCATTLE_LIVE_ORIGINS` (default: local Vite ports). It listens on `127.0.0.1` unless `--host 0.0.0.0` is given.

```powershell
python probe_camera.py --duration 10 --reconnect-test   # check the stream first
python live.py                                          # camera + YOLO + video server
powershell -ExecutionPolicy Bypass -File scripts\start_local.ps1   # backend, live.py and frontend together
```

`python live.py --tunnel` (or `SMARTCATTLE_LIVE_TUNNEL=1`) also starts a Cloudflare quick tunnel (`cloudflared` on PATH, in `%LOCALAPPDATA%\cloudflared\`, or `SMARTCATTLE_CLOUDFLARED`) and reports its `https://*.trycloudflare.com` URL to the backend as the camera's `stream_url`, so a hosted frontend finds the video without a rebuild. The URL changes on every start, and anyone who has it can watch the camera. `--public-url` reports a fixed URL instead.

The frontend uses the camera's `stream_url` from the backend and falls back to `VITE_AI_SERVICE_URL` (default setup: `http://localhost:8090`). The service runs on the PC that shares the camera's network; a hosted frontend can only show the video in a browser that can reach this PC. Test the model on footage from this camera before relying on its counts.

## Credits and licenses

[Ultralytics YOLO](https://github.com/ultralytics/ultralytics) is distributed under AGPL-3.0; review its obligations before deployment or redistribution. Open Images V7 images here are CC BY 2.0, with per-image author and source links in [`data/ATTRIBUTION.csv`](data/ATTRIBUTION.csv); Open Images annotations are CC BY 4.0. Scenario image credits are in [`data/samples/manifest.csv`](data/samples/manifest.csv) and the scenario reports. Video credits appear in the evaluation section.
