# live.py in the cloud: reads the camera through the Imou Open Platform (IMOU_* variables) and serves MJPEG.
FROM python:3.13-slim

# OpenCV (pulled in by ultralytics) needs these shared libraries even when no window is opened.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
# CPU-only PyTorch first: the default wheel bundles CUDA and is several GB.
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY smartcattle_ai smartcattle_ai
COPY live.py .
COPY models/cattle_coco_yolo11n_best.pt models/

ENV PYTHONUNBUFFERED=1 \
    YOLO_CONFIG_DIR=/tmp/ultralytics \
    SMARTCATTLE_DEVICE=cpu \
    SMARTCATTLE_LIVE_HOST=0.0.0.0
CMD ["python", "live.py"]
