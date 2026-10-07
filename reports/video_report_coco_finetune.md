# Video report

| Weights | Video | Frames processed | Max cattle | Avg cattle | Max persons | Avg persons | Frames with cattle | Mean ms/frame |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| yolo11n.pt | data\raw\videos\Herding_the_cows_video.webm | 198 | 22 | 11.44 | 4 | 1.08 | 193 | 113.94 |
| yolo11n.pt | data\raw\videos\Moving_cows_to_the_summer_range__42877666722_.webm | 418 | 13 | 7.63 | 3 | 0.94 | 398 | 97.91 |
| models/cattle_coco_yolo11n_best.pt | data\raw\videos\Herding_the_cows_video.webm | 198 | 18 | 10.61 | 4 | 0.87 | 198 | 99.24 |
| models/cattle_coco_yolo11n_best.pt | data\raw\videos\Moving_cows_to_the_summer_range__42877666722_.webm | 418 | 13 | 7.01 | 2 | 0.76 | 404 | 98.54 |

Video stride: 2. Averages use processed frames only. Mean ms/frame includes model loading, decoding, inference, annotation and video writing; contact-sheet generation is excluded. Empty videos report zero averages and timing.

- yolo11n.pt / Herding_the_cows_video.webm: [Contact sheet](video_frames/yolo11n_Herding_the_cows_video.jpg) / [Annotated video](../outputs/videos/yolo11n/Herding_the_cows_video_annotated.mp4)
- yolo11n.pt / Moving_cows_to_the_summer_range__42877666722_.webm: [Contact sheet](video_frames/yolo11n_Moving_cows_to_the_summer_range__42877666722_.jpg) / [Annotated video](../outputs/videos/yolo11n/Moving_cows_to_the_summer_range__42877666722__annotated.mp4)
- cattle_coco_yolo11n_best.pt / Herding_the_cows_video.webm: [Contact sheet](video_frames/cattle_coco_yolo11n_best_Herding_the_cows_video.jpg) / [Annotated video](../outputs/videos/cattle_coco_yolo11n_best/Herding_the_cows_video_annotated.mp4)
- cattle_coco_yolo11n_best.pt / Moving_cows_to_the_summer_range__42877666722_.webm: [Contact sheet](video_frames/cattle_coco_yolo11n_best_Moving_cows_to_the_summer_range__42877666722_.jpg) / [Annotated video](../outputs/videos/cattle_coco_yolo11n_best/Moving_cows_to_the_summer_range__42877666722__annotated.mp4)
