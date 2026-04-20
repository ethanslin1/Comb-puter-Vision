# BeeVision

Computer vision system for automated colony health assessment from beehive frame photographs.

## Pipeline

```
frame photo
   │
   ▼  preprocessing/  white-balance → frame detection → homography rectification
rectified frame (canonical rectangle)
   │
   ▼  models/segmentation/  UNet + DINOv2 ViT-S encoder → per-cell class map
segmentation mask (7 classes: capped brood, uncapped brood, honey, nectar, pollen, empty, other)
   │
   ├─▶ models/mite/        crop brood cells → ResNet-50 binary classifier → mite counts
   │
   └─▶ metrics/            brood regularity, honey/pollen ratio, mite load → composite health score 1–10
                             │
                             ▼ explainability/ Grad-CAM + LIME overlays
                           pipeline/  HealthReport JSON + streamlit dashboard
```

## Quick start

```bash
cd beevision
pip install -e .
make data     # download + preprocess datasets
make train-seg
make train-mite
make infer IMAGE=data/samples/example_frame.jpg
```

## Layout

See `CLAUDE.md` in each subdirectory for local context. The main code lives in `src/beevision/`.
