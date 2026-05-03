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
make env              # create/update conda env from environment.yml
make preprocess       # run all ingest_* modules in order
make sbatch-ingest SRC=deepbee_cls   # alt: submit any single ingest as a batch job
make sanity           # regenerate interim/_sanity/*.{png,json}
make test             # run pytest -q
make train-seg
make train-mite
make sbatch-train-cells   # 7-class cell head — GPU SLURM (6h); needs deepbee_cls ingest
make infer IMAGE=data/samples/example_frame.jpg
```

All paths resolve under `$BEEVISION_ROOT` (default
`/oscar/scratch/$USER/beevision`) via `configs/data.yaml`. No module hard-codes
an absolute path.

## Dataset Statistics

The ingest pipeline normalizes four raw datasets into unified Pydantic records
under `$BEEVISION_ROOT/interim/`, one Parquet per source plus PNG images/masks.
Splits are reproducible (seed=1337) and materialized as a `split` column; a
live `pytest` check asserts zero record-id overlap across train/val/test for
every source.

| Source         | Model feed          | Records | Image size           | Train  | Val   | Test   |
| -------------- | ------------------- | ------: | -------------------- | -----: | ----: | -----: |
| `varroa`       | mite classifier     | 13,507  | 224-short-edge RGB   |  8,223 | 1,876 |  3,408 |
| `beeimage`     | mite classifier     |  4,435  | 224-short-edge RGB   |  3,104 |   665 |    666 |
| `deepbee_cls`  | cell classifier     | 63,640  | 224 × 224 center-crop| 46,104 | 8,136 |  9,400 |
| `deepbee_seg`  | segmentation UNet   |     61  | 768 × 512 RGB + mask |     42 |     9 |     10 |

Counts are refreshed by `make sanity`, which writes:

* `interim/_sanity/stats.json`        — exact numbers + per-split/per-class breakdowns
* `interim/_sanity/class_balance.png` — stacked bar per source × split × class
* `interim/_sanity/sample_grid_<source>_<split>.png` — 8×8 thumbnail grids
  (segmentation grids overlay the binary mask in translucent red)
* `interim/_sanity/splits_summary.json` — from `make ... splits validation`

### Per-source notes

**`varroa` (Zenodo 4085044).** Single bee crops from in-hive video; each labeled
with a mite count. We binarize to `{mite, no_mite}` (count>0 → mite), honor the
zip's own train/val/test dirs, and de-duplicate the ~2 exact-duplicate rows in
`gt.csv` on generated record id (source+video+frame+bee id).

**`beeimage` (Kaggle).** Kaggle bee image set; the `health` column covers
several diagnoses. We keep only `healthy` → `no_mite` and `varroa`/`varrao`
(including the dataset's misspelling) → `mite`, drop all other diagnoses, and
carve a fresh stratified 70/15/15 split on the unified label.

**`deepbee_cls` (DS-COMB-PT).** Per-cell labels for 1,202 BEE_HOPE frames:
Portuguese→English label map (`eggs`→`egg`, `larves`→`larva`,
`capped`→`capped_brood`, …), `dontcare` dropped. `labels_test.csv`'s 10 DSC_*
frames aren't in our archive so their 30,133 test cells are dropped (with a
loud warning); 3 BEE_HOPE frames appear in both train and test, and test wins.
Val is a 15% stratified slice of the remaining train. The `other` class only
exists in `labels_test.csv`, so it legitimately has 0 rows in train/val.

**`deepbee_seg` (DS-COMB segmentation).** 61 frames + 61 annotation JPEGs.
Despite being stored as 3-channel JPEGs, the annotations are effectively
binary grayscale (R=G=B, ~50% of pixels at gray≈0, ~50% at gray≈255, 0.2%
mid-gray JPEG artifacts at boundaries). We therefore emit **binary** masks
(`{background, comb}`) with threshold `gray ≥ 128`. The decoder aborts if a
frame has >2% ambiguous pixels or if the p99 channel disagreement exceeds 8
(wall-to-wall true color would fail loudly). Before resize, frames get
**gray-world white balance** (per-channel rescale so all three channel means
match the overall mean) — the hive photos vary a lot in warm cast, and
equalizing the means gives the segmentation UNet a consistent color prior.
Per-frame RGB means are recorded in the parquet's `meta.wb_mean_rgb_*`
fields; side-by-side before/after grids for the first 50 frames are saved
to `interim/_debug/whitebalance/whitebalance_{00..04}.png`. Cell crops are
**not** white-balanced (they go through per-cell normalization at train
time instead). 70/15/15 split stratified on foreground-fraction tercile.

### Running the pipeline

```bash
# Foreground (small / iterative):
make ingest SRC=varroa
make ingest SRC=beeimage

# Batch on SLURM (for long runs — deepbee_cls is ~34 min, varroa ~11 min):
make sbatch-ingest SRC=varroa
make sbatch-ingest SRC=deepbee_cls

# Then build shards, validate splits, produce sanity plots:
make shards
python -m beevision.data.splits --config configs/data.yaml
make sanity

# Poster-ready figures (after training jobs finish):
make poster-results
```

## Layout

See `CLAUDE.md` in each subdirectory for local context. The main code lives in `src/beevision/`.
