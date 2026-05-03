# Poster results (generated assets)

This directory holds **print-resolution figures** and a short **numeric summary** for course posters and reports. Files are produced by the repo script, not edited by hand.

## Regenerate

From the `beevision/` checkout (same layout as training):

```bash
export PYTHONPATH=src   # if you are not using pip install -e .
make poster-results
# optional: custom output dir or data root
make poster-results POSTER_OUT=/path/to/out
python scripts/make_poster_figures.py --repo . --out poster_results --bevision-root /oscar/scratch/$USER/beevision
```

The script auto-detects `$BEEVISION_ROOT` by searching, in order: the `BEEVISION_ROOT` environment variable, `/oscar/scratch/$USER/beevision`, then `beevision/data/` (symlinked checkouts). It reads `processed/checkpoints/*/metrics.jsonl` and loads `best.pt` from the **same directory** as each metrics file.

## Contents

| File | Description |
|------|-------------|
| `segmentation_training_curves.png` | Train/val loss and foreground IoU (DS-COMB / DINOv2-UNet). |
| `segmentation_test_metrics_bar.png` | Held-out test IoU/Dice/mIoU/pixel accuracy. |
| `segmentation_qualitative_test.png` | RGB / ground truth / prediction / error map (test set). |
| `mite_training_curves.png` | Train loss; val balanced accuracy and AUROC. |
| `mite_confusion_matrix.png` | Confusion counts and row-normalized view. |
| `mite_test_metrics_bar.png` | Held-out test headline metrics. |
| `mite_roc_curve.png` | ROC curve on test (re-inference from `best.pt`). |
| `cells_training_curves.png` | Cell head: loss + val macro-F1 / acc / balanced acc. |
| `cells_test_headline_bar.png` | Held-out test accuracy / balanced acc / macro-F1. |
| `cells_per_class_f1.png` | Per-class F1 bar chart (7 classes). |
| `cells_confusion_matrix.png` | 7×7 confusion: counts + row-normalized. |
| `SUMMARY.md` / `summary.json` | Copy-paste friendly numbers. |

`SUMMARY.md` is overwritten every run. Edit the poster in your design tool; keep this folder as the **export target** for reproducibility.
