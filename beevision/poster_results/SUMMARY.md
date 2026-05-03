# BeeVision — poster results summary

Auto-generated; re-run `make poster-results` after new training runs.

## Segmentation (DS-COMB / `deepbee_seg`, DINOv2 ViT-S + UNet)

- **Best checkpoint (val IoU comb):** epoch 47
- **Test IoU (comb):** 0.9643
- **Test mIoU:** 0.9696
- **Test Dice (comb):** 0.9818
- **Test pixel accuracy:** 0.9850

## Mite detector (Varroa + BeeImage, ResNet-50)

- **Best checkpoint (val balanced accuracy):** epoch 9
- **Test accuracy:** 0.9359
- **Test balanced accuracy:** 0.9238
- **Test AUROC:** 0.9754
- **Test F1 (mite):** 0.8832

## Cell classifier (`deepbee_cls`, 7-class ResNet-50)

- **Best checkpoint (val macro-F1):** epoch 12
- **Test accuracy:** 0.5629
- **Test balanced accuracy:** 0.5789
- **Test macro-F1:** 0.3749

## Figure files

| File | Description |
|------|-------------|
| `segmentation_training_curves.png` | Train/val loss & IoU (comb) |
| `segmentation_test_metrics_bar.png` | Held-out test scalar metrics |
| `mite_training_curves.png` | Train loss; val balanced acc & AUROC |
| `mite_confusion_matrix.png` | Confusion counts + row-normalized |
| `mite_test_metrics_bar.png` | Held-out test headline metrics |
| `mite_roc_curve.png` | ROC on test (re-eval from `best.pt`) |
| `segmentation_qualitative_test.png` | RGB / GT / pred / error |
| `cells_training_curves.png` | Loss + val macro-F1 / acc / bal-acc |
| `cells_test_headline_bar.png` | Test accuracy / bal-acc / macro-F1 |
| `cells_per_class_f1.png` | Per-class F1 (horizontal bars) |
| `cells_confusion_matrix.png` | 7×7 counts + row-normalized |
