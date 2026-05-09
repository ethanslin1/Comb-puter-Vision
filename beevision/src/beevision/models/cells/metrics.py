"""7-class cell-classifier metrics.

Accumulates a 7×7 confusion matrix across an eval pass, then derives accuracy,
balanced accuracy, macro-F1, and per-class precision/recall/F1.

We do **not** report AUROC: it's defined for binary tasks; one-vs-rest AUROC
for multi-class is rarely informative on this kind of data and would mostly
restate macro-F1.

Best-metric for checkpointing: ``macro_f1``. Plain accuracy can be
hijacked by the dominant ``capped_brood`` / ``nectar`` / ``honey`` classes;
macro-F1 weighs all 7 classes equally, including the rare ``egg`` and
``larva`` classes that matter for brood-regularity scoring downstream.

The ``other`` class is empty in train/val (only present in test). The
accumulator handles class-absent-in-this-split gracefully via clamped
denominators — precision/recall return 0 for absent classes and balanced
accuracy averages only over classes that *did* appear in the eval set.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class CellMetricAccumulator:
    num_classes: int = 7
    eps: float = 1e-9
    _confusion: torch.Tensor = field(init=False)

    def __post_init__(self) -> None:
        self._confusion = torch.zeros(
            self.num_classes, self.num_classes, dtype=torch.int64
        )

    def update(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        """Accumulate from logits ``(B, C)`` and target ``(B,)``."""
        pred = logits.argmax(dim=1)
        target = target.long()
        flat = target.view(-1) * self.num_classes + pred.view(-1)
        binc = torch.bincount(
            flat.detach().cpu(), minlength=self.num_classes * self.num_classes
        )
        self._confusion += binc.view(self.num_classes, self.num_classes).long()

    def compute(self) -> dict[str, float | list[float] | list[list[int]]]:
        cm = self._confusion.double()
        total = cm.sum().clamp_min(1)
        diag = cm.diag()

        pred_sum = cm.sum(dim=0).clamp_min(self.eps)  
        true_sum = cm.sum(dim=1).clamp_min(self.eps)  
        precision = (diag / pred_sum).tolist()
        recall = (diag / true_sum).tolist()

        accuracy = float(diag.sum() / total)

        present = (cm.sum(dim=1) > 0).double()
        n_present = present.sum().clamp_min(1)
        bal_acc = float(((diag / true_sum) * present).sum() / n_present)

        f1_per: list[float] = []
        for c in range(self.num_classes):
            p = precision[c]
            r = recall[c]
            f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
            f1_per.append(float(f1)) # nacro f1 average f1 over classes present in truths; skip absent.
        present_mask = (cm.sum(dim=1) > 0).cpu().numpy()
        if present_mask.any():
            macro_f1 = float(np.mean([f1_per[c] for c in range(self.num_classes) if present_mask[c]]))
        else:
            macro_f1 = 0.0

        return {
            "accuracy": accuracy,
            "balanced_accuracy": bal_acc,
            "macro_f1": macro_f1,
            "precision_per_class": [float(v) for v in precision],
            "recall_per_class": [float(v) for v in recall],
            "f1_per_class": f1_per,
            "confusion": [[int(v) for v in row] for row in cm.long().tolist()],
            "classes_present": [bool(v) for v in present_mask.tolist()],
        }
