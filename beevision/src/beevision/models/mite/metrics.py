"""Binary mite-classifier metrics.

We accumulate a 2×2 confusion matrix plus the per-sample positive-class
probability and target across an eval pass, then derive accuracy, balanced
accuracy, macro-F1, AUROC, and per-class precision/recall at the end.

Why store probs/targets instead of computing AUROC online: AUROC is rank-
based across the full set, not decomposable per-batch without a streaming
estimator. For our scale (test ~4k samples) the buffer is tiny.

Best-metric convention for checkpointing: ``balanced_accuracy``. Plain
accuracy lies under any class skew; balanced accuracy averages per-class
recall, so it can't be gamed by always predicting the majority class.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class ClassificationMetricAccumulator:
    num_classes: int = 2
    eps: float = 1e-9
    _confusion: torch.Tensor = field(init=False)
    _pos_probs: list[float] = field(init=False, default_factory=list)
    _targets: list[int] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        self._confusion = torch.zeros(
            self.num_classes, self.num_classes, dtype=torch.int64
        )
        self._pos_probs = []
        self._targets = []

    def update(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        """Accumulate from logits ``(B, C)`` and target ``(B,)``."""
        pred = logits.argmax(dim=1)
        target = target.long()

        for t, p in zip(target.view(-1).cpu().tolist(), pred.view(-1).cpu().tolist()):
            self._confusion[t, p] += 1

        if self.num_classes == 2:
            probs = torch.softmax(logits.float(), dim=1)[:, 1]
            self._pos_probs.extend(probs.detach().cpu().tolist())
            self._targets.extend(target.detach().cpu().tolist())

    def compute(self) -> dict[str, float | list[float] | list[list[int]]]:
        cm = self._confusion.double()
        total = cm.sum().clamp_min(1)
        diag = cm.diag()

        pred_sum = cm.sum(dim=0).clamp_min(self.eps)  # over true → predictions per class
        true_sum = cm.sum(dim=1).clamp_min(self.eps)  # over pred → truths per class
        precision = (diag / pred_sum).tolist()
        recall = (diag / true_sum).tolist()

        accuracy = float(diag.sum() / total)
        # Balanced accuracy = mean per-class recall (where the class is present).
        present = (cm.sum(dim=1) > 0).double()
        n_present = present.sum().clamp_min(1)
        bal_acc = float(((diag / true_sum) * present).sum() / n_present)

        f1_per = [] # pipelien for macro F-1
        for c in range(self.num_classes):
            p = precision[c]
            r = recall[c]
            f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
            f1_per.append(float(f1))
        macro_f1 = float(np.mean(f1_per))

        out: dict[str, float | list[float] | list[list[int]]] = {
            "accuracy": accuracy,
            "balanced_accuracy": bal_acc,
            "macro_f1": macro_f1,
            "precision_per_class": [float(v) for v in precision],
            "recall_per_class": [float(v) for v in recall],
            "f1_per_class": f1_per,
            "confusion": [[int(v) for v in row] for row in cm.long().tolist()],
        }

        if self.num_classes == 2 and self._pos_probs:
            out["auroc"] = _binary_auroc(self._pos_probs, self._targets)
            out["precision_mite"] = float(precision[1])
            out["recall_mite"] = float(recall[1])
            out["f1_mite"] = float(f1_per[1])

        return out


def _binary_auroc(scores: list[float], targets: list[int]) -> float:
    """ROC-AUC for binary classification via the rank formula.

    AUROC = (sum_of_positive_ranks - n_pos*(n_pos+1)/2) / (n_pos * n_neg).
    Average ranks for ties so the estimator is unbiased on tied scores.
    """
    s = np.asarray(scores, dtype=np.float64)
    t = np.asarray(targets, dtype=np.int64)
    n_pos = int((t == 1).sum())
    n_neg = int((t == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(s, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1, dtype=np.float64)
    
    sorted_scores = s[order]
    i = 0
    while i < len(sorted_scores):
        j = i + 1
        while j < len(sorted_scores) and sorted_scores[j] == sorted_scores[i]:
            j += 1
        if j > i + 1:
            avg = ranks[order[i:j]].mean()
            ranks[order[i:j]] = avg
        i = j

    pos_rank_sum = ranks[t == 1].sum()
    auc = (pos_rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)
