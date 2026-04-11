from __future__ import annotations

from typing import Sequence

import numpy as np


def evaluate_multilabel_head(
    probabilities: np.ndarray,
    targets: np.ndarray,
    masks: np.ndarray,
    labels: Sequence[str],
    thresholds: dict[str, float],
) -> dict[str, object]:
    if probabilities.size == 0:
        return {
            "coverage": 0.0,
            "micro": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
            "macro_f1": 0.0,
            "per_label": {},
        }

    per_label: dict[str, dict[str, float | None]] = {}
    macro_f1_values: list[float] = []
    micro_tp = 0.0
    micro_fp = 0.0
    micro_fn = 0.0
    total_masked = float(masks.sum())

    for label_index, label in enumerate(labels):
        label_mask = masks[:, label_index].astype(bool)
        coverage = float(label_mask.mean()) if label_mask.size else 0.0

        if not label_mask.any():
            per_label[label] = {
                "threshold": float(thresholds[label]),
                "coverage": coverage,
                "precision": None,
                "recall": None,
                "f1": None,
                "auroc": None,
                "support": 0.0,
            }
            continue

        label_targets = targets[label_mask, label_index]
        label_scores = probabilities[label_mask, label_index]
        label_predictions = (label_scores >= thresholds[label]).astype(np.float32)
        tp = float(np.sum(label_predictions * label_targets))
        fp = float(np.sum(label_predictions * (1.0 - label_targets)))
        fn = float(np.sum((1.0 - label_predictions) * label_targets))

        precision = tp / max(1.0, tp + fp)
        recall = tp / max(1.0, tp + fn)
        f1 = 0.0 if precision + recall == 0 else (2.0 * precision * recall) / (precision + recall)
        auroc = compute_binary_auroc(label_targets, label_scores)

        per_label[label] = {
            "threshold": float(thresholds[label]),
            "coverage": coverage,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "auroc": None if auroc is None else round(auroc, 4),
            "support": float(np.sum(label_targets)),
        }

        macro_f1_values.append(f1)
        micro_tp += tp
        micro_fp += fp
        micro_fn += fn

    micro_precision = micro_tp / max(1.0, micro_tp + micro_fp)
    micro_recall = micro_tp / max(1.0, micro_tp + micro_fn)
    micro_f1 = 0.0 if micro_precision + micro_recall == 0 else (
        2.0 * micro_precision * micro_recall
    ) / (micro_precision + micro_recall)

    return {
        "coverage": round(total_masked / max(1.0, float(masks.size)), 4),
        "micro": {
            "precision": round(micro_precision, 4),
            "recall": round(micro_recall, 4),
            "f1": round(micro_f1, 4),
        },
        "macro_f1": round(float(np.mean(macro_f1_values)) if macro_f1_values else 0.0, 4),
        "per_label": per_label,
    }


def tune_thresholds(
    probabilities: np.ndarray,
    targets: np.ndarray,
    masks: np.ndarray,
    labels: Sequence[str],
    defaults: dict[str, float],
    candidate_thresholds: Sequence[float] | None = None,
) -> dict[str, float]:
    if candidate_thresholds is None:
        candidate_thresholds = tuple(np.arange(0.2, 0.81, 0.05).round(2))

    tuned = dict(defaults)

    for label_index, label in enumerate(labels):
        label_mask = masks[:, label_index].astype(bool)

        if not label_mask.any():
            continue

        label_targets = targets[label_mask, label_index]
        label_scores = probabilities[label_mask, label_index]

        if np.unique(label_targets).size < 2:
            continue

        best_threshold = defaults[label]
        best_f1 = -1.0

        for threshold in candidate_thresholds:
            predictions = (label_scores >= threshold).astype(np.float32)
            tp = float(np.sum(predictions * label_targets))
            fp = float(np.sum(predictions * (1.0 - label_targets)))
            fn = float(np.sum((1.0 - predictions) * label_targets))

            precision = tp / max(1.0, tp + fp)
            recall = tp / max(1.0, tp + fn)
            f1 = 0.0 if precision + recall == 0 else (2.0 * precision * recall) / (precision + recall)

            if f1 > best_f1:
                best_f1 = f1
                best_threshold = float(threshold)

        tuned[label] = round(best_threshold, 2)

    return tuned


def compute_binary_auroc(targets: np.ndarray, scores: np.ndarray) -> float | None:
    binary_targets = targets.astype(np.int32)
    positives = int(binary_targets.sum())
    negatives = int(binary_targets.shape[0] - positives)

    if positives == 0 or negatives == 0:
        return None

    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, order.size + 1, dtype=np.float64)
    positive_rank_sum = float(ranks[binary_targets == 1].sum())
    return (
        positive_rank_sum
        - (positives * (positives + 1) / 2.0)
    ) / float(positives * negatives)
