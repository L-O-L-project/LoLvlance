from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import onnxruntime as ort

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.label_schema import (  # noqa: E402
    DEFAULT_ISSUE_THRESHOLDS,
    DEFAULT_SOURCE_THRESHOLDS,
    ISSUE_LABELS,
    SOURCE_LABELS,
)
from ml.preprocessing import (  # noqa: E402
    DEFAULT_MEL_BIN_COUNT,
    PreprocessingConfig,
    extract_audio_features_from_path,
)

SMALL_GOLDEN_SET_WARNING = "golden set is small; results are for regression detection only"
BASELINE_SCHEMA_VERSION = "2.0"
NONE_LABEL = "<none>"


@dataclass(frozen=True)
class GoldenSample:
    sample_id: str
    audio_path: Path
    metadata_path: Path
    expected_issue: tuple[str, ...]
    expected_source: tuple[str, ...]
    severity: str


@dataclass(frozen=True)
class SamplePrediction:
    sample: GoldenSample
    predicted_issue: tuple[str, ...]
    predicted_source: tuple[str, ...]
    issue_probs: dict[str, float]
    source_probs: dict[str, float]


class OnnxAudioModel:
    def __init__(self, model_path: Path) -> None:
        self.model_path = model_path
        self.session = ort.InferenceSession(model_path.as_posix(), providers=["CPUExecutionProvider"])
        self.preprocessing_config = PreprocessingConfig(mel_bin_count=DEFAULT_MEL_BIN_COUNT)

    def predict(self, audio_path: Path) -> dict[str, np.ndarray]:
        features = extract_audio_features_from_path(audio_path, config=self.preprocessing_config)
        outputs = self.session.run(
            None,
            {"log_mel_spectrogram": features.log_mel_spectrogram[np.newaxis, :, :].astype(np.float32)},
        )
        output_names = [output.name for output in self.session.get_outputs()]
        output_map = {name: value for name, value in zip(output_names, outputs)}

        issue_probs = np.asarray(output_map["issue_probs"], dtype=np.float32)
        source_probs = np.asarray(output_map["source_probs"], dtype=np.float32)

        if issue_probs.ndim != 2 or issue_probs.shape[0] != 1 or issue_probs.shape[1] != len(ISSUE_LABELS):
            raise RuntimeError(
                f"Expected issue_probs shape (1, {len(ISSUE_LABELS)}), received {tuple(issue_probs.shape)}."
            )
        if source_probs.ndim != 2 or source_probs.shape[0] != 1 or source_probs.shape[1] != len(SOURCE_LABELS):
            raise RuntimeError(
                f"Expected source_probs shape (1, {len(SOURCE_LABELS)}), received {tuple(source_probs.shape)}."
            )
        if not np.isfinite(issue_probs).all():
            raise RuntimeError("issue_probs contains non-finite values.")
        if not np.isfinite(source_probs).all():
            raise RuntimeError("source_probs contains non-finite values.")

        return {
            "issue_probs": issue_probs[0],
            "source_probs": source_probs[0],
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate LoLvlance ONNX checkpoints against golden audio samples.")
    parser.add_argument("--goldens-dir", type=Path, default=Path("eval/goldens"))
    parser.add_argument("--model-path", type=Path, default=Path("ml/checkpoints/lightweight_audio_model.onnx"))
    parser.add_argument("--thresholds-path", type=Path, default=Path("ml/checkpoints/label_thresholds.json"))
    parser.add_argument("--baseline-path", type=Path, default=Path("ml/eval/baseline.json"))
    parser.add_argument("--macro-epsilon", type=float, default=0.02)
    parser.add_argument("--per-label-epsilon", type=float, default=0.03)
    parser.add_argument("--weak-label-f1-threshold", type=float, default=0.4)
    parser.add_argument("--weak-label-epsilon", type=float, default=0.02)
    parser.add_argument("--max-ratio-per-label", type=float, default=0.5)
    parser.add_argument("--distribution-slack", type=float, default=0.2)
    parser.add_argument("--entropy-epsilon", type=float, default=0.12)
    parser.add_argument("--top-confusion-deltas", type=int, default=10)
    parser.add_argument("--report-json-path", type=Path, default=None)
    parser.add_argument("--write-baseline", action="store_true")
    return parser.parse_args()


def load_thresholds(thresholds_path: Path) -> tuple[dict[str, float], dict[str, float]]:
    if not thresholds_path.exists():
        return dict(DEFAULT_ISSUE_THRESHOLDS), dict(DEFAULT_SOURCE_THRESHOLDS)

    payload = json.loads(thresholds_path.read_text(encoding="utf-8"))
    issue_thresholds = dict(DEFAULT_ISSUE_THRESHOLDS)
    source_thresholds = dict(DEFAULT_SOURCE_THRESHOLDS)
    issue_thresholds.update(payload.get("issue_thresholds", {}))
    source_thresholds.update(payload.get("source_thresholds", {}))
    return issue_thresholds, source_thresholds


def discover_golden_samples(goldens_dir: Path) -> list[GoldenSample]:
    if not goldens_dir.exists():
        raise FileNotFoundError(f"Golden directory does not exist: {goldens_dir}")

    metadata_paths = sorted(goldens_dir.rglob("metadata.json"))
    if not metadata_paths:
        raise RuntimeError(f"No metadata.json files found under: {goldens_dir}")

    samples: list[GoldenSample] = []
    for metadata_path in metadata_paths:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        file_name = payload.get("file")
        if not isinstance(file_name, str) or not file_name.strip():
            raise ValueError(f"metadata.json is missing a valid 'file' entry: {metadata_path}")

        audio_path = metadata_path.parent / file_name
        if not audio_path.exists():
            raise FileNotFoundError(f"Golden audio file not found: {audio_path}")

        samples.append(
            GoldenSample(
                sample_id=metadata_path.parent.name,
                audio_path=audio_path,
                metadata_path=metadata_path,
                expected_issue=normalize_expected_labels(
                    payload.get("expected_issue", []),
                    ISSUE_LABELS,
                    field_name="expected_issue",
                    metadata_path=metadata_path,
                ),
                expected_source=normalize_expected_labels(
                    payload.get("expected_source", []),
                    SOURCE_LABELS,
                    field_name="expected_source",
                    metadata_path=metadata_path,
                ),
                severity=str(payload.get("severity", "unknown")).strip() or "unknown",
            )
        )

    return samples


def normalize_expected_labels(
    values: Any,
    allowed_labels: Iterable[str],
    *,
    field_name: str,
    metadata_path: Path,
) -> tuple[str, ...]:
    if not isinstance(values, list):
        raise ValueError(f"{field_name} must be a list in {metadata_path}")

    allowed = set(allowed_labels)
    normalized: list[str] = []
    seen: set[str] = set()
    for entry in values:
        if not isinstance(entry, str):
            raise ValueError(f"{field_name} must contain only strings in {metadata_path}")
        label = entry.strip()
        if label not in allowed:
            raise ValueError(f"Unknown label '{label}' in {field_name} for {metadata_path}")
        if label not in seen:
            normalized.append(label)
            seen.add(label)
    return tuple(normalized)


def select_labels(probabilities: dict[str, float], thresholds: dict[str, float]) -> tuple[str, ...]:
    return tuple(label for label, score in probabilities.items() if score >= float(thresholds.get(label, 0.5)))


def evaluate_samples(
    samples: list[GoldenSample],
    model: OnnxAudioModel,
    issue_thresholds: dict[str, float],
    source_thresholds: dict[str, float],
) -> list[SamplePrediction]:
    predictions: list[SamplePrediction] = []
    for sample in samples:
        output = model.predict(sample.audio_path)
        issue_probs = {label: float(score) for label, score in zip(ISSUE_LABELS, output["issue_probs"], strict=True)}
        source_probs = {label: float(score) for label, score in zip(SOURCE_LABELS, output["source_probs"], strict=True)}
        predictions.append(
            SamplePrediction(
                sample=sample,
                predicted_issue=select_labels(issue_probs, issue_thresholds),
                predicted_source=select_labels(source_probs, source_thresholds),
                issue_probs=issue_probs,
                source_probs=source_probs,
            )
        )
    return predictions


def compute_metrics(
    predictions: list[SamplePrediction],
    label_namespace: str,
    labels: tuple[str, ...],
    selector: str,
) -> dict[str, Any]:
    totals = {"tp": 0, "fp": 0, "fn": 0}
    macro_precision_values: list[float] = []
    macro_recall_values: list[float] = []
    macro_f1_values: list[float] = []
    per_label: dict[str, dict[str, float | int]] = {}

    for label in labels:
        tp = 0
        fp = 0
        fn = 0
        support = 0
        predicted_positive = 0

        for prediction in predictions:
            expected = set(getattr(prediction.sample, f"expected_{selector}"))
            observed = set(getattr(prediction, f"predicted_{selector}"))
            expected_has_label = label in expected
            observed_has_label = label in observed
            if expected_has_label:
                support += 1
            if observed_has_label:
                predicted_positive += 1
            if expected_has_label and observed_has_label:
                tp += 1
            elif not expected_has_label and observed_has_label:
                fp += 1
            elif expected_has_label and not observed_has_label:
                fn += 1

        precision, recall, f1 = compute_prf(tp, fp, fn)
        totals["tp"] += tp
        totals["fp"] += fp
        totals["fn"] += fn
        macro_precision_values.append(precision)
        macro_recall_values.append(recall)
        macro_f1_values.append(f1)
        per_label[f"{label_namespace}:{label}"] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": support,
            "predicted_positive": predicted_positive,
        }
    precision, recall, f1 = compute_prf(totals["tp"], totals["fp"], totals["fn"])
    return {
        "overall": {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "tp": totals["tp"],
            "fp": totals["fp"],
            "fn": totals["fn"],
        },
        "macro": {
            "precision": round(mean(macro_precision_values), 4),
            "recall": round(mean(macro_recall_values), 4),
            "f1": round(mean(macro_f1_values), 4),
            "label_count": len(labels),
        },
        "per_label": per_label,
    }


def compute_combined_metrics(issue_metrics: dict[str, Any], source_metrics: dict[str, Any]) -> dict[str, Any]:
    tp = int(issue_metrics["overall"]["tp"]) + int(source_metrics["overall"]["tp"])
    fp = int(issue_metrics["overall"]["fp"]) + int(source_metrics["overall"]["fp"])
    fn = int(issue_metrics["overall"]["fn"]) + int(source_metrics["overall"]["fn"])
    precision, recall, f1 = compute_prf(tp, fp, fn)
    per_label = {**issue_metrics["per_label"], **source_metrics["per_label"]}
    return {
        "overall": {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "tp": tp,
            "fp": fp,
            "fn": fn,
        },
        "macro": {
            "precision": round(mean(float(metrics["precision"]) for metrics in per_label.values()), 4),
            "recall": round(mean(float(metrics["recall"]) for metrics in per_label.values()), 4),
            "f1": round(mean(float(metrics["f1"]) for metrics in per_label.values()), 4),
            "label_count": len(per_label),
        },
        "per_label": per_label,
    }


def compute_prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = float(tp) / float(tp + fp) if tp + fp > 0 else 0.0
    recall = float(tp) / float(tp + fn) if tp + fn > 0 else 0.0
    if precision + recall == 0.0:
        return precision, recall, 0.0
    f1 = 2.0 * precision * recall / (precision + recall)
    return precision, recall, f1


def build_prediction_distribution(
    predictions: list[SamplePrediction],
    label_namespace: str,
    labels: tuple[str, ...],
    selector: str,
) -> dict[str, Any]:
    predicted_positive_counts: Counter[str] = Counter()
    expected_counts: Counter[str] = Counter()
    top1_counts: Counter[str] = Counter()

    for prediction in predictions:
        predicted_labels = tuple(getattr(prediction, f"predicted_{selector}"))
        expected_labels = tuple(getattr(prediction.sample, f"expected_{selector}"))
        probability_map = getattr(prediction, f"{selector}_probs")
        top_label = max(probability_map.items(), key=lambda entry: entry[1])[0]

        for label in predicted_labels:
            predicted_positive_counts[f"{label_namespace}:{label}"] += 1
        for label in expected_labels:
            expected_counts[f"{label_namespace}:{label}"] += 1
        top1_counts[f"{label_namespace}:{top_label}"] += 1

    sample_count = len(predictions)
    by_label: dict[str, dict[str, float | int]] = {}
    for label in labels:
        label_key = f"{label_namespace}:{label}"
        predicted_positive = int(predicted_positive_counts.get(label_key, 0))
        expected_positive = int(expected_counts.get(label_key, 0))
        top1_count = int(top1_counts.get(label_key, 0))
        by_label[label_key] = {
            "predicted_positive": predicted_positive,
            "predicted_ratio": round(predicted_positive / sample_count, 4) if sample_count else 0.0,
            "expected_positive": expected_positive,
            "expected_ratio": round(expected_positive / sample_count, 4) if sample_count else 0.0,
            "top1_count": top1_count,
            "top1_ratio": round(top1_count / sample_count, 4) if sample_count else 0.0,
        }

    dominant_label, dominant_ratio = max(
        ((label, float(metrics["predicted_ratio"])) for label, metrics in by_label.items()),
        key=lambda entry: entry[1],
        default=(None, 0.0),
    )
    return {
        "by_label": by_label,
        "positive_entropy": {
            "normalized": round(normalized_entropy(list(predicted_positive_counts.values()), len(labels)), 4),
            "counts": [int(predicted_positive_counts.get(f"{label_namespace}:{label}", 0)) for label in labels],
        },
        "top1_entropy": {
            "normalized": round(normalized_entropy(list(top1_counts.values()), len(labels)), 4),
            "counts": [int(top1_counts.get(f"{label_namespace}:{label}", 0)) for label in labels],
        },
        "dominant_label": dominant_label,
        "dominant_ratio": round(dominant_ratio, 4),
    }


def build_confusion_summary(predictions: list[SamplePrediction]) -> dict[str, Any]:
    false_positive_counts: Counter[str] = Counter()
    false_negative_counts: Counter[str] = Counter()
    confusion_pairs: Counter[str] = Counter()
    mismatched_samples: list[dict[str, Any]] = []

    for prediction in predictions:
        issue_summary = build_selector_confusion(
            expected_labels=prediction.sample.expected_issue,
            predicted_labels=prediction.predicted_issue,
            namespace="issue",
        )
        source_summary = build_selector_confusion(
            expected_labels=prediction.sample.expected_source,
            predicted_labels=prediction.predicted_source,
            namespace="source",
        )

        merge_counter(false_positive_counts, issue_summary["false_positives"])
        merge_counter(false_positive_counts, source_summary["false_positives"])
        merge_counter(false_negative_counts, issue_summary["false_negatives"])
        merge_counter(false_negative_counts, source_summary["false_negatives"])
        merge_counter(confusion_pairs, issue_summary["confusion_pairs"])
        merge_counter(confusion_pairs, source_summary["confusion_pairs"])

        if (
            issue_summary["issue_false_positives"]
            or issue_summary["issue_false_negatives"]
            or source_summary["source_false_positives"]
            or source_summary["source_false_negatives"]
        ):
            mismatched_samples.append(
                {
                    "sample_id": prediction.sample.sample_id,
                    "severity": prediction.sample.severity,
                    "expected_issue": list(prediction.sample.expected_issue),
                    "predicted_issue": list(prediction.predicted_issue),
                    "expected_source": list(prediction.sample.expected_source),
                    "predicted_source": list(prediction.predicted_source),
                    "issue_false_positives": issue_summary["issue_false_positives"],
                    "issue_false_negatives": issue_summary["issue_false_negatives"],
                    "source_false_positives": source_summary["source_false_positives"],
                    "source_false_negatives": source_summary["source_false_negatives"],
                }
            )

    return {
        "false_positives": dict(sorted(false_positive_counts.items())),
        "false_negatives": dict(sorted(false_negative_counts.items())),
        "confusion_matrix": dict(sorted(confusion_pairs.items())),
        "mismatched_samples": mismatched_samples,
    }


def build_selector_confusion(
    *,
    expected_labels: tuple[str, ...],
    predicted_labels: tuple[str, ...],
    namespace: str,
) -> dict[str, Any]:
    expected = set(expected_labels)
    predicted = set(predicted_labels)
    false_positives = sorted(predicted - expected)
    false_negatives = sorted(expected - predicted)
    confusion_pairs: Counter[str] = Counter()

    for label in false_positives:
        confusion_pairs[f"{namespace}:{NONE_LABEL}->{label}"] += 1
    for label in false_negatives:
        confusion_pairs[f"{namespace}:{label}->{NONE_LABEL}"] += 1
    if false_negatives and false_positives:
        for expected_label in false_negatives:
            for predicted_label in false_positives:
                confusion_pairs[f"{namespace}:{expected_label}->{predicted_label}"] += 1

    return {
        f"{namespace}_false_positives": false_positives,
        f"{namespace}_false_negatives": false_negatives,
        "false_positives": Counter(f"{namespace}:{label}" for label in false_positives),
        "false_negatives": Counter(f"{namespace}:{label}" for label in false_negatives),
        "confusion_pairs": confusion_pairs,
    }


def build_confusion_delta_summary(
    current_confusion_summary: dict[str, Any],
    baseline_payload: dict[str, Any] | None,
    *,
    top_n: int,
) -> list[dict[str, Any]]:
    baseline_confusion = extract_baseline_confusion_matrix(baseline_payload)
    if not baseline_confusion:
        return []

    current_confusion = current_confusion_summary.get("confusion_matrix", {})
    if not isinstance(current_confusion, dict):
        return []

    deltas: list[dict[str, Any]] = []
    for key in set(current_confusion.keys()) | set(baseline_confusion.keys()):
        current_count = int(current_confusion.get(key, 0))
        baseline_count = int(baseline_confusion.get(key, 0))
        delta = current_count - baseline_count
        if delta == 0:
            continue
        deltas.append({"pair": key, "baseline": baseline_count, "current": current_count, "delta": delta})

    deltas.sort(key=lambda entry: (abs(int(entry["delta"])), int(entry["current"])), reverse=True)
    return deltas[:top_n]


def build_report(
    predictions: list[SamplePrediction],
    model_path: Path,
    issue_thresholds: dict[str, float],
    source_thresholds: dict[str, float],
) -> dict[str, Any]:
    issue_metrics = compute_metrics(predictions, "issue", ISSUE_LABELS, "issue")
    source_metrics = compute_metrics(predictions, "source", SOURCE_LABELS, "source")
    combined_metrics = compute_combined_metrics(issue_metrics, source_metrics)

    return {
        "schema_version": BASELINE_SCHEMA_VERSION,
        "note": SMALL_GOLDEN_SET_WARNING,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_path": model_path.as_posix(),
        "sample_count": len(predictions),
        "thresholds": {
            "issue_thresholds": issue_thresholds,
            "source_thresholds": source_thresholds,
        },
        "groups": {
            "issue": issue_metrics,
            "source": source_metrics,
            "combined": combined_metrics,
        },
        "prediction_distribution": {
            "issue": build_prediction_distribution(predictions, "issue", ISSUE_LABELS, "issue"),
            "source": build_prediction_distribution(predictions, "source", SOURCE_LABELS, "source"),
        },
        "confusion_summary": build_confusion_summary(predictions),
        "samples": [
            {
                "sample_id": prediction.sample.sample_id,
                "severity": prediction.sample.severity,
                "file": prediction.sample.audio_path.name,
                "expected_issue": list(prediction.sample.expected_issue),
                "predicted_issue": list(prediction.predicted_issue),
                "expected_source": list(prediction.sample.expected_source),
                "predicted_source": list(prediction.predicted_source),
                "top_issue_probs": top_k_scores(prediction.issue_probs),
                "top_source_probs": top_k_scores(prediction.source_probs),
            }
            for prediction in predictions
        ],
    }


def top_k_scores(scores: dict[str, float], k: int = 3) -> list[dict[str, float | str]]:
    return [
        {"label": label, "score": round(score, 4)}
        for label, score in sorted(scores.items(), key=lambda entry: entry[1], reverse=True)[:k]
    ]


def load_baseline(baseline_path: Path) -> dict[str, Any] | None:
    if not baseline_path.exists():
        return None
    return json.loads(baseline_path.read_text(encoding="utf-8"))


def build_gate_config(args: argparse.Namespace) -> dict[str, float | int]:
    return {
        "macro_epsilon": round(args.macro_epsilon, 4),
        "per_label_epsilon": round(args.per_label_epsilon, 4),
        "weak_label_f1_threshold": round(args.weak_label_f1_threshold, 4),
        "weak_label_epsilon": round(args.weak_label_epsilon, 4),
        "max_ratio_per_label": round(args.max_ratio_per_label, 4),
        "distribution_slack": round(args.distribution_slack, 4),
        "entropy_epsilon": round(args.entropy_epsilon, 4),
        "top_confusion_deltas": int(args.top_confusion_deltas),
    }


def build_gate_report(
    report: dict[str, Any],
    baseline_payload: dict[str, Any] | None,
    *,
    gate_config: dict[str, float | int],
) -> dict[str, Any]:
    failures: list[str] = []
    regression_flags = {
        "macro_regression": False,
        "per_label_regressions": [],
        "weak_label_regressions": [],
        "bias_violations": [],
        "entropy_regressions": [],
    }

    current_macro_f1 = float(report["groups"]["combined"]["macro"]["f1"])
    baseline_macro_f1 = extract_baseline_macro_f1(baseline_payload)
    macro_check = {
        "current_macro_f1": round(current_macro_f1, 4),
        "baseline_macro_f1": round(baseline_macro_f1, 4) if baseline_macro_f1 is not None else None,
        "epsilon": float(gate_config["macro_epsilon"]),
        "passed": True,
    }

    if baseline_payload is None:
        failures.append("baseline file is missing; relative regression checks require a baseline artifact")
        regression_flags["macro_regression"] = True
        macro_check["passed"] = False
    elif baseline_macro_f1 is not None and current_macro_f1 < baseline_macro_f1 - float(gate_config["macro_epsilon"]):
        failures.append(
            "combined macro F1 regressed "
            f"from {baseline_macro_f1:.4f} to {current_macro_f1:.4f} "
            f"(epsilon={float(gate_config['macro_epsilon']):.4f})"
        )
        regression_flags["macro_regression"] = True
        macro_check["passed"] = False

    per_label_gate = build_per_label_gate(report, baseline_payload, gate_config=gate_config)
    bias_gate = build_bias_gate(report, baseline_payload, gate_config=gate_config)
    failures.extend(per_label_gate["failures"])
    failures.extend(bias_gate["failures"])
    regression_flags["per_label_regressions"] = per_label_gate["regressions"]
    regression_flags["weak_label_regressions"] = per_label_gate["weak_label_regressions"]
    regression_flags["bias_violations"] = bias_gate["violations"]
    regression_flags["entropy_regressions"] = bias_gate["entropy_regressions"]

    return {
        "status": "fail" if failures else "pass",
        "warnings": [SMALL_GOLDEN_SET_WARNING],
        "failures": failures,
        "regression_flags": regression_flags,
        "checks": {
            "macro_non_regression": macro_check,
            "per_label_non_regression": per_label_gate["checks"],
            "bias_distribution": bias_gate["distribution_checks"],
            "entropy_non_regression": bias_gate["entropy_checks"],
            "confusion_shift_top_changes": build_confusion_delta_summary(
                report["confusion_summary"],
                baseline_payload,
                top_n=int(gate_config["top_confusion_deltas"]),
            ),
        },
    }


def build_per_label_gate(
    report: dict[str, Any],
    baseline_payload: dict[str, Any] | None,
    *,
    gate_config: dict[str, float | int],
) -> dict[str, Any]:
    failures: list[str] = []
    regressions: list[dict[str, Any]] = []
    weak_label_regressions: list[dict[str, Any]] = []
    checks: list[dict[str, Any]] = []
    if baseline_payload is None:
        return {
            "failures": failures,
            "regressions": regressions,
            "weak_label_regressions": weak_label_regressions,
            "checks": checks,
        }

    baseline_per_label = extract_baseline_per_label_metrics(baseline_payload)
    current_per_label = report["groups"]["combined"]["per_label"]
    labels = sorted(set(current_per_label.keys()) | set(baseline_per_label.keys()))

    for label in labels:
        baseline_metrics = baseline_per_label.get(label)
        current_metrics = current_per_label.get(label)
        baseline_f1 = float(baseline_metrics["f1"]) if baseline_metrics is not None else 0.0
        current_f1 = float(current_metrics["f1"]) if current_metrics is not None else 0.0
        is_weak_label = baseline_f1 <= float(gate_config["weak_label_f1_threshold"])
        allowed_drop = float(gate_config["weak_label_epsilon"]) if is_weak_label else float(gate_config["per_label_epsilon"])
        passed = current_f1 >= baseline_f1 - allowed_drop
        check = {
            "label": label,
            "baseline_f1": round(baseline_f1, 4),
            "current_f1": round(current_f1, 4),
            "allowed_drop": round(allowed_drop, 4),
            "is_weak_label": is_weak_label,
            "passed": passed,
        }
        checks.append(check)
        if not passed:
            failures.append(
                f"{label} F1 regressed from {baseline_f1:.4f} to {current_f1:.4f} "
                f"(allowed_drop={allowed_drop:.4f})"
            )
            regressions.append(check)
            if is_weak_label:
                weak_label_regressions.append(check)

    return {
        "failures": failures,
        "regressions": regressions,
        "weak_label_regressions": weak_label_regressions,
        "checks": checks,
    }


def build_bias_gate(
    report: dict[str, Any],
    baseline_payload: dict[str, Any] | None,
    *,
    gate_config: dict[str, float | int],
) -> dict[str, Any]:
    failures: list[str] = []
    violations: list[dict[str, Any]] = []
    entropy_regressions: list[dict[str, Any]] = []
    distribution_checks: list[dict[str, Any]] = []
    entropy_checks: list[dict[str, Any]] = []
    baseline_distribution = extract_baseline_prediction_distribution(baseline_payload)

    for namespace in ("issue", "source"):
        namespace_distribution = report["prediction_distribution"][namespace]
        baseline_namespace_distribution = baseline_distribution.get(namespace, {})

        for label, metrics in namespace_distribution.get("by_label", {}).items():
            predicted_ratio = float(metrics["predicted_ratio"])
            expected_ratio = float(metrics["expected_ratio"])
            allowed_ratio = min(
                1.0,
                max(float(gate_config["max_ratio_per_label"]), expected_ratio + float(gate_config["distribution_slack"])),
            )
            passed = predicted_ratio <= allowed_ratio
            check = {
                "label": label,
                "predicted_ratio": round(predicted_ratio, 4),
                "expected_ratio": round(expected_ratio, 4),
                "allowed_ratio": round(allowed_ratio, 4),
                "passed": passed,
            }
            distribution_checks.append(check)
            if not passed:
                failures.append(
                    f"{label} predicted-positive ratio {predicted_ratio:.4f} exceeds allowed ratio {allowed_ratio:.4f} "
                    f"(gold prevalence={expected_ratio:.4f})"
                )
                violations.append(check)

        baseline_entropy = extract_nested_float(baseline_namespace_distribution, ("positive_entropy", "normalized"))
        current_entropy = float(namespace_distribution["positive_entropy"]["normalized"])
        entropy_check = {
            "namespace": namespace,
            "baseline_positive_entropy": round(baseline_entropy, 4) if baseline_entropy is not None else None,
            "current_positive_entropy": round(current_entropy, 4),
            "epsilon": float(gate_config["entropy_epsilon"]),
            "passed": True,
        }
        if baseline_entropy is not None and current_entropy < baseline_entropy - float(gate_config["entropy_epsilon"]):
            failures.append(
                f"{namespace} prediction entropy regressed from {baseline_entropy:.4f} "
                f"to {current_entropy:.4f} "
                f"(epsilon={float(gate_config['entropy_epsilon']):.4f})"
            )
            entropy_check["passed"] = False
            entropy_regressions.append(entropy_check)
        entropy_checks.append(entropy_check)

    return {
        "failures": failures,
        "violations": violations,
        "entropy_regressions": entropy_regressions,
        "distribution_checks": distribution_checks,
        "entropy_checks": entropy_checks,
    }


def normalized_entropy(counts: Iterable[int], label_count: int) -> float:
    values = [max(0, int(count)) for count in counts]
    total = sum(values)
    if total <= 0 or label_count <= 1:
        return 0.0

    entropy = 0.0
    for value in values:
        if value <= 0:
            continue
        probability = value / total
        entropy -= probability * math.log(probability)

    max_entropy = math.log(label_count)
    if max_entropy <= 0.0:
        return 0.0
    return entropy / max_entropy


def mean(values: Iterable[float]) -> float:
    numeric_values = [float(value) for value in values]
    if not numeric_values:
        return 0.0
    return sum(numeric_values) / len(numeric_values)


def merge_counter(target: Counter[str], source: Counter[str]) -> None:
    target.update(source)


def strip_samples_from_report(report: dict[str, Any]) -> dict[str, Any]:
    baseline_report = json.loads(json.dumps(report))
    baseline_report.pop("samples", None)
    baseline_report.pop("gate", None)
    return baseline_report


def extract_baseline_macro_f1(baseline_payload: dict[str, Any] | None) -> float | None:
    return extract_nested_float(
        baseline_payload,
        ("groups", "combined", "macro", "f1"),
        ("groups", "combined", "overall", "f1"),
        ("overall", "f1"),
    )


def extract_baseline_per_label_metrics(baseline_payload: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    if baseline_payload is None:
        return {}

    per_label = baseline_payload.get("groups", {}).get("combined", {}).get("per_label")
    if isinstance(per_label, dict):
        return {str(label): metrics for label, metrics in per_label.items() if isinstance(metrics, dict)}

    legacy_per_label = baseline_payload.get("per_label_f1")
    if isinstance(legacy_per_label, dict):
        return {
            str(label): {"f1": float(score), "precision": float(score), "recall": float(score), "support": 0}
            for label, score in legacy_per_label.items()
            if isinstance(score, (int, float))
        }

    return {}


def extract_baseline_prediction_distribution(baseline_payload: dict[str, Any] | None) -> dict[str, Any]:
    if baseline_payload is None:
        return {}
    distribution = baseline_payload.get("prediction_distribution")
    return distribution if isinstance(distribution, dict) else {}


def extract_baseline_confusion_matrix(baseline_payload: dict[str, Any] | None) -> dict[str, int]:
    if baseline_payload is None:
        return {}

    confusion_matrix = baseline_payload.get("confusion_summary", {}).get("confusion_matrix")
    if not isinstance(confusion_matrix, dict):
        return {}

    return {str(key): int(value) for key, value in confusion_matrix.items()}


def extract_nested_float(
    payload: dict[str, Any] | None,
    *paths: tuple[str, ...],
) -> float | None:
    if payload is None:
        return None

    for path in paths:
        cursor: Any = payload
        for segment in path:
            if not isinstance(cursor, dict) or segment not in cursor:
                cursor = None
                break
            cursor = cursor[segment]
        if isinstance(cursor, (int, float)):
            return float(cursor)
    return None


def write_baseline(
    baseline_path: Path,
    report: dict[str, Any],
    *,
    gate_config: dict[str, float | int],
) -> None:
    baseline_payload = strip_samples_from_report(report)
    baseline_payload["gate_config"] = gate_config
    baseline_payload["baseline_written_at_utc"] = datetime.now(timezone.utc).isoformat()
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    baseline_path.write_text(json.dumps(baseline_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_report_json(report_json_path: Path, payload: dict[str, Any]) -> None:
    report_json_path.parent.mkdir(parents=True, exist_ok=True)
    report_json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def print_report(report: dict[str, Any]) -> None:
    gate = report["gate"]
    combined_macro = report["groups"]["combined"]["macro"]
    combined_overall = report["groups"]["combined"]["overall"]

    print("LoLvlance golden evaluation")
    print(f"warning: {SMALL_GOLDEN_SET_WARNING}")
    print(f"status: {gate['status'].upper()}")
    print(
        "combined macro F1: "
        f"{combined_macro['f1']:.4f} "
        f"(precision={combined_macro['precision']:.4f}, recall={combined_macro['recall']:.4f})"
    )
    print(
        "combined micro F1: "
        f"{combined_overall['f1']:.4f} "
        f"(precision={combined_overall['precision']:.4f}, recall={combined_overall['recall']:.4f})"
    )

    macro_check = gate["checks"]["macro_non_regression"]
    if macro_check["baseline_macro_f1"] is not None:
        print(
            "baseline macro F1: "
            f"{macro_check['baseline_macro_f1']:.4f} "
            f"(epsilon={macro_check['epsilon']:.4f})"
        )
    else:
        print("baseline macro F1: unavailable")

    print("per-label metrics:")
    for label, metrics in sorted(report["groups"]["combined"]["per_label"].items()):
        print(
            f"  {label}: "
            f"f1={float(metrics['f1']):.4f} "
            f"precision={float(metrics['precision']):.4f} "
            f"recall={float(metrics['recall']):.4f} "
            f"support={int(metrics['support'])} "
            f"predicted_positive={int(metrics['predicted_positive'])}"
        )

    print("prediction distribution:")
    for namespace in ("issue", "source"):
        namespace_distribution = report["prediction_distribution"][namespace]
        print(
            f"  {namespace}: dominant={namespace_distribution['dominant_label']} "
            f"(ratio={float(namespace_distribution['dominant_ratio']):.4f}), "
            f"positive_entropy={float(namespace_distribution['positive_entropy']['normalized']):.4f}"
        )
        for label, metrics in sorted(namespace_distribution["by_label"].items()):
            print(
                f"    {label}: "
                f"predicted_ratio={float(metrics['predicted_ratio']):.4f} "
                f"expected_ratio={float(metrics['expected_ratio']):.4f} "
                f"top1_ratio={float(metrics['top1_ratio']):.4f}"
            )

    confusion_changes = gate["checks"]["confusion_shift_top_changes"]
    if confusion_changes:
        print("top confusion shifts vs baseline:")
        for entry in confusion_changes:
            print(
                f"  {entry['pair']}: "
                f"baseline={int(entry['baseline'])} "
                f"current={int(entry['current'])} "
                f"delta={int(entry['delta'])}"
            )

    if gate["failures"]:
        print("gate failures:")
        for failure in gate["failures"]:
            print(f"  - {failure}")
    else:
        print("gate result: pass")


def main() -> int:
    args = parse_args()

    issue_thresholds, source_thresholds = load_thresholds(args.thresholds_path)
    samples = discover_golden_samples(args.goldens_dir)
    model = OnnxAudioModel(args.model_path)
    predictions = evaluate_samples(samples, model, issue_thresholds, source_thresholds)

    report = build_report(predictions, args.model_path, issue_thresholds, source_thresholds)
    baseline_payload = strip_samples_from_report(report) if args.write_baseline else load_baseline(args.baseline_path)
    gate_config = build_gate_config(args)
    report["gate"] = build_gate_report(report, baseline_payload, gate_config=gate_config)

    print_report(report)

    if args.report_json_path is not None:
        write_report_json(args.report_json_path, report)
        print(f"json report written to {args.report_json_path.as_posix()}")

    if args.write_baseline:
        write_baseline(args.baseline_path, report, gate_config=gate_config)
        print(f"baseline written to {args.baseline_path.as_posix()}")
        return 0

    return 1 if report["gate"]["status"] == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
