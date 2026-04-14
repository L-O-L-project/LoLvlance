from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
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
    parser.add_argument(
        "--goldens-dir",
        type=Path,
        default=Path("eval/goldens"),
        help="Directory containing golden sample subdirectories with metadata.json and audio files.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("ml/checkpoints/lightweight_audio_model.onnx"),
        help="Path to the ONNX model used for evaluation.",
    )
    parser.add_argument(
        "--thresholds-path",
        type=Path,
        default=Path("ml/checkpoints/label_thresholds.json"),
        help="Path to threshold configuration JSON.",
    )
    parser.add_argument(
        "--baseline-path",
        type=Path,
        default=Path("ml/eval/baseline.json"),
        help="Path to the baseline performance JSON used for regression gating.",
    )
    parser.add_argument(
        "--min-f1",
        type=float,
        default=0.65,
        help="Minimum allowed combined overall F1 before CI fails.",
    )
    parser.add_argument(
        "--regression-tolerance",
        type=float,
        default=0.02,
        help="Allowed F1 drop versus baseline before CI fails.",
    )
    parser.add_argument(
        "--critical-label",
        action="append",
        default=[],
        help="Optional critical label override. Use namespaced labels such as issue:harsh or source:vocal.",
    )
    parser.add_argument(
        "--write-baseline",
        action="store_true",
        help="Write the current evaluation result to --baseline-path instead of enforcing regression checks.",
    )
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

        expected_issue = normalize_expected_labels(
            payload.get("expected_issue", []),
            ISSUE_LABELS,
            field_name="expected_issue",
            metadata_path=metadata_path,
        )
        expected_source = normalize_expected_labels(
            payload.get("expected_source", []),
            SOURCE_LABELS,
            field_name="expected_source",
            metadata_path=metadata_path,
        )
        severity = payload.get("severity", "unknown")
        if not isinstance(severity, str) or not severity.strip():
            raise ValueError(f"metadata.json has an invalid severity value: {metadata_path}")

        samples.append(
            GoldenSample(
                sample_id=metadata_path.parent.name,
                audio_path=audio_path,
                metadata_path=metadata_path,
                expected_issue=expected_issue,
                expected_source=expected_source,
                severity=severity.strip(),
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
    selected = [
        label
        for label, score in probabilities.items()
        if score >= float(thresholds.get(label, 0.5))
    ]
    return tuple(selected)


def evaluate_samples(
    samples: list[GoldenSample],
    model: OnnxAudioModel,
    issue_thresholds: dict[str, float],
    source_thresholds: dict[str, float],
) -> list[SamplePrediction]:
    predictions: list[SamplePrediction] = []

    for sample in samples:
        output = model.predict(sample.audio_path)
        issue_probs = {
            label: float(score)
            for label, score in zip(ISSUE_LABELS, output["issue_probs"], strict=True)
        }
        source_probs = {
            label: float(score)
            for label, score in zip(SOURCE_LABELS, output["source_probs"], strict=True)
        }
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
        "per_label": per_label,
    }


def compute_combined_metrics(
    issue_metrics: dict[str, Any],
    source_metrics: dict[str, Any],
) -> dict[str, Any]:
    tp = int(issue_metrics["overall"]["tp"]) + int(source_metrics["overall"]["tp"])
    fp = int(issue_metrics["overall"]["fp"]) + int(source_metrics["overall"]["fp"])
    fn = int(issue_metrics["overall"]["fn"]) + int(source_metrics["overall"]["fn"])
    precision, recall, f1 = compute_prf(tp, fp, fn)

    per_label = {}
    per_label.update(issue_metrics["per_label"])
    per_label.update(source_metrics["per_label"])

    return {
        "overall": {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "tp": tp,
            "fp": fp,
            "fn": fn,
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


def build_confusion_summary(predictions: list[SamplePrediction]) -> dict[str, Any]:
    false_positive_counts: Counter[str] = Counter()
    false_negative_counts: Counter[str] = Counter()
    mismatched_samples: list[dict[str, Any]] = []

    for prediction in predictions:
        expected_issues = set(prediction.sample.expected_issue)
        predicted_issues = set(prediction.predicted_issue)
        expected_sources = set(prediction.sample.expected_source)
        predicted_sources = set(prediction.predicted_source)

        issue_fp = sorted(predicted_issues - expected_issues)
        issue_fn = sorted(expected_issues - predicted_issues)
        source_fp = sorted(predicted_sources - expected_sources)
        source_fn = sorted(expected_sources - predicted_sources)

        for label in issue_fp:
            false_positive_counts[f"issue:{label}"] += 1
        for label in issue_fn:
            false_negative_counts[f"issue:{label}"] += 1
        for label in source_fp:
            false_positive_counts[f"source:{label}"] += 1
        for label in source_fn:
            false_negative_counts[f"source:{label}"] += 1

        if issue_fp or issue_fn or source_fp or source_fn:
            mismatched_samples.append(
                {
                    "sample_id": prediction.sample.sample_id,
                    "severity": prediction.sample.severity,
                    "expected_issue": list(prediction.sample.expected_issue),
                    "predicted_issue": list(prediction.predicted_issue),
                    "expected_source": list(prediction.sample.expected_source),
                    "predicted_source": list(prediction.predicted_source),
                    "issue_false_positives": issue_fp,
                    "issue_false_negatives": issue_fn,
                    "source_false_positives": source_fp,
                    "source_false_negatives": source_fn,
                }
            )

    return {
        "false_positives": dict(sorted(false_positive_counts.items())),
        "false_negatives": dict(sorted(false_negative_counts.items())),
        "mismatched_samples": mismatched_samples,
    }


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
        "per_label_f1": {
            label: metrics["f1"]
            for label, metrics in combined_metrics["per_label"].items()
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


def resolve_critical_labels(
    explicit_labels: list[str],
    baseline_payload: dict[str, Any] | None,
) -> list[str]:
    if explicit_labels:
        return sorted(dict.fromkeys(explicit_labels))

    if baseline_payload:
        critical_labels = baseline_payload.get("critical_labels")
        if isinstance(critical_labels, list) and critical_labels:
            return [label for label in critical_labels if isinstance(label, str)]

        baseline_per_label = baseline_payload.get("per_label_f1", {})
        if isinstance(baseline_per_label, dict):
            return sorted(label for label in baseline_per_label.keys() if isinstance(label, str))

    return []


def check_gate(
    report: dict[str, Any],
    baseline_payload: dict[str, Any] | None,
    *,
    min_f1: float,
    regression_tolerance: float,
    critical_labels: list[str],
) -> list[str]:
    failures: list[str] = []
    overall_f1 = float(report["groups"]["combined"]["overall"]["f1"])

    if overall_f1 < min_f1:
        failures.append(
            f"overall combined F1 {overall_f1:.4f} is below min_f1 {min_f1:.4f}"
        )

    if not baseline_payload:
        return failures

    baseline_per_label = baseline_payload.get("per_label_f1", {})
    if not isinstance(baseline_per_label, dict):
        return failures

    current_per_label = report["per_label_f1"]
    for label in critical_labels:
        baseline_f1 = baseline_per_label.get(label)
        if baseline_f1 is None:
            continue
        current_f1 = float(current_per_label.get(label, 0.0))
        if current_f1 < float(baseline_f1) - regression_tolerance:
            failures.append(
                f"{label} F1 regressed from {float(baseline_f1):.4f} to {current_f1:.4f}"
            )

    return failures


def write_baseline(
    baseline_path: Path,
    report: dict[str, Any],
    *,
    min_f1: float,
    regression_tolerance: float,
    critical_labels: list[str],
) -> None:
    payload = {
        "min_f1": round(min_f1, 4),
        "regression_tolerance": round(regression_tolerance, 4),
        "critical_labels": critical_labels,
        "sample_count": report["sample_count"],
        "overall": report["groups"]["combined"]["overall"],
        "per_label_f1": report["per_label_f1"],
        "groups": {
            "issue": report["groups"]["issue"],
            "source": report["groups"]["source"],
            "combined": report["groups"]["combined"],
        },
    }
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    baseline_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def print_report(report: dict[str, Any], failures: list[str]) -> None:
    combined = report["groups"]["combined"]["overall"]
    issue = report["groups"]["issue"]["overall"]
    source = report["groups"]["source"]["overall"]

    print("=== LoLvlance Golden Evaluation ===")
    print(f"Model: {report['model_path']}")
    print(f"Samples: {report['sample_count']}")
    print(
        "Overall Combined F1: "
        f"{combined['f1']:.4f} (precision={combined['precision']:.4f}, recall={combined['recall']:.4f})"
    )
    print(
        "Issue F1: "
        f"{issue['f1']:.4f} (precision={issue['precision']:.4f}, recall={issue['recall']:.4f})"
    )
    print(
        "Source F1: "
        f"{source['f1']:.4f} (precision={source['precision']:.4f}, recall={source['recall']:.4f})"
    )
    print()
    print("Per-label F1:")
    for label, metrics in sorted(report["groups"]["combined"]["per_label"].items()):
        print(
            f"  {label:<18} "
            f"f1={float(metrics['f1']):.4f} "
            f"precision={float(metrics['precision']):.4f} "
            f"recall={float(metrics['recall']):.4f} "
            f"support={int(metrics['support'])}"
        )

    print()
    print("Confusion Summary:")
    false_positives = report["confusion_summary"]["false_positives"]
    false_negatives = report["confusion_summary"]["false_negatives"]
    mismatched_samples = report["confusion_summary"]["mismatched_samples"]

    if false_positives:
        print("  False positives:")
        for label, count in false_positives.items():
            print(f"    {label}: {count}")
    else:
        print("  False positives: none")

    if false_negatives:
        print("  False negatives:")
        for label, count in false_negatives.items():
            print(f"    {label}: {count}")
    else:
        print("  False negatives: none")

    if mismatched_samples:
        print("  Mismatched samples:")
        for sample in mismatched_samples:
            print(
                "    "
                f"{sample['sample_id']} "
                f"issues(expected={sample['expected_issue']}, predicted={sample['predicted_issue']}) "
                f"sources(expected={sample['expected_source']}, predicted={sample['predicted_source']})"
            )
    else:
        print("  Mismatched samples: none")

    if failures:
        print()
        print("Gate Failures:")
        for failure in failures:
            print(f"  - {failure}")


def main() -> None:
    args = parse_args()
    issue_thresholds, source_thresholds = load_thresholds(args.thresholds_path)
    samples = discover_golden_samples(args.goldens_dir)
    model = OnnxAudioModel(args.model_path)
    predictions = evaluate_samples(samples, model, issue_thresholds, source_thresholds)
    report = build_report(predictions, args.model_path, issue_thresholds, source_thresholds)
    baseline_payload = load_baseline(args.baseline_path)
    critical_labels = resolve_critical_labels(args.critical_label, baseline_payload)

    if args.write_baseline:
        write_baseline(
            args.baseline_path,
            report,
            min_f1=args.min_f1,
            regression_tolerance=args.regression_tolerance,
            critical_labels=critical_labels or sorted(report["per_label_f1"].keys()),
        )
        print_report(report, [])
        return

    failures = check_gate(
        report,
        baseline_payload,
        min_f1=args.min_f1,
        regression_tolerance=args.regression_tolerance,
        critical_labels=critical_labels,
    )
    print_report(report, failures)

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
