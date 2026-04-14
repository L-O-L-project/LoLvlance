"""
ingest_feedback.py

Converts user-exported feedback JSONL (from the browser's FeedbackWidget)
into training manifest entries that can be merged with the existing manifest
and used in the next training run.

Usage:
    python ml/ingest_feedback.py \\
        --feedback lolvlance_feedback_2024-01-15.jsonl \\
        --output   ml/artifacts/feedback_manifest.jsonl

    # Merge with existing manifest and retrain
    cat ml/artifacts/public_dataset_manifest.jsonl \\
        ml/artifacts/feedback_manifest.jsonl \\
        > ml/artifacts/merged_manifest.jsonl

    python -m ml.train \\
        --manifest-path ml/artifacts/merged_manifest.jsonl \\
        --epochs 10 --export-onnx

Schema of a feedback entry (v1.0, produced by feedbackStore.ts):
    feedback_schema_version: "1.0"
    entry_id:        str
    session_id:      str
    timestamp_ms:    int
    verdict:         "correct" | "wrong"
    corrected_labels: list[str]   # user-chosen actual issues ("none" = no issue)
    analysis:
        engine:    str
        problems:  [{type, confidence, sources}]
        ml_issues: {label: float}
        ml_sources:{label: float}
    audio_features:
        rms:               float | null
        spectrogram_shape: [int, int] | null

What this script produces:
    A JSONL manifest compatible with LoLvlanceAudioDataset / RealAudioDegradationDataset.
    Because we don't have raw audio, entries carry audio_path=null and use only
    the label information — the dataset loader will skip null-path entries gracefully.
    The label quality is set to "reviewed" for corrected entries and "weak" for
    entries where the model was confirmed correct.
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path
from typing import Any

try:
    from .label_schema import ISSUE_LABELS, SCHEMA_VERSION, SOURCE_LABELS
except ImportError:
    from label_schema import ISSUE_LABELS, SCHEMA_VERSION, SOURCE_LABELS

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FEEDBACK_SCHEMA_VERSION = "1.0"

# Minimum ML confidence to trust a model prediction as a soft label
_ML_CONFIDENCE_THRESHOLD = 0.35

# How much to down-weight "weak" (model-confirmed) labels vs "reviewed" ones
_WEAK_MASK_VALUE = 0.6
_REVIEWED_MASK_VALUE = 1.0


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------

def _all_zeros(labels: tuple[str, ...]) -> list[float]:
    return [0.0] * len(labels)


def _build_issue_targets(
    corrected_labels: list[str],
    ml_issues: dict[str, float],
    verdict: str,
) -> dict[str, Any]:
    """
    Build issue_targets from a feedback entry.

    - verdict='correct' → use ml_issues as soft labels with weak mask
    - verdict='wrong' + corrected_labels provided → hard labels from user, reviewed mask
    - verdict='wrong' + corrected_labels=['none'] → all zeros, reviewed mask
    - verdict='wrong' + no corrected_labels → skip issue head (mask=0)
    """
    labels = list(ISSUE_LABELS)

    if verdict == "correct":
        values = [
            round(float(ml_issues.get(label, 0.0)), 4)
            for label in labels
        ]
        mask = [_WEAK_MASK_VALUE if v > _ML_CONFIDENCE_THRESHOLD else 0.0 for v in values]
        quality = ["weak"] * len(labels)
        return {"labels": labels, "values": values, "mask": mask, "quality": quality}

    # verdict == "wrong"
    user_labels = set(corrected_labels)

    if "none" in user_labels or not user_labels:
        # User said there's no issue, or didn't specify → zero out all, mask=1
        values = _all_zeros(labels)
        mask = [_REVIEWED_MASK_VALUE] * len(labels) if "none" in user_labels else [0.0] * len(labels)
        quality = ["reviewed" if "none" in user_labels else "unavailable"] * len(labels)
        return {"labels": labels, "values": values, "mask": mask, "quality": quality}

    # User selected specific wrong issues
    values = [1.0 if label in user_labels else 0.0 for label in labels]
    mask = [_REVIEWED_MASK_VALUE] * len(labels)
    quality = ["reviewed"] * len(labels)
    return {"labels": labels, "values": values, "mask": mask, "quality": quality}


def _build_source_targets(
    ml_sources: dict[str, float],
    verdict: str,
) -> dict[str, Any]:
    """
    Build source_targets from feedback.
    We only have ML-predicted source confidences (no user correction for sources yet).
    For correct verdicts we carry them as weak labels; for wrong we mask them out.
    """
    labels = list(SOURCE_LABELS)

    if verdict == "correct":
        values = [round(float(ml_sources.get(label, 0.0)), 4) for label in labels]
        mask = [_WEAK_MASK_VALUE if v > _ML_CONFIDENCE_THRESHOLD else 0.0 for v in values]
        quality = ["weak"] * len(labels)
    else:
        values = _all_zeros(labels)
        mask = [0.0] * len(labels)
        quality = ["unavailable"] * len(labels)

    return {"labels": labels, "values": values, "mask": mask, "quality": quality}


def feedback_entry_to_manifest(entry: dict[str, Any]) -> dict[str, Any] | None:
    """
    Convert one feedback entry to a manifest dict.
    Returns None if the entry should be skipped (e.g. no useful signal).
    """
    if entry.get("feedback_schema_version") != FEEDBACK_SCHEMA_VERSION:
        return None

    verdict: str = entry.get("verdict", "")
    if verdict not in ("correct", "wrong"):
        return None

    analysis = entry.get("analysis", {})
    ml_issues: dict[str, float] = analysis.get("ml_issues", {})
    ml_sources: dict[str, float] = analysis.get("ml_sources", {})
    corrected_labels: list[str] = entry.get("corrected_labels", [])

    issue_targets = _build_issue_targets(corrected_labels, ml_issues, verdict)
    source_targets = _build_source_targets(ml_sources, verdict)

    # Skip entries that carry zero signal in both heads
    has_issue_signal = any(m > 0 for m in issue_targets["mask"])
    has_source_signal = any(m > 0 for m in source_targets["mask"])
    if not has_issue_signal and not has_source_signal:
        return None

    audio_features = entry.get("audio_features", {})

    return {
        "schema_version": SCHEMA_VERSION,
        "clip_id": f"feedback:{entry.get('entry_id', uuid.uuid4().hex)}",
        "track_group_id": f"feedback:{entry.get('session_id', 'unknown')}",
        "dataset": "feedback",
        "audio_path": None,             # no raw audio available
        "start_seconds": 0.0,
        "duration_seconds": 3.0,
        "split": "train",               # feedback always goes to train
        "issue_targets": issue_targets,
        "source_targets": source_targets,
        "metadata": {
            "verdict": verdict,
            "corrected_labels": corrected_labels,
            "original_engine": analysis.get("engine", "unknown"),
            "rms": audio_features.get("rms"),
            "spectrogram_shape": audio_features.get("spectrogram_shape"),
            "timestamp_ms": entry.get("timestamp_ms"),
            "session_id": entry.get("session_id"),
        },
    }


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def _deduplicate(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep only the last feedback entry per session × original analysis."""
    seen: dict[str, dict[str, Any]] = {}
    for entry in entries:
        key = entry["clip_id"]
        seen[key] = entry  # last one wins
    return list(seen.values())


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def _print_stats(converted: list[dict[str, Any]], skipped: int) -> None:
    total = len(converted) + skipped
    print(f"\n  Total feedback entries : {total}")
    print(f"  Skipped (no signal)    : {skipped}")
    print(f"  Converted              : {len(converted)}")

    issue_counts: dict[str, int] = {}
    for entry in converted:
        for label, value, mask in zip(
            entry["issue_targets"]["labels"],
            entry["issue_targets"]["values"],
            entry["issue_targets"]["mask"],
        ):
            if value > 0.5 and mask > 0:
                issue_counts[label] = issue_counts.get(label, 0) + 1

    if issue_counts:
        print("\n  Issue label distribution:")
        for label, count in sorted(issue_counts.items(), key=lambda x: -x[1]):
            print(f"    {label:16s} {count}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert browser feedback JSONL into a training manifest.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--feedback",
        type=Path,
        required=True,
        help="Path to the JSONL file exported from the browser (lolvlance_feedback_*.jsonl).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("ml/artifacts/feedback_manifest.jsonl"),
        help="Output manifest path (default: ml/artifacts/feedback_manifest.jsonl).",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to an existing output manifest instead of overwriting.",
    )
    parser.add_argument(
        "--min-entries",
        type=int,
        default=10,
        help="Warn if fewer than this many entries are produced (default: 10).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    feedback_path: Path = args.feedback
    output_path: Path = args.output

    if not feedback_path.exists():
        print(f"ERROR: feedback file not found: {feedback_path}", file=sys.stderr)
        sys.exit(1)

    raw_entries: list[dict[str, Any]] = []
    with open(feedback_path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                raw_entries.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"  WARNING: skipping malformed line {line_num}: {exc}", file=sys.stderr)

    print(f"Loaded {len(raw_entries)} raw feedback entries from {feedback_path}")

    converted: list[dict[str, Any]] = []
    skipped = 0
    for entry in raw_entries:
        result = feedback_entry_to_manifest(entry)
        if result is None:
            skipped += 1
        else:
            converted.append(result)

    converted = _deduplicate(converted)
    _print_stats(converted, skipped)

    if len(converted) < args.min_entries:
        print(
            f"\n  WARNING: only {len(converted)} entries produced. "
            f"Consider collecting more feedback before retraining."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.append else "w"
    with open(output_path, mode, encoding="utf-8") as f:
        for entry in converted:
            f.write(json.dumps(entry, ensure_ascii=True) + "\n")

    print(f"\n  Manifest written to: {output_path.resolve()}")
    print("\nNext steps:")
    print(f"  cat ml/artifacts/public_dataset_manifest.jsonl {output_path} > ml/artifacts/merged_manifest.jsonl")
    print("  python -m ml.train --manifest-path ml/artifacts/merged_manifest.jsonl --epochs 10 --export-onnx")


if __name__ == "__main__":
    main()
