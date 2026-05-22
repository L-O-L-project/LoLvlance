from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.eval.evaluate import dataset_load_report_to_dict, load_golden_dataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate LoLvlance golden labels without running model inference.")
    parser.add_argument("--goldens-dir", type=Path, default=Path("eval/goldens"))
    parser.add_argument("--labels-path", type=Path, default=Path("eval/goldens/labels.json"))
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    samples, report = load_golden_dataset(args.goldens_dir, args.labels_path)
    payload = dataset_load_report_to_dict(report)

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print("LoLvlance golden labels validation")
        print(f"status: ok")
        print(f"source: {payload['source']}")
        print(f"labels_path: {payload['labels_path']}")
        print(f"loaded_sample_count: {len(samples)}")
        print(f"skipped_samples: {len(payload['skipped_samples'])}")
        print(f"issue_label_distribution: {payload['issue_label_distribution']}")
        print(f"source_label_distribution: {payload['source_label_distribution']}")
        print(f"label_quality_distribution: {payload['label_quality_distribution']}")
        print(f"audio_quality_flags_summary: {payload['audio_quality_flags_summary']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
