from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_VERSION = "v0.1-real-data"
DEFAULT_DATASET_ROOT = Path("data/datasets")
DEFAULT_BASELINE_PATH = Path("ml/eval/baseline.json")
DEFAULT_BROWSER_PRODUCTION_MODEL_PATH = Path("public/models/lightweight_audio_model.production.onnx")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train, evaluate, and optionally promote the LoLvlance real-data checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--version", default=DEFAULT_MODEL_VERSION, help="Version tag for the promoted checkpoint.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT, help="Root containing musan/fsd50k/openmic.")
    parser.add_argument("--musan-root", type=Path, default=None, help="Override MUSAN root. Required for real-data training.")
    parser.add_argument("--fsd50k-root", type=Path, default=None, help="Override FSD50K root.")
    parser.add_argument("--openmic-root", type=Path, default=None, help="Override OpenMIC root.")
    parser.add_argument("--download-missing", action="store_true", help="Download missing datasets before training.")
    parser.add_argument("--include-fsd50k", action="store_true", help="Download/use FSD50K when available.")
    parser.add_argument("--include-openmic", action="store_true", help="Download/use OpenMIC when available.")
    parser.add_argument("--keep-archives", action="store_true", help="Keep dataset archives after extraction.")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=12, help="Training batch size.")
    parser.add_argument("--clips-per-file", type=int, default=2, help="Clips sampled per source file.")
    parser.add_argument("--max-files-per-dataset", type=int, default=None, help="Optional cap for faster experimental runs.")
    parser.add_argument("--model-variant", choices=("student", "teacher"), default="student", help="Model size for browser deployment.")
    parser.add_argument(
        "--teacher-checkpoint",
        type=Path,
        default=Path("ml/checkpoints/best_sound_issue_model.pt"),
        help="Optional teacher checkpoint for student distillation.",
    )
    parser.add_argument("--device", default="auto", help="Training device: auto, cpu, cuda, or mps.")
    parser.add_argument("--checkpoint-dir", type=Path, default=None, help="Directory for training outputs.")
    parser.add_argument("--onnx-output", type=Path, default=None, help="Versioned ONNX artifact path.")
    parser.add_argument("--manifest-path", type=Path, default=None, help="Manifest path for the real-data run.")
    parser.add_argument("--report-json-path", type=Path, default=None, help="Machine-readable evaluation report path.")
    parser.add_argument("--summary-path", type=Path, default=None, help="Training/evaluation summary path.")
    parser.add_argument("--baseline-path", type=Path, default=DEFAULT_BASELINE_PATH, help="Regression baseline used by evaluator.")
    parser.add_argument("--update-baseline", action="store_true", help="Rewrite baseline.json after a passing run.")
    parser.add_argument("--promote-browser-model", action="store_true", help="Copy the ONNX artifact into the production browser path.")
    parser.add_argument(
        "--browser-production-model-path",
        type=Path,
        default=DEFAULT_BROWSER_PRODUCTION_MODEL_PATH,
        help="Where to copy the promoted production ONNX model.",
    )
    parser.add_argument(
        "--required-free-gb-for-training",
        type=float,
        default=8.0,
        help="Minimum free disk space required before training starts.",
    )
    parser.add_argument(
        "--required-free-gb-for-download",
        type=float,
        default=60.0,
        help="Minimum free disk space required before downloading public datasets.",
    )
    return parser.parse_args()


def resolve_repo_path(path: Path | None, fallback: Path | None = None) -> Path | None:
    resolved = path if path is not None else fallback
    if resolved is None:
        return None
    return resolved if resolved.is_absolute() else (PROJECT_ROOT / resolved)


def find_existing_parent(path: Path) -> Path:
    current = path
    while not current.exists():
        if current.parent == current:
            return PROJECT_ROOT
        current = current.parent
    return current


def get_free_space_gb(path: Path) -> float:
    anchor = find_existing_parent(path)
    usage = shutil.disk_usage(anchor)
    return usage.free / (1024 ** 3)


def require_free_space(path: Path, minimum_gb: float, *, purpose: str) -> None:
    free_gb = get_free_space_gb(path)
    if free_gb < minimum_gb:
        raise RuntimeError(
            f"Not enough free disk space to {purpose}: {free_gb:.2f} GB available, "
            f"{minimum_gb:.2f} GB required."
        )


def run_command(command: list[str]) -> None:
    command_display = subprocess.list2cmdline(command)
    print(f"\n[run] {command_display}")
    env = dict(os.environ)
    env.update({"PYTHONIOENCODING": "utf-8"})
    subprocess.run(command, cwd=PROJECT_ROOT, check=True, env=env, text=True)


def resolve_dataset_roots(args: argparse.Namespace) -> dict[str, Path | None]:
    dataset_root = resolve_repo_path(args.dataset_root)
    roots = {
        "musan": resolve_repo_path(args.musan_root, dataset_root / "musan" if dataset_root is not None else None),
        "fsd50k": resolve_repo_path(args.fsd50k_root, dataset_root / "fsd50k" if dataset_root is not None else None),
        "openmic": resolve_repo_path(args.openmic_root, dataset_root / "openmic" if dataset_root is not None else None),
    }
    return {name: path if path is not None and path.exists() else None for name, path in roots.items()}


def download_missing_datasets(args: argparse.Namespace, dataset_roots: dict[str, Path | None]) -> dict[str, Path | None]:
    dataset_root = resolve_repo_path(args.dataset_root)
    assert dataset_root is not None

    missing: list[str] = []
    if dataset_roots["musan"] is None:
        missing.append("musan")
    if args.include_fsd50k and dataset_roots["fsd50k"] is None:
        missing.append("fsd50k")
    if args.include_openmic and dataset_roots["openmic"] is None:
        missing.append("openmic")

    if not missing:
        return dataset_roots

    require_free_space(dataset_root, float(args.required_free_gb_for_download), purpose="download real datasets")

    command = [
        sys.executable,
        "ml/download_datasets.py",
        "--datasets",
        *missing,
        "--output-root",
        str(dataset_root),
    ]
    if args.keep_archives:
        command.append("--keep-archives")

    run_command(command)
    return resolve_dataset_roots(args)


def ensure_required_datasets(dataset_roots: dict[str, Path | None], *, include_fsd50k: bool, include_openmic: bool) -> None:
    if dataset_roots["musan"] is None:
        raise FileNotFoundError(
            "MUSAN is required for v0.1-real-data. Provide --musan-root or use --download-missing once enough disk is available."
        )
    if include_fsd50k and dataset_roots["fsd50k"] is None:
        raise FileNotFoundError("FSD50K was requested but is not available.")
    if include_openmic and dataset_roots["openmic"] is None:
        raise FileNotFoundError("OpenMIC was requested but is not available.")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_summary(
    *,
    version: str,
    dataset_roots: dict[str, Path | None],
    onnx_output: Path,
    checkpoint_dir: Path,
    report_json_path: Path,
    summary_path: Path,
    promoted_browser_model_path: Path | None,
) -> None:
    training_history = load_json(checkpoint_dir / "training_history.json")
    evaluation_report = load_json(report_json_path)

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_version": version,
        "datasets": {name: path.as_posix() for name, path in dataset_roots.items() if path is not None},
        "artifacts": {
            "onnx": onnx_output.as_posix(),
            "checkpoint_dir": checkpoint_dir.as_posix(),
            "report_json": report_json_path.as_posix(),
            "browser_production_model": None if promoted_browser_model_path is None else promoted_browser_model_path.as_posix(),
        },
        "training": {
            "best_epoch": training_history.get("best_epoch"),
            "selection_score": training_history.get("selection_score"),
            "manifest": training_history.get("manifest"),
            "tuned_metrics": training_history.get("tuned_metrics"),
        },
        "evaluation": {
            "status": evaluation_report.get("gate", {}).get("status"),
            "warning": evaluation_report.get("note"),
            "combined_macro": evaluation_report.get("groups", {}).get("combined", {}).get("macro"),
            "combined_overall": evaluation_report.get("groups", {}).get("combined", {}).get("overall"),
            "per_label": evaluation_report.get("groups", {}).get("combined", {}).get("per_label"),
            "prediction_distribution": evaluation_report.get("prediction_distribution"),
            "gate_failures": evaluation_report.get("gate", {}).get("failures"),
        },
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def copy_if_exists(source: Path, destination: Path) -> None:
    if not source.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def main() -> int:
    args = parse_args()
    version = args.version.strip() or DEFAULT_MODEL_VERSION

    checkpoint_dir = resolve_repo_path(args.checkpoint_dir, Path("ml/checkpoints") / version)
    onnx_output = resolve_repo_path(args.onnx_output, Path("ml/checkpoints") / f"{version}.onnx")
    manifest_path = resolve_repo_path(args.manifest_path, Path("ml/artifacts") / f"{version}.manifest.jsonl")
    report_json_path = resolve_repo_path(args.report_json_path, Path("ml/eval") / f"{version}.report.json")
    summary_path = resolve_repo_path(args.summary_path, Path("ml/checkpoints") / f"{version}.summary.json")
    baseline_path = resolve_repo_path(args.baseline_path)
    browser_production_model_path = resolve_repo_path(args.browser_production_model_path)

    assert checkpoint_dir is not None
    assert onnx_output is not None
    assert manifest_path is not None
    assert report_json_path is not None
    assert summary_path is not None
    assert baseline_path is not None

    dataset_roots = resolve_dataset_roots(args)
    if args.download_missing:
        dataset_roots = download_missing_datasets(args, dataset_roots)
    ensure_required_datasets(
        dataset_roots,
        include_fsd50k=bool(args.include_fsd50k),
        include_openmic=bool(args.include_openmic),
    )
    require_free_space(checkpoint_dir, float(args.required_free_gb_for_training), purpose="train the real-data checkpoint")

    train_command = [
        sys.executable,
        "-m",
        "ml.train",
        "--musan-root",
        str(dataset_roots["musan"]),
        "--manifest-path",
        str(manifest_path),
        "--checkpoint-dir",
        str(checkpoint_dir),
        "--onnx-output",
        str(onnx_output),
        "--rebuild-manifest",
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--clips-per-file",
        str(args.clips_per_file),
        "--model-variant",
        args.model_variant,
        "--device",
        args.device,
        "--export-onnx",
    ]
    if args.max_files_per_dataset is not None:
        train_command.extend(["--max-files-per-dataset", str(args.max_files_per_dataset)])
    if dataset_roots["fsd50k"] is not None:
        train_command.extend(["--fsd50k-root", str(dataset_roots["fsd50k"])])
    if dataset_roots["openmic"] is not None:
        train_command.extend(["--openmic-root", str(dataset_roots["openmic"])])

    teacher_checkpoint = resolve_repo_path(args.teacher_checkpoint)
    if args.model_variant == "student" and teacher_checkpoint is not None and teacher_checkpoint.exists():
        train_command.extend(["--teacher-checkpoint", str(teacher_checkpoint)])

    run_command(train_command)

    thresholds_path = checkpoint_dir / "label_thresholds.json"
    evaluate_command = [
        sys.executable,
        "ml/eval/evaluate.py",
        "--goldens-dir",
        "eval/goldens",
        "--model-path",
        str(onnx_output),
        "--thresholds-path",
        str(thresholds_path),
        "--baseline-path",
        str(baseline_path),
        "--report-json-path",
        str(report_json_path),
    ]
    run_command(evaluate_command)

    if args.update_baseline:
        run_command(
            [
                sys.executable,
                "ml/eval/evaluate.py",
                "--goldens-dir",
                "eval/goldens",
                "--model-path",
                str(onnx_output),
                "--thresholds-path",
                str(thresholds_path),
                "--baseline-path",
                str(baseline_path),
                "--write-baseline",
            ]
        )

    promoted_browser_model = None
    if args.promote_browser_model:
        assert browser_production_model_path is not None
        copy_if_exists(onnx_output, browser_production_model_path)
        copy_if_exists(onnx_output.with_suffix(".metadata.json"), browser_production_model_path.with_suffix(".metadata.json"))
        promoted_browser_model = browser_production_model_path

    write_summary(
        version=version,
        dataset_roots=dataset_roots,
        onnx_output=onnx_output,
        checkpoint_dir=checkpoint_dir,
        report_json_path=report_json_path,
        summary_path=summary_path,
        promoted_browser_model_path=promoted_browser_model,
    )

    print(f"\nReal-data training completed for {version}.")
    print(f"  ONNX: {onnx_output}")
    print(f"  Evaluation report: {report_json_path}")
    print(f"  Summary: {summary_path}")
    if promoted_browser_model is not None:
        print(f"  Browser production model: {promoted_browser_model}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
