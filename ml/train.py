from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    from .dataset import (
        DatasetRoots,
        LoLvlanceAudioDataset,
        build_public_manifest,
        load_manifest,
        summarize_manifest,
    )
    from .model import LightweightAudioAnalysisNet, ModelConfig
    from .preprocessing import PreprocessingConfig
except ImportError:
    from dataset import (
        DatasetRoots,
        LoLvlanceAudioDataset,
        build_public_manifest,
        load_manifest,
        summarize_manifest,
    )
    from model import LightweightAudioAnalysisNet, ModelConfig
    from preprocessing import PreprocessingConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a lightweight LoLvlance multi-label sound issue classifier."
    )
    parser.add_argument("--openmic-root", type=Path, default=None, help="OpenMIC-2018 dataset root.")
    parser.add_argument("--slakh-root", type=Path, default=None, help="Slakh2100 dataset root.")
    parser.add_argument("--musan-root", type=Path, default=None, help="MUSAN dataset root.")
    parser.add_argument("--fsd50k-root", type=Path, default=None, help="FSD50K dataset root.")
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=Path("ml/artifacts/public_dataset_manifest.jsonl"),
        help="Where to write/read the derived LoLvlance manifest.",
    )
    parser.add_argument(
        "--rebuild-manifest",
        action="store_true",
        help="Re-scan the public datasets even if a manifest already exists.",
    )
    parser.add_argument("--clips-per-file", type=int, default=2, help="How many clips to sample per source file.")
    parser.add_argument(
        "--max-files-per-dataset",
        type=int,
        default=None,
        help="Optional cap to keep the first run fast while bootstrapping.",
    )
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=16, help="Training batch size.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay.")
    parser.add_argument("--dropout", type=float, default=0.15, help="Classifier dropout.")
    parser.add_argument("--num-workers", type=int, default=0, help="Dataloader workers.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("ml/checkpoints"),
        help="Directory for last/best checkpoints and training metrics.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Training device: auto, cpu, cuda, or mps.",
    )
    parser.add_argument(
        "--export-onnx",
        action="store_true",
        help="Export the best checkpoint to ONNX when training finishes.",
    )
    parser.add_argument(
        "--onnx-output",
        type=Path,
        default=Path("ml/checkpoints/lightweight_audio_model.onnx"),
        help="Output path used with --export-onnx.",
    )
    return parser.parse_args()


def run_training(args: argparse.Namespace) -> tuple[Path, dict[str, object]]:
    seed_everything(args.seed)
    device = resolve_device(args.device)
    preprocessing_config = PreprocessingConfig()

    manifest_path = prepare_manifest(args, preprocessing_config)
    manifest_entries = load_manifest(manifest_path)
    manifest_summary = summarize_manifest(manifest_entries)

    train_dataset = LoLvlanceAudioDataset(
        manifest_path=manifest_path,
        split="train",
        preprocessing_config=preprocessing_config,
    )
    val_dataset = LoLvlanceAudioDataset(
        manifest_path=manifest_path,
        split="val",
        preprocessing_config=preprocessing_config,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model_config = ModelConfig(dropout=args.dropout)
    model = LightweightAudioAnalysisNet(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    pos_weight = compute_pos_weight(train_dataset).to(device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, float]] = []
    best_metric = float("-inf")
    best_checkpoint_path = args.checkpoint_dir / "best_sound_issue_model.pt"

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model=model,
            data_loader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
        )
        val_metrics = run_epoch(
            model=model,
            data_loader=val_loader,
            criterion=criterion,
            device=device,
            optimizer=None,
        )

        epoch_record = {
            "epoch": float(epoch),
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_micro_f1": train_metrics["micro_f1"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_micro_f1": val_metrics["micro_f1"],
        }
        history.append(epoch_record)

        last_checkpoint_path = args.checkpoint_dir / "last_sound_issue_model.pt"
        save_checkpoint(
            checkpoint_path=last_checkpoint_path,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            model_config=model_config,
            metrics=epoch_record,
            manifest_summary=manifest_summary,
        )

        if val_metrics["micro_f1"] >= best_metric:
            best_metric = val_metrics["micro_f1"]
            save_checkpoint(
                checkpoint_path=best_checkpoint_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                model_config=model_config,
                metrics=epoch_record,
                manifest_summary=manifest_summary,
            )

        print(
            f"epoch={epoch} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_f1={train_metrics['micro_f1']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_f1={val_metrics['micro_f1']:.4f}"
        )

    history_path = args.checkpoint_dir / "training_history.json"
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

    if args.export_onnx:
        try:
            from .export_to_onnx import export_to_onnx
        except ImportError:
            from export_to_onnx import export_to_onnx

        sample_time_steps = int(train_dataset[0]["log_mel_spectrogram"].shape[0])
        export_args = argparse.Namespace(
            checkpoint=best_checkpoint_path,
            output=args.onnx_output,
            time_steps=sample_time_steps,
            opset=18,
            verify=True,
            mel_bins=model_config.mel_bins,
        )
        export_to_onnx(export_args)

    summary = {
        "manifest": manifest_summary,
        "history": history,
        "best_checkpoint": best_checkpoint_path.as_posix(),
        "device": device.type,
    }
    return best_checkpoint_path, summary


def prepare_manifest(args: argparse.Namespace, preprocessing_config: PreprocessingConfig) -> Path:
    dataset_roots = DatasetRoots(
        openmic=args.openmic_root,
        slakh=args.slakh_root,
        musan=args.musan_root,
        fsd50k=args.fsd50k_root,
    )

    if args.rebuild_manifest or not args.manifest_path.exists():
        build_public_manifest(
            dataset_roots=dataset_roots,
            output_path=args.manifest_path,
            preprocessing_config=preprocessing_config,
            clips_per_file=args.clips_per_file,
            max_files_per_dataset=args.max_files_per_dataset,
        )

    if not args.manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found at {args.manifest_path}. Provide dataset roots or enable --rebuild-manifest."
        )

    return args.manifest_path


def run_epoch(
    model: LightweightAudioAnalysisNet,
    data_loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, float]:
    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    total_examples = 0
    total_accuracy = 0.0
    true_positives = 0.0
    false_positives = 0.0
    false_negatives = 0.0

    for batch in data_loader:
        inputs = batch["log_mel_spectrogram"].to(device)
        labels = batch["labels"].to(device)

        if is_training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_training):
            outputs = model(inputs)
            logits = outputs["problem_logits"]
            probabilities = outputs["problem_probs"]
            loss = criterion(logits, labels)

            if is_training:
                loss.backward()
                optimizer.step()

        predictions = (probabilities >= 0.5).float()
        total_loss += float(loss.item()) * labels.size(0)
        total_examples += labels.size(0)
        total_accuracy += float((predictions == labels).float().mean(dim=1).sum().item())
        true_positives += float((predictions * labels).sum().item())
        false_positives += float((predictions * (1.0 - labels)).sum().item())
        false_negatives += float(((1.0 - predictions) * labels).sum().item())

    precision = true_positives / max(1.0, true_positives + false_positives)
    recall = true_positives / max(1.0, true_positives + false_negatives)
    micro_f1 = 0.0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall)

    return {
        "loss": total_loss / max(1, total_examples),
        "accuracy": total_accuracy / max(1, total_examples),
        "micro_f1": micro_f1,
    }


def compute_pos_weight(dataset: LoLvlanceAudioDataset) -> torch.Tensor:
    labels = torch.tensor([entry["label_vector"] for entry in dataset.entries], dtype=torch.float32)
    positive_counts = labels.sum(dim=0)
    negative_counts = labels.size(0) - positive_counts
    return negative_counts / positive_counts.clamp(min=1.0)


def save_checkpoint(
    checkpoint_path: Path,
    model: LightweightAudioAnalysisNet,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    model_config: ModelConfig,
    metrics: dict[str, float],
    manifest_summary: dict[str, object],
) -> None:
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "config": model_config.to_dict(),
        "metrics": metrics,
        "manifest_summary": manifest_summary,
    }
    torch.save(checkpoint, checkpoint_path)


def resolve_device(device_name: str) -> torch.device:
    normalized = device_name.lower()

    if normalized == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    return torch.device(normalized)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main() -> None:
    args = parse_args()
    best_checkpoint_path, summary = run_training(args)
    print(json.dumps(summary, indent=2))
    print(f"Best checkpoint saved to: {best_checkpoint_path}")


if __name__ == "__main__":
    main()
