from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    from .dataset import (
        DatasetRoots,
        LoLvlanceAudioDataset,
        build_public_manifest,
        load_manifest,
        summarize_manifest,
    )
    from .label_schema import (
        DEFAULT_ISSUE_THRESHOLDS,
        DEFAULT_SOURCE_THRESHOLDS,
        ISSUE_LABELS,
        SCHEMA_VERSION,
        SOURCE_LABELS,
    )
    from .metrics import evaluate_multilabel_head, tune_thresholds
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
    from label_schema import (
        DEFAULT_ISSUE_THRESHOLDS,
        DEFAULT_SOURCE_THRESHOLDS,
        ISSUE_LABELS,
        SCHEMA_VERSION,
        SOURCE_LABELS,
    )
    from metrics import evaluate_multilabel_head, tune_thresholds
    from model import LightweightAudioAnalysisNet, ModelConfig
    from preprocessing import PreprocessingConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the hierarchical LoLvlance audio diagnosis model."
    )
    parser.add_argument("--openmic-root", type=Path, default=None, help="OpenMIC-2018 dataset root.")
    parser.add_argument("--slakh-root", type=Path, default=None, help="Slakh2100 dataset root.")
    parser.add_argument("--musan-root", type=Path, default=None, help="MUSAN dataset root.")
    parser.add_argument("--fsd50k-root", type=Path, default=None, help="FSD50K dataset root.")
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=Path("ml/artifacts/public_dataset_manifest.jsonl"),
        help="Where to store the derived hierarchical label manifest.",
    )
    parser.add_argument(
        "--rebuild-manifest",
        action="store_true",
        help="Re-scan public datasets even if the manifest already exists.",
    )
    parser.add_argument("--clips-per-file", type=int, default=2, help="Number of clips to sample per file.")
    parser.add_argument(
        "--max-files-per-dataset",
        type=int,
        default=None,
        help="Optional cap to keep early experiments fast.",
    )
    parser.add_argument("--epochs", type=int, default=6, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=16, help="Training batch size.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="AdamW learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay.")
    parser.add_argument("--dropout", type=float, default=0.15, help="Head dropout.")
    parser.add_argument("--issue-loss-weight", type=float, default=1.0, help="Weight for issue-head loss.")
    parser.add_argument("--source-loss-weight", type=float, default=0.6, help="Weight for source-head loss.")
    parser.add_argument("--num-workers", type=int, default=0, help="Dataloader workers.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("ml/checkpoints"),
        help="Directory for checkpoints and metric artifacts.",
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
        help="Export the best checkpoint to ONNX after training.",
    )
    parser.add_argument(
        "--onnx-output",
        type=Path,
        default=Path("ml/checkpoints/lightweight_audio_model.onnx"),
        help="ONNX output path used with --export-onnx.",
    )
    return parser.parse_args()


def run_training(args: argparse.Namespace) -> tuple[Path, dict[str, object]]:
    seed_everything(args.seed)
    device = resolve_device(args.device)
    preprocessing_config = PreprocessingConfig()
    manifest_path = prepare_manifest(args, preprocessing_config)
    manifest_summary = summarize_manifest(load_manifest(manifest_path))

    train_dataset = LoLvlanceAudioDataset(manifest_path=manifest_path, split="train", preprocessing_config=preprocessing_config)
    val_dataset = LoLvlanceAudioDataset(manifest_path=manifest_path, split="val", preprocessing_config=preprocessing_config)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model_config = ModelConfig(dropout=args.dropout)
    model = LightweightAudioAnalysisNet(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    issue_pos_weight = compute_pos_weight(train_dataset.entries, "issue_targets", ISSUE_LABELS).to(device)
    source_pos_weight = compute_pos_weight(train_dataset.entries, "source_targets", SOURCE_LABELS).to(device)
    has_source_supervision = float(source_pos_weight.numel()) > 0 and float(
        sum(sum(entry["source_targets"]["mask"]) for entry in train_dataset.entries)
    ) > 0

    default_issue_thresholds = dict(DEFAULT_ISSUE_THRESHOLDS)
    default_source_thresholds = dict(DEFAULT_SOURCE_THRESHOLDS)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    history: list[dict[str, object]] = []
    best_score = float("-inf")
    best_epoch = 0
    best_state_dict = copy.deepcopy(model.state_dict())
    best_checkpoint_path = args.checkpoint_dir / "best_sound_issue_model.pt"

    for epoch in range(1, args.epochs + 1):
        train_epoch = run_epoch(
            model=model,
            data_loader=train_loader,
            device=device,
            optimizer=optimizer,
            issue_pos_weight=issue_pos_weight,
            source_pos_weight=source_pos_weight,
            issue_loss_weight=args.issue_loss_weight,
            source_loss_weight=args.source_loss_weight if has_source_supervision else 0.0,
            collect_outputs=False,
        )
        val_epoch = run_epoch(
            model=model,
            data_loader=val_loader,
            device=device,
            optimizer=None,
            issue_pos_weight=issue_pos_weight,
            source_pos_weight=source_pos_weight,
            issue_loss_weight=args.issue_loss_weight,
            source_loss_weight=args.source_loss_weight if has_source_supervision else 0.0,
            collect_outputs=True,
        )

        val_metrics = build_validation_metrics(
            epoch_outputs=val_epoch["outputs"],
            issue_thresholds=default_issue_thresholds,
            source_thresholds=default_source_thresholds,
        )
        composite_score = float(val_metrics["issue_head"]["macro_f1"]) + (
            float(val_metrics["source_head"]["macro_f1"]) * 0.35 if has_source_supervision else 0.0
        )
        epoch_record = {
            "epoch": epoch,
            "train": {
                "total_loss": round(float(train_epoch["total_loss"]), 4),
                "issue_loss": round(float(train_epoch["issue_loss"]), 4),
                "source_loss": round(float(train_epoch["source_loss"]), 4),
            },
            "val": {
                "total_loss": round(float(val_epoch["total_loss"]), 4),
                "issue_loss": round(float(val_epoch["issue_loss"]), 4),
                "source_loss": round(float(val_epoch["source_loss"]), 4),
                "metrics": val_metrics,
            },
            "selection_score": round(composite_score, 4),
        }
        history.append(epoch_record)

        save_checkpoint(
            checkpoint_path=args.checkpoint_dir / "last_sound_issue_model.pt",
            model_state_dict=model.state_dict(),
            optimizer_state_dict=optimizer.state_dict(),
            epoch=epoch,
            model_config=model_config,
            manifest_summary=manifest_summary,
            metrics=epoch_record,
            thresholds={
                "issue_thresholds": default_issue_thresholds,
                "source_thresholds": default_source_thresholds,
            },
        )

        if composite_score >= best_score:
            best_score = composite_score
            best_epoch = epoch
            best_state_dict = copy.deepcopy(model.state_dict())

        print(
            f"epoch={epoch} "
            f"train_loss={train_epoch['total_loss']:.4f} "
            f"val_loss={val_epoch['total_loss']:.4f} "
            f"issue_macro_f1={val_metrics['issue_head']['macro_f1']:.4f} "
            f"source_macro_f1={val_metrics['source_head']['macro_f1']:.4f}"
        )

    model.load_state_dict(best_state_dict)
    best_val_epoch = run_epoch(
        model=model,
        data_loader=val_loader,
        device=device,
        optimizer=None,
        issue_pos_weight=issue_pos_weight,
        source_pos_weight=source_pos_weight,
        issue_loss_weight=args.issue_loss_weight,
        source_loss_weight=args.source_loss_weight if has_source_supervision else 0.0,
        collect_outputs=True,
    )
    tuned_issue_thresholds = tune_thresholds(
        probabilities=best_val_epoch["outputs"]["issue_probs"],
        targets=best_val_epoch["outputs"]["issue_targets"],
        masks=best_val_epoch["outputs"]["issue_mask"],
        labels=ISSUE_LABELS,
        defaults=default_issue_thresholds,
    )
    tuned_source_thresholds = tune_thresholds(
        probabilities=best_val_epoch["outputs"]["source_probs"],
        targets=best_val_epoch["outputs"]["source_targets"],
        masks=best_val_epoch["outputs"]["source_mask"],
        labels=SOURCE_LABELS,
        defaults=default_source_thresholds,
    )
    tuned_metrics = build_validation_metrics(
        epoch_outputs=best_val_epoch["outputs"],
        issue_thresholds=tuned_issue_thresholds,
        source_thresholds=tuned_source_thresholds,
    )
    threshold_path = args.checkpoint_dir / "label_thresholds.json"
    threshold_payload = {
        "schema_version": SCHEMA_VERSION,
        "issue_thresholds": tuned_issue_thresholds,
        "source_thresholds": tuned_source_thresholds,
        "best_epoch": best_epoch,
    }
    threshold_path.write_text(json.dumps(threshold_payload, indent=2), encoding="utf-8")

    final_summary = {
        "schema_version": SCHEMA_VERSION,
        "manifest": manifest_summary,
        "best_epoch": best_epoch,
        "selection_score": round(best_score, 4),
        "tuned_metrics": tuned_metrics,
        "thresholds": threshold_payload,
        "device": device.type,
        "history": history,
    }
    history_path = args.checkpoint_dir / "training_history.json"
    history_path.write_text(json.dumps(final_summary, indent=2), encoding="utf-8")
    (args.checkpoint_dir / "config.json").write_text(
        json.dumps(model_config.to_dict(), indent=2),
        encoding="utf-8",
    )
    (args.checkpoint_dir / "thresholds.json").write_text(
        json.dumps(threshold_payload, indent=2),
        encoding="utf-8",
    )
    torch.save(best_state_dict, args.checkpoint_dir / "model.pt")

    save_checkpoint(
        checkpoint_path=best_checkpoint_path,
        model_state_dict=best_state_dict,
        optimizer_state_dict=optimizer.state_dict(),
        epoch=best_epoch,
        model_config=model_config,
        manifest_summary=manifest_summary,
        metrics=final_summary,
        thresholds=threshold_payload,
    )

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

    return best_checkpoint_path, final_summary


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
    *,
    model: LightweightAudioAnalysisNet,
    data_loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    issue_pos_weight: torch.Tensor,
    source_pos_weight: torch.Tensor,
    issue_loss_weight: float,
    source_loss_weight: float,
    collect_outputs: bool,
) -> dict[str, object]:
    is_training = optimizer is not None
    model.train(is_training)
    total_loss = 0.0
    total_issue_loss = 0.0
    total_source_loss = 0.0
    total_batches = 0

    collected_outputs = {
        "issue_probs": [],
        "issue_targets": [],
        "issue_mask": [],
        "source_probs": [],
        "source_targets": [],
        "source_mask": [],
    }

    for batch in data_loader:
        inputs = batch["log_mel_spectrogram"].to(device)
        issue_targets = batch["issue_targets"].to(device)
        issue_mask = batch["issue_target_mask"].to(device)
        source_targets = batch["source_targets"].to(device)
        source_mask = batch["source_target_mask"].to(device)

        if is_training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_training):
            outputs = model(inputs)
            issue_logits = outputs["issue_logits"]
            source_logits = outputs["source_logits"]
            issue_loss = masked_bce_with_logits(
                logits=issue_logits,
                targets=issue_targets,
                mask=issue_mask,
                pos_weight=issue_pos_weight,
            )
            source_loss = masked_bce_with_logits(
                logits=source_logits,
                targets=source_targets,
                mask=source_mask,
                pos_weight=source_pos_weight,
            )
            loss = issue_loss * issue_loss_weight + source_loss * source_loss_weight

            if is_training:
                loss.backward()
                optimizer.step()

        total_loss += float(loss.item())
        total_issue_loss += float(issue_loss.item())
        total_source_loss += float(source_loss.item())
        total_batches += 1

        if collect_outputs:
            collected_outputs["issue_probs"].append(outputs["issue_probs"].detach().cpu().numpy())
            collected_outputs["issue_targets"].append(issue_targets.detach().cpu().numpy())
            collected_outputs["issue_mask"].append(issue_mask.detach().cpu().numpy())
            collected_outputs["source_probs"].append(outputs["source_probs"].detach().cpu().numpy())
            collected_outputs["source_targets"].append(source_targets.detach().cpu().numpy())
            collected_outputs["source_mask"].append(source_mask.detach().cpu().numpy())

    output_arrays = {
        key: (
            np.concatenate(value, axis=0)
            if value
            else np.zeros((0, len(ISSUE_LABELS) if "issue" in key else len(SOURCE_LABELS)), dtype=np.float32)
        )
        for key, value in collected_outputs.items()
    }

    return {
        "total_loss": total_loss / max(1, total_batches),
        "issue_loss": total_issue_loss / max(1, total_batches),
        "source_loss": total_source_loss / max(1, total_batches),
        "outputs": output_arrays,
    }


def build_validation_metrics(
    *,
    epoch_outputs: dict[str, np.ndarray],
    issue_thresholds: dict[str, float],
    source_thresholds: dict[str, float],
) -> dict[str, object]:
    return {
        "issue_head": evaluate_multilabel_head(
            probabilities=epoch_outputs["issue_probs"],
            targets=epoch_outputs["issue_targets"],
            masks=epoch_outputs["issue_mask"],
            labels=ISSUE_LABELS,
            thresholds=issue_thresholds,
        ),
        "source_head": evaluate_multilabel_head(
            probabilities=epoch_outputs["source_probs"],
            targets=epoch_outputs["source_targets"],
            masks=epoch_outputs["source_mask"],
            labels=SOURCE_LABELS,
            thresholds=source_thresholds,
        ),
    }


def masked_bce_with_logits(
    *,
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    pos_weight: torch.Tensor,
) -> torch.Tensor:
    if torch.count_nonzero(mask).item() == 0:
        return logits.sum() * 0.0

    loss = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction="none",
        pos_weight=pos_weight,
    )
    masked_loss = loss * mask
    return masked_loss.sum() / mask.sum().clamp(min=1.0)


def compute_pos_weight(entries: list[dict], target_key: str, labels: tuple[str, ...]) -> torch.Tensor:
    values = torch.tensor([entry[target_key]["values"] for entry in entries], dtype=torch.float32)
    masks = torch.tensor([entry[target_key]["mask"] for entry in entries], dtype=torch.float32)
    positives = (values * masks).sum(dim=0)
    negatives = masks.sum(dim=0) - positives
    safe_weight = negatives / positives.clamp(min=1.0)
    safe_weight = torch.where(masks.sum(dim=0) > 0, safe_weight, torch.ones_like(safe_weight))
    if safe_weight.numel() != len(labels):
        raise RuntimeError(f"Expected pos_weight length {len(labels)}, received {safe_weight.numel()}")
    return safe_weight


def save_checkpoint(
    *,
    checkpoint_path: Path,
    model_state_dict: dict[str, torch.Tensor],
    optimizer_state_dict: dict,
    epoch: int,
    model_config: ModelConfig,
    manifest_summary: dict[str, object],
    metrics: dict[str, object],
    thresholds: dict[str, object],
) -> None:
    checkpoint = {
        "state_dict": model_state_dict,
        "optimizer_state_dict": optimizer_state_dict,
        "epoch": epoch,
        "config": model_config.to_dict(),
        "manifest_summary": manifest_summary,
        "metrics": metrics,
        "thresholds": thresholds,
        "schema_version": SCHEMA_VERSION,
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
