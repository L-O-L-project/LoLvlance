from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    from .dataset import DatasetRoots, build_public_manifest, load_manifest, summarize_manifest
    from .degradation import DegradationConfig, RealAudioDegradationDataset, build_real_source_manifest
    from .label_schema import (
        DEFAULT_ISSUE_THRESHOLDS,
        DEFAULT_SOURCE_THRESHOLDS,
        ISSUE_LABELS,
        SCHEMA_VERSION,
        SOURCE_LABELS,
    )
    from .losses import (
        LossBreakdown,
        class_balanced_weights,
        distillation_kl_loss,
        masked_smooth_l1_loss,
        sigmoid_focal_loss,
        source_classification_loss,
    )
    from .metrics import evaluate_multilabel_head, tune_thresholds
    from .model import AudioIntelligenceNet, ModelConfig, load_model_from_checkpoint
    from .preprocessing import PreprocessingConfig
except ImportError:
    from dataset import DatasetRoots, build_public_manifest, load_manifest, summarize_manifest
    from degradation import DegradationConfig, RealAudioDegradationDataset, build_real_source_manifest
    from label_schema import (
        DEFAULT_ISSUE_THRESHOLDS,
        DEFAULT_SOURCE_THRESHOLDS,
        ISSUE_LABELS,
        SCHEMA_VERSION,
        SOURCE_LABELS,
    )
    from losses import (
        LossBreakdown,
        class_balanced_weights,
        distillation_kl_loss,
        masked_smooth_l1_loss,
        sigmoid_focal_loss,
        source_classification_loss,
    )
    from metrics import evaluate_multilabel_head, tune_thresholds
    from model import AudioIntelligenceNet, ModelConfig, load_model_from_checkpoint
    from preprocessing import PreprocessingConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the production LoLvlance audio intelligence model.")
    parser.add_argument("--audio-root", action="append", type=Path, default=[], help="Root directory of real source audio.")
    parser.add_argument("--openmic-root", type=Path, default=None, help="Compatibility dataset root.")
    parser.add_argument("--slakh-root", type=Path, default=None, help="Compatibility dataset root.")
    parser.add_argument("--musan-root", type=Path, default=None, help="Compatibility dataset root.")
    parser.add_argument("--fsd50k-root", type=Path, default=None, help="Compatibility dataset root.")
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=Path("ml/artifacts/public_dataset_manifest.jsonl"),
        help="Where to store the derived training manifest.",
    )
    parser.add_argument("--rebuild-manifest", action="store_true", help="Rebuild the training manifest.")
    parser.add_argument("--clips-per-file", type=int, default=2, help="Number of clips sampled per source file.")
    parser.add_argument("--max-files-per-dataset", type=int, default=None, help="Optional cap for compatibility roots.")
    parser.add_argument("--epochs", type=int, default=6, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=12, help="Training batch size.")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="AdamW learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay.")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout for heads.")
    parser.add_argument("--hidden-dim", type=int, default=256, help="Head hidden dimension.")
    parser.add_argument("--embedding-dim", type=int, default=192, help="Shared embedding dimension.")
    parser.add_argument("--model-variant", choices=("teacher", "student"), default="teacher", help="teacher or student.")
    parser.add_argument("--teacher-checkpoint", type=Path, default=None, help="Teacher checkpoint for student distillation.")
    parser.add_argument("--eq-bands", type=int, default=5, help="Number of learned EQ bands.")
    parser.add_argument("--issue-loss-weight", type=float, default=1.0, help="Issue loss weight.")
    parser.add_argument("--source-loss-weight", type=float, default=0.45, help="Source loss weight.")
    parser.add_argument("--eq-loss-weight", type=float, default=1.2, help="EQ regression loss weight.")
    parser.add_argument("--distillation-weight", type=float, default=0.35, help="Student distillation loss weight.")
    parser.add_argument("--distillation-temperature", type=float, default=2.0, help="KL temperature.")
    parser.add_argument("--num-workers", type=int, default=0, help="Dataloader workers.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("ml/checkpoints"), help="Directory for checkpoints.")
    parser.add_argument("--device", type=str, default="auto", help="auto, cpu, cuda, or mps.")
    parser.add_argument("--export-onnx", action="store_true", help="Export the best checkpoint to ONNX after training.")
    parser.add_argument("--onnx-output", type=Path, default=Path("ml/checkpoints/lightweight_audio_model.onnx"))
    return parser.parse_args()


def run_training(args: argparse.Namespace) -> tuple[Path, dict[str, object]]:
    seed_everything(int(getattr(args, "seed", 7)))
    device = resolve_device(str(getattr(args, "device", "auto")))
    preprocessing_config = PreprocessingConfig()
    manifest_path = prepare_manifest(args, preprocessing_config)
    manifest_entries = load_manifest(manifest_path)
    manifest_summary = summarize_manifest(manifest_entries)

    degradation_config = DegradationConfig(
        sample_rate=preprocessing_config.sample_rate,
        clip_seconds=preprocessing_config.clip_seconds,
        eq_band_count=int(getattr(args, "eq_bands", 5)),
        seed=int(getattr(args, "seed", 7)),
    )
    train_dataset = RealAudioDegradationDataset(
        manifest_path=manifest_path,
        split="train",
        preprocessing_config=preprocessing_config,
        degradation_config=degradation_config,
    )
    val_dataset = RealAudioDegradationDataset(
        manifest_path=manifest_path,
        split="val",
        preprocessing_config=preprocessing_config,
        degradation_config=degradation_config,
    )
    train_loader = DataLoader(train_dataset, batch_size=int(getattr(args, "batch_size", 12)), shuffle=True, num_workers=int(getattr(args, "num_workers", 0)))
    val_loader = DataLoader(val_dataset, batch_size=int(getattr(args, "batch_size", 12)), shuffle=False, num_workers=int(getattr(args, "num_workers", 0)))

    model_config = ModelConfig(
        mel_bins=preprocessing_config.mel_bin_count,
        model_variant=str(getattr(args, "model_variant", "teacher")),
        dropout=float(getattr(args, "dropout", 0.2)),
        hidden_dim=int(getattr(args, "hidden_dim", 256)),
        embedding_dim=int(getattr(args, "embedding_dim", 192)),
        eq_band_count=int(getattr(args, "eq_bands", 5)),
        freeze_foundation=False,
    )
    model = AudioIntelligenceNet(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(getattr(args, "learning_rate", 3e-4)), weight_decay=float(getattr(args, "weight_decay", 1e-4)))
    source_pos_weight = compute_source_pos_weight(manifest_entries).to(device)
    issue_class_weights = estimate_issue_class_weights(train_dataset.issue_sampling_weights, len(train_dataset)).to(device)

    teacher_model = None
    teacher_checkpoint = getattr(args, "teacher_checkpoint", None)
    if model_config.model_variant == "student" and teacher_checkpoint is not None:
        teacher_model = load_model_from_checkpoint(torch.load(teacher_checkpoint, map_location="cpu"), override_variant="teacher", mel_bins=preprocessing_config.mel_bin_count).to(device).eval()
        for parameter in teacher_model.parameters():
            parameter.requires_grad = False

    default_issue_thresholds = dict(DEFAULT_ISSUE_THRESHOLDS)
    default_source_thresholds = dict(DEFAULT_SOURCE_THRESHOLDS)
    checkpoint_dir = Path(getattr(args, "checkpoint_dir", Path("ml/checkpoints")))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    history: list[dict[str, object]] = []
    best_score = float("-inf")
    best_epoch = 0
    best_state_dict = copy.deepcopy(model.state_dict())
    best_checkpoint_path = checkpoint_dir / "best_sound_issue_model.pt"

    for epoch in range(1, int(getattr(args, "epochs", 6)) + 1):
        train_epoch = run_epoch(
            model=model,
            teacher_model=teacher_model,
            data_loader=train_loader,
            device=device,
            optimizer=optimizer,
            issue_class_weights=issue_class_weights,
            source_pos_weight=source_pos_weight,
            issue_loss_weight=float(getattr(args, "issue_loss_weight", 1.0)),
            source_loss_weight=float(getattr(args, "source_loss_weight", 0.45)),
            eq_loss_weight=float(getattr(args, "eq_loss_weight", 1.2)),
            distillation_weight=float(getattr(args, "distillation_weight", 0.35)),
            distillation_temperature=float(getattr(args, "distillation_temperature", 2.0)),
            source_mode=model_config.source_head_mode,
            collect_outputs=False,
        )
        val_epoch = run_epoch(
            model=model,
            teacher_model=teacher_model,
            data_loader=val_loader,
            device=device,
            optimizer=None,
            issue_class_weights=issue_class_weights,
            source_pos_weight=source_pos_weight,
            issue_loss_weight=float(getattr(args, "issue_loss_weight", 1.0)),
            source_loss_weight=float(getattr(args, "source_loss_weight", 0.45)),
            eq_loss_weight=float(getattr(args, "eq_loss_weight", 1.2)),
            distillation_weight=float(getattr(args, "distillation_weight", 0.35)),
            distillation_temperature=float(getattr(args, "distillation_temperature", 2.0)),
            source_mode=model_config.source_head_mode,
            collect_outputs=True,
        )

        val_metrics = build_validation_metrics(
            epoch_outputs=val_epoch["outputs"],
            issue_thresholds=default_issue_thresholds,
            source_thresholds=default_source_thresholds,
        )
        composite_score = float(val_metrics["issue_head"]["macro_f1"]) + float(val_metrics["source_head"]["macro_f1"]) * 0.35 - float(val_metrics["eq_head"]["normalized_mae"]) * 0.1
        epoch_record = {
            "epoch": epoch,
            "train": {
                "total_loss": round(float(train_epoch["total_loss"]), 4),
                "issue_loss": round(float(train_epoch["issue_loss"]), 4),
                "source_loss": round(float(train_epoch["source_loss"]), 4),
                "eq_loss": round(float(train_epoch["eq_loss"]), 4),
                "distillation_loss": round(float(train_epoch["distillation_loss"]), 4),
            },
            "val": {
                "total_loss": round(float(val_epoch["total_loss"]), 4),
                "issue_loss": round(float(val_epoch["issue_loss"]), 4),
                "source_loss": round(float(val_epoch["source_loss"]), 4),
                "eq_loss": round(float(val_epoch["eq_loss"]), 4),
                "distillation_loss": round(float(val_epoch["distillation_loss"]), 4),
                "metrics": val_metrics,
            },
            "selection_score": round(composite_score, 4),
        }
        history.append(epoch_record)

        save_checkpoint(
            checkpoint_path=checkpoint_dir / "last_sound_issue_model.pt",
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

    model.load_state_dict(best_state_dict)
    best_val_epoch = run_epoch(
        model=model,
        teacher_model=teacher_model,
        data_loader=val_loader,
        device=device,
        optimizer=None,
        issue_class_weights=issue_class_weights,
        source_pos_weight=source_pos_weight,
        issue_loss_weight=float(getattr(args, "issue_loss_weight", 1.0)),
        source_loss_weight=float(getattr(args, "source_loss_weight", 0.45)),
        eq_loss_weight=float(getattr(args, "eq_loss_weight", 1.2)),
        distillation_weight=float(getattr(args, "distillation_weight", 0.35)),
        distillation_temperature=float(getattr(args, "distillation_temperature", 2.0)),
        source_mode=model_config.source_head_mode,
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
    threshold_payload = {
        "schema_version": SCHEMA_VERSION,
        "issue_thresholds": tuned_issue_thresholds,
        "source_thresholds": tuned_source_thresholds,
        "best_epoch": best_epoch,
    }
    (checkpoint_dir / "label_thresholds.json").write_text(json.dumps(threshold_payload, indent=2), encoding="utf-8")
    (checkpoint_dir / "config.json").write_text(json.dumps(model_config.to_dict(), indent=2), encoding="utf-8")
    (checkpoint_dir / "thresholds.json").write_text(json.dumps(threshold_payload, indent=2), encoding="utf-8")

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
    (checkpoint_dir / "training_history.json").write_text(json.dumps(final_summary, indent=2), encoding="utf-8")
    torch.save(best_state_dict, checkpoint_dir / "model.pt")

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

    if bool(getattr(args, "export_onnx", False)):
        try:
            from .export_to_onnx import export_to_onnx
        except ImportError:
            from export_to_onnx import export_to_onnx

        sample_time_steps = int(train_dataset[0]["log_mel_spectrogram"].shape[0])
        export_args = argparse.Namespace(
            checkpoint=best_checkpoint_path,
            output=Path(getattr(args, "onnx_output", checkpoint_dir / "lightweight_audio_model.onnx")),
            time_steps=sample_time_steps,
            opset=18,
            verify=True,
            mel_bins=model_config.mel_bins,
        )
        export_to_onnx(export_args)

    return best_checkpoint_path, final_summary


def prepare_manifest(args: argparse.Namespace, preprocessing_config: PreprocessingConfig) -> Path:
    manifest_path = Path(getattr(args, "manifest_path", Path("ml/artifacts/public_dataset_manifest.jsonl")))
    audio_roots = [Path(path) for path in getattr(args, "audio_root", []) if path]
    compatibility_roots = [
        getattr(args, "openmic_root", None),
        getattr(args, "slakh_root", None),
        getattr(args, "musan_root", None),
        getattr(args, "fsd50k_root", None),
    ]
    compatibility_roots = [Path(path) for path in compatibility_roots if path is not None]

    if bool(getattr(args, "rebuild_manifest", False)) or not manifest_path.exists():
        if compatibility_roots:
            build_public_manifest(
                dataset_roots=DatasetRoots(
                    openmic=getattr(args, "openmic_root", None),
                    slakh=getattr(args, "slakh_root", None),
                    musan=getattr(args, "musan_root", None),
                    fsd50k=getattr(args, "fsd50k_root", None),
                ),
                output_path=manifest_path,
                preprocessing_config=preprocessing_config,
                clips_per_file=int(getattr(args, "clips_per_file", 2)),
                max_files_per_dataset=getattr(args, "max_files_per_dataset", None),
            )
        elif audio_roots:
            build_real_source_manifest(
                audio_roots=audio_roots,
                output_path=manifest_path,
                clip_seconds=preprocessing_config.clip_seconds,
                clips_per_file=int(getattr(args, "clips_per_file", 2)),
            )
        else:
            raise FileNotFoundError("Provide --audio-root or dataset compatibility roots to build a manifest.")

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found at {manifest_path}.")

    return manifest_path


def run_epoch(
    *,
    model: AudioIntelligenceNet,
    teacher_model: AudioIntelligenceNet | None,
    data_loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    issue_class_weights: torch.Tensor,
    source_pos_weight: torch.Tensor,
    issue_loss_weight: float,
    source_loss_weight: float,
    eq_loss_weight: float,
    distillation_weight: float,
    distillation_temperature: float,
    source_mode: str,
    collect_outputs: bool,
) -> dict[str, object]:
    is_training = optimizer is not None
    model.train(is_training)
    total_loss = 0.0
    total_issue_loss = 0.0
    total_source_loss = 0.0
    total_eq_loss = 0.0
    total_distillation_loss = 0.0
    total_batches = 0

    collected_outputs = {
        "issue_probs": [],
        "issue_targets": [],
        "issue_mask": [],
        "source_probs": [],
        "source_targets": [],
        "source_mask": [],
        "eq_params_normalized": [],
        "eq_targets_normalized": [],
        "eq_mask": [],
    }

    for batch in data_loader:
        inputs = batch["log_mel_spectrogram"].to(device)
        issue_targets = batch["issue_targets"].to(device)
        issue_mask = batch["issue_target_mask"].to(device)
        source_targets = batch["source_targets"].to(device)
        source_mask = batch["source_target_mask"].to(device)
        eq_targets_normalized = batch["eq_params_normalized"].to(device)
        eq_mask = batch["eq_mask"].to(device)

        if is_training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_training):
            outputs = model(inputs)
            teacher_outputs = None
            if teacher_model is not None:
                with torch.no_grad():
                    teacher_outputs = teacher_model(inputs)

            losses = compute_losses(
                outputs=outputs,
                issue_targets=issue_targets,
                issue_mask=issue_mask,
                source_targets=source_targets,
                source_mask=source_mask,
                eq_targets_normalized=eq_targets_normalized,
                eq_mask=eq_mask,
                issue_class_weights=issue_class_weights,
                source_pos_weight=source_pos_weight,
                source_mode=source_mode,
                teacher_outputs=teacher_outputs,
                distillation_temperature=distillation_temperature,
                issue_loss_weight=issue_loss_weight,
                source_loss_weight=source_loss_weight,
                eq_loss_weight=eq_loss_weight,
                distillation_weight=distillation_weight,
            )

            if is_training:
                losses.total.backward()
                optimizer.step()

        total_loss += float(losses.total.item())
        total_issue_loss += float(losses.issue_loss.item())
        total_source_loss += float(losses.source_loss.item())
        total_eq_loss += float(losses.eq_loss.item())
        total_distillation_loss += float(losses.distillation_loss.item())
        total_batches += 1

        if collect_outputs:
            collected_outputs["issue_probs"].append(outputs["issue_probs"].detach().cpu().numpy())
            collected_outputs["issue_targets"].append(issue_targets.detach().cpu().numpy())
            collected_outputs["issue_mask"].append(issue_mask.detach().cpu().numpy())
            collected_outputs["source_probs"].append(outputs["source_probs"].detach().cpu().numpy())
            collected_outputs["source_targets"].append(source_targets.detach().cpu().numpy())
            collected_outputs["source_mask"].append(source_mask.detach().cpu().numpy())
            collected_outputs["eq_params_normalized"].append(outputs["eq_params_normalized"].detach().cpu().numpy())
            collected_outputs["eq_targets_normalized"].append(eq_targets_normalized.detach().cpu().numpy())
            collected_outputs["eq_mask"].append(eq_mask.detach().cpu().numpy())

    output_arrays = {
        key: (np.concatenate(value, axis=0) if value else np.zeros((0,), dtype=np.float32))
        for key, value in collected_outputs.items()
    }

    return {
        "total_loss": total_loss / max(1, total_batches),
        "issue_loss": total_issue_loss / max(1, total_batches),
        "source_loss": total_source_loss / max(1, total_batches),
        "eq_loss": total_eq_loss / max(1, total_batches),
        "distillation_loss": total_distillation_loss / max(1, total_batches),
        "outputs": output_arrays,
    }


def compute_losses(
    *,
    outputs: dict[str, torch.Tensor],
    issue_targets: torch.Tensor,
    issue_mask: torch.Tensor,
    source_targets: torch.Tensor,
    source_mask: torch.Tensor,
    eq_targets_normalized: torch.Tensor,
    eq_mask: torch.Tensor,
    issue_class_weights: torch.Tensor,
    source_pos_weight: torch.Tensor,
    source_mode: str,
    teacher_outputs: dict[str, torch.Tensor] | None,
    distillation_temperature: float,
    issue_loss_weight: float,
    source_loss_weight: float,
    eq_loss_weight: float,
    distillation_weight: float,
) -> LossBreakdown:
    issue_loss = sigmoid_focal_loss(
        outputs["issue_logits"],
        issue_targets,
        mask=issue_mask,
        class_weights=issue_class_weights,
    ) * issue_loss_weight
    source_loss = source_classification_loss(
        outputs["source_logits"],
        source_targets,
        mask=source_mask,
        mode=source_mode,
        pos_weight=source_pos_weight,
    ) * source_loss_weight
    eq_loss = masked_smooth_l1_loss(
        outputs["eq_params_normalized"],
        eq_targets_normalized,
        mask=eq_mask,
        beta=0.25,
    ) * eq_loss_weight

    distillation_loss = outputs["issue_logits"].sum() * 0.0
    if teacher_outputs is not None and distillation_weight > 0:
        issue_distillation = distillation_kl_loss(
            outputs["issue_logits"],
            teacher_outputs["issue_logits"],
            temperature=distillation_temperature,
            mask=issue_mask,
        )
        source_distillation = distillation_kl_loss(
            outputs["source_logits"],
            teacher_outputs["source_logits"],
            temperature=distillation_temperature,
            mask=source_mask,
        )
        eq_distillation = masked_smooth_l1_loss(
            outputs["eq_params_normalized"],
            teacher_outputs["eq_params_normalized"],
            mask=eq_mask,
            beta=0.25,
        )
        distillation_loss = (issue_distillation + source_distillation + eq_distillation) * distillation_weight

    return LossBreakdown(
        issue_loss=issue_loss,
        source_loss=source_loss,
        eq_loss=eq_loss,
        distillation_loss=distillation_loss,
    )


def build_validation_metrics(
    *,
    epoch_outputs: dict[str, np.ndarray],
    issue_thresholds: dict[str, float],
    source_thresholds: dict[str, float],
) -> dict[str, object]:
    eq_prediction = epoch_outputs["eq_params_normalized"]
    eq_target = epoch_outputs["eq_targets_normalized"]
    eq_mask = epoch_outputs["eq_mask"]
    if eq_prediction.size == 0:
        eq_mae = 0.0
    else:
        eq_mae = float(np.abs((eq_prediction - eq_target) * eq_mask).sum() / max(eq_mask.sum(), 1.0))

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
        "eq_head": {
            "normalized_mae": round(eq_mae, 4),
        },
    }


def estimate_issue_class_weights(issue_sampling_weights: dict[str, float], dataset_size: int) -> torch.Tensor:
    counts = torch.tensor(
        [max(1.0, issue_sampling_weights[label] * max(1, dataset_size)) for label in ISSUE_LABELS],
        dtype=torch.float32,
    )
    return class_balanced_weights(counts)


def compute_source_pos_weight(entries: list[dict]) -> torch.Tensor:
    positives = torch.zeros(len(SOURCE_LABELS), dtype=torch.float32)
    support = torch.zeros(len(SOURCE_LABELS), dtype=torch.float32)

    for entry in entries:
        source_targets = entry.get("source_targets", {})
        values = torch.tensor(source_targets.get("values", [0.0 for _ in SOURCE_LABELS]), dtype=torch.float32)
        mask = torch.tensor(source_targets.get("mask", [0.0 for _ in SOURCE_LABELS]), dtype=torch.float32)
        positives += values * mask
        support += mask

    negatives = support - positives
    pos_weight = negatives / positives.clamp(min=1.0)
    pos_weight = torch.where(support > 0, pos_weight, torch.ones_like(pos_weight))
    return pos_weight


def save_checkpoint(
    *,
    checkpoint_path: Path,
    model_state_dict: dict[str, torch.Tensor],
    optimizer_state_dict: dict[str, Any],
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
