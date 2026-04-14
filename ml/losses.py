from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class LossBreakdown:
    issue_loss: torch.Tensor
    source_loss: torch.Tensor
    eq_loss: torch.Tensor
    distillation_loss: torch.Tensor

    @property
    def total(self) -> torch.Tensor:
        return self.issue_loss + self.source_loss + self.eq_loss + self.distillation_loss


def class_balanced_weights(
    class_counts: torch.Tensor,
    *,
    beta: float = 0.999,
) -> torch.Tensor:
    counts = class_counts.float().clamp(min=1.0)
    effective_num = 1.0 - torch.pow(torch.full_like(counts, beta), counts)
    weights = (1.0 - beta) / effective_num.clamp(min=1e-6)
    return weights / weights.mean().clamp(min=1e-6)


def sigmoid_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
    alpha: float = 0.25,
    gamma: float = 2.0,
    class_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    targets = targets.float()
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    probabilities = torch.sigmoid(logits)
    pt = torch.where(targets > 0.5, probabilities, 1.0 - probabilities)
    focal_term = torch.pow((1.0 - pt).clamp(min=1e-6), gamma)
    alpha_factor = torch.where(targets > 0.5, alpha, 1.0 - alpha)
    loss = loss * focal_term * alpha_factor

    if class_weights is not None:
        loss = loss * class_weights.view(1, -1)

    if mask is None:
        return loss.mean()

    masked = loss * mask
    return masked.sum() / mask.sum().clamp(min=1.0)


def source_classification_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    mask: torch.Tensor | None,
    mode: str,
    pos_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    if mode == "softmax":
        if mask is None:
            active_rows = torch.ones(logits.shape[0], dtype=torch.bool, device=logits.device)
        else:
            active_rows = mask.sum(dim=-1) > 0

        if not bool(active_rows.any()):
            return logits.sum() * 0.0

        target_indices = targets[active_rows].argmax(dim=-1)
        return F.cross_entropy(logits[active_rows], target_indices)

    loss = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction="none",
        pos_weight=pos_weight,
    )

    if mask is None:
        return loss.mean()

    masked = loss * mask
    return masked.sum() / mask.sum().clamp(min=1.0)


def masked_smooth_l1_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
    beta: float = 1.0,
) -> torch.Tensor:
    loss = F.smooth_l1_loss(prediction, target, reduction="none", beta=beta)
    if mask is None:
        return loss.mean()

    masked = loss * mask
    return masked.sum() / mask.sum().clamp(min=1.0)


def distillation_kl_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    *,
    temperature: float = 2.0,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    student = F.log_softmax(student_logits / temperature, dim=-1)
    teacher = F.softmax(teacher_logits / temperature, dim=-1)
    kl = F.kl_div(student, teacher, reduction="none") * (temperature ** 2)

    if mask is None:
        return kl.sum(dim=-1).mean()

    if mask.dim() == kl.dim() - 1:
        active_rows = mask.sum(dim=-1) > 0
        if not bool(active_rows.any()):
            return student_logits.sum() * 0.0
        return kl.sum(dim=-1)[active_rows].mean()

    masked = kl * mask
    return masked.sum() / mask.sum().clamp(min=1.0)
