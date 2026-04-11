from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import torch
import torch.nn as nn

try:
    from .label_schema import ISSUE_LABELS, SOURCE_LABELS
except ImportError:
    from label_schema import ISSUE_LABELS, SOURCE_LABELS


@dataclass
class ModelConfig:
    mel_bins: int = 64
    conv_channels: tuple[int, ...] = field(default_factory=lambda: (24, 48, 72, 96))
    hidden_dim: int = 96
    dropout: float = 0.15
    enable_cause_head: bool = False

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "ModelConfig":
        if not payload:
            return cls()

        normalized = dict(payload)
        conv_channels = normalized.get("conv_channels")

        if isinstance(conv_channels, list):
            normalized["conv_channels"] = tuple(int(value) for value in conv_channels)

        return cls(**normalized)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ClassificationHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class LightweightAudioAnalysisNet(nn.Module):
    """
    Lightweight hierarchical audio diagnosis model.

    Input:
        x: [batch, time, mel_bins] or [time, mel_bins]

    Outputs:
        - issue_logits / issue_probs: [batch, 9]
        - source_logits / source_probs: [batch, 5]
        - embedding: [batch, hidden]
    """

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()

        channels = (1, *self.config.conv_channels)
        self.encoder = nn.Sequential(
            *[ConvBlock(channels[index], channels[index + 1]) for index in range(len(channels) - 1)]
        )

        embedding_dim = self.config.conv_channels[-1] * 2
        self.issue_head = ClassificationHead(
            input_dim=embedding_dim,
            hidden_dim=self.config.hidden_dim,
            output_dim=len(ISSUE_LABELS),
            dropout=self.config.dropout,
        )
        self.source_head = ClassificationHead(
            input_dim=embedding_dim,
            hidden_dim=self.config.hidden_dim,
            output_dim=len(SOURCE_LABELS),
            dropout=self.config.dropout,
        )
        self.cause_head = (
            ClassificationHead(
                input_dim=embedding_dim,
                hidden_dim=self.config.hidden_dim,
                output_dim=0,
                dropout=self.config.dropout,
            )
            if self.config.enable_cause_head
            else None
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self._prepare_input(x)
        x = self.encoder(x)

        average_pool = x.mean(dim=(2, 3))
        max_pool = x.amax(dim=(2, 3))
        embedding = torch.cat([average_pool, max_pool], dim=-1)

        issue_logits = self.issue_head(embedding)
        source_logits = self.source_head(embedding)
        issue_probs = torch.sigmoid(issue_logits)
        source_probs = torch.sigmoid(source_logits)

        outputs = {
            "issue_logits": issue_logits,
            "issue_probs": issue_probs,
            "source_logits": source_logits,
            "source_probs": source_probs,
            # Backward-compatible aliases for older helper code.
            "problem_logits": issue_logits,
            "problem_probs": issue_probs,
            "embedding": embedding,
        }

        if self.cause_head is not None:
            cause_logits = self.cause_head(embedding)
            outputs["cause_logits"] = cause_logits
            outputs["cause_probs"] = torch.sigmoid(cause_logits)

        return outputs

    def _prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(0)

        if x.dim() != 3:
            raise ValueError(
                f"Expected input with shape [batch, time, mel_bins] or [time, mel_bins], got {tuple(x.shape)}"
            )

        if x.size(-1) != self.config.mel_bins:
            raise ValueError(f"Expected mel dimension {self.config.mel_bins}, got {x.size(-1)}")

        return x.unsqueeze(1)


if __name__ == "__main__":
    model = LightweightAudioAnalysisNet()
    dummy_input = torch.randn(2, 298, 64)
    outputs = model(dummy_input)

    for key, value in outputs.items():
        print(f"{key}: {tuple(value.shape)}")
