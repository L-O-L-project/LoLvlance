from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import torch
import torch.nn as nn

PROBLEM_LABELS = ("muddy", "harsh", "buried")


@dataclass
class ModelConfig:
    mel_bins: int = 64
    conv_channels: tuple[int, ...] = field(default_factory=lambda: (24, 48, 72, 96))
    hidden_dim: int = 96
    dropout: float = 0.15

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "ModelConfig":
        if not payload:
            return cls()

        payload = dict(payload)
        conv_channels = payload.get("conv_channels")

        if isinstance(conv_channels, list):
            payload["conv_channels"] = tuple(int(value) for value in conv_channels)

        return cls(**payload)

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


class LightweightAudioAnalysisNet(nn.Module):
    """
    Lightweight CNN for LoLvlance issue detection.

    Input:
        x: [batch, time, mel_bins] or [time, mel_bins]

    Outputs:
        - problem_logits: [batch, 3]
        - problem_probs: [batch, 3] with sigmoid activation
    """

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()

        channels = (1, *self.config.conv_channels)
        encoder_layers = [
            ConvBlock(channels[index], channels[index + 1]) for index in range(len(channels) - 1)
        ]
        self.encoder = nn.Sequential(*encoder_layers)

        embedding_dim = self.config.conv_channels[-1] * 2
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, self.config.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_dim, len(PROBLEM_LABELS)),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self._prepare_input(x)
        x = self.encoder(x)

        avg_pool = x.mean(dim=(2, 3))
        max_pool = x.amax(dim=(2, 3))
        embedding = torch.cat([avg_pool, max_pool], dim=-1)
        problem_logits = self.classifier(embedding)

        return {
            "problem_logits": problem_logits,
            "problem_probs": torch.sigmoid(problem_logits),
            "embedding": embedding,
        }

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
    dummy_input = torch.randn(4, 298, 64)
    output = model(dummy_input)

    for key, value in output.items():
        print(f"{key}: {tuple(value.shape)}")
