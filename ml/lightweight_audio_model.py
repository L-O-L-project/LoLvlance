from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    mel_bins: int = 64
    encoder_dim: int = 128
    transformer_layers: int = 2
    transformer_heads: int = 4
    transformer_ffn_dim: int = 256
    dropout: float = 0.1
    max_gain_db: float = 6.0
    use_transformer: bool = True


class ConvStem(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: tuple[int, int]) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=(5, 5),
                stride=stride,
                padding=(2, 2),
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DepthwiseSeparableBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: tuple[int, int]) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=(3, 3),
            stride=stride,
            padding=(1, 1),
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.SiLU(inplace=True)

        if stride != (1, 1) or in_channels != out_channels:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.residual = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual(x)
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.norm(x)
        x = x + residual
        return self.activation(x)


class TemporalTransformer(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.encoder_dim,
            nhead=config.transformer_heads,
            dim_feedforward=config.transformer_ffn_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.transformer_layers,
            norm=nn.LayerNorm(config.encoder_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class HeadMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LightweightAudioAnalysisNet(nn.Module):
    """
    Multi-head audio analysis network for log-mel spectrogram input.

    Input:
        x: [batch, time, mel_bins] or [time, mel_bins]

    Outputs:
        - problem classification: 4-way softmax
        - instrument estimation: 5-way sigmoid
        - EQ recommendation:
            - freq in [0, 1]
            - gain in [-6, 6] dB
    """

    def __init__(self, config: Optional[ModelConfig] = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()

        self.conv_encoder = nn.Sequential(
            ConvStem(1, 24, stride=(2, 2)),
            DepthwiseSeparableBlock(24, 32, stride=(1, 1)),
            DepthwiseSeparableBlock(32, 48, stride=(2, 2)),
            DepthwiseSeparableBlock(48, 64, stride=(1, 1)),
            DepthwiseSeparableBlock(64, 96, stride=(2, 2)),
            DepthwiseSeparableBlock(96, self.config.encoder_dim, stride=(1, 1)),
        )

        self.temporal_transformer = (
            TemporalTransformer(self.config) if self.config.use_transformer else nn.Identity()
        )
        self.sequence_norm = nn.LayerNorm(self.config.encoder_dim)

        pooled_dim = self.config.encoder_dim * 2
        self.problem_head = HeadMLP(
            input_dim=pooled_dim,
            hidden_dim=self.config.encoder_dim,
            output_dim=4,
            dropout=self.config.dropout,
        )
        self.instrument_head = HeadMLP(
            input_dim=pooled_dim,
            hidden_dim=self.config.encoder_dim,
            output_dim=5,
            dropout=self.config.dropout,
        )
        self.eq_head = HeadMLP(
            input_dim=pooled_dim,
            hidden_dim=self.config.encoder_dim,
            output_dim=2,
            dropout=self.config.dropout,
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self._prepare_input(x)

        # CNN encoder extracts compact local time-frequency patterns.
        x = self.conv_encoder(x)

        # Collapse the frequency axis so the temporal model only tracks time steps.
        x = x.mean(dim=-1)  # [batch, channels, time']
        x = x.transpose(1, 2)  # [batch, time', channels]

        x = self.temporal_transformer(x)
        x = self.sequence_norm(x)

        mean_pool = x.mean(dim=1)
        max_pool = x.amax(dim=1)
        pooled = torch.cat([mean_pool, max_pool], dim=-1)

        problem_logits = self.problem_head(pooled)
        instrument_logits = self.instrument_head(pooled)
        eq_raw = self.eq_head(pooled)

        problem_probs = F.softmax(problem_logits, dim=-1)
        instrument_probs = torch.sigmoid(instrument_logits)

        # Frequency is normalized to [0, 1], gain is scaled to [-6, 6] dB.
        eq_freq = torch.sigmoid(eq_raw[..., 0:1])
        eq_gain_db = torch.tanh(eq_raw[..., 1:2]) * self.config.max_gain_db

        return {
            "problem_logits": problem_logits,
            "problem_probs": problem_probs,
            "instrument_logits": instrument_logits,
            "instrument_probs": instrument_probs,
            "eq_raw": eq_raw,
            "eq_freq": eq_freq,
            "eq_gain_db": eq_gain_db,
            "embedding": pooled,
        }

    def _prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(0)

        if x.dim() != 3:
            raise ValueError(
                f"Expected input with shape [batch, time, mel_bins] or [time, mel_bins], got {tuple(x.shape)}"
            )

        if x.size(-1) != self.config.mel_bins:
            raise ValueError(
                f"Expected mel dimension {self.config.mel_bins}, got {x.size(-1)}"
            )

        # Conv2d expects [batch, channels, time, mel_bins].
        return x.unsqueeze(1)


if __name__ == "__main__":
    model = LightweightAudioAnalysisNet()
    dummy_input = torch.randn(4, 128, 64)
    output = model(dummy_input)

    for key, value in output.items():
        print(f"{key}: {tuple(value.shape)}")
