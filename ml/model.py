from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .label_schema import ISSUE_LABELS, SOURCE_LABELS
except ImportError:
    from label_schema import ISSUE_LABELS, SOURCE_LABELS


def _log_frequency_to_hz(freq_norm: torch.Tensor, min_hz: float, max_hz: float) -> torch.Tensor:
    return min_hz * torch.pow(max_hz / min_hz, freq_norm.clamp(0.0, 1.0))


def _build_2d_sincos_positional_encoding(
    height: int,
    width: int,
    channels: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if channels % 4 != 0:
        raise ValueError(f"Expected positional channel dimension divisible by 4, received {channels}.")

    y_positions = torch.arange(height, device=device, dtype=dtype)
    x_positions = torch.arange(width, device=device, dtype=dtype)
    y_grid, x_grid = torch.meshgrid(y_positions, x_positions, indexing="ij")
    half_channels = channels // 2
    omega = torch.arange(half_channels // 2, device=device, dtype=dtype)
    omega = 1.0 / (10_000 ** (omega / max(1.0, float(half_channels // 2 - 1))))

    y_embed = y_grid.reshape(-1, 1) * omega.reshape(1, -1)
    x_embed = x_grid.reshape(-1, 1) * omega.reshape(1, -1)
    return torch.cat(
        [torch.sin(y_embed), torch.cos(y_embed), torch.sin(x_embed), torch.cos(x_embed)],
        dim=-1,
    )


@dataclass
class ModelConfig:
    mel_bins: int = 64
    model_variant: str = "student"
    foundation_channels: int = 192
    foundation_layers: int = 6
    foundation_heads: int = 8
    foundation_ff_dim: int = 768
    patch_time_stride: int = 8
    patch_mel_stride: int = 8
    conv_channels: tuple[int, ...] = field(default_factory=lambda: (32, 64, 96, 128))
    hidden_dim: int = 256
    embedding_dim: int = 192
    dropout: float = 0.2
    eq_band_count: int = 5
    min_eq_hz: float = 60.0
    max_eq_hz: float = 8_000.0
    max_eq_gain_db: float = 9.0
    freeze_foundation: bool = True
    source_head_mode: str = "multilabel"
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
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class MlpHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class StudentCnnEncoder(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        channels = (1, *config.conv_channels)
        self.blocks = nn.Sequential(
            *[ConvBlock(channels[index], channels[index + 1]) for index in range(len(channels) - 1)]
        )
        pooled_dim = config.conv_channels[-1] * 2
        self.projection = nn.Sequential(
            nn.LayerNorm(pooled_dim),
            nn.Linear(pooled_dim, config.embedding_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.blocks(x)
        average_pool = encoded.mean(dim=(2, 3))
        max_pool = encoded.amax(dim=(2, 3))
        return self.projection(torch.cat([average_pool, max_pool], dim=-1))


class SpectrogramFoundationEncoder(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.patch_embed = nn.Conv2d(
            in_channels=1,
            out_channels=config.foundation_channels,
            kernel_size=(config.patch_time_stride, config.patch_mel_stride),
            stride=(config.patch_time_stride, config.patch_mel_stride),
            bias=False,
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.foundation_channels,
            nhead=config.foundation_heads,
            dim_feedforward=config.foundation_ff_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.foundation_layers)
        self.output_projection = nn.Sequential(
            nn.LayerNorm(config.foundation_channels * 2),
            nn.Linear(config.foundation_channels * 2, config.embedding_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patches = self.patch_embed(x)
        batch_size, channels, time_steps, mel_steps = patches.shape
        del batch_size
        tokens = patches.flatten(2).transpose(1, 2)
        position = _build_2d_sincos_positional_encoding(
            height=time_steps,
            width=mel_steps,
            channels=channels,
            device=tokens.device,
            dtype=tokens.dtype,
        ).unsqueeze(0)
        contextualized = self.transformer(tokens + position)
        pooled_mean = contextualized.mean(dim=1)
        pooled_max = contextualized.amax(dim=1)
        return self.output_projection(torch.cat([pooled_mean, pooled_max], dim=-1))


class MultiBandEqHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        band_count: int,
        dropout: float,
        *,
        min_hz: float,
        max_hz: float,
        max_gain_db: float,
    ) -> None:
        super().__init__()
        self.band_count = band_count
        self.min_hz = min_hz
        self.max_hz = max_hz
        self.max_gain_db = max_gain_db
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, band_count * 2),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        raw = self.network(x).view(x.shape[0], self.band_count, 2)
        freq_norm = torch.sigmoid(raw[..., 0])
        gain_norm = torch.tanh(raw[..., 1])
        sort_indices = torch.argsort(freq_norm, dim=1)
        sorted_freq_norm = torch.gather(freq_norm, 1, sort_indices)
        sorted_gain_norm = torch.gather(gain_norm, 1, sort_indices)
        eq_frequency_hz = _log_frequency_to_hz(sorted_freq_norm, self.min_hz, self.max_hz)
        eq_gain_db = sorted_gain_norm * self.max_gain_db
        eq_params = torch.stack([eq_frequency_hz, eq_gain_db], dim=-1)
        eq_params_normalized = torch.stack([sorted_freq_norm, sorted_gain_norm], dim=-1)
        return {
            "eq_params": eq_params,
            "eq_params_normalized": eq_params_normalized,
        }


class AudioIntelligenceNet(nn.Module):
    """
    Production-ready audio intelligence model with:
    - source classification
    - source-conditioned issue classification
    - learned multi-band EQ regression
    - selectable teacher/student encoders for distillation workflows
    """

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        if self.config.model_variant not in {"student", "teacher"}:
            raise ValueError(f"Unsupported model_variant '{self.config.model_variant}'.")

        if self.config.source_head_mode not in {"multilabel", "softmax"}:
            raise ValueError(f"Unsupported source_head_mode '{self.config.source_head_mode}'.")

        self.encoder = (
            StudentCnnEncoder(self.config)
            if self.config.model_variant == "student"
            else SpectrogramFoundationEncoder(self.config)
        )

        if self.config.freeze_foundation and self.config.model_variant == "teacher":
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False

        self.source_head = MlpHead(
            input_dim=self.config.embedding_dim,
            hidden_dim=self.config.hidden_dim,
            output_dim=len(SOURCE_LABELS),
            dropout=self.config.dropout,
        )
        self.issue_head = MlpHead(
            input_dim=self.config.embedding_dim + len(SOURCE_LABELS),
            hidden_dim=self.config.hidden_dim,
            output_dim=len(ISSUE_LABELS),
            dropout=self.config.dropout,
        )
        self.eq_head = MultiBandEqHead(
            input_dim=self.config.embedding_dim + len(ISSUE_LABELS) + len(SOURCE_LABELS),
            hidden_dim=self.config.hidden_dim,
            band_count=self.config.eq_band_count,
            dropout=self.config.dropout,
            min_hz=self.config.min_eq_hz,
            max_hz=self.config.max_eq_hz,
            max_gain_db=self.config.max_eq_gain_db,
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        prepared = self._prepare_input(x)
        embedding = self.encoder(prepared)
        source_logits = self.source_head(embedding)
        if self.config.source_head_mode == "softmax":
            source_probs = F.softmax(source_logits, dim=-1)
        else:
            source_probs = torch.sigmoid(source_logits)

        issue_logits = self.issue_head(torch.cat([embedding, source_probs], dim=-1))
        issue_probs = torch.sigmoid(issue_logits)
        eq_outputs = self.eq_head(torch.cat([embedding, issue_probs, source_probs], dim=-1))
        eq_params = eq_outputs["eq_params"]

        return {
            "issue_logits": issue_logits,
            "issue_probs": issue_probs,
            "source_logits": source_logits,
            "source_probs": source_probs,
            "eq_params": eq_params,
            "eq_params_normalized": eq_outputs["eq_params_normalized"],
            "eq_freq": eq_params[..., 0],
            "eq_gain_db": eq_params[..., 1],
            "problem_logits": issue_logits,
            "problem_probs": issue_probs,
            "embedding": embedding,
        }

    def _prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(0)

        if x.dim() != 3:
            raise ValueError(
                f"Expected input with shape [batch, time, mel_bins] or [time, mel_bins], got {tuple(x.shape)}"
            )

        if (not torch.onnx.is_in_onnx_export()) and x.size(-1) != self.config.mel_bins:
            raise ValueError(f"Expected mel dimension {self.config.mel_bins}, got {x.size(-1)}")

        return x.unsqueeze(1)


class ProductionAudioIntelligenceNet(AudioIntelligenceNet):
    pass


class LightweightAudioAnalysisNet(AudioIntelligenceNet):
    def __init__(self, config: ModelConfig | None = None) -> None:
        active_config = config or ModelConfig(model_variant="student")
        if active_config.model_variant != "student":
            active_config = ModelConfig.from_dict(active_config.to_dict())
            active_config.model_variant = "student"
        super().__init__(active_config)


def load_model_from_checkpoint(
    checkpoint: dict[str, Any],
    *,
    override_variant: str | None = None,
    mel_bins: int | None = None,
) -> AudioIntelligenceNet:
    config = ModelConfig.from_dict(checkpoint.get("config") if isinstance(checkpoint, dict) else None)
    if override_variant is not None:
        config.model_variant = override_variant
    if mel_bins is not None:
        config.mel_bins = mel_bins

    model = AudioIntelligenceNet(config)
    state_dict = checkpoint.get("state_dict", checkpoint.get("model_state_dict", checkpoint))
    normalized_state_dict = {key.removeprefix("module."): value for key, value in state_dict.items()}
    missing_keys, unexpected_keys = model.load_state_dict(normalized_state_dict, strict=False)

    if missing_keys:
        raise RuntimeError(f"Missing keys while loading checkpoint: {missing_keys}")
    if unexpected_keys:
        raise RuntimeError(f"Unexpected keys while loading checkpoint: {unexpected_keys}")

    return model


if __name__ == "__main__":
    torch.manual_seed(7)
    model = LightweightAudioAnalysisNet()
    dummy_input = torch.randn(2, 298, 64)
    outputs = model(dummy_input)

    for key, value in outputs.items():
        print(f"{key}: {tuple(value.shape)}")
