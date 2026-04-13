from __future__ import annotations

import math
import tempfile
from pathlib import Path

import onnx
import torch
from onnx import compose

try:
    from .label_schema import ISSUE_FALLBACK_EQ, ISSUE_LABELS, SOURCE_LABELS
except ImportError:
    from label_schema import ISSUE_FALLBACK_EQ, ISSUE_LABELS, SOURCE_LABELS

MODEL_MIN_HZ = 80.0
MODEL_MAX_HZ = 8000.0
MAX_GAIN_DB = 6.0
EPSILON = 1e-6

LEGACY_OUTPUT_NAMES = (
    "problem_probs",
    "instrument_probs",
    "eq_freq",
    "eq_gain_db",
)

LEGACY_INPUT_NAMES = tuple(f"legacy_{name}" for name in LEGACY_OUTPUT_NAMES)
ADAPTER_PREFIX = "adapter_"

PAIR_EQ_OVERRIDES: dict[tuple[str, str], tuple[float, float]] = {
    ("muddy", "vocal"): (250.0, -3.5),
    ("muddy", "guitar"): (275.0, -2.5),
    ("muddy", "bass"): (250.0, -3.0),
    ("muddy", "keys"): (350.0, -2.5),
    ("harsh", "vocal"): (4000.0, -3.0),
    ("harsh", "guitar"): (3500.0, -3.0),
    ("harsh", "drums"): (6250.0, -3.0),
    ("buried", "vocal"): (2200.0, 3.0),
    ("buried", "guitar"): (2150.0, 2.5),
    ("buried", "keys"): (1750.0, 2.5),
}

LEGACY_EQ_WIDTHS = {
    "muddy": 0.12,
    "harsh": 0.12,
    "buried": 0.12,
    "boomy": 0.09,
    "thin": 0.09,
    "boxy": 0.10,
    "nasal": 0.10,
    "sibilant": 0.10,
    "dull": 0.10,
}


def normalize_frequency_hz(frequency_hz: float) -> float:
    clamped_frequency_hz = min(max(frequency_hz, MODEL_MIN_HZ), MODEL_MAX_HZ)
    return math.log(clamped_frequency_hz / MODEL_MIN_HZ) / math.log(MODEL_MAX_HZ / MODEL_MIN_HZ)


def build_issue_eq_tensors() -> tuple[torch.Tensor, torch.Tensor]:
    frequencies = []
    gains = []

    for issue in ISSUE_LABELS:
        fallback = ISSUE_FALLBACK_EQ[issue]
        frequencies.append(normalize_frequency_hz(float(fallback["frequency_hz"])))
        gains.append(float(fallback["gain_db"]))

    return torch.tensor(frequencies, dtype=torch.float32), torch.tensor(gains, dtype=torch.float32)


def build_pair_eq_tensors() -> tuple[torch.Tensor, torch.Tensor]:
    frequency_rows: list[list[float]] = []
    gain_rows: list[list[float]] = []

    for issue in ISSUE_LABELS:
        issue_fallback = ISSUE_FALLBACK_EQ[issue]
        fallback_frequency = float(issue_fallback["frequency_hz"])
        fallback_gain = float(issue_fallback["gain_db"])
        frequency_row: list[float] = []
        gain_row: list[float] = []

        for source in SOURCE_LABELS:
            frequency_hz, gain_db = PAIR_EQ_OVERRIDES.get((issue, source), (fallback_frequency, fallback_gain))
            frequency_row.append(normalize_frequency_hz(frequency_hz))
            gain_row.append(gain_db)

        frequency_rows.append(frequency_row)
        gain_rows.append(gain_row)

    return (
        torch.tensor(frequency_rows, dtype=torch.float32),
        torch.tensor(gain_rows, dtype=torch.float32),
    )


class HierarchicalEqProjection(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        issue_frequencies, issue_gains = build_issue_eq_tensors()
        pair_frequencies, pair_gains = build_pair_eq_tensors()
        self.register_buffer("issue_freq_norm", issue_frequencies)
        self.register_buffer("issue_gain_db", issue_gains)
        self.register_buffer("pair_freq_norm", pair_frequencies)
        self.register_buffer("pair_gain_db", pair_gains)

    def forward(
        self,
        issue_probs: torch.Tensor,
        source_probs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        normalized_issue_probs = issue_probs / issue_probs.sum(dim=-1, keepdim=True).clamp(min=EPSILON)
        normalized_source_probs = source_probs / source_probs.sum(dim=-1, keepdim=True).clamp(min=EPSILON)
        pair_weights = normalized_issue_probs.unsqueeze(-1) * normalized_source_probs.unsqueeze(-2)

        fallback_eq_freq = (normalized_issue_probs * self.issue_freq_norm.unsqueeze(0)).sum(dim=-1, keepdim=True)
        fallback_eq_gain = (normalized_issue_probs * self.issue_gain_db.unsqueeze(0)).sum(dim=-1, keepdim=True)
        pair_eq_freq = (pair_weights * self.pair_freq_norm.unsqueeze(0)).sum(dim=(-2, -1)).unsqueeze(-1)
        pair_eq_gain = (pair_weights * self.pair_gain_db.unsqueeze(0)).sum(dim=(-2, -1)).unsqueeze(-1)
        use_pair_projection = (source_probs.sum(dim=-1, keepdim=True) > EPSILON).to(issue_probs.dtype)

        eq_freq = pair_eq_freq * use_pair_projection + fallback_eq_freq * (1.0 - use_pair_projection)
        eq_gain = pair_eq_gain * use_pair_projection + fallback_eq_gain * (1.0 - use_pair_projection)
        return eq_freq.clamp(0.0, 1.0), eq_gain.clamp(-MAX_GAIN_DB, MAX_GAIN_DB)


class LegacyHierarchicalOutputAdapter(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        issue_frequencies, _ = build_issue_eq_tensors()
        issue_bandwidths = torch.tensor(
            [LEGACY_EQ_WIDTHS[issue] for issue in ISSUE_LABELS],
            dtype=torch.float32,
        )
        self.register_buffer("issue_freq_norm", issue_frequencies)
        self.register_buffer("issue_bandwidth_norm", issue_bandwidths)
        self.eq_projection = HierarchicalEqProjection()

    def forward(
        self,
        legacy_problem_probs: torch.Tensor,
        legacy_instrument_probs: torch.Tensor,
        legacy_eq_freq: torch.Tensor,
        legacy_eq_gain_db: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        muddy_problem = legacy_problem_probs[..., 0:1]
        harsh_problem = legacy_problem_probs[..., 1:2]
        buried_problem = legacy_problem_probs[..., 2:3]
        normal_problem = legacy_problem_probs[..., 3:4]

        source_probs = legacy_instrument_probs.clamp(0.0, 1.0)
        vocal = source_probs[..., 0:1]
        guitar = source_probs[..., 1:2]
        bass = source_probs[..., 2:3]
        drums = source_probs[..., 3:4]
        keys = source_probs[..., 4:5]

        vocal_or_guitar = torch.maximum(vocal, guitar)
        melodic_sources = torch.maximum(torch.maximum(vocal, guitar), keys)
        low_end_sources = torch.maximum(bass, drums)
        bright_sources = torch.maximum(torch.maximum(guitar, keys), drums)
        thin_sources = torch.maximum(bass, guitar)

        clamped_eq_freq = legacy_eq_freq.clamp(0.0, 1.0)
        clamped_eq_gain = legacy_eq_gain_db.clamp(-MAX_GAIN_DB, MAX_GAIN_DB)
        cut_strength = (-clamped_eq_gain / MAX_GAIN_DB).clamp(0.0, 1.0)
        boost_strength = (clamped_eq_gain / MAX_GAIN_DB).clamp(0.0, 1.0)
        freq_match = (
            1.0
            - (clamped_eq_freq - self.issue_freq_norm.unsqueeze(0)).abs()
            / self.issue_bandwidth_norm.unsqueeze(0)
        ).clamp(0.0, 1.0)

        muddy = (muddy_problem + 0.12 * cut_strength * freq_match[..., 0:1]).clamp(0.0, 1.0)
        harsh = (harsh_problem + 0.10 * cut_strength * freq_match[..., 1:2]).clamp(0.0, 1.0)
        buried = (buried_problem + 0.10 * boost_strength * freq_match[..., 2:3]).clamp(0.0, 1.0)
        boomy = (
            muddy_problem
            * (0.55 + 0.25 * low_end_sources)
            * cut_strength
            * freq_match[..., 3:4]
        ).clamp(0.0, 1.0)
        thin = (
            (0.45 * buried_problem + 0.15 * normal_problem)
            * (0.60 + 0.20 * thin_sources)
            * boost_strength
            * freq_match[..., 4:5]
        ).clamp(0.0, 1.0)
        boxy = (
            muddy_problem
            * (0.50 + 0.30 * melodic_sources)
            * cut_strength
            * freq_match[..., 5:6]
        ).clamp(0.0, 1.0)
        nasal = (
            (0.20 * harsh_problem + 0.25 * buried_problem)
            * (0.55 + 0.25 * vocal_or_guitar)
            * cut_strength
            * freq_match[..., 6:7]
        ).clamp(0.0, 1.0)
        sibilant = (
            harsh_problem
            * (0.55 + 0.35 * vocal)
            * cut_strength
            * freq_match[..., 7:8]
        ).clamp(0.0, 1.0)
        dull = (
            (0.55 * buried_problem + 0.15 * normal_problem)
            * (0.55 + 0.25 * bright_sources)
            * boost_strength
            * freq_match[..., 8:9]
        ).clamp(0.0, 1.0)

        issue_probs = torch.cat(
            [muddy, harsh, buried, boomy, thin, boxy, nasal, sibilant, dull],
            dim=-1,
        ).clamp(0.0, 1.0)
        eq_freq, eq_gain_db = self.eq_projection(issue_probs, source_probs)
        return issue_probs, source_probs, eq_freq, eq_gain_db


def rename_terminal_outputs_to_legacy_names(model: onnx.ModelProto) -> onnx.ModelProto:
    output_renames = dict(zip(LEGACY_OUTPUT_NAMES, LEGACY_INPUT_NAMES))

    for node in model.graph.node:
        node.output[:] = [output_renames.get(name, name) for name in node.output]

    for graph_output in model.graph.output:
        if graph_output.name in output_renames:
            graph_output.name = output_renames[graph_output.name]

    return model


def rename_graph_value_names(model: onnx.ModelProto, rename_map: dict[str, str]) -> onnx.ModelProto:
    for node in model.graph.node:
        node.input[:] = [rename_map.get(name, name) for name in node.input]
        node.output[:] = [rename_map.get(name, name) for name in node.output]

    for graph_input in model.graph.input:
        if graph_input.name in rename_map:
            graph_input.name = rename_map[graph_input.name]

    for graph_output in model.graph.output:
        if graph_output.name in rename_map:
            graph_output.name = rename_map[graph_output.name]

    return model


def export_legacy_adapter_to_onnx(output_path: Path, *, opset_version: int = 18) -> Path:
    adapter = LegacyHierarchicalOutputAdapter().eval().cpu()
    dummy_problem_probs = torch.tensor([[0.25, 0.25, 0.25, 0.25]], dtype=torch.float32)
    dummy_source_probs = torch.full((1, len(SOURCE_LABELS)), 0.2, dtype=torch.float32)
    dummy_eq_freq = torch.tensor([[0.5]], dtype=torch.float32)
    dummy_eq_gain_db = torch.tensor([[0.0]], dtype=torch.float32)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        adapter,
        (dummy_problem_probs, dummy_source_probs, dummy_eq_freq, dummy_eq_gain_db),
        output_path.as_posix(),
        export_params=True,
        do_constant_folding=True,
        opset_version=opset_version,
        dynamo=False,
        input_names=list(LEGACY_INPUT_NAMES),
        output_names=["issue_probs", "source_probs", "eq_freq", "eq_gain_db"],
        dynamic_axes={
            LEGACY_INPUT_NAMES[0]: {0: "batch_size"},
            LEGACY_INPUT_NAMES[1]: {0: "batch_size"},
            LEGACY_INPUT_NAMES[2]: {0: "batch_size"},
            LEGACY_INPUT_NAMES[3]: {0: "batch_size"},
            "issue_probs": {0: "batch_size"},
            "source_probs": {0: "batch_size"},
            "eq_freq": {0: "batch_size"},
            "eq_gain_db": {0: "batch_size"},
        },
    )

    return output_path


def adapt_legacy_browser_onnx_to_hierarchical_schema(
    *,
    legacy_onnx_path: Path,
    output_path: Path,
    opset_version: int = 18,
) -> Path:
    renamed_legacy_model = rename_terminal_outputs_to_legacy_names(
        onnx.load(legacy_onnx_path.as_posix(), load_external_data=True)
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        adapter_path = Path(temp_dir) / "legacy_adapter.onnx"
        export_legacy_adapter_to_onnx(adapter_path, opset_version=opset_version)
        adapter_model = compose.add_prefix(
            onnx.load(adapter_path.as_posix(), load_external_data=False),
            ADAPTER_PREFIX,
        )
        adapter_model.ir_version = renamed_legacy_model.ir_version
        merged_model = compose.merge_models(
            renamed_legacy_model,
            adapter_model,
            io_map=[(name, f"{ADAPTER_PREFIX}{name}") for name in LEGACY_INPUT_NAMES],
        )
        rename_graph_value_names(
            merged_model,
            {
                f"{ADAPTER_PREFIX}issue_probs": "issue_probs",
                f"{ADAPTER_PREFIX}source_probs": "source_probs",
                f"{ADAPTER_PREFIX}eq_freq": "eq_freq",
                f"{ADAPTER_PREFIX}eq_gain_db": "eq_gain_db",
            },
        )
        hierarchical_outputs = [
            output
            for output in merged_model.graph.output
            if output.name in {"issue_probs", "source_probs", "eq_freq", "eq_gain_db"}
        ]
        merged_model.graph.ClearField("output")
        merged_model.graph.output.extend(hierarchical_outputs)
        onnx.checker.check_model(merged_model)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        onnx.save_model(
            merged_model,
            output_path.as_posix(),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=f"{output_path.name}.data",
        )

    return output_path
