from __future__ import annotations

try:
    from .label_schema import ISSUE_LABELS, SOURCE_LABELS
    from .model import LightweightAudioAnalysisNet, ModelConfig
except ImportError:
    from label_schema import ISSUE_LABELS, SOURCE_LABELS
    from model import LightweightAudioAnalysisNet, ModelConfig

PROBLEM_LABELS = ISSUE_LABELS

__all__ = [
    "ISSUE_LABELS",
    "SOURCE_LABELS",
    "PROBLEM_LABELS",
    "LightweightAudioAnalysisNet",
    "ModelConfig",
]
