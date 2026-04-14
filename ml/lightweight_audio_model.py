from __future__ import annotations

try:
    from .label_schema import ISSUE_LABELS, SOURCE_LABELS
    from .model import (
        AudioIntelligenceNet,
        LightweightAudioAnalysisNet,
        ModelConfig,
        ProductionAudioIntelligenceNet,
    )
except ImportError:
    from label_schema import ISSUE_LABELS, SOURCE_LABELS
    from model import (
        AudioIntelligenceNet,
        LightweightAudioAnalysisNet,
        ModelConfig,
        ProductionAudioIntelligenceNet,
    )

PROBLEM_LABELS = ISSUE_LABELS

__all__ = [
    "ISSUE_LABELS",
    "SOURCE_LABELS",
    "PROBLEM_LABELS",
    "AudioIntelligenceNet",
    "ProductionAudioIntelligenceNet",
    "LightweightAudioAnalysisNet",
    "ModelConfig",
]
