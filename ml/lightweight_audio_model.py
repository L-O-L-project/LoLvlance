from __future__ import annotations

try:
    from .model import LightweightAudioAnalysisNet, ModelConfig, PROBLEM_LABELS
except ImportError:
    from model import LightweightAudioAnalysisNet, ModelConfig, PROBLEM_LABELS

__all__ = ["LightweightAudioAnalysisNet", "ModelConfig", "PROBLEM_LABELS"]
