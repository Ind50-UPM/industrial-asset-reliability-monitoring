"""Configuration objects for Model_D processing.

Model_D combines window construction, deviation scoring, temporal segmentation,
family assignment, and occurrence summarization. This module centralizes the
configuration contract so CLIs, experiments, and tests can share the same typed
settings surface.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class DeviationWeights:
    """Weights used to aggregate window-level deviation indicators.

    Attributes:
        sequence: Weight assigned to sequence-divergence evidence.
        duration: Weight assigned to duration-drift evidence.
        recurrence: Weight assigned to recurrence-based evidence.
        persistence: Weight assigned to persistence-based evidence.
        consumption: Weight assigned to analogue consumption deviation.
        state_error: Weight assigned to semantic or state error signals.
        mode: Weight assigned to operating-mode divergence.
        command_response: Weight assigned to command-response residual norms.
        transient_concentration: Weight assigned to fourth-order excess evidence.
    """

    sequence: float = 1.0
    duration: float = 1.0
    recurrence: float = 1.0
    persistence: float = 1.0
    consumption: float = 1.0
    state_error: float = 1.0
    mode: float = 1.0
    command_response: float = 1.0
    transient_concentration: float = 1.0


@dataclass(frozen=True)
class WindowConfig:
    """Definition of the rolling analysis window policy."""

    length: pd.Timedelta = pd.Timedelta(hours=24)
    step: pd.Timedelta = pd.Timedelta(hours=1)
    min_coverage: float = 0.90
    min_active_sequences: int = 1


@dataclass(frozen=True)
class EpisodeDetectionConfig:
    """Parameters governing onset persistence and recovery segmentation."""

    onset_threshold: float = 1.0
    recovery_threshold: float = 0.5
    onset_windows: int = 2
    recovery_windows: int = 3
    minimum_duration: pd.Timedelta = pd.Timedelta(minutes=10)
    maximum_gap: pd.Timedelta = pd.Timedelta(hours=2)
    maximum_duration: pd.Timedelta = pd.Timedelta(days=7)


@dataclass(frozen=True)
class FamilyAssignmentConfig:
    """Configuration for the explainable family-assignment stage."""

    method: str = "rule_based"
    minimum_confidence: float = 0.60


@dataclass(frozen=True)
class OccurrenceConfig:
    """Configuration for family-level occurrence modelling outputs."""

    minimum_events_for_fit: int = 5


@dataclass(frozen=True)
class ModelDConfig:
    """Top-level configuration for the Model_D pipeline.

    Attributes:
        window: Rolling-window construction policy.
        detection: Episode-onset and recovery segmentation policy.
        weights: Aggregation weights for deviation indicators.
        family_assignment: Rule-based family-assignment settings.
        occurrence: Occurrence-summary settings.
    """

    window: WindowConfig = WindowConfig()
    detection: EpisodeDetectionConfig = EpisodeDetectionConfig()
    weights: DeviationWeights = DeviationWeights()
    family_assignment: FamilyAssignmentConfig = FamilyAssignmentConfig()
    occurrence: OccurrenceConfig = OccurrenceConfig()

    def to_dict(self) -> dict[str, Any]:
        """Serialize the configuration into JSON-friendly values.

        Returns:
            Nested dictionary ready to be written into metadata artefacts.

        Notes:
            Timedelta objects are converted downstream by ``_serialize_dataclass``
            so metadata remains portable across command-line runs and notebooks.
        """
        return {
            "window": _serialize_dataclass(self.window),
            "detection": _serialize_dataclass(self.detection),
            "weights": _serialize_dataclass(self.weights),
            "family_assignment": _serialize_dataclass(self.family_assignment),
            "occurrence": _serialize_dataclass(self.occurrence),
        }


def load_model_d_config(file_path: str | Path | None) -> ModelDConfig:
    """Load a Model_D configuration file when provided.

    Args:
        file_path: Optional JSON configuration path.

    Returns:
        A fully typed ``ModelDConfig`` instance.

    Notes:
        The loader keeps backward compatibility with both ``detection`` and the
        older ``episode_detection`` key so historical configuration files remain
        valid after refactors.
    """
    if file_path is None:
        return ModelDConfig()

    path = Path(file_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    return ModelDConfig(
        window=_load_window_config(payload.get("window", {})),
        detection=_load_detection_config(payload.get("episode_detection", payload.get("detection", {}))),
        weights=_load_weights_config(payload.get("weights", {})),
        family_assignment=_load_family_assignment_config(payload.get("family_assignment", {})),
        occurrence=_load_occurrence_config(payload.get("occurrence", {})),
    )


def _load_window_config(payload: dict[str, Any]) -> WindowConfig:
    """Hydrate ``WindowConfig`` from a JSON payload fragment."""
    return WindowConfig(
        length=pd.Timedelta(hours=payload.get("length_hours", 24)),
        step=pd.Timedelta(hours=payload.get("step_hours", 1)),
        min_coverage=float(payload.get("min_coverage", 0.90)),
        min_active_sequences=int(payload.get("min_active_sequences", 1)),
    )


def _load_detection_config(payload: dict[str, Any]) -> EpisodeDetectionConfig:
    """Hydrate ``EpisodeDetectionConfig`` from a JSON payload fragment."""
    return EpisodeDetectionConfig(
        onset_threshold=float(payload.get("onset_threshold", payload.get("onset_quantile", 1.0))),
        recovery_threshold=float(payload.get("recovery_threshold", payload.get("recovery_quantile", 0.5))),
        onset_windows=int(payload.get("onset_windows", 2)),
        recovery_windows=int(payload.get("recovery_windows", 3)),
        minimum_duration=pd.Timedelta(hours=payload.get("minimum_duration_hours", 0.5)),
        maximum_gap=pd.Timedelta(hours=payload.get("maximum_gap_hours", 2)),
        maximum_duration=pd.Timedelta(hours=payload.get("maximum_duration_hours", 24 * 7)),
    )


def _load_weights_config(payload: dict[str, Any]) -> DeviationWeights:
    """Hydrate ``DeviationWeights`` from a JSON payload fragment."""
    return DeviationWeights(
        sequence=float(payload.get("sequence", 1.0)),
        duration=float(payload.get("duration", 1.0)),
        recurrence=float(payload.get("recurrence", 1.0)),
        persistence=float(payload.get("persistence", 1.0)),
        consumption=float(payload.get("consumption", 1.0)),
        state_error=float(payload.get("state_error", 1.0)),
        mode=float(payload.get("mode", 1.0)),
        command_response=float(payload.get("command_response", 1.0)),
        transient_concentration=float(payload.get("transient_concentration", 1.0)),
    )


def _load_family_assignment_config(payload: dict[str, Any]) -> FamilyAssignmentConfig:
    """Hydrate ``FamilyAssignmentConfig`` from a JSON payload fragment."""
    return FamilyAssignmentConfig(
        method=str(payload.get("method", "rule_based")),
        minimum_confidence=float(payload.get("minimum_confidence", 0.60)),
    )


def _load_occurrence_config(payload: dict[str, Any]) -> OccurrenceConfig:
    """Hydrate ``OccurrenceConfig`` from a JSON payload fragment."""
    return OccurrenceConfig(
        minimum_events_for_fit=int(payload.get("minimum_events_for_fit", 5)),
    )


def _serialize_dataclass(instance: Any) -> dict[str, Any]:
    """Convert dataclass values into JSON-friendly primitives.

    Args:
        instance: Dataclass instance to serialize.

    Returns:
        Dictionary containing only JSON-friendly primitive values.
    """
    serialized: dict[str, Any] = {}
    for key, value in asdict(instance).items():
        if isinstance(value, pd.Timedelta):
            serialized[key] = value.isoformat()
        else:
            serialized[key] = value
    return serialized
