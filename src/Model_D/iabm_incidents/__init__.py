"""Public package surface for Model_D incident and episode analysis.

The package exports the typed configuration objects, builders, evaluators,
pipeline orchestrators, and registry helpers required to assemble longitudinal
incident-processing workflows from either semantic assignments or richer
indicator-driven inputs.
"""

from .ablation import (AblationConfig, AblationLevelResult, FourLevelAblationResult, FourLevelAblationRunner)
from .behavioral import BehavioralWindowEncoder
from .anticipation import AnticipationConfig, AnticipationResult, AnticipatoryMaintenanceForecaster
from .config import ModelDConfig
from .electrical import (ElectricalConfig, CommandResponseObserver, TransientConcentration,
                         TransientReference, VQEDirectionSolver, DominantDirectionSolver,
                         causal_delay, fourth_scatter)
from .episodes import CandidateSegment, IncidentEpisodeBuilder
from .experimental import ExperimentResult, ModelDExperimentRunner
from .evaluation import EpisodeEvaluator
from .exposure import ExposureConfig, derive_observation_periods
from .indicators import IndicatorBaseline, compute_window_indicators, fit_indicator_baseline
from .interpretation import (EventFrequencyResponseInterpreter, EventFrequencyInterpretationResult,
                             FrequencyInterpretationConfig, residual_delay_taps,
                             residual_frequency_response)
from .multiscale import (MultiscaleSensitivityConfig, MultiscaleSensitivityResult, MultiscaleSensitivityRunner)
from .occurrence import OccurrenceModeler
from .pipeline import IndicatorPipeline, IndicatorPipelineResult
from .registry import IncidentRecord, IncidentRegistry
from .registry_builder import CandidateIncidentRegistryBuilder, RegistryGenerationConfig
from .scoring import DeviationScorer, WeightedDeviationScorer
from .taxonomy import DEFAULT_INCIDENT_TAXONOMY, FAMILY_SIGNATURES, FamilySignature
from .windows import WindowBuildConfig, build_rolling_windows

__all__ = [
    "MultiscaleSensitivityConfig", "MultiscaleSensitivityResult", "MultiscaleSensitivityRunner",
    "AblationConfig", "AblationLevelResult", "FourLevelAblationResult",
    "FourLevelAblationRunner", "BehavioralWindowEncoder",
    "EventFrequencyResponseInterpreter", "EventFrequencyInterpretationResult",
    "FrequencyInterpretationConfig", "residual_delay_taps", "residual_frequency_response",
    "ElectricalConfig", "CommandResponseObserver", "TransientConcentration",
    "TransientReference", "VQEDirectionSolver", "DominantDirectionSolver",
    "causal_delay", "fourth_scatter",
    "AnticipationConfig",
    "AnticipationResult",
    "AnticipatoryMaintenanceForecaster",
    "CandidateSegment",
    "compute_window_indicators",
    "DEFAULT_INCIDENT_TAXONOMY",
    "derive_observation_periods",
    "ExperimentResult",
    "DeviationScorer",
    "EpisodeEvaluator",
    "ExposureConfig",
    "FAMILY_SIGNATURES",
    "FamilySignature",
    "fit_indicator_baseline",
    "IndicatorBaseline",
    "ModelDExperimentRunner",
    "IndicatorPipeline",
    "IndicatorPipelineResult",
    "CandidateIncidentRegistryBuilder",
    "IncidentEpisodeBuilder",
    "IncidentRecord",
    "RegistryGenerationConfig",
    "IncidentRegistry",
    "ModelDConfig",
    "OccurrenceModeler",
    "build_rolling_windows",
    "WeightedDeviationScorer",
    "WindowBuildConfig",
]
