"""Deviation scoring utilities for Model_D."""

from __future__ import annotations

from dataclasses import dataclass
import ast
import math

import numpy as np
import pandas as pd

from .config import DeviationWeights, WindowConfig


INDICATOR_COLUMNS = {
    "sequence_divergence": "sequence",
    "duration_drift": "duration",
    "recurrence_excess": "recurrence",
    "persistence_excess": "persistence",
    "consumption_deviation": "consumption",
    "state_error_rate": "state_error",
    "mode_divergence": "mode",
    "command_response_deviation": "command_response",
    "transient_concentration": "transient_concentration",
}
PREWINDOWED_COLUMNS = {"window_start", "window_end", "data_coverage", "sequence_count", "active_sequence_count"}
TIME_COLUMN_ALIASES = {
    "window_start": "start_time",
    "window_end": "end_time",
}
COUNT_COLUMN_ALIASES = {
    "active_sequence_count": "run_count",
}


@dataclass(frozen=True)
class WindowIndicators:
    """Container describing a single window-level anomaly profile."""

    start_time: pd.Timestamp
    end_time: pd.Timestamp
    sequence_divergence: float
    duration_drift: float
    recurrence_excess: float
    persistence_excess: float
    consumption_deviation: float
    state_error_rate: float
    mode_divergence: float
    command_response_deviation: float = 0.0
    transient_concentration: float = 0.0


def compute_deviation_score(frame: pd.DataFrame, weights: DeviationWeights) -> pd.Series:
    """Compute a composite deviation score from available indicators."""

    if any(column in frame.columns for column in INDICATOR_COLUMNS):
        score = pd.Series(0.0, index=frame.index, dtype=float)
        for column, weight_name in INDICATOR_COLUMNS.items():
            values = pd.to_numeric(
                frame[column] if column in frame.columns else pd.Series(0.0, index=frame.index),
                errors="coerce",
            ).fillna(0.0)
            score = score + getattr(weights, weight_name) * values
        return score

    if "semantic_status" in frame.columns:
        semantic_score = frame["semantic_status"].eq("ANOMALOUS").astype(float)
        fallback_score = pd.Series(0.0, index=frame.index, dtype=float)
        if "word_regime_shift_score" in frame.columns:
            fallback_score = frame.apply(
                lambda row: float(_infer_fallback_family(row.to_dict()) is not None),
                axis=1,
            )
        return pd.concat([semantic_score, fallback_score], axis=1).max(axis=1)

    raise ValueError(
        "Unable to compute deviation score: expected semantic_status or at least one deviation indicator column."
    )


def build_window_scores(
    sequences: pd.DataFrame,
    assignments: pd.DataFrame,
    weights: DeviationWeights,
    window_config: WindowConfig,
) -> pd.DataFrame:
    """Build scored windows from raw timed rows or accept pre-windowed inputs."""

    sequence_frame = _normalize_window_input(sequences).reset_index(drop=True).copy()
    assignment_frame = _normalize_window_input(assignments).reset_index(drop=True).copy()

    if _is_prewindowed(sequence_frame):
        merged = sequence_frame.copy()
        for column in assignment_frame.columns:
            if column not in merged.columns:
                merged[column] = assignment_frame[column]
        return _finalize_window_frame(merged, weights)

    overlapping = [column for column in assignment_frame.columns if column in sequence_frame.columns]
    protected = {'semantic_status', 'incident_family', 'anomaly_score'}
    drop_from_assignment = [column for column in overlapping if column not in protected]
    if drop_from_assignment:
        assignment_frame = assignment_frame.drop(columns=drop_from_assignment)

    merged = pd.concat([sequence_frame, assignment_frame], axis=1)
    merged = merged.loc[:, ~merged.columns.duplicated()].copy()
    return _build_rolling_windows(merged, weights, window_config)


def _build_rolling_windows(
    merged: pd.DataFrame,
    weights: DeviationWeights,
    window_config: WindowConfig,
) -> pd.DataFrame:
    """Construct explicit rolling windows from timed sequence rows."""

    frame = merged.loc[:, ~merged.columns.duplicated()].copy()
    frame["start_time"] = pd.to_datetime(frame["start_time"], errors="coerce", utc=True).dt.tz_convert(None)
    frame["end_time"] = pd.to_datetime(frame["end_time"], errors="coerce", utc=True).dt.tz_convert(None)
    frame = frame.dropna(subset=["start_time", "end_time"]).reset_index(drop=True)
    if frame.empty:
        return _finalize_window_frame(frame, weights)

    baseline_profiles = _fit_behavioral_baseline(frame)

    if not any(column in frame.columns for column in INDICATOR_COLUMNS):
        return _finalize_window_frame(_build_semantic_windows_fast(frame, window_config, baseline_profiles), weights)

    min_start = frame["start_time"].min().floor(window_config.step)
    max_end = frame["end_time"].max()
    rows: list[dict[str, object]] = []
    current_start = min_start
    while current_start <= max_end:
        current_end = current_start + window_config.length
        window_rows = frame[(frame["start_time"] < current_end) & (frame["end_time"] > current_start)]
        window_payload = _summarize_window(window_rows, current_start, current_end, window_config, baseline_profiles)
        if window_payload is not None:
            rows.append(window_payload)
        current_start = current_start + window_config.step

    return _finalize_window_frame(pd.DataFrame(rows), weights)


def _build_semantic_windows_fast(
    frame: pd.DataFrame,
    window_config: WindowConfig,
    baseline_profiles: dict[str, object],
) -> pd.DataFrame:
    """Build semantic-only windows using array operations instead of DataFrame slicing."""

    if frame.empty:
        return pd.DataFrame()

    starts = frame["start_time"].to_numpy(dtype="datetime64[ns]")
    ends = frame["end_time"].to_numpy(dtype="datetime64[ns]")
    semantic_values = (
        frame["semantic_status"].fillna("NORMAL").astype(str).to_numpy()
        if "semantic_status" in frame.columns
        else np.full(len(frame), "NORMAL", dtype=object)
    )
    family_values = (
        frame["incident_family"].fillna("unclassified_incident").astype(str).to_numpy()
        if "incident_family" in frame.columns
        else np.full(len(frame), "unclassified_incident", dtype=object)
    )
    asset_values = (
        frame["asset_id"].fillna("asset-unknown").astype(str).to_numpy()
        if "asset_id" in frame.columns
        else np.full(len(frame), "asset-unknown", dtype=object)
    )

    starts_ns = starts.astype("int64")
    ends_ns = ends.astype("int64")
    length_seconds = window_config.length.total_seconds()
    step = window_config.step
    min_start = frame["start_time"].min().floor(step)
    max_end = frame["end_time"].max()

    rows: list[dict[str, object]] = []
    current_start = min_start
    while current_start <= max_end:
        current_end = current_start + window_config.length
        current_start_ns = current_start.to_datetime64().astype("datetime64[ns]").astype("int64")
        current_end_ns = current_end.to_datetime64().astype("datetime64[ns]").astype("int64")
        mask = (starts_ns < current_end_ns) & (ends_ns > current_start_ns)
        if not bool(mask.any()):
            current_start = current_start + step
            continue

        overlap_start = np.maximum(starts_ns[mask], current_start_ns)
        overlap_end = np.minimum(ends_ns[mask], current_end_ns)
        overlap_seconds = np.maximum((overlap_end - overlap_start) / 1_000_000_000.0, 0.0)
        active_mask = overlap_seconds > 0.0
        if not bool(active_mask.any()):
            current_start = current_start + step
            continue

        overlap_seconds = overlap_seconds[active_mask]
        coverage = min(float(overlap_seconds.sum()) / length_seconds, 1.0)
        if coverage < window_config.min_coverage:
            current_start = current_start + step
            continue
        active_count = int(active_mask.sum())
        if active_count < window_config.min_active_sequences:
            current_start = current_start + step
            continue

        active_semantic = semantic_values[mask][active_mask]
        active_family = family_values[mask][active_mask]
        anomalous_mask = active_semantic == "ANOMALOUS"
        family_source = active_family[anomalous_mask] if anomalous_mask.any() else active_family
        if family_source.size:
            family_mode = pd.Series(family_source).mode(dropna=True)
            dominant_family = str(family_mode.iloc[0]) if not family_mode.empty else "unclassified_incident"
        else:
            dominant_family = "unclassified_incident"

        semantic_window = {
                "asset_id": str(asset_values[mask][active_mask][0]) if active_count else None,
                "start_time": current_start,
                "end_time": current_end,
                "data_coverage": coverage,
                "sequence_count": active_count,
                "active_sequence_count": active_count,
                "semantic_status": "ANOMALOUS" if anomalous_mask.any() else "NORMAL",
                "incident_family": dominant_family,
            }
        active_rows = frame.loc[mask].iloc[np.flatnonzero(active_mask)].copy()
        word_summary = _summarize_state_words(active_rows, baseline_profiles)
        semantic_window.update(word_summary)
        fallback_family = _infer_fallback_family(semantic_window)
        if semantic_window["semantic_status"] == "NORMAL" and fallback_family is not None:
            semantic_window["semantic_status"] = "ANOMALOUS"
            semantic_window["incident_family"] = fallback_family
        rows.append(semantic_window)
        current_start = current_start + step

    return pd.DataFrame(rows)


def _summarize_window(
    window_rows: pd.DataFrame,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    window_config: WindowConfig,
    baseline_profiles: dict[str, object],
) -> dict[str, object] | None:
    """Summarize a single rolling window."""

    if window_rows.empty:
        return None

    start_clip = window_rows["start_time"].where(window_rows["start_time"] > window_start, window_start)
    end_clip = window_rows["end_time"].where(window_rows["end_time"] < window_end, window_end)
    overlap_seconds = (end_clip - start_clip).dt.total_seconds().clip(lower=0.0)
    active_mask = overlap_seconds.gt(0)
    if not bool(active_mask.any()):
        return None

    active_rows = window_rows.loc[active_mask]
    overlap_seconds = overlap_seconds.loc[active_mask]
    coverage = min(float(overlap_seconds.sum()) / window_config.length.total_seconds(), 1.0)
    if coverage < window_config.min_coverage:
        return None
    if len(active_rows) < window_config.min_active_sequences:
        return None

    weighted = {
        column: _weighted_mean(active_rows[column], overlap_seconds)
        for column in INDICATOR_COLUMNS
        if column in active_rows.columns
    }

    # Precomputed transient subwindows retain their strongest concentration;
    # command-response magnitudes retain duration-weighted aggregation.
    if "transient_concentration" in active_rows:
        weighted["transient_concentration"] = float(pd.to_numeric(active_rows["transient_concentration"], errors="coerce").max())

    if "semantic_status" in active_rows.columns:
        semantic_series = active_rows["semantic_status"].fillna("NORMAL")
    else:
        semantic_series = pd.Series("NORMAL", index=active_rows.index)
    is_anomalous = semantic_series.eq("ANOMALOUS")

    dominant_family = None
    if "incident_family" in active_rows.columns:
        family_source = active_rows.loc[is_anomalous, "incident_family"] if bool(is_anomalous.any()) else active_rows["incident_family"]
        family_mode = family_source.dropna().mode()
        dominant_family = family_mode.iloc[0] if not family_mode.empty else None

    word_summary = _summarize_state_words(active_rows, baseline_profiles)
    semantic_status = "ANOMALOUS" if bool(is_anomalous.any()) else "NORMAL"
    incident_family = dominant_family or "unclassified_incident"
    fallback_context = {
        **weighted,
        **word_summary,
        "semantic_status": semantic_status,
        "incident_family": incident_family,
    }
    fallback_family = _infer_fallback_family(fallback_context)
    if semantic_status == "NORMAL" and fallback_family is not None:
        semantic_status = "ANOMALOUS"
        incident_family = fallback_family

    return {
        "asset_id": active_rows["asset_id"].iloc[0] if "asset_id" in active_rows.columns else None,
        "start_time": window_start,
        "end_time": window_end,
        "data_coverage": coverage,
        "sequence_count": int(len(active_rows)),
        "active_sequence_count": int(len(active_rows)),
        "semantic_status": semantic_status,
        "incident_family": incident_family,
        **weighted,
        **word_summary,
    }


def _finalize_window_frame(frame: pd.DataFrame, weights: DeviationWeights) -> pd.DataFrame:
    """Normalize and score a window frame regardless of its source."""

    if frame.empty:
        return pd.DataFrame(
            columns=[
                "asset_id",
                "start_time",
                "end_time",
                "data_coverage",
                "sequence_count",
                "active_sequence_count",
                "semantic_status",
                "incident_family",
                *INDICATOR_COLUMNS.keys(),
                "dominant_state_word",
                "state_word_diversity",
                "dominant_state_word_fraction",
                "state_word_transition_rate",
                "nominal_state_word_match_fraction",
                "mean_state_distance",
                "mean_dtw_distance",
                "mean_nominal_anomaly_score",
                "rare_word_fraction",
                "rare_state_fraction",
                "state_entropy",
                "state_17_fraction",
                "off_nominal_state_fraction",
                "word_regime_shift_score",
                "deviation_score",
            ]
        )

    window_frame = frame.copy()
    window_frame["start_time"] = pd.to_datetime(window_frame["start_time"])
    window_frame["end_time"] = pd.to_datetime(window_frame["end_time"])
    data_coverage = window_frame["data_coverage"] if "data_coverage" in window_frame.columns else pd.Series(1.0, index=window_frame.index)
    active_sequence_count = window_frame["run_count"] if "run_count" in window_frame.columns else pd.Series(1, index=window_frame.index)
    sequence_count = window_frame["sequence_count"] if "sequence_count" in window_frame.columns else pd.Series(1, index=window_frame.index)
    window_frame["data_coverage"] = pd.to_numeric(data_coverage, errors="coerce").fillna(1.0)
    window_frame["sequence_count"] = pd.to_numeric(sequence_count, errors="coerce").fillna(1).astype(int)
    window_frame["active_sequence_count"] = pd.to_numeric(active_sequence_count, errors="coerce").fillna(1).astype(int)
    window_frame["deviation_score"] = compute_deviation_score(window_frame, weights)
    return window_frame.sort_values("start_time").reset_index(drop=True)


def _normalize_window_input(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize supported window input column conventions."""

    normalized = frame.copy()
    for source, target in TIME_COLUMN_ALIASES.items():
        if source in normalized.columns and target not in normalized.columns:
            normalized[target] = normalized[source]
    for source, target in COUNT_COLUMN_ALIASES.items():
        if source in normalized.columns and target not in normalized.columns:
            normalized[target] = normalized[source]
    return normalized


def _is_prewindowed(frame: pd.DataFrame) -> bool:
    """Return whether the input already represents analysis windows."""

    return bool(PREWINDOWED_COLUMNS.intersection(frame.columns))


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    """Compute a weighted mean with safe fallbacks for empty or invalid data."""

    numeric_values = pd.to_numeric(values, errors="coerce")
    numeric_weights = pd.to_numeric(weights, errors="coerce").fillna(0.0)
    mask = numeric_values.notna() & numeric_weights.gt(0)
    if not bool(mask.any()):
        return 0.0
    return float((numeric_values[mask] * numeric_weights[mask]).sum() / numeric_weights[mask].sum())




def _fit_behavioral_baseline(frame: pd.DataFrame) -> dict[str, object]:
    profiles = {
        "global": _fit_single_behavioral_profile(frame),
        "by_regime": {},
    }
    if frame.empty:
        return profiles

    enriched = frame.copy()
    enriched["_baseline_regime"] = _infer_baseline_regime(enriched)
    regime_profiles: dict[str, dict[str, object]] = {}
    for regime, regime_frame in enriched.groupby("_baseline_regime", dropna=False, sort=False):
        if len(regime_frame) < 2:
            continue
        regime_profiles[str(regime)] = _fit_single_behavioral_profile(regime_frame)
    profiles["by_regime"] = regime_profiles
    return profiles


def _fit_single_behavioral_profile(frame: pd.DataFrame) -> dict[str, object]:
    state_words = frame["states"].dropna().astype(str) if "states" in frame.columns else pd.Series(dtype=str)
    if state_words.empty:
        return {
            "word_frequency": {},
            "state_frequency": {},
            "rare_word_threshold": 0.0,
            "rare_state_threshold": 0.0,
        }

    word_counts = state_words.value_counts(dropna=True)
    word_frequency = (word_counts / max(int(word_counts.sum()), 1)).to_dict()
    state_counts: dict[int, int] = {}
    for word in state_words:
        for state in _parse_state_word(word):
            state_counts[state] = state_counts.get(state, 0) + 1
    total_states = max(sum(state_counts.values()), 1)
    state_frequency = {state: count / total_states for state, count in state_counts.items()}
    rare_word_threshold = float((word_counts / max(int(word_counts.sum()), 1)).quantile(0.2)) if not word_counts.empty else 0.0
    rare_state_threshold = float(pd.Series(state_frequency).quantile(0.2)) if state_frequency else 0.0
    return {
        "word_frequency": word_frequency,
        "state_frequency": state_frequency,
        "rare_word_threshold": rare_word_threshold,
        "rare_state_threshold": rare_state_threshold,
    }


def _summarize_state_words(active_rows: pd.DataFrame, baseline_profiles: dict[str, object]) -> dict[str, object]:
    """Summarize the local sequence-word regime inside one window."""

    default = {
        "dominant_state_word": "",
        "state_word_diversity": 0.0,
        "dominant_state_word_fraction": 0.0,
        "state_word_transition_rate": 0.0,
        "nominal_state_word_match_fraction": 0.0,
        "mean_state_distance": 0.0,
        "mean_dtw_distance": 0.0,
        "mean_nominal_anomaly_score": 0.0,
        "rare_word_fraction": 0.0,
        "rare_state_fraction": 0.0,
        "state_entropy": 0.0,
        "state_17_fraction": 0.0,
        "off_nominal_state_fraction": 0.0,
        "word_regime_shift_score": 0.0,
    }
    if "states" not in active_rows.columns:
        return default

    state_words = active_rows["states"].dropna().astype(str)
    if state_words.empty:
        return default

    baseline_profile = _resolve_baseline_profile(active_rows, baseline_profiles)

    counts = state_words.value_counts(dropna=True)
    dominant_word = str(counts.index[0]) if not counts.empty else ""
    total = int(counts.sum())
    transitions = int((state_words != state_words.shift(1)).sum() - 1) if len(state_words) > 1 else 0
    transition_rate = float(transitions / max(len(state_words) - 1, 1))
    nominal_match_fraction = _nominal_match_fraction(active_rows)
    rare_word_fraction = _rare_word_fraction(state_words, baseline_profile)
    state_tokens = [state for word in state_words for state in _parse_state_word(word)]
    rare_state_fraction = _rare_state_fraction(state_tokens, baseline_profile)
    state_entropy = _state_entropy(state_tokens)
    state_17_fraction = float(sum(1 for state in state_tokens if state == 17) / max(len(state_tokens), 1))
    off_nominal_state_fraction = _off_nominal_state_fraction(active_rows)
    word_regime_shift_score = _word_regime_shift_score(state_words, baseline_profile)
    payload = default.copy()
    payload.update({
        "dominant_state_word": dominant_word,
        "state_word_diversity": float(state_words.nunique(dropna=True)),
        "dominant_state_word_fraction": float(counts.iloc[0] / total) if total else 0.0,
        "state_word_transition_rate": transition_rate,
        "nominal_state_word_match_fraction": nominal_match_fraction,
        "mean_state_distance": _mean_numeric(active_rows, "state_distance"),
        "mean_dtw_distance": _mean_numeric(active_rows, "dtw_distance"),
        "mean_nominal_anomaly_score": _mean_numeric(active_rows, "anomaly_score"),
        "rare_word_fraction": rare_word_fraction,
        "rare_state_fraction": rare_state_fraction,
        "state_entropy": state_entropy,
        "state_17_fraction": state_17_fraction,
        "off_nominal_state_fraction": off_nominal_state_fraction,
        "word_regime_shift_score": word_regime_shift_score,
    })
    return payload




def _infer_fallback_family(window_summary: dict[str, object]) -> str | None:
    shift = float(window_summary.get("word_regime_shift_score", 0.0) or 0.0)
    dominant_fraction = float(window_summary.get("dominant_state_word_fraction", 0.0) or 0.0)
    rare_word = float(window_summary.get("rare_word_fraction", 0.0) or 0.0)
    rare_state = float(window_summary.get("rare_state_fraction", 0.0) or 0.0)
    state17 = float(window_summary.get("state_17_fraction", 0.0) or 0.0)
    off_nominal = float(window_summary.get("off_nominal_state_fraction", 0.0) or 0.0)
    nominal_match = float(window_summary.get("nominal_state_word_match_fraction", 0.0) or 0.0)
    transition_rate = float(window_summary.get("state_word_transition_rate", 0.0) or 0.0)
    diversity = float(window_summary.get("state_word_diversity", 0.0) or 0.0)

    if not _has_localized_stable_shift_signature(
        shift=shift,
        dominant_fraction=dominant_fraction,
        rare_word=rare_word,
        rare_state=rare_state,
        transition_rate=transition_rate,
        diversity=diversity,
    ):
        return None
    if state17 > 0.0 or off_nominal >= 0.2 or nominal_match <= 0.35 or transition_rate <= 0.35:
        return "process_saturation"
    if rare_state >= 0.2 or rare_word >= 0.5:
        return "external_ambiguous_disturbance"
    return "external_ambiguous_disturbance"


def _has_localized_stable_shift_signature(
    *,
    shift: float,
    dominant_fraction: float,
    rare_word: float,
    rare_state: float,
    transition_rate: float,
    diversity: float,
) -> bool:
    return bool(
        shift >= 0.995
        and dominant_fraction >= 0.999
        and rare_word <= 0.05
        and 0.10 <= rare_state <= 0.20
        and transition_rate <= 0.05
        and diversity <= 1.5
    )


def _resolve_baseline_profile(active_rows: pd.DataFrame, baseline_profiles: dict[str, object]) -> dict[str, object]:
    global_profile = baseline_profiles.get("global", baseline_profiles)
    by_regime = baseline_profiles.get("by_regime", {})
    if active_rows.empty or not by_regime:
        return global_profile
    regime = _infer_baseline_regime(active_rows).mode(dropna=True)
    if regime.empty:
        return global_profile
    return by_regime.get(str(regime.iloc[0]), global_profile)


def _infer_baseline_regime(frame: pd.DataFrame) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=str)
    if "start_time" in frame.columns:
        timestamps = pd.to_datetime(frame["start_time"], errors="coerce")
    else:
        timestamps = pd.Series(pd.NaT, index=frame.index)
    hours = timestamps.dt.hour.fillna(12).astype(int)
    day_phase = pd.Series("day", index=frame.index, dtype=str)
    day_phase.loc[(hours < 7) | (hours >= 22)] = "night"
    day_phase.loc[(hours >= 7) & (hours < 10)] = "morning"
    day_phase.loc[(hours >= 18) & (hours < 22)] = "evening"

    if "run_count" in frame.columns:
        activity_values = pd.to_numeric(frame["run_count"], errors="coerce")
    elif "sequence_count" in frame.columns:
        activity_values = pd.to_numeric(frame["sequence_count"], errors="coerce")
    else:
        activity_values = pd.Series(1.0, index=frame.index)
    activity_values = activity_values.fillna(activity_values.median() if activity_values.notna().any() else 1.0)
    median_activity = float(activity_values.median()) if not activity_values.empty else 1.0
    high_activity = activity_values > max(median_activity, 1.0)
    activity_band = pd.Series("low_activity", index=frame.index, dtype=str)
    activity_band.loc[high_activity] = "high_activity"

    return day_phase.str.cat(activity_band, sep="__")

def _parse_state_word(value: object) -> tuple[int, ...]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return tuple()
    if isinstance(value, tuple):
        return tuple(int(item) for item in value)
    if isinstance(value, list):
        return tuple(int(item) for item in value)
    text = str(value).strip()
    if not text:
        return tuple()
    try:
        parsed = ast.literal_eval(text)
    except Exception:
        return tuple()
    if isinstance(parsed, (list, tuple)):
        return tuple(int(item) for item in parsed)
    return tuple()


def _nominal_match_fraction(active_rows: pd.DataFrame) -> float:
    if "states" not in active_rows.columns or "nominal_match" not in active_rows.columns or active_rows.empty:
        return 0.0
    return float(active_rows["states"].astype(str).eq(active_rows["nominal_match"].astype(str)).mean())


def _off_nominal_state_fraction(active_rows: pd.DataFrame) -> float:
    if "states" not in active_rows.columns or "nominal_match" not in active_rows.columns or active_rows.empty:
        return 0.0
    fractions: list[float] = []
    for _, row in active_rows.iterrows():
        observed = _parse_state_word(row.get("states"))
        nominal = _parse_state_word(row.get("nominal_match"))
        if not observed:
            continue
        mismatches = sum(1 for idx, state in enumerate(observed) if idx >= len(nominal) or nominal[idx] != state)
        mismatches += max(len(nominal) - len(observed), 0)
        fractions.append(float(mismatches / max(len(observed), len(nominal), 1)))
    return float(sum(fractions) / len(fractions)) if fractions else 0.0


def _rare_word_fraction(state_words: pd.Series, baseline_profiles: dict[str, object]) -> float:
    freq = baseline_profiles.get("word_frequency", {})
    threshold = float(baseline_profiles.get("rare_word_threshold", 0.0))
    if state_words.empty or not freq:
        return 0.0
    rare = [freq.get(word, 0.0) < threshold for word in state_words.astype(str)]
    return float(sum(rare) / len(rare)) if rare else 0.0


def _rare_state_fraction(state_tokens: list[int], baseline_profiles: dict[str, object]) -> float:
    freq = baseline_profiles.get("state_frequency", {})
    threshold = float(baseline_profiles.get("rare_state_threshold", 0.0))
    if not state_tokens or not freq:
        return 0.0
    rare = [freq.get(state, 0.0) < threshold for state in state_tokens]
    return float(sum(rare) / len(rare)) if rare else 0.0


def _state_entropy(state_tokens: list[int]) -> float:
    if not state_tokens:
        return 0.0
    series = pd.Series(state_tokens)
    probs = series.value_counts(normalize=True)
    return float(-sum(p * math.log(p + 1e-12) for p in probs))


def _word_regime_shift_score(state_words: pd.Series, baseline_profiles: dict[str, object]) -> float:
    global_freq = baseline_profiles.get("word_frequency", {})
    if state_words.empty or not global_freq:
        return 0.0
    local_freq = state_words.value_counts(normalize=True).to_dict()
    union = set(global_freq) | set(local_freq)
    return float(0.5 * sum(abs(local_freq.get(word, 0.0) - global_freq.get(word, 0.0)) for word in union))


def _mean_numeric(frame: pd.DataFrame, column: str) -> float:
    if column not in frame.columns or frame.empty:
        return 0.0
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return float(values.mean()) if not values.empty else 0.0

def _overlap_seconds(
    left_start: pd.Timestamp,
    left_end: pd.Timestamp,
    right_start: pd.Timestamp,
    right_end: pd.Timestamp,
) -> float:
    """Compute the temporal overlap in seconds between two intervals."""

    overlap_start = max(left_start, pd.to_datetime(right_start))
    overlap_end = min(left_end, pd.to_datetime(right_end))
    return max((overlap_end - overlap_start).total_seconds(), 0.0)

