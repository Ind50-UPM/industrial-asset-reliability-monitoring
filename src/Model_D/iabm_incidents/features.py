"""Episode-level feature extraction for Model_D classification and reporting."""

from __future__ import annotations

import pandas as pd


FEATURE_COLUMNS = [
    "episode_id",
    "asset_id",
    "duration_hours",
    "recovery_hours",
    "onset_slope",
    "peak_score",
    "mean_score",
    "max_sequence_divergence",
    "median_sequence_divergence",
    "max_duration_drift",
    "median_duration_drift",
    "max_recurrence_excess",
    "median_recurrence_excess",
    "max_persistence_excess",
    "median_persistence_excess",
    "signed_consumption_deviation",
    "min_consumption_deviation",
    "max_state_error_rate",
    "mean_state_error_rate",
    "max_mode_divergence",
    "mean_mode_divergence",
    "max_command_response_deviation",
    "mean_command_response_deviation",
    "max_transient_concentration",
    "mean_transient_concentration",
    "dominant_family",
    "mixed_family_evidence",
    "max_state_word_diversity",
    "median_state_word_diversity",
    "mean_dominant_state_word_fraction",
    "mean_state_word_transition_rate",
    "dominant_state_word_count",
    "mean_nominal_state_word_match_fraction",
    "mean_state_distance_to_nominal",
    "mean_dtw_distance_to_nominal",
    "mean_nominal_word_anomaly_score",
    "mean_rare_word_fraction",
    "mean_rare_state_fraction",
    "max_state_entropy",
    "mean_state_entropy",
    "mean_state_17_fraction",
    "mean_off_nominal_state_fraction",
    "mean_word_regime_shift_score",
]


def build_episode_features(episodes: pd.DataFrame, window_scores: pd.DataFrame) -> pd.DataFrame:
    """Convert scored windows plus episode spans into one feature row per episode."""

    if episodes.empty:
        return pd.DataFrame(columns=FEATURE_COLUMNS)

    rows: list[dict[str, object]] = []
    for _, episode in episodes.iterrows():
        window_ids = episode.get("source_window_ids", tuple())
        if isinstance(window_ids, str):
            window_ids = (window_ids,)
        episode_windows = _slice_episode_windows(window_scores, window_ids, episode)
        if episode_windows.empty:
            episode_windows = window_scores.iloc[0:0].copy()
        row = {
            "episode_id": episode["episode_id"],
            "asset_id": episode.get("asset_id"),
            "duration_hours": float(pd.to_numeric(pd.Series([episode.get("duration_seconds")]), errors="coerce").fillna(0.0).iloc[0] / 3600.0),
            "recovery_hours": _seconds_to_hours(episode.get("time_to_recovery_seconds")),
            "onset_slope": _onset_slope(episode_windows),
            "peak_score": _safe_float(episode.get("peak_score")),
            "mean_score": _safe_float(episode.get("mean_score")),
            "max_sequence_divergence": _agg(episode_windows, "sequence_divergence", "max"),
            "median_sequence_divergence": _agg(episode_windows, "sequence_divergence", "median"),
            "max_duration_drift": _agg(episode_windows, "duration_drift", "max"),
            "median_duration_drift": _agg(episode_windows, "duration_drift", "median"),
            "max_recurrence_excess": _agg(episode_windows, "recurrence_excess", "max"),
            "median_recurrence_excess": _agg(episode_windows, "recurrence_excess", "median"),
            "max_persistence_excess": _agg(episode_windows, "persistence_excess", "max"),
            "median_persistence_excess": _agg(episode_windows, "persistence_excess", "median"),
            "signed_consumption_deviation": _agg(episode_windows, "consumption_deviation", "mean"),
            "min_consumption_deviation": _agg(episode_windows, "consumption_deviation", "min"),
            "max_state_error_rate": _agg(episode_windows, "state_error_rate", "max"),
            "mean_state_error_rate": _agg(episode_windows, "state_error_rate", "mean"),
            "max_mode_divergence": _agg(episode_windows, "mode_divergence", "max"),
            "mean_mode_divergence": _agg(episode_windows, "mode_divergence", "mean"),
            "max_command_response_deviation": _agg(episode_windows, "command_response_deviation", "max"),
            "mean_command_response_deviation": _agg(episode_windows, "command_response_deviation", "mean"),
            "max_transient_concentration": _agg(episode_windows, "transient_concentration", "max"),
            "mean_transient_concentration": _agg(episode_windows, "transient_concentration", "mean"),
            "dominant_family": _dominant_family(episode_windows),
            "mixed_family_evidence": _mixed_family_evidence(episode_windows),
            "max_state_word_diversity": _agg(episode_windows, "state_word_diversity", "max"),
            "median_state_word_diversity": _agg(episode_windows, "state_word_diversity", "median"),
            "mean_dominant_state_word_fraction": _agg(episode_windows, "dominant_state_word_fraction", "mean"),
            "mean_state_word_transition_rate": _agg(episode_windows, "state_word_transition_rate", "mean"),
            "dominant_state_word_count": _dominant_state_word_count(episode_windows),
            "mean_nominal_state_word_match_fraction": _agg(episode_windows, "nominal_state_word_match_fraction", "mean"),
            "mean_state_distance_to_nominal": _agg(episode_windows, "mean_state_distance", "mean"),
            "mean_dtw_distance_to_nominal": _agg(episode_windows, "mean_dtw_distance", "mean"),
            "mean_nominal_word_anomaly_score": _agg(episode_windows, "mean_nominal_anomaly_score", "mean"),
            "mean_rare_word_fraction": _agg(episode_windows, "rare_word_fraction", "mean"),
            "mean_rare_state_fraction": _agg(episode_windows, "rare_state_fraction", "mean"),
            "max_state_entropy": _agg(episode_windows, "state_entropy", "max"),
            "mean_state_entropy": _agg(episode_windows, "state_entropy", "mean"),
            "mean_state_17_fraction": _agg(episode_windows, "state_17_fraction", "mean"),
            "mean_off_nominal_state_fraction": _agg(episode_windows, "off_nominal_state_fraction", "mean"),
            "mean_word_regime_shift_score": _agg(episode_windows, "word_regime_shift_score", "mean"),
        }
        for column in episode_windows.columns:
            if column.startswith("command_response_signed_") or column == "command_response_imbalance":
                row["mean_" + column] = _agg(episode_windows, column, "mean")
        rows.append(row)
    extra = sorted({key for row in rows for key in row} - set(FEATURE_COLUMNS))
    return pd.DataFrame(rows, columns=FEATURE_COLUMNS + extra)


def _slice_episode_windows(window_scores: pd.DataFrame, window_ids: tuple[str, ...], episode: pd.Series) -> pd.DataFrame:
    if window_scores.empty:
        return window_scores.copy()
    if window_ids:
        index_labels = [int(item) for item in window_ids if str(item).isdigit()]
        if index_labels:
            existing = [index for index in index_labels if index in window_scores.index]
            if existing:
                return window_scores.loc[existing].copy()
    start = pd.to_datetime(episode.get("event_start"))
    end = pd.to_datetime(episode.get("event_end"))
    frame = window_scores.copy()
    frame["start_time"] = pd.to_datetime(frame["start_time"])
    frame["end_time"] = pd.to_datetime(frame["end_time"])
    return frame[(frame["start_time"] <= end) & (frame["end_time"] >= start)].copy()


def _agg(frame: pd.DataFrame, column: str, op: str) -> float:
    if column not in frame.columns or frame.empty:
        return 0.0
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    if values.empty:
        return 0.0
    if op == "max":
        return float(values.max())
    if op == "min":
        return float(values.min())
    if op == "median":
        return float(values.median())
    return float(values.mean())


def _dominant_family(frame: pd.DataFrame) -> str:
    if "incident_family" not in frame.columns or frame.empty:
        return "unclassified_incident"
    mode = frame["incident_family"].mode(dropna=True)
    return str(mode.iloc[0]) if not mode.empty else "unclassified_incident"


def _mixed_family_evidence(frame: pd.DataFrame) -> float:
    if "incident_family" not in frame.columns or frame.empty:
        return 0.0
    return 1.0 if frame["incident_family"].nunique(dropna=True) >= 2 else 0.0




def _dominant_state_word_count(frame: pd.DataFrame) -> float:
    if "dominant_state_word" not in frame.columns or frame.empty:
        return 0.0
    words = frame["dominant_state_word"].replace("", pd.NA).dropna()
    if words.empty:
        return 0.0
    return float(words.nunique(dropna=True))

def _onset_slope(frame: pd.DataFrame) -> float:
    if frame.empty or "deviation_score" not in frame.columns or len(frame) < 2:
        return 0.0
    scores = pd.to_numeric(frame["deviation_score"], errors="coerce").fillna(0.0).reset_index(drop=True)
    return float(scores.iloc[1] - scores.iloc[0])


def _seconds_to_hours(value: object) -> float | None:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return None
    return float(numeric) / 3600.0


def _safe_float(value: object) -> float:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").fillna(0.0).iloc[0]
    return float(numeric)
