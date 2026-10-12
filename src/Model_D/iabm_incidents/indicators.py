"""Indicator aggregation utilities for the indicator-driven Model_D route."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class IndicatorBaseline:
    """Simple nominal baseline used to center window indicators."""

    medians: dict[str, float]


def fit_indicator_baseline(frame: pd.DataFrame, columns: list[str] | None = None) -> IndicatorBaseline:
    """Fit a robust median baseline from nominal rows when available."""

    candidate_columns = columns or [
        'sequence_divergence', 'duration_drift', 'recurrence_excess', 'persistence_excess',
        'consumption_deviation', 'state_error_rate', 'mode_divergence',
        'command_response_deviation', 'transient_concentration',
    ]
    nominal = frame.copy()
    if 'semantic_status' in nominal.columns:
        nominal = nominal[nominal['semantic_status'].ne('ANOMALOUS')]
    if nominal.empty:
        nominal = frame.copy()
    medians = {
        column: float(pd.to_numeric(nominal[column], errors='coerce').dropna().median())
        for column in candidate_columns
        if column in nominal.columns and nominal[column].notna().any()
    }
    return IndicatorBaseline(medians=medians)


def compute_window_indicators(source_rows: pd.DataFrame, windows: pd.DataFrame, baseline: IndicatorBaseline | None = None) -> pd.DataFrame:
    """Aggregate row-level indicators into the supplied windows."""

    if source_rows.empty or windows.empty:
        return pd.DataFrame(columns=['window_id', 'asset_id', 'start_time', 'end_time'])

    frame = source_rows.copy()
    frame = frame.loc[:, ~frame.columns.duplicated()].copy()
    if 'window_start' in frame.columns and 'start_time' not in frame.columns:
        frame['start_time'] = frame['window_start']
    if 'window_end' in frame.columns and 'end_time' not in frame.columns:
        frame['end_time'] = frame['window_end']
    frame['start_time'] = _normalize_timestamp_series(frame['start_time'])
    frame['end_time'] = _normalize_timestamp_series(frame['end_time'])
    frame = frame.dropna(subset=['start_time', 'end_time'])
    baseline_values = (baseline or fit_indicator_baseline(frame)).medians

    window_frame = windows.copy()
    window_frame = window_frame.loc[:, ~window_frame.columns.duplicated()].copy()
    window_frame['start_time'] = _normalize_timestamp_series(window_frame['start_time'])
    window_frame['end_time'] = _normalize_timestamp_series(window_frame['end_time'])
    window_frame = window_frame.dropna(subset=['start_time', 'end_time']).reset_index(drop=True)
    if window_frame.empty:
        return pd.DataFrame(columns=['window_id', 'asset_id', 'start_time', 'end_time'])

    if 'asset_id' not in frame.columns:
        frame['asset_id'] = 'asset-unknown'
    frame['asset_id'] = frame['asset_id'].fillna('asset-unknown').astype(str)

    if 'asset_id' not in window_frame.columns:
        window_frame['asset_id'] = 'asset-unknown'
    window_frame['asset_id'] = window_frame['asset_id'].fillna('asset-unknown').astype(str)

    rows: list[dict[str, object]] = []
    for asset_id, asset_windows in window_frame.groupby('asset_id', dropna=False, sort=False):
        asset_rows = frame[frame['asset_id'] == asset_id]
        if asset_rows.empty:
            continue
        rows.extend(_compute_asset_window_indicators(asset_rows, asset_windows, baseline_values))
    return pd.DataFrame(rows)


def _compute_asset_window_indicators(
    asset_rows: pd.DataFrame,
    asset_windows: pd.DataFrame,
    baseline_values: dict[str, float],
) -> list[dict[str, object]]:
    """Aggregate indicators for one asset using NumPy overlap masks."""

    starts = asset_rows['start_time'].to_numpy(dtype='datetime64[ns]').astype('int64')
    ends = asset_rows['end_time'].to_numpy(dtype='datetime64[ns]').astype('int64')
    semantic_values = (
        asset_rows['semantic_status'].fillna('NORMAL').astype(str).to_numpy()
        if 'semantic_status' in asset_rows.columns
        else np.full(len(asset_rows), 'NORMAL', dtype=object)
    )
    family_values = (
        asset_rows['incident_family'].fillna('unclassified_incident').astype(str).to_numpy()
        if 'incident_family' in asset_rows.columns
        else np.full(len(asset_rows), 'unclassified_incident', dtype=object)
    )
    indicator_arrays = {
        column: pd.to_numeric(asset_rows[column], errors='coerce').fillna(0.0).to_numpy(dtype=float)
        for column in baseline_values
        if column in asset_rows.columns
    }

    rows: list[dict[str, object]] = []
    for window in asset_windows.itertuples(index=False):
        window_start = _normalize_scalar_timestamp(getattr(window, 'start_time'))
        window_end = _normalize_scalar_timestamp(getattr(window, 'end_time'))
        window_start_ns = window_start.to_datetime64().astype('datetime64[ns]').astype('int64')
        window_end_ns = window_end.to_datetime64().astype('datetime64[ns]').astype('int64')
        mask = (starts < window_end_ns) & (ends > window_start_ns)
        if not bool(mask.any()):
            continue

        active_semantic = semantic_values[mask]
        active_family = family_values[mask]
        row = {
            'window_id': getattr(window, 'window_id', None),
            'asset_id': getattr(window, 'asset_id', None),
            'start_time': window_start,
            'end_time': window_end,
            'data_coverage': getattr(window, 'data_coverage', 1.0),
            'sequence_count': int(mask.sum()),
            'active_sequence_count': int(mask.sum()),
            'semantic_status': 'ANOMALOUS' if bool((active_semantic == 'ANOMALOUS').any()) else 'NORMAL',
            'incident_family': _dominant_family_from_arrays(active_semantic, active_family),
        }
        for column, baseline_value in baseline_values.items():
            if column not in indicator_arrays:
                continue
            row[column] = float(indicator_arrays[column][mask].mean() - baseline_value)
        # Electrical scores already encode a frozen nominal reference.
        # Preserve raw nonnegative evidence and missing support, without recentering.
        for column in ("command_response_deviation", "transient_concentration"):
            if column in asset_rows:
                values = pd.to_numeric(asset_rows.loc[mask, column], errors="coerce")
                row[column] = float(values.max() if column == "transient_concentration" else values.mean())
        for column in asset_rows.columns:
            if column.startswith("command_response_signed_") or column == "command_response_imbalance":
                row[column] = float(pd.to_numeric(asset_rows.loc[mask, column], errors="coerce").mean())
        rows.append(row)
    return rows


def _dominant_family(frame: pd.DataFrame) -> str:
    if 'incident_family' not in frame.columns:
        return 'unclassified_incident'
    anomalous = frame[frame.get('semantic_status', pd.Series(index=frame.index)).eq('ANOMALOUS')]
    target = anomalous if not anomalous.empty else frame
    mode = target['incident_family'].mode(dropna=True)
    return str(mode.iloc[0]) if not mode.empty else 'unclassified_incident'


def _dominant_family_from_arrays(semantic_values: np.ndarray, family_values: np.ndarray) -> str:
    """Return the dominant family giving priority to anomalous rows."""

    anomalous = family_values[semantic_values == 'ANOMALOUS']
    target = anomalous if anomalous.size else family_values
    if not target.size:
        return 'unclassified_incident'
    mode = pd.Series(target).mode(dropna=True)
    return str(mode.iloc[0]) if not mode.empty else 'unclassified_incident'


def _normalize_timestamp_series(values: pd.Series) -> pd.Series:
    converted = pd.to_datetime(values, errors='coerce', utc=True)
    if getattr(converted.dtype, 'tz', None) is not None:
        converted = converted.dt.tz_convert(None)
    return converted


def _normalize_scalar_timestamp(value: object) -> pd.Timestamp:
    timestamp = pd.to_datetime(value, errors='coerce', utc=True)
    if pd.isna(timestamp):
        return timestamp
    if getattr(timestamp, 'tzinfo', None) is not None:
        return timestamp.tz_convert(None)
    return timestamp
