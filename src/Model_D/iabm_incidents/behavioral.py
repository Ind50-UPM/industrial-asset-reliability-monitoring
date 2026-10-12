"""Chronologically fitted seven-coordinate behavioral encoder for raw telemetry.

This supplies a complete M0 when retained Model D tables contain only semantic
scores. It is a new matched raw-command implementation, not a replay of the
historical semantic baseline. All references are frozen before evaluated windows.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .electrical import ElectricalConfig


def _runs(states: np.ndarray, gap: np.ndarray, dt: float):
    change = np.r_[True, states[1:] != states[:-1]] | gap
    starts = np.flatnonzero(change)
    counts = np.diff(np.r_[starts, len(states)])
    return states[starts], counts*dt, ~gap[starts[1:]]


def _distribution(values: np.ndarray, size: int):
    counts = np.bincount(values, minlength=size).astype(float)
    return counts/max(counts.sum(), 1.)


def _js(p, q):
    mid = (p+q)/2
    def term(a):
        use = a > 0
        return np.sum(a[use]*np.log2(a[use]/mid[use]))
    return float((term(p)+term(q))/2)


class BehavioralWindowEncoder:
    """Fit command-run transition, duration, recurrence, persistence/mode references.

    Dseq is Jensen--Shannon divergence of compressed command-run transitions;
    Ddur is frequency-weighted per-state median dwell drift/MAD; Rexc is the
    maximum positive run-count excess against frozen observed-time rates;
    Pexc is maximum occupancy excess against nominal 95th percentiles; Dcons
    is mean signed state-standardized electrical residual; Estate is actual
    command versus retained Model A disagreement on available matched records;
    Dmode is Jensen--Shannon divergence of active-control-bit-count distribution.

    Estate retains its prediction-coverage fraction: unavailable predictions
    are not counted as correct. Mode is an explicit command-activity proxy,
    not a reconstruction of Model C's complete semantic mode ontology.
    """

    def __init__(self, config: ElectricalConfig = ElectricalConfig()):
        self.config = config
        self.fitted = False

    def fit(self, nominal: pd.DataFrame, window_length: pd.Timedelta = pd.Timedelta(hours=24),
            step: pd.Timedelta = pd.Timedelta(hours=1)) -> BehavioralWindowEncoder:
        """Fit only the supplied nominal training samples, excluding timestamp gaps."""
        u = nominal[self.config.state_column].to_numpy(float)
        if not np.isfinite(u).all() or ((u < 0) | (u >= 64) | (u != np.round(u))).any():
            raise ValueError('Behavioral encoder requires known six-bit controller states')
        u = u.astype(int)
        gaps = nominal[self.config.timestamp_column].diff().dt.total_seconds().gt(self.config.maximum_gap_seconds).to_numpy()
        states, durations, pair_valid = _runs(u, gaps, self.config.sample_period_seconds)
        pairs = (states[:-1]*64+states[1:])[pair_valid]
        self.transitions = _distribution(pairs,4096)
        self.dwell = {}
        for state in np.unique(states):
            values = durations[states == state]
            median = float(np.median(values)); mad = float(1.4826*np.median(abs(values-median)))
            self.dwell[int(state)] = median, max(mad,self.config.sample_period_seconds)
        self.run_rates = np.bincount(states,minlength=64)/(len(u)*self.config.sample_period_seconds)
        modes = np.array([int(state).bit_count() for state in u])
        self.modes = _distribution(modes,7)
        occupancies = []
        times = nominal[self.config.timestamp_column].to_numpy(dtype='datetime64[ns]')
        current = pd.Timestamp(times[0]).ceil(step)
        while current+window_length <= pd.Timestamp(times[-1]):
            lo, hi = np.searchsorted(times,[current.to_datetime64(), (current+window_length).to_datetime64()])
            if hi > lo:
                occupancies.append(_distribution(u[lo:hi],64))
            current += step
        if not occupancies:
            raise ValueError('Nominal partition too short for behavioral occupancy reference')
        self.occupancy95 = np.quantile(occupancies,.95,axis=0)
        self.fitted = True
        return self

    def encode(self, samples: pd.DataFrame, minimum_predictions: int = 32) -> dict:
        """Encode one already sliced window; NaN marks missing model support."""
        if not self.fitted:
            raise ValueError('Fit the nominal behavioral encoder first')
        cfg = self.config
        u = samples[cfg.state_column].to_numpy(float)
        finite = np.isfinite(u) & (u >= 0) & (u < 64) & (u == np.round(u))
        known_fraction = float(finite.mean()) if len(u) else 0.
        original_positions = np.flatnonzero(finite)
        if known_fraction < cfg.minimum_safe_fraction or finite.sum() < cfg.minimum_samples:
            return {c:np.nan for c in ('sequence_divergence','duration_drift','recurrence_excess','persistence_excess','consumption_deviation','state_error_rate','mode_divergence')}
        if not finite.all():
            samples = samples.loc[finite]
            u = u[finite]
        u = u.astype(int)
        gaps = samples[cfg.timestamp_column].diff().dt.total_seconds().gt(cfg.maximum_gap_seconds).to_numpy()
        gaps[1:] |= np.diff(original_positions) > 1
        states, durations, pair_valid = _runs(u,gaps,cfg.sample_period_seconds)
        pairdist = _distribution((states[:-1]*64+states[1:])[pair_valid],4096)
        nominal_state_fraction = float(np.isin(u, list(self.dwell)).mean())
        if nominal_state_fraction < cfg.minimum_safe_fraction:
            return {c:np.nan for c in ('sequence_divergence','duration_drift','recurrence_excess','persistence_excess','consumption_deviation','state_error_rate','mode_divergence')}
        dwell = []
        for state in np.unique(states):
            if state not in self.dwell:
                continue
            median, scale = self.dwell[state]
            dwell.extend([abs(np.median(durations[states == state])-median)/scale]*int((states == state).sum()))
        counts = np.bincount(states,minlength=64)
        expected = self.run_rates*len(u)*cfg.sample_period_seconds
        occupancy = _distribution(u,64)
        active = expected > 0
        recurrence = float(np.maximum((counts[active]-expected[active])/expected[active],0).max())
        persistence = float(np.maximum((occupancy-self.occupancy95)/np.maximum(self.occupancy95,1/len(u)),0).max())
        residual = samples[['residual_1','residual_2','residual_3']].to_numpy(float)
        safe = samples.safe.to_numpy(bool) & np.isfinite(residual).all(1)
        pred = samples.pred_estado.to_numpy(float)
        predicted = np.isfinite(pred)
        error = float(np.mean(pred[predicted] != u[predicted])) if predicted.sum() >= minimum_predictions else np.nan
        return dict(sequence_divergence=_js(pairdist,self.transitions), duration_drift=float(np.mean(dwell)),
            recurrence_excess=recurrence,persistence_excess=persistence,
            consumption_deviation=float(residual[safe].mean()) if safe.any() else np.nan,
            state_error_rate=error, mode_divergence=_js(_distribution(np.array([int(state).bit_count() for state in u]),7),self.modes),
            controller_known_fraction=known_fraction,
            behavioral_nominal_state_fraction=nominal_state_fraction,
            state_prediction_fraction=float(predicted.mean()), state_prediction_samples=int(predicted.sum()))
