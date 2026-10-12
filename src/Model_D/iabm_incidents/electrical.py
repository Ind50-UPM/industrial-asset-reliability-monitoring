"""Frozen command-response residuals and solver-independent fourth-order evidence.

The eight-coordinate causal representation and normalized scatter follow the
QIP branch of CReST_Bode_and_Ablation_Atlas. Fit on declared nominal partitions;
neither observer nor reference is fitted implicitly by the incident pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ElectricalConfig:
    """Telemetry contract and support policy (timestamps use half-open windows)."""

    channels: tuple[str, str, str] = ("RP1", "RP2", "RP3")
    state_column: str = "controller_state"
    timestamp_column: str = "timestamp"
    minimum_samples: int = 32
    minimum_safe_fraction: float = 0.9
    settling_samples: int = 3
    maximum_gap_seconds: float = 2.0
    sample_period_seconds: float = 1.0
    shrinkage: float = 0.1
    inverse_floor: float = 1e-6

    def __post_init__(self) -> None:
        if len(self.channels) != 3 or len(set(self.channels)) != 3:
            raise ValueError("Three distinct electrical phases required")
        if self.minimum_samples < 4 or not 0 < self.minimum_safe_fraction <= 1:
            raise ValueError("Invalid electrical support policy")
        if self.settling_samples < 0 or self.maximum_gap_seconds <= 0 or self.sample_period_seconds <= 0:
            raise ValueError("Invalid settling/gap policy")
        if not 0 <= self.shrinkage <= 1 or self.inverse_floor <= 0:
            raise ValueError("Invalid covariance regularization")


class CommandResponseObserver:
    """Robust same-state observer; known switching is excluded by settling guards.

    This deliberately simple observer must be validated on separate nominal data
    before use. More detailed command/age/FIR observers can supply residuals to
    ``TransientConcentration`` directly. No measured-output lags are used.
    """

    def __init__(self, config: ElectricalConfig = ElectricalConfig()) -> None:
        self.config = config
        self.profiles: dict[tuple[str, object], tuple[np.ndarray, np.ndarray]] = {}

    def fit(self, nominal: pd.DataFrame) -> CommandResponseObserver:
        frame = _ordered(nominal, self.config)
        self.profiles = {}
        for asset, group in frame.groupby("asset_id", sort=False):
            stable = _safe_history(group, self.config)
            for state, rows in group.loc[stable].groupby(self.config.state_column):
                y = rows[list(self.config.channels)].to_numpy(float)
                y = y[np.isfinite(y).all(1)]
                if len(y) < self.config.minimum_samples:
                    continue
                center = np.median(y, axis=0)
                scale = np.maximum(1.4826 * np.median(abs(y-center), axis=0),
                                   np.maximum(abs(center)*0.01, 1e-6))
                self.profiles[(str(asset), state)] = (center, scale)
        if not self.profiles:
            raise ValueError("No supported nominal command states")
        return self

    def residuals(self, samples: pd.DataFrame) -> pd.DataFrame:
        """Return signed phase residuals; unknown states remain NaN/unsafe."""
        frame = _ordered(samples, self.config)
        r = np.full((len(frame), 3), np.nan)
        if not self.profiles:
            raise ValueError("Fit the command observer on nominal data first")
        for (asset, state), indices in frame.groupby(["asset_id", self.config.state_column]).groups.items():
            profile = self.profiles.get((str(asset), state))
            if profile is not None:
                center, scale = profile
                r[indices] = (frame.loc[indices, list(self.config.channels)].to_numpy(float)-center)/scale
        frame[["residual_1", "residual_2", "residual_3"]] = r
        safe = np.zeros(len(frame), bool)
        for _, group in frame.groupby("asset_id", sort=False):
            safe[group.index] = _safe_history(group, self.config)
        frame["safe"] = safe & np.isfinite(r).all(1) & (np.abs(r).max(1) < 1e6)
        return frame


def _ordered(samples: pd.DataFrame, config: ElectricalConfig) -> pd.DataFrame:
    frame = samples.copy()
    if "asset_id" not in frame:
        frame["asset_id"] = "asset-unknown"
    frame["asset_id"] = frame.asset_id.fillna("asset-unknown").astype(str)
    frame[config.timestamp_column] = pd.to_datetime(frame[config.timestamp_column], utc=True, errors="raise").dt.tz_convert(None)
    frame = frame.sort_values(["asset_id", config.timestamp_column]).reset_index(drop=True)
    if frame.duplicated(["asset_id", config.timestamp_column]).any():
        raise ValueError("Duplicate telemetry timestamps within an asset")
    return frame


def _safe_history(group: pd.DataFrame, config: ElectricalConfig) -> np.ndarray:
    states = group[config.state_column].to_numpy()
    dt = group[config.timestamp_column].diff().dt.total_seconds().to_numpy()
    if not len(group):
        return np.empty(0, bool)
    regular = (dt > 0) & (dt <= config.maximum_gap_seconds)
    reset = ~regular
    reset[1:] |= states[1:] != states[:-1]
    positions = np.arange(len(group))
    age = positions-np.maximum.accumulate(np.where(reset, positions, 0))
    return regular & (age >= config.settling_samples)



def causal_delay(residual: np.ndarray, safe: np.ndarray) -> np.ndarray:
    """RP1–3 now/lag1, common mode lag2/lag3; never bridge unsafe samples."""
    r = np.asarray(residual, float)
    ok = np.asarray(safe, bool)
    if r.ndim != 2 or r.shape[1] != 3 or ok.shape != (len(r),):
        raise ValueError("Expected N x 3 residuals and N safety flags")
    if len(r) < 4:
        return np.empty((0, 8))
    x = np.column_stack((r[3:], r[2:-1], r[1:-2].sum(1)/np.sqrt(3), r[:-3].sum(1)/np.sqrt(3)))
    valid = ok[3:] & ok[2:-1] & ok[1:-2] & ok[:-3] & np.isfinite(x).all(1)
    return x[valid]


def fourth_scatter(z: np.ndarray) -> np.ndarray:
    """F = mean[(||z||²/d) zzᵀ], a PSD degree-four scatter."""
    z = np.asarray(z, float)
    if z.ndim != 2 or not len(z) or not z.shape[1] or not np.isfinite(z).all():
        raise ValueError("Finite nonempty coordinate matrix required")
    return z.T @ ((np.sum(z*z, axis=1)/z.shape[1])[:, None]*z)/len(z)


class DominantDirectionSolver(Protocol):
    """Approximation contract; return a normalized direction for the same D."""

    def solve(self, operator: np.ndarray) -> np.ndarray: ...


class VQEDirectionSolver:
    """Three-qubit real RY/CX statevector VQE maximizing <D> by Rotosolve.

    Independent starts, no eigenvector initialization or exact-solver fallback.
    This is a classical simulator, not a claim of quantum advantage.
    """

    def __init__(self, layers: int = 3, sweeps: int = 8, restarts: int = 2, seed: int = 0) -> None:
        if min(layers, sweeps, restarts) < 1:
            raise ValueError("Positive VQE resource budgets required")
        self.layers, self.sweeps, self.restarts, self.seed = layers, sweeps, restarts, seed

    def _state(self, theta: np.ndarray) -> np.ndarray:
        psi = np.zeros(8); psi[0] = 1
        basis = np.arange(8)
        for layer, angles in enumerate(theta.reshape(self.layers, 3)):
            for q, angle in enumerate(angles):
                bit = 1 << (2-q); zero = basis[(basis & bit) == 0]; one = zero | bit
                a, b = psi[zero].copy(), psi[one].copy()
                c, s = np.cos(angle/2), np.sin(angle/2)
                psi[zero], psi[one] = c*a-s*b, s*a+c*b
            pairs = ((0, 1), (1, 2)) if layer % 2 == 0 else ((2, 1), (1, 0))
            for control, target in pairs:
                psi = psi[basis ^ (((basis >> (2-control)) & 1) << (2-target))]
        return psi

    def solve(self, operator: np.ndarray) -> np.ndarray:
        """Return one variational direction; see ``solve_batch`` for resource policy."""
        return self.solve_batch(np.asarray(operator)[None])[0]

    def solve_batch(self, operators: np.ndarray) -> np.ndarray:
        """Solve independent identical-budget RY/CX circuits in vectorized batches.

        Each operator has its own angles and updates. Restart initial angles
        match the single-operator seeded policy. No classical eigenvectors or
        best-case filtering enter optimization, including difficult windows.
        """
        d = np.asarray(operators, float)
        if d.ndim != 3 or d.shape[1:] != (8, 8) or not np.isfinite(d).all():
            raise ValueError("VQE requires finite B x 8 x 8 real operators")
        n = len(d)
        if not n:
            return np.empty((0, 8))
        rng = np.random.default_rng(self.seed)
        h = -d/np.maximum(np.linalg.norm(d, axis=(1, 2)), 1e-12)[:, None, None]
        basis = np.arange(8)
        def state(theta: np.ndarray) -> np.ndarray:
            psi = np.zeros((n, 8)); psi[:, 0] = 1
            for layer in range(self.layers):
                for q in range(3):
                    angle = theta[:, 3*layer+q]
                    bit = 1 << (2-q); zero = basis[(basis & bit) == 0]; one = zero | bit
                    a, b = psi[:, zero].copy(), psi[:, one].copy()
                    c, sn = np.cos(angle/2)[:, None], np.sin(angle/2)[:, None]
                    psi[:, zero], psi[:, one] = c*a-sn*b, sn*a+c*b
                pairs = ((0, 1), (1, 2)) if layer % 2 == 0 else ((2, 1), (1, 0))
                for control, target in pairs:
                    psi = psi[:, basis ^ (((basis >> (2-control)) & 1) << (2-target))]
            return psi
        def energy(theta: np.ndarray) -> np.ndarray:
            psi = state(theta)
            return np.einsum("bi,bij,bj->b", psi, h, psi)
        best = np.full(n, np.inf); best_theta = np.zeros((n, 3*self.layers))
        for restart in range(self.restarts):
            initial = np.zeros(3*self.layers) if restart == 0 else rng.uniform(-np.pi, np.pi, 3*self.layers)
            theta = np.tile(initial, (n, 1)); e = energy(theta)
            for _ in range(self.sweeps):
                for j in range(theta.shape[1]):
                    old, old_e = theta[:, j].copy(), e.copy()
                    theta[:, j] = 0; e0 = energy(theta)
                    theta[:, j] = np.pi/2; ep = energy(theta)
                    theta[:, j] = -np.pi/2; em = energy(theta)
                    theta[:, j] = np.arctan2((ep-em)/2, e0-(ep+em)/2)+np.pi
                    e = energy(theta)
                    reject = e > old_e
                    theta[reject, j], e[reject] = old[reject], old_e[reject]
            improve = e < best
            best[improve], best_theta[improve] = e[improve], theta[improve]
        return state(best_theta)


@dataclass(frozen=True)
class TransientReference:
    """Frozen context-specific whitening and fourth-order nominal operator."""

    center: np.ndarray
    whitening: np.ndarray
    scatter: np.ndarray
    scale: float


class TransientConcentration:
    """Define Tconc independently of the optional variational direction solver."""

    def __init__(self, config: ElectricalConfig = ElectricalConfig(), solver: DominantDirectionSolver | None = None) -> None:
        self.config, self.solver = config, solver
        self.references: dict[tuple[str, object], TransientReference] = {}

    def fit(self, nominal_residuals: pd.DataFrame) -> TransientConcentration:
        """Fit each asset/state on complete contiguous reference runs only."""
        frame = _ordered(nominal_residuals, self.config)
        self.references = {}
        for asset, group in frame.groupby("asset_id", sort=False):
            r = group[["residual_1", "residual_2", "residual_3"]].to_numpy(float)
            states = group[self.config.state_column].to_numpy()
            safe = group.safe.to_numpy(bool) & _safe_history(group, self.config)
            for state in pd.unique(states):
                x = causal_delay(r, safe & (states == state))
                if len(x) < self.config.minimum_samples:
                    continue
                center = np.median(x, axis=0); xc = x-center
                cov = xc.T @ xc/len(xc); tau = max(np.trace(cov)/8, 1e-8)
                reg = (1-self.config.shrinkage)*cov+(self.config.shrinkage+self.config.inverse_floor)*tau*np.eye(8)
                values, vectors = np.linalg.eigh(reg)
                w = (vectors/np.sqrt(np.maximum(values, self.config.inverse_floor*tau))) @ vectors.T
                f0 = fourth_scatter(xc @ w)
                self.references[(str(asset), state)] = TransientReference(center, w, f0, max(np.trace(f0)/8, 1e-8))
        if not self.references:
            raise ValueError("No supported transient reference contexts")
        return self

    @staticmethod
    def operator(x: np.ndarray, reference: TransientReference) -> np.ndarray:
        """D = (F((x-center)W)-F0)/s0; shared by exact and VQE solvers."""
        d = (fourth_scatter((x-reference.center) @ reference.whitening)-reference.scatter)/reference.scale
        return (d+d.T)/2

    def score_windows(self, residuals: pd.DataFrame, windows: pd.DataFrame) -> pd.DataFrame:
        """Mean residual norm, signed phases and max supported context Tconc.

        Unsupported windows retain NaN and explicit coverage; the optional VQE
        Rayleigh score is exported separately and never replaces exact Tconc.
        """
        if not self.references:
            raise ValueError("Fit the transient reference on nominal data first")
        frame = _ordered(residuals, self.config)
        rows = []
        for window in windows.itertuples(index=False):
            start = pd.to_datetime(window.start_time, utc=True).tz_convert(None)
            end = pd.to_datetime(window.end_time, utc=True).tz_convert(None)
            part = frame[(frame.asset_id == str(window.asset_id)) & (frame[self.config.timestamp_column] >= start) & (frame[self.config.timestamp_column] < end)]
            r = part[["residual_1", "residual_2", "residual_3"]].to_numpy(float)
            safe = part.safe.to_numpy(bool) & _safe_history(part, self.config) & np.isfinite(r).all(1)
            duration = (end-start).total_seconds()
            if duration <= 0:
                raise ValueError("Positive window duration required")
            coverage = min(float(safe.mean()), float(safe.sum()*self.config.sample_period_seconds/duration), 1.0) if len(safe) else 0.0
            row = {"window_id": window.window_id, "asset_id": window.asset_id,
                   "command_response_deviation": np.nan, "transient_concentration": np.nan,
                   "electrical_safe_fraction": coverage, "electrical_status": "unsupported"}
            scores, approximate = [], []
            if safe.sum() >= self.config.minimum_samples and coverage >= self.config.minimum_safe_fraction:
                rr = r[safe]
                row["command_response_deviation"] = float(np.linalg.norm(rr, axis=1).mean())
                for j, channel in enumerate(self.config.channels):
                    row[f"command_response_signed_{channel}"] = float(rr[:, j].mean())
                row["command_response_imbalance"] = float(np.std(rr, axis=1).mean())
                states = part[self.config.state_column].to_numpy()
                supported = np.zeros(len(part), bool)
                for state in pd.unique(states):
                    reference = self.references.get((str(window.asset_id), state))
                    x = causal_delay(r, safe & (states == state))
                    if reference is None or len(x) < self.config.minimum_samples:
                        continue
                    supported |= safe & (states == state)
                    d = self.operator(x, reference)
                    exact = float(np.linalg.eigvalsh(d)[-1])
                    scores.append(max(exact, 0.0)**0.25)
                    if self.solver is not None:
                        v = np.asarray(self.solver.solve(d), float)
                        if v.shape != (8,) or not np.isfinite(v).all() or not np.isclose(v @ v, 1):
                            raise ValueError("Solver returned an invalid normalized direction")
                        rayleigh = float(v @ d @ v)
                        approximate.append(max(rayleigh, 0.0)**0.25)
                row["transient_supported_fraction"] = min(float(supported.mean()), float(supported.sum()*self.config.sample_period_seconds/duration), 1.0)
                if scores and row["transient_supported_fraction"] >= self.config.minimum_safe_fraction:
                    row["transient_concentration"] = max(scores)
                    row["electrical_status"] = "supported"
                    if approximate:
                        row["transient_concentration_vqe"] = max(approximate)
                        row["transient_solver_deficit"] = max(scores)-max(approximate)
            rows.append(row)
        return pd.DataFrame(rows)
