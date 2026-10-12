"""Experimental evaluation utilities for Model_D."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import html
import json

import pandas as pd

from .config import ModelDConfig
from .electrical import CommandResponseObserver, TransientConcentration
from .episodes import IncidentEpisodeBuilder
from .evaluation import EpisodeEvaluator
from .interpretation import EventFrequencyInterpretationResult, FrequencyInterpretationConfig
from .metrics import summarize_episode_metrics
from .pipeline import IndicatorPipeline
from .registry import IncidentRegistry


@dataclass(frozen=True)
class ExperimentResult:
    """Artifacts produced by one experiment mode."""

    mode: str
    episodes: pd.DataFrame
    window_scores: pd.DataFrame
    evaluation_summary: pd.DataFrame
    registry_matches: pd.DataFrame
    family_summary: pd.DataFrame
    lead_time_summary: pd.DataFrame
    event_interpretation: EventFrequencyInterpretationResult | None = None


class ModelDExperimentRunner:
    """Run comparative experiments for semantic and indicator-driven detection."""

    def __init__(
        self, config: ModelDConfig | None = None, *,
        command_observer: CommandResponseObserver | None = None,
        transient_model: TransientConcentration | None = None,
        interpretation_config: FrequencyInterpretationConfig | None = None,
    ) -> None:
        self._config = config or ModelDConfig()
        self._builder = IncidentEpisodeBuilder(self._config)
        self._pipeline = IndicatorPipeline(self._config, command_observer=command_observer, transient_model=transient_model, interpretation_config=interpretation_config)
        self._evaluator = EpisodeEvaluator()
        self._registry = IncidentRegistry()

    def run_semantic(
        self,
        *,
        sequences: pd.DataFrame,
        assignments: pd.DataFrame,
        registry: pd.DataFrame | None = None,
    ) -> ExperimentResult:
        window_scores = self._builder.build_window_scores(sequences, assignments)
        episodes = self._builder.build_episodes_from_semantic_assignments(
            sequences,
            assignments,
            window_scores=window_scores,
        )
        return self._assemble_result("semantic", episodes, window_scores, registry)

    def run_indicators(
        self,
        *,
        sequences: pd.DataFrame,
        assignments: pd.DataFrame,
        registry: pd.DataFrame | None = None,
        analogue: pd.DataFrame | None = None,
        digital: pd.DataFrame | None = None,
    ) -> ExperimentResult:
        pipeline_result = self._pipeline.run(
            sequences=sequences,
            assignments=assignments,
            registry=registry,
            analogue=analogue,
            digital=digital,
        )
        return self._assemble_result("indicators", pipeline_result.episodes, pipeline_result.window_scores, registry, pipeline_result.event_interpretation)

    def run_comparison(
        self,
        *,
        sequences: pd.DataFrame,
        assignments: pd.DataFrame,
        registry: pd.DataFrame,
        analogue: pd.DataFrame | None = None,
        digital: pd.DataFrame | None = None,
    ) -> dict[str, ExperimentResult]:
        normalized_registry = self._registry.normalize(registry)
        return {
            "semantic": self.run_semantic(sequences=sequences, assignments=assignments, registry=normalized_registry),
            "indicators": self.run_indicators(
                sequences=sequences,
                assignments=assignments,
                registry=normalized_registry,
                analogue=analogue,
                digital=digital,
            ),
        }

    def export_report(self, results: dict[str, ExperimentResult], output_dir: str | Path) -> dict[str, Path]:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        written: dict[str, Path] = {}

        comparison_summary = self._comparison_summary(results)
        comparison_path = output_path / "comparison_summary.csv"
        comparison_summary.to_csv(comparison_path, index=False)
        written["comparison_summary"] = comparison_path

        for mode, result in results.items():
            mode_prefix = output_path / mode
            mode_prefix.mkdir(parents=True, exist_ok=True)
            for label, frame in {
                "episodes": result.episodes,
                "window_scores": result.window_scores,
                "evaluation_summary": result.evaluation_summary,
                "registry_matches": result.registry_matches,
                "family_summary": result.family_summary,
                "lead_time_summary": result.lead_time_summary,
            }.items():
                path = mode_prefix / f"{label}.csv"
                frame.to_csv(path, index=False)
                written[f"{mode}_{label}"] = path

            if result.event_interpretation is not None:
                for label, path in result.event_interpretation.export(mode_prefix / "event_interpretation").items():
                    written[f"{mode}_interpretation_{label}"] = path

        report_path = output_path / "experiment_report.html"
        report_path.write_text(self._build_html_report(results, comparison_summary), encoding="utf-8")
        written["html_report"] = report_path

        metadata_path = output_path / "experiment_metadata.json"
        metadata_path.write_text(
            json.dumps(
                {
                    "modes": list(results.keys()),
                    "artifacts": {key: str(path) for key, path in written.items()},
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        written["metadata"] = metadata_path
        return written

    def _assemble_result(
        self,
        mode: str,
        episodes: pd.DataFrame,
        window_scores: pd.DataFrame,
        registry: pd.DataFrame | None,
        event_interpretation: EventFrequencyInterpretationResult | None = None,
    ) -> ExperimentResult:
        normalized_registry = self._registry.normalize(registry) if registry is not None else pd.DataFrame()
        matches = self._evaluator.match(normalized_registry, episodes) if not normalized_registry.empty else pd.DataFrame()
        evaluation_summary = self._evaluator.summarize(normalized_registry, episodes) if not normalized_registry.empty else pd.DataFrame()
        family_summary = summarize_episode_metrics(episodes)
        lead_time_summary = self._lead_time_summary(matches)
        return ExperimentResult(mode, episodes, window_scores, evaluation_summary, matches, family_summary, lead_time_summary, event_interpretation)

    @staticmethod
    def _lead_time_summary(matches: pd.DataFrame) -> pd.DataFrame:
        if matches.empty or "lead_time_seconds" not in matches.columns:
            return pd.DataFrame(
                columns=[
                    "reference_family",
                    "match_count",
                    "mean_lead_time_hours",
                    "median_lead_time_hours",
                    "positive_lead_fraction",
                ]
            )
        frame = matches.copy()
        frame["lead_time_seconds"] = pd.to_numeric(frame["lead_time_seconds"], errors="coerce").fillna(0.0)
        rows: list[dict[str, Any]] = []
        for family, family_frame in frame.groupby("reference_family", dropna=False):
            rows.append(
                {
                    "reference_family": family,
                    "match_count": int(len(family_frame)),
                    "mean_lead_time_hours": float(family_frame["lead_time_seconds"].mean() / 3600.0),
                    "median_lead_time_hours": float(family_frame["lead_time_seconds"].median() / 3600.0),
                    "positive_lead_fraction": float(family_frame["lead_time_seconds"].gt(0).mean()),
                }
            )
        return pd.DataFrame(rows).sort_values("match_count", ascending=False)

    @staticmethod
    def _comparison_summary(results: dict[str, ExperimentResult]) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for mode, result in results.items():
            summary_row = result.evaluation_summary.iloc[0].to_dict() if not result.evaluation_summary.empty else {}
            rows.append(
                {
                    "mode": mode,
                    "episode_count": int(len(result.episodes)),
                    "window_count": int(len(result.window_scores)),
                    **summary_row,
                }
            )
        return pd.DataFrame(rows)

    def _build_html_report(self, results: dict[str, ExperimentResult], comparison_summary: pd.DataFrame) -> str:
        sections = [
            "<html><head><meta charset='utf-8'><title>Model_D Experimental Report</title>",
            "<style>body{font-family:Arial,sans-serif;margin:24px;}table{border-collapse:collapse;margin:16px 0;}th,td{border:1px solid #ccc;padding:6px 10px;}h1,h2{margin-top:24px;} .chart{margin:18px 0;padding:12px;border:1px solid #ddd;} svg text{font-size:12px;font-family:Arial,sans-serif;}</style>",
            "</head><body>",
            "<h1>Model_D Experimental Report</h1>",
            "<h2>Comparison Summary</h2>",
            comparison_summary.to_html(index=False),
            self._svg_bar_chart(comparison_summary, "mode", "episode_precision", "Episode Precision by Mode"),
            self._svg_bar_chart(comparison_summary, "mode", "family_precision", "Family Precision by Mode"),
        ]
        for mode, result in results.items():
            sections.extend(
                [
                    f"<h2>Mode: {html.escape(mode)}</h2>",
                    f"<p>Episodes: {len(result.episodes)} | Windows: {len(result.window_scores)}</p>",
                    "<h3>Evaluation Summary</h3>",
                    result.evaluation_summary.to_html(index=False) if not result.evaluation_summary.empty else "<p>No evaluation summary available.</p>",
                    "<h3>Family Summary</h3>",
                    result.family_summary.to_html(index=False) if not result.family_summary.empty else "<p>No family summary available.</p>",
                    self._svg_bar_chart(result.family_summary, "primary_family", "episode_count", f"Episode Count by Family ({mode})"),
                    self._svg_bar_chart(result.lead_time_summary, "reference_family", "mean_lead_time_hours", f"Mean Lead Time by Family ({mode})"),
                ]
            )
        sections.append("</body></html>")
        return "\n".join(sections)

    @staticmethod
    def _svg_bar_chart(frame: pd.DataFrame, category_col: str, value_col: str, title: str) -> str:
        if frame.empty or category_col not in frame.columns or value_col not in frame.columns:
            return f"<div class='chart'><strong>{html.escape(title)}</strong><p>No data.</p></div>"
        plotting = frame[[category_col, value_col]].copy()
        plotting[value_col] = pd.to_numeric(plotting[value_col], errors="coerce").fillna(0.0)
        if plotting.empty:
            return f"<div class='chart'><strong>{html.escape(title)}</strong><p>No data.</p></div>"
        max_value = float(plotting[value_col].max())
        max_value = max(max_value, 1e-9)
        width = 700
        bar_height = 26
        gap = 12
        left_pad = 180
        top_pad = 40
        height = top_pad + len(plotting) * (bar_height + gap) + 20
        bars: list[str] = [f"<div class='chart'><strong>{html.escape(title)}</strong><svg width='{width}' height='{height}'>"]
        bars.append(f"<text x='10' y='22'>{html.escape(title)}</text>")
        for idx, row in plotting.reset_index(drop=True).iterrows():
            y = top_pad + idx * (bar_height + gap)
            label = html.escape(str(row[category_col]))
            value = float(row[value_col])
            bar_width = 0 if max_value == 0 else int((width - left_pad - 80) * (value / max_value))
            bars.append(f"<text x='10' y='{y + 18}'>{label}</text>")
            bars.append(f"<rect x='{left_pad}' y='{y}' width='{bar_width}' height='{bar_height}' fill='#3b82f6'></rect>")
            bars.append(f"<text x='{left_pad + bar_width + 8}' y='{y + 18}'>{value:.3f}</text>")
        bars.append("</svg></div>")
        return "".join(bars)
