"""
K2 Aerospace — Trade Study Engine
====================================
Multi-configuration comparison for engineering decision-making.

Features:
  • Evaluate rocket configurations via Monte-Carlo
  • Min-max normalisation across metrics
  • Weighted composite scoring & ranking
  • CSV / JSON export

No Qt imports — pure computation, thread-safe.
"""

from __future__ import annotations

import csv
import json
import copy
import logging
from dataclasses import dataclass, field

import numpy as np

from core.batch_simulation import BatchSimConfig, BatchSimResult, run_batch_simulation

logger = logging.getLogger("K2.TradeStudy")


# ══════════════════════════════════════════════════════════════════════════════
#  DATA  CLASSES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class RocketConfiguration:
    name: str = ""
    config: BatchSimConfig = None
    mc_results: dict = field(default_factory=dict)
    # Expected keys: mean_apogee, std_apogee, mean_mach, mean_stability,
    #   mean_landing_dist, mean_rail_exit, success_rate, mean_accel, mass


@dataclass
class TradeStudyConfig:
    configurations: list = field(default_factory=list)  # list[RocketConfiguration]
    metrics: list = field(default_factory=lambda: [
        "mean_apogee", "mean_mach", "mean_stability",
        "mean_landing_dist", "success_rate", "mass",
    ])
    # direction: True = higher is better, False = lower is better
    metric_directions: dict = field(default_factory=lambda: {
        "mean_apogee": True,
        "mean_mach": False,      # lower Mach → less structural load
        "mean_stability": True,
        "mean_landing_dist": False,
        "mean_rail_exit": True,
        "success_rate": True,
        "mass": False,
    })
    weights: dict = field(default_factory=lambda: {
        "mean_apogee": 1.0,
        "mean_mach": 0.5,
        "mean_stability": 0.8,
        "mean_landing_dist": 0.7,
        "mean_rail_exit": 0.6,
        "success_rate": 1.0,
        "mass": 0.6,
    })


@dataclass
class TradeStudyResult:
    comparison_matrix: dict = field(default_factory=dict)
    normalized_scores: dict = field(default_factory=dict)
    weighted_scores: dict = field(default_factory=dict)
    rankings: list = field(default_factory=list)
    best_config: str = ""


# ══════════════════════════════════════════════════════════════════════════════
#  EVALUATE  A  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_configuration(name: str, config: BatchSimConfig,
                           mc_sims: int = 50) -> RocketConfiguration:
    """Run Monte-Carlo evaluation of a single rocket configuration."""
    apogees, machs, stabs, landings, rail_exits, accels, successes = (
        [], [], [], [], [], [], [])

    for i in range(mc_sims):
        try:
            res = run_batch_simulation(config, seed=42 + i * 7)
            apogees.append(res.apogee)
            machs.append(res.max_mach)
            stabs.append(res.min_stability_margin)
            landings.append(res.landing_distance)
            rail_exits.append(res.rail_exit_velocity)
            accels.append(res.max_acceleration)
            successes.append(1.0 if res.success else 0.0)
        except Exception:
            apogees.append(0.0)
            machs.append(0.0)
            stabs.append(0.0)
            landings.append(9999.0)
            rail_exits.append(0.0)
            accels.append(0.0)
            successes.append(0.0)

    total_mass = config.dry_mass + config.propellant_mass

    mc = {
        "mean_apogee": float(np.mean(apogees)),
        "std_apogee": float(np.std(apogees)),
        "mean_mach": float(np.mean(machs)),
        "mean_stability": float(np.mean(stabs)),
        "mean_landing_dist": float(np.mean(landings)),
        "mean_rail_exit": float(np.mean(rail_exits)),
        "mean_accel": float(np.mean(accels)),
        "success_rate": float(np.mean(successes)),
        "mass": total_mass,
        "n_sims": mc_sims,
    }

    logger.info(f"Evaluated '{name}': apogee={mc['mean_apogee']:.1f}m, "
                f"success={mc['success_rate']*100:.0f}%")

    return RocketConfiguration(name=name, config=config, mc_results=mc)


# ══════════════════════════════════════════════════════════════════════════════
#  NORMALISATION  &  SCORING
# ══════════════════════════════════════════════════════════════════════════════

def normalize_metrics(configurations: list, metric_directions: dict) -> dict:
    """Min-max normalisation per metric to [0, 1] range.

    Higher = better for all metrics after normalisation, i.e.
    "lower is better" metrics are flipped.

    Returns {config_name: {metric: normalised_value}}.
    """
    metrics = set()
    for cfg in configurations:
        metrics.update(cfg.mc_results.keys())
    metrics = [m for m in metrics if m in metric_directions]

    result = {}
    for cfg in configurations:
        result[cfg.name] = {}

    for m in metrics:
        values = [cfg.mc_results.get(m, 0) for cfg in configurations]
        vmin, vmax = min(values), max(values)
        spread = vmax - vmin if vmax != vmin else 1.0

        higher_is_better = metric_directions.get(m, True)

        for cfg in configurations:
            raw = cfg.mc_results.get(m, 0)
            normed = (raw - vmin) / spread
            if not higher_is_better:
                normed = 1.0 - normed
            result[cfg.name][m] = float(normed)

    return result


def compute_weighted_scores(normalized: dict, weights: dict) -> dict:
    """Compute weighted composite score per configuration.

    Returns {config_name: composite_score}.
    """
    scores = {}
    for name, metrics in normalized.items():
        total = 0.0
        weight_sum = 0.0
        for m, val in metrics.items():
            w = weights.get(m, 1.0)
            total += w * val
            weight_sum += w
        scores[name] = total / weight_sum if weight_sum > 0 else 0.0
    return scores


def rank_configurations(study_config: TradeStudyConfig) -> TradeStudyResult:
    """Full trade-study comparison with ranking.

    Returns a TradeStudyResult with raw comparison, normalised scores,
    weighted composite scores, and overall ranking.
    """
    configs = study_config.configurations
    if not configs:
        return TradeStudyResult()

    # Raw comparison
    comparison = {cfg.name: dict(cfg.mc_results) for cfg in configs}

    # Normalise
    normalised = normalize_metrics(configs, study_config.metric_directions)

    # Weighted scores
    weighted = compute_weighted_scores(normalised, study_config.weights)

    # Rank (highest score first)
    ranking = sorted(weighted.items(), key=lambda x: x[1], reverse=True)
    ranking_names = [name for name, _ in ranking]
    best = ranking_names[0] if ranking_names else ""

    logger.info(f"Trade study ranking: {ranking_names}")

    return TradeStudyResult(
        comparison_matrix=comparison,
        normalized_scores=normalised,
        weighted_scores=weighted,
        rankings=ranking_names,
        best_config=best,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  EXPORTS
# ══════════════════════════════════════════════════════════════════════════════

def export_comparison_csv(result: TradeStudyResult, filepath: str):
    """Export trade-study comparison matrix to CSV."""
    if not result.comparison_matrix:
        return

    configs = list(result.comparison_matrix.keys())
    all_metrics = set()
    for m in result.comparison_matrix.values():
        all_metrics.update(m.keys())
    metrics = sorted(all_metrics)

    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Configuration"] + metrics +
                         ["Weighted Score", "Rank"])
        for rank, name in enumerate(result.rankings, 1):
            raw = result.comparison_matrix.get(name, {})
            row = [name]
            row += [f"{raw.get(m, 0):.4f}" for m in metrics]
            row.append(f"{result.weighted_scores.get(name, 0):.4f}")
            row.append(rank)
            writer.writerow(row)

    logger.info(f"Trade study CSV exported: {filepath}")


def export_comparison_json(result: TradeStudyResult, filepath: str):
    """Export trade-study results to JSON."""
    data = {
        "rankings": result.rankings,
        "best_configuration": result.best_config,
        "weighted_scores": result.weighted_scores,
        "normalized_scores": result.normalized_scores,
        "raw_metrics": result.comparison_matrix,
    }
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2, default=str)
    logger.info(f"Trade study JSON exported: {filepath}")
