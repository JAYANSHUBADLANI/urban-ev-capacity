"""Resolution of the three pre-registered claims.

Each function returns the numbers the claim was registered against and a verdict
of ``supported`` or ``refuted``, applying the refutation condition exactly as it
was written in ``docs/preregistration.md``. Nothing is softened because a claim
failed: a refuted claim is a result, and reporting it is the reason for
registering it.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .logging_utils import get_logger

LOG = get_logger("claims")

GLOBAL_FAMILIES = ("global_ridge", "global_gbm")
ZONE_FAMILIES = ("zone_ridge", "zone_gbm")


def resolve_claim_1(grid_metrics: pd.DataFrame, zone_metrics: pd.DataFrame,
                    deciles: pd.DataFrame) -> Tuple[Dict, pd.DataFrame, pd.DataFrame]:
    """A pooled model beats per zone models, and more so for quieter zones."""
    pooled = (grid_metrics.groupby(["target", "horizon", "family"])["mase"]
              .mean().reset_index())

    rows = []
    for (target, horizon), part in pooled.groupby(["target", "horizon"]):
        indexed = part.set_index("family")["mase"]
        best_global = indexed.reindex(GLOBAL_FAMILIES).dropna()
        best_zone = indexed.reindex(ZONE_FAMILIES).dropna()
        if best_global.empty or best_zone.empty:
            continue
        rows.append({
            "target": target, "horizon": horizon,
            "best_global_family": best_global.idxmin(),
            "best_global_mase": float(best_global.min()),
            "best_per_zone_family": best_zone.idxmin(),
            "best_per_zone_mase": float(best_zone.min()),
            "global_advantage_mase": float(best_zone.min() - best_global.min()),
            "global_wins": bool(best_global.min() < best_zone.min()),
        })
    comparison = pd.DataFrame(rows)

    per_zone = (zone_metrics.groupby(["zone_id", "family"])["mase"].mean()
                .unstack("family"))
    available_global = [f for f in GLOBAL_FAMILIES if f in per_zone.columns]
    available_zone = [f for f in ZONE_FAMILIES if f in per_zone.columns]
    advantage = (per_zone[available_zone].min(axis=1)
                 - per_zone[available_global].min(axis=1)).rename("advantage")
    by_zone = advantage.reset_index().merge(
        deciles[["zone_id", "volume_decile", "total_volume"]], on="zone_id")

    decile_table = (by_zone.groupby("volume_decile")
                    .agg(n_zones=("zone_id", "nunique"),
                         mean_advantage=("advantage", "mean"),
                         median_advantage=("advantage", "median"),
                         share_favouring_pooling=("advantage", lambda s: float((s > 0).mean())))
                    .reset_index())

    rho = float(spearmanr(by_zone["volume_decile"], by_zone["advantage"]).statistic)
    lowest = decile_table.loc[decile_table["volume_decile"].idxmin(), "mean_advantage"]
    highest = decile_table.loc[decile_table["volume_decile"].idxmax(), "mean_advantage"]

    n_combinations = len(comparison)
    n_global_wins = int(comparison["global_wins"].sum())
    global_wins_majority = n_global_wins > n_combinations / 2
    advantage_increases_as_volume_falls = rho < 0

    verdict = ("supported" if (global_wins_majority and advantage_increases_as_volume_falls)
               else "refuted")
    summary = {
        "claim": 1,
        "verdict": verdict,
        "n_target_horizon_combinations": n_combinations,
        "n_won_by_pooled_model": n_global_wins,
        "pooled_model_wins_majority": bool(global_wins_majority),
        "spearman_decile_vs_advantage": rho,
        "advantage_increases_as_volume_falls": bool(advantage_increases_as_volume_falls),
        "mean_advantage_lowest_decile": float(lowest),
        "mean_advantage_highest_decile": float(highest),
        "n_zones": int(by_zone["zone_id"].nunique()),
    }
    return summary, comparison, decile_table


def resolve_claim_2(verdicts: pd.DataFrame) -> Tuple[Dict, pd.DataFrame]:
    """The input monitor alerts later than the residual monitor."""
    primary = (verdicts[verdicts["variant"] == "preregistered"]
               if "variant" in verdicts.columns else verdicts)
    n_rates = len(primary)
    n_later = int(primary["input_later_than_residual"].sum())
    verdict = "supported" if n_later > n_rates / 2 else "refuted"

    summary = {
        "claim": 2,
        "verdict": verdict,
        "n_matched_rates": n_rates,
        "n_rates_where_input_is_later": n_later,
        "best_input_delay_hours": float(primary["best_input_delay_hours"].min()),
        "best_residual_delay_hours": float(primary["best_residual_delay_hours"].min()),
        "residual_monitor_ever_detected": bool(
            np.isfinite(primary["best_residual_delay_hours"]).any()),
    }
    return summary, verdicts


def resolve_claim_3(ranking: pd.DataFrame, capacity: pd.DataFrame,
                    estimate: str = "rated") -> Tuple[Dict, pd.DataFrame]:
    """The two rankings disagree, and the marginal rule serves more energy."""
    rank_row = ranking[ranking["estimate"] == estimate].iloc[0]
    rho = float(rank_row["spearman_utilisation_vs_simulated_marginal"])
    rankings_disagree = rho <= 0.9

    budgets = capacity[capacity["estimate"] == estimate]
    n_budgets = len(budgets)
    n_marginal_wins = int(budgets["marginal_rule_serves_more"].sum())
    marginal_wins_majority = n_marginal_wins > n_budgets / 2

    verdict = "supported" if (rankings_disagree and marginal_wins_majority) else "refuted"
    summary = {
        "claim": 3,
        "verdict": verdict,
        "estimate": estimate,
        "spearman_utilisation_vs_marginal": rho,
        "rankings_disagree_materially": bool(rankings_disagree),
        "n_budgets": n_budgets,
        "n_budgets_won_by_marginal_rule": n_marginal_wins,
        "marginal_rule_wins_majority": bool(marginal_wins_majority),
        "mean_extra_energy_kwh": float(budgets["extra_energy_from_marginal_rule_kwh"].mean()),
        "top20_overlap": int(rank_row["top20_overlap"]),
        "n_zones_with_no_marginal_value": int(
            rank_row.get("n_zones_with_no_marginal_value", np.nan))
        if "n_zones_with_no_marginal_value" in rank_row else -1,
    }
    return summary, budgets
