"""Phase 10: queueing model, capacity allocation and budget scenarios.

Offered load is recovered per zone by inverting the Erlang loss formula on the
observed occupancy, the marginal value of one more charging point is measured
both analytically and by simulation across seeds, and a fixed budget is then
allocated under three rules and evaluated by simulation.

    python scripts/run_phase10_capacity.py
"""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from uev import progress
from uev.config import CONFIG
from uev.graph import neighbour_weights
from uev.io_load import load_bundle
from uev.logging_utils import get_logger, timed
from uev.paths import ensure_dirs
from uev.queueing import (allocate_by_weight, analytic_marginal, calibrate,
                          greedy_marginal_allocation, peak_window, simulate,
                          simulated_marginal)
from uev.resultsio import write_table

LOG = get_logger("phase10")


def main() -> int:
    parser = argparse.ArgumentParser(description="capacity model and allocation")
    parser.add_argument("--seeds", type=int, default=CONFIG.capacity.n_seeds)
    parser.add_argument("--estimate", default="rated", choices=["rated", "11kw"])
    args = parser.parse_args()

    ensure_dirs()
    progress.mark_phase(10, "running")
    settings = CONFIG.capacity
    bundle = load_bundle()

    with timed("queueing calibration", LOG):
        calibration = calibrate(bundle)
    weights = neighbour_weights(bundle.adjacency, bundle.distance).to_numpy(dtype=float)

    hours = peak_window(calibration, settings.sim_hours)
    LOG.info("scenarios replay the busiest %d hours: %s to %s", settings.sim_hours,
             calibration.index[hours.start].strftime("%Y-%m-%d"),
             calibration.index[hours.stop - 1].strftime("%Y-%m-%d"))

    utilisation = (calibration.carried / calibration.capacity[None, :]).mean(axis=0)
    write_table("queue_calibration.csv", pd.DataFrame({
        "zone_id": calibration.zone_ids,
        "capacity_points": calibration.capacity,
        "mean_service_hours": calibration.service_hours,
        "kw_per_point_hour_rated": calibration.kw_rated,
        "kw_per_point_hour_11kw": calibration.kw_11,
        "mean_utilisation": utilisation,
        "mean_offered_erlangs": calibration.offered.mean(axis=0),
        "mean_carried_erlangs": calibration.carried.mean(axis=0),
        "peak_offered_erlangs": calibration.offered.max(axis=0),
    }))

    marginal_rows = {}
    for estimate in ("rated", "11kw"):
        with timed(f"simulated marginal value, {estimate} estimate, {args.seeds} seeds", LOG):
            simulated = simulated_marginal(calibration, hours, weights, args.seeds, estimate)
        simulated["analytic_marginal_kwh"] = analytic_marginal(calibration, hours, estimate)
        simulated["estimate"] = estimate
        simulated["mean_utilisation"] = utilisation
        marginal_rows[estimate] = simulated

        # Most zones never fill, so an extra point there changes nothing and both
        # methods return zero. A correlation over all zones is therefore mostly
        # a correlation between ties, and the agreement that matters is among
        # the zones where capacity actually binds.
        active = ((simulated["marginal_energy_kwh_mean"] > 0)
                  | (simulated["analytic_marginal_kwh"] >= 1.0))
        rho_all = spearmanr(simulated["marginal_energy_kwh_mean"],
                            simulated["analytic_marginal_kwh"]).statistic
        rho_active = spearmanr(simulated.loc[active, "marginal_energy_kwh_mean"],
                               simulated.loc[active, "analytic_marginal_kwh"]).statistic
        simulated["is_active_zone"] = active
        LOG.info("%s estimate: simulated and analytic marginals agree at Spearman "
                 "%.4f over all zones, %.4f over the %d zones where capacity binds",
                 estimate, rho_all, rho_active, int(active.sum()))

    marginal = pd.concat(marginal_rows.values(), ignore_index=True)
    write_table("marginal_value.csv", marginal)

    # Claim 3, first half: do the two rankings disagree.
    ranking_rows = []
    for estimate, frame in marginal_rows.items():
        rho = spearmanr(frame["mean_utilisation"], frame["marginal_energy_kwh_mean"]).statistic
        rho_analytic = spearmanr(frame["mean_utilisation"],
                                 frame["analytic_marginal_kwh"]).statistic
        top_util = set(frame.nlargest(20, "mean_utilisation")["zone_id"])
        top_marginal = set(frame.nlargest(20, "marginal_energy_kwh_mean")["zone_id"])
        active = frame["is_active_zone"]
        ranking_rows.append({
            "estimate": estimate,
            "n_zones_with_binding_capacity": int(active.sum()),
            "n_zones_with_no_marginal_value": int((~active).sum()),
            "spearman_methods_all_zones": spearmanr(
                frame["marginal_energy_kwh_mean"], frame["analytic_marginal_kwh"]).statistic,
            "spearman_methods_active_zones": spearmanr(
                frame.loc[active, "marginal_energy_kwh_mean"],
                frame.loc[active, "analytic_marginal_kwh"]).statistic,
            "spearman_utilisation_vs_simulated_marginal": rho,
            "spearman_utilisation_vs_analytic_marginal": rho_analytic,
            "spearman_simulated_vs_analytic_marginal": spearmanr(
                frame["marginal_energy_kwh_mean"], frame["analytic_marginal_kwh"]).statistic,
            "top20_overlap": len(top_util & top_marginal),
            "rankings_disagree_materially": bool(rho <= 0.9),
        })
        LOG.info("%s estimate: Spearman(utilisation, marginal energy) = %.4f, "
                 "top twenty overlap %d of 20", estimate, rho, len(top_util & top_marginal))
    write_table("ranking_comparison.csv", pd.DataFrame(ranking_rows))

    # Claim 3, second half: does the allocation rule change served energy.
    scenario_rows = []
    allocation_rows = []
    for estimate in ("rated", "11kw"):
        frame = marginal_rows[estimate]
        for budget in settings.budget_points:
            rules = {
                "by_utilisation": allocate_by_weight(utilisation, budget),
                "by_marginal_energy": greedy_marginal_allocation(calibration, hours,
                                                                 budget, estimate),
            }
            rules["balanced"] = (allocate_by_weight(utilisation, budget // 2)
                                 + greedy_marginal_allocation(calibration, hours,
                                                              budget - budget // 2, estimate))
            rules["no_investment"] = np.zeros(len(calibration.zone_ids), dtype=int)

            names = list(rules)
            capacities = np.vstack([calibration.capacity + rules[name] for name in names])
            with timed(f"{estimate} budget {budget}: {args.seeds} seeds", LOG):
                per_seed = np.zeros((args.seeds, len(names)))
                unserved = np.zeros((args.seeds, len(names)))
                for seed in range(args.seeds):
                    outcome = simulate(calibration, capacities, hours, weights,
                                       f"allocate:{estimate}:{budget}:{seed}",
                                       estimate=estimate)
                    per_seed[seed] = outcome["served_energy"].sum(axis=1)
                    unserved[seed] = outcome["unserved_sessions"]

            for i, name in enumerate(names):
                scenario_rows.append({
                    "estimate": estimate, "budget_points": budget, "rule": name,
                    "n_seeds": args.seeds,
                    "served_energy_kwh_mean": float(per_seed[:, i].mean()),
                    "served_energy_kwh_std": float(per_seed[:, i].std()),
                    "served_energy_kwh_p10": float(np.quantile(per_seed[:, i], 0.10)),
                    "served_energy_kwh_p90": float(np.quantile(per_seed[:, i], 0.90)),
                    "unserved_sessions_mean": float(unserved[:, i].mean()),
                    "unserved_sessions_p90": float(np.quantile(unserved[:, i], 0.90)),
                    "zones_receiving_points": int((rules[name] > 0).sum()),
                    "max_points_to_one_zone": int(rules[name].max()),
                })
                for zone_index, points in enumerate(rules[name]):
                    if points > 0:
                        allocation_rows.append({
                            "estimate": estimate, "budget_points": budget, "rule": name,
                            "zone_id": calibration.zone_ids[zone_index],
                            "points_added": int(points),
                        })

    scenarios = pd.DataFrame(scenario_rows)
    write_table("capacity_scenarios.csv", scenarios)
    write_table("capacity_allocations.csv", pd.DataFrame(allocation_rows))

    verdict_rows = []
    for estimate in ("rated", "11kw"):
        for budget in settings.budget_points:
            part = scenarios[(scenarios["estimate"] == estimate)
                             & (scenarios["budget_points"] == budget)].set_index("rule")
            by_util = part.loc["by_utilisation", "served_energy_kwh_mean"]
            by_marginal = part.loc["by_marginal_energy", "served_energy_kwh_mean"]
            base = part.loc["no_investment", "served_energy_kwh_mean"]
            verdict_rows.append({
                "estimate": estimate, "budget_points": budget,
                "served_no_investment_kwh": base,
                "served_by_utilisation_kwh": by_util,
                "served_by_marginal_kwh": by_marginal,
                "served_balanced_kwh": part.loc["balanced", "served_energy_kwh_mean"],
                "gain_by_utilisation_kwh": by_util - base,
                "gain_by_marginal_kwh": by_marginal - base,
                "marginal_rule_serves_more": bool(by_marginal > by_util),
                "extra_energy_from_marginal_rule_kwh": by_marginal - by_util,
            })
    verdict = pd.DataFrame(verdict_rows)
    write_table("capacity_claim3.csv", verdict)

    headline = verdict[verdict["estimate"] == "rated"]
    wins = int(headline["marginal_rule_serves_more"].sum())
    LOG.info("claim 3: the marginal energy rule serves more at %d of %d budgets "
             "under the rated power estimate", wins, len(headline))
    for _, row in headline.iterrows():
        LOG.info("budget %4d: utilisation rule %+.0f kWh, marginal rule %+.0f kWh, "
                 "difference %+.0f kWh", int(row["budget_points"]),
                 row["gain_by_utilisation_kwh"], row["gain_by_marginal_kwh"],
                 row["extra_energy_from_marginal_rule_kwh"])

    # Sensitivity on the one parameter the data cannot identify. Mean session
    # length is assumed from the power of the zone's points.
    #
    # The analytic marginal is invariant to this assumption by construction:
    # offered load is recovered in erlangs, which carry no session length, and
    # energy is obtained per point hour. The simulation is not invariant,
    # because session length sets both the arrival rate and the energy each
    # session delivers, so the sensitivity is run on the simulated marginal.
    sensitivity_rows = []
    sensitivity_seeds = max(4, args.seeds // 2)
    for scale in (0.5, 1.0, 1.5):
        scaled = calibrate(bundle, service_scale=scale)
        simulated = simulated_marginal(scaled, hours, weights, sensitivity_seeds, "rated")
        values = simulated["marginal_energy_kwh_mean"]
        rho = spearmanr(utilisation, values).statistic
        active = values > 0
        sensitivity_rows.append({
            "service_hours_scale": scale,
            "median_assumed_session_hours": float(np.median(scaled.service_hours)),
            "n_seeds": sensitivity_seeds,
            "spearman_utilisation_vs_simulated_marginal": rho,
            "spearman_utilisation_vs_analytic_marginal": spearmanr(
                utilisation, analytic_marginal(scaled, hours, "rated")).statistic,
            "n_zones_with_marginal_value": int(active.sum()),
            "rankings_disagree_materially": bool(rho <= 0.9),
            "total_marginal_energy_kwh": float(values.sum()),
        })
        LOG.info("session length scale %.1f (median %.2f h): Spearman %.4f, "
                 "%d zones with any marginal value, total marginal %.3e kWh",
                 scale, float(np.median(scaled.service_hours)), rho,
                 int(active.sum()), float(values.sum()))
    write_table("capacity_sensitivity.csv", pd.DataFrame(sensitivity_rows))

    progress.mark_phase(10, "complete",
                        f"marginal rule wins at {wins} of {len(headline)} budgets")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
