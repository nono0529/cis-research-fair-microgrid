import numpy as np
import pandas as pd
from pyswarm import pso
import warnings
warnings.filterwarnings("ignore")

# Import simulation functions and parameters from step2
from step2_model import (
    compute_pv_output, simulate_system, compute_lpsp, compute_annualized_cost,
    generate_load_profile,
    n_households, daily_demand_per_household, peak_factor,
    pv_unit_cost, pv_lifetime, temp_coeff, derating_factor,
    battery_unit_cost, battery_lifetime, battery_rt_efficiency,
    depth_of_discharge, soc_min, soc_initial,
    project_lifetime, discount_rate, maintenance_ratio
)

# Load data
print("Loading data...")
weather = pd.read_csv("data/nasa_power_data.csv", index_col=0, parse_dates=True)
ghi_data = weather["GHI"].values
temp_data = weather["T_amb"].values
load_data = generate_load_profile(n_households, daily_demand_per_household, peak_factor)
annual_total_load = load_data.sum()

print(f"  Load data: {n_households} households, {annual_total_load:,.0f} kWh/year")
print(f"  Weather data: {len(ghi_data)} hours\n")


# Objective function for PSO
def objective(x, cost_weight=1.0, lpsp_weight=1.0):
    """Weighted-sum objective for PSO.  x = [pv_capacity_kWp, battery_capacity_kWh]."""

    pv_cap, bat_cap = x

    # Sanity check
    if pv_cap < 0 or bat_cap < 0:
        return 1e12

    # Compute PV output
    pv_output = compute_pv_output(ghi_data, temp_data, pv_cap, temp_coeff, derating_factor)

    # Simulate system
    soc, deficit, curtailment = simulate_system(
        pv_output, load_data, bat_cap, soc_min, soc_initial, battery_rt_efficiency
    )

    # Compute metrics
    lpsp = compute_lpsp(deficit, load_data)
    annual_cost = compute_annualized_cost(
        pv_cap, pv_unit_cost, pv_lifetime,
        bat_cap, battery_unit_cost, battery_lifetime,
        project_lifetime, discount_rate, maintenance_ratio
    )

    # Normalize cost to 0-1 range
    cost_norm = annual_cost / 100000.0
    lpsp_norm = lpsp

    # Penalty for exceeding 5% LPSP
    penalty = 0
    if lpsp > 0.05:
        penalty = (lpsp - 0.05) * 100

    return cost_weight * cost_norm + lpsp_weight * lpsp_norm + penalty


# Search bounds
lower_bound = [20.0,  50.0]     # [PV (kWp), Battery (kWh)]
upper_bound = [300.0, 800.0]

# Weight combinations for Pareto front tracing
weight_combinations = [
    (1.0, 0.0),     # cost only (LPSP constraint enforced via penalty)
    (0.8, 0.2),
    (0.5, 0.5),
    (0.2, 0.8),
    (0.0, 1.0),     # LPSP only
]

print("Running PSO optimization...")
print(f"{'Weight (cost, LPSP)':<28s} {'PV (kWp)':>10s} {'Battery (kWh)':>10s} {'ACS ($/yr)':>12s} {'LPSP (%)':>10s}")
print("-" * 75)

results_list = []

for cost_w, lpsp_w in weight_combinations:
    opt_result = pso(
        func=lambda x: objective(x, cost_w, lpsp_w),
        lb=lower_bound,
        ub=upper_bound,
        swarmsize=30,
        maxiter=50,
        debug=False,
    )
    pv_opt, bat_opt = opt_result.x

    # Recompute to get accurate metrics
    pv_output = compute_pv_output(ghi_data, temp_data, pv_opt, temp_coeff, derating_factor)
    soc, deficit, curtailment = simulate_system(
        pv_output, load_data, bat_opt, soc_min, soc_initial, battery_rt_efficiency
    )
    lpsp_opt = compute_lpsp(deficit, load_data)
    cost_opt = compute_annualized_cost(
        pv_opt, pv_unit_cost, pv_lifetime,
        bat_opt, battery_unit_cost, battery_lifetime,
        project_lifetime, discount_rate, maintenance_ratio
    )

    results_list.append({
        "w_cost": cost_w,
        "w_lpsp": lpsp_w,
        "pv_kwp": pv_opt,
        "bat_kwh": bat_opt,
        "acs_usd_per_year": cost_opt,
        "lpsp_pct": lpsp_opt * 100,
    })

    print(f"  ({cost_w:.1f}, {lpsp_w:.1f})                     {pv_opt:>8.1f}   {bat_opt:>8.1f}   ${cost_opt:>10,.0f}   {lpsp_opt*100:>8.4f}")

# Save results
results_df = pd.DataFrame(results_list)
results_df.to_csv("data/optimization_results.csv", index=False)
print(f"\nResults saved to data/optimization_results.csv")

# Identify Pareto-optimal solutions
pareto_idx = []
for i, row1 in results_df.iterrows():
    dominated = False
    for j, row2 in results_df.iterrows():
        if (row2["acs_usd_per_year"] <= row1["acs_usd_per_year"] and
            row2["lpsp_pct"] <= row1["lpsp_pct"] and
            (row2["acs_usd_per_year"] < row1["acs_usd_per_year"] or
             row2["lpsp_pct"] < row1["lpsp_pct"])):
            dominated = True
            break
    if not dominated:
        pareto_idx.append(i)

print(f"\nPareto optimal solutions (indices): {pareto_idx}")
for i in pareto_idx:
    row = results_df.iloc[i]
    print(f"  PV={row['pv_kwp']:.0f} kWp, Battery={row['bat_kwh']:.0f} kWh, "
          f"ACS=${row['acs_usd_per_year']:,.0f}/yr, LPSP={row['lpsp_pct']:.4f}%")
