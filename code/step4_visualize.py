import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Chart style
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


# Figure 1: Pareto front
def plot_pareto_front(results_csv="data/optimization_results.csv"):
    df = pd.read_csv(results_csv)

    fig, ax = plt.subplots(figsize=(7, 5))

    # Scatter: all solutions
    ax.scatter(df["acs_usd_per_year"] / 1000, df["lpsp_pct"],
               c="steelblue", s=60, zorder=3, edgecolors="white", linewidth=0.5)

    # Line: Pareto front
    pareto_df = df.sort_values("acs_usd_per_year")
    ax.plot(pareto_df["acs_usd_per_year"] / 1000, pareto_df["lpsp_pct"],
            "o-", color="darkorange", linewidth=1.5, markersize=8, zorder=4,
            label="Pareto front (approx.)")

    # LPSP = 5% target line
    ax.axhline(y=5.0, color="red", linestyle="--", linewidth=0.8, alpha=0.6,
               label="Target: LPSP = 5%")

    ax.set_xlabel("Annualized System Cost (k$ / year)")
    ax.set_ylabel("Loss of Power Supply Probability LPSP (%)")
    ax.set_title("Pareto Front: System Cost vs Supply Reliability")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    for fmt in ["png", "pdf"]:
        fig.savefig(f"figures/fig_pareto_front.{fmt}")
    print("  Saved: figures/fig_pareto_front.png / .pdf")
    plt.close(fig)


# Figure 2: Annual SOC (summer & winter weeks)
def plot_annual_soc():
    weather = pd.read_csv("data/nasa_power_data.csv", index_col=0, parse_dates=True)
    ghi_data = weather["GHI"].values
    temp_data = weather["T_amb"].values

    from step2_model import (
        generate_load_profile, compute_pv_output, simulate_system,
        n_households, daily_demand_per_household, peak_factor,
        temp_coeff, derating_factor, soc_min, soc_initial,
        battery_rt_efficiency
    )

    load_data = generate_load_profile(n_households, daily_demand_per_household, peak_factor)

    # Optimal design from step 3
    pv_opt = 271.0
    bat_opt = 483.0

    pv_output = compute_pv_output(ghi_data, temp_data, pv_opt, temp_coeff, derating_factor)
    soc, deficit, curtailment = simulate_system(
        pv_output, load_data, bat_opt, soc_min, soc_initial, battery_rt_efficiency
    )

    time_index = weather.index

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=False)

    for ax, (label, start_date) in zip(
        axes,
        [("Summer week (Jul 15-21)", "2023-07-15"),
         ("Winter week (Jan 15-21)", "2023-01-15")]
    ):
        mask = (time_index >= start_date) & (time_index < pd.Timestamp(start_date) + pd.Timedelta(days=7))
        week_time = time_index[mask]
        week_soc = soc[mask]
        week_pv = pv_output[mask]
        week_load = load_data[mask]

        ax.plot(week_time, week_soc * 100, color="darkgreen", linewidth=0.8, label="SOC (%)")
        ax.fill_between(week_time, 0, week_pv, alpha=0.15, color="orange", label="PV output (kW)")
        ax.fill_between(week_time, 0, -week_load, alpha=0.1, color="red", label="Load (kW, negative)")

        ax.set_ylabel("SOC / Power")
        ax.set_title(label)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%a %H:%M"))

    fig.suptitle(f"Microgrid Operation: PV={pv_opt:.0f} kWp, Battery={bat_opt:.0f} kWh",
                 fontsize=14, y=1.01)
    plt.tight_layout()

    for fmt in ["png", "pdf"]:
        fig.savefig(f"figures/fig_annual_soc.{fmt}")
    print("  Saved: figures/fig_annual_soc.png / .pdf")
    plt.close(fig)


# Figure 3: Sensitivity heatmap
def plot_sensitivity():
    weather = pd.read_csv("data/nasa_power_data.csv", index_col=0, parse_dates=True)
    ghi_raw = weather["GHI"].values
    temp_data = weather["T_amb"].values

    from step2_model import (
        generate_load_profile, compute_pv_output, simulate_system,
        compute_lpsp,
        n_households, daily_demand_per_household, peak_factor,
        pv_unit_cost, pv_lifetime, temp_coeff, derating_factor,
        battery_unit_cost, battery_lifetime, battery_rt_efficiency,
        soc_min, soc_initial,
        project_lifetime, discount_rate, maintenance_ratio
    )

    load_data = generate_load_profile(n_households, daily_demand_per_household, peak_factor)

    # Parameter ranges
    battery_price_mult = np.array([0.5, 0.75, 1.0, 1.25, 1.5])
    solar_mult = np.array([0.7, 0.85, 1.0, 1.15, 1.3])

    # Fixed PV capacity
    pv_fixed = 270.0

    # Grid search: find the smallest battery that meets LPSP ≤ 5%
    battery_options = np.arange(50, 601, 25)

    result_grid = np.zeros((len(battery_price_mult), len(solar_mult)))

    print("  Running sensitivity grid search...")
    for i, price_mult in enumerate(battery_price_mult):
        for j, solar_mult_val in enumerate(solar_mult):
            ghi_adjusted = ghi_raw * solar_mult_val
            pv_output = compute_pv_output(ghi_adjusted, temp_data, pv_fixed, temp_coeff, derating_factor)

            best_battery = battery_options[-1]
            for bat_cap in battery_options:
                soc, deficit, curtailment = simulate_system(
                    pv_output, load_data, bat_cap, soc_min,
                    soc_initial, battery_rt_efficiency
                )
                lpsp_val = compute_lpsp(deficit, load_data)
                if lpsp_val <= 0.05:
                    best_battery = bat_cap
                    break

            result_grid[i, j] = best_battery

    # Draw heatmap
    fig, ax = plt.subplots(figsize=(7, 5.5))

    mesh = ax.pcolormesh(solar_mult * 100, battery_price_mult * 100,
                         result_grid, cmap="YlOrRd", shading="auto",
                         edgecolors="white", linewidth=0.5)

    # Annotate cells
    for i in range(len(battery_price_mult)):
        for j in range(len(solar_mult)):
            val = result_grid[i, j]
            ax.text(solar_mult[j] * 100, battery_price_mult[i] * 100,
                    f"{val:.0f}", ha="center", va="center", fontsize=9,
                    color="black" if val < 300 else "white")

    ax.set_xlabel("Solar irradiance (% of baseline)")
    ax.set_ylabel("Battery price (% of baseline)")
    ax.set_title(f"Required battery capacity (kWh) for LPSP <= 5%\nPV fixed at {pv_fixed:.0f} kWp")
    cbar = fig.colorbar(mesh, ax=ax, label="Battery capacity (kWh)")
    plt.tight_layout()

    for fmt in ["png", "pdf"]:
        fig.savefig(f"figures/fig_sensitivity_heatmap.{fmt}")
    print("  Saved: figures/fig_sensitivity_heatmap.png / .pdf")
    plt.close(fig)


# Figure 4: Load profile
def plot_load_profile():
    from step2_model import (
        generate_load_profile,
        n_households, daily_demand_per_household, peak_factor
    )

    load_data = generate_load_profile(n_households, daily_demand_per_household, peak_factor)

    # Average daily load (by hour)
    avg_hourly = np.zeros(24)
    for h in range(24):
        avg_hourly[h] = load_data.reshape(-1, 24)[:, h].mean()

    # Monthly totals
    days_per_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    monthly_total = []
    cursor = 0
    for days in days_per_month:
        monthly_total.append(load_data[cursor * 24:(cursor + days) * 24].sum())
        cursor = cursor + days

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # Daily load curve
    ax1.plot(range(24), avg_hourly, "o-", color="steelblue", linewidth=2, markersize=6)
    ax1.fill_between(range(24), 0, avg_hourly, alpha=0.15, color="steelblue")
    ax1.set_xlabel("Hour of day")
    ax1.set_ylabel("Average power (kW)")
    ax1.set_title("Typical daily load profile")
    ax1.set_xticks(range(0, 24, 3))
    ax1.grid(True, alpha=0.3)

    # Monthly load
    ax2.bar(month_names, np.array(monthly_total) / 1000, color="steelblue", alpha=0.8)
    ax2.set_xlabel("Month")
    ax2.set_ylabel("Monthly consumption (MWh)")
    ax2.set_title("Monthly energy consumption")
    ax2.tick_params(axis="x", rotation=45)
    ax2.grid(True, alpha=0.3, axis="y")

    fig.suptitle(f"Village load profile ({n_households} households)", fontsize=13)
    plt.tight_layout()

    for fmt in ["png", "pdf"]:
        fig.savefig(f"figures/fig_load_profile.{fmt}")
    print("  Saved: figures/fig_load_profile.png / .pdf")
    plt.close(fig)


if __name__ == "__main__":
    print("Generating charts...\n")

    try:
        plot_pareto_front()
    except FileNotFoundError:
        print("  [Skipped] Pareto front -- run step3_optimize.py first")

    try:
        plot_annual_soc()
    except Exception as e:
        print(f"  [Skipped] Annual SOC -- {e}")

    try:
        plot_sensitivity()
    except Exception as e:
        print(f"  [Skipped] Sensitivity -- {e}")

    plot_load_profile()

    print("\nAll charts generated. Ready for paper and poster.")
