import numpy as np
import pandas as pd
import requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pyswarm import pso
import warnings
warnings.filterwarnings("ignore")

# Import simulation functions from step2
from step2_model import (
    compute_pv_output, simulate_system, compute_lpsp, compute_annualized_cost,
    generate_load_profile,
    temp_coeff, derating_factor, battery_rt_efficiency,
    soc_min, soc_initial,
    project_lifetime, discount_rate, maintenance_ratio
)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

# Cameroon parameters
CAM_LAT = 5.0
CAM_LON = 12.0

CAM_START = "20230101"
CAM_END = "20231231"
CAM_PARAMS = "ALLSKY_SFC_SW_DWN,T2M"

cam_households = 200
cam_daily_demand = 2.5               # kWh per household per day
cam_peak_factor = 1.6

cam_pv_unit_cost = 650               # USD/kWp (higher logistics costs in Africa)
cam_battery_unit_cost = 260           # USD/kWh
cam_pv_lifetime = 25
cam_battery_lifetime = 10
cam_project_lifetime = 20
cam_discount_rate = 0.08              # higher cost of capital in Africa
cam_maintenance_ratio = 0.025


# ------------------------------------------------------------------
# Cameroon load profile
# ------------------------------------------------------------------
def generate_cam_load_profile(n_households, daily_demand, peak_factor):
    """Generate Cameroon rural load profile with distinct wet/dry seasons."""

    hour_index = np.arange(8760)
    hour_of_day = hour_index % 24

    avg_hourly_per_household = daily_demand / 24

    # Cameroon rural load shape (Central Africa literature)
    # Morning cooking peak 5-7am, evening lighting peak 18-21pm
    shape_factor = np.array([
        0.25, 0.22, 0.20, 0.22, 0.30, 0.55,   # 0-5h
        0.80, 0.65, 0.55, 0.50, 0.52, 0.55,   # 6-11h
        0.58, 0.60, 0.62, 0.60, 0.65, 0.72,   # 12-17h
        0.90, 1.20, 1.45, 1.35, 1.10, 0.60,   # 18-23h
    ])

    shape_factor = shape_factor / shape_factor.mean()

    hourly_load = np.zeros(8760)
    for h in range(8760):
        base = avg_hourly_per_household * n_households
        h_idx = hour_of_day[h]
        hourly_load[h] = base * shape_factor[h_idx]

    # Tropical wet/dry season variation
    day_of_year = (hour_index / 24).astype(int)

    # Central Africa: long rainy season (May-Oct), dry season (Nov-Apr)
    seasonal_variation = 1.0 + 0.08 * np.sin(2 * np.pi * (day_of_year - 30) / 365)

    hourly_load = hourly_load * seasonal_variation

    # Scale to target
    target_annual = n_households * daily_demand * 365
    hourly_load = hourly_load * (target_annual / hourly_load.sum())

    return hourly_load


# ------------------------------------------------------------------
# Fetch Cameroon weather data
# ------------------------------------------------------------------
def fetch_cam_weather():
    """Download Cameroon weather data from NASA POWER."""

    url = (
        "https://power.larc.nasa.gov/api/temporal/hourly/point"
        f"?parameters={CAM_PARAMS}"
        "&community=RE"
        f"&longitude={CAM_LON}"
        f"&latitude={CAM_LAT}"
        f"&start={CAM_START}"
        f"&end={CAM_END}"
        "&format=JSON"
    )

    print("Downloading Cameroon data from NASA POWER...")
    print(f"  Location: ({CAM_LAT}, {CAM_LON})")

    response = requests.get(url, timeout=60)
    response.raise_for_status()
    raw_data = response.json()

    records = raw_data["properties"]["parameter"]
    df = pd.DataFrame(records)
    df.index = pd.to_datetime(df.index, format="%Y%m%d%H")
    df.index.name = "datetime"

    df.rename(columns={
        "ALLSKY_SFC_SW_DWN": "GHI",
        "T2M": "T_amb"
    }, inplace=True)

    df["GHI"] = df["GHI"].clip(lower=0)

    annual_ghi = df["GHI"].mean()
    annual_temp = df["T_amb"].mean()

    print(f"  Done. Retrieved {len(df)} hourly records")
    print(f"  Annual mean GHI: {annual_ghi:.1f} W/m^2")
    print(f"  Annual mean temperature: {annual_temp:.1f} °C")
    print(f"  (Gansu reference: GHI 186.3 W/m^2, temperature 4.6 °C)\n")

    return df


# ------------------------------------------------------------------
# Cameroon PSO optimization
# ------------------------------------------------------------------
def optimize_cam(ghi_data, temp_data, load_data):
    """PSO optimization for Cameroon."""

    def objective_cam(x, cost_weight=1.0, lpsp_weight=1.0):
        pv_cap, bat_cap = x
        if pv_cap < 0 or bat_cap < 0:
            return 1e12

        pv_output = compute_pv_output(ghi_data, temp_data, pv_cap, temp_coeff, derating_factor)
        soc, deficit, curtailment = simulate_system(
            pv_output, load_data, bat_cap, soc_min, soc_initial, battery_rt_efficiency
        )

        lpsp_val = compute_lpsp(deficit, load_data)
        annual_cost = compute_annualized_cost(
            pv_cap, cam_pv_unit_cost, cam_pv_lifetime,
            bat_cap, cam_battery_unit_cost, cam_battery_lifetime,
            cam_project_lifetime, cam_discount_rate, cam_maintenance_ratio
        )

        cost_norm = annual_cost / 100000.0
        lpsp_norm = lpsp_val

        penalty = 0
        if lpsp_val > 0.05:
            penalty = (lpsp_val - 0.05) * 100

        return cost_weight * cost_norm + lpsp_weight * lpsp_norm + penalty

    lb = [20.0, 50.0]
    ub = [300.0, 800.0]

    weight_combos = [
        (1.0, 0.0),
        (0.8, 0.2),
        (0.5, 0.5),
        (0.2, 0.8),
        (0.0, 1.0),
    ]

    print("Running Cameroon PSO optimization...")
    print(f"{'Weight (cost, LPSP)':<28s} {'PV (kWp)':>10s} {'Battery (kWh)':>10s} {'ACS ($/yr)':>12s} {'LPSP (%)':>10s}")
    print("-" * 75)

    results_list = []

    for cost_w, lpsp_w in weight_combos:
        opt_result = pso(
            func=lambda x: objective_cam(x, cost_w, lpsp_w),
            lb=lb,
            ub=ub,
            swarmsize=30,
            maxiter=50,
            debug=False,
        )
        pv_opt, bat_opt = opt_result.x

        pv_output = compute_pv_output(ghi_data, temp_data, pv_opt, temp_coeff, derating_factor)
        soc, deficit, curtailment = simulate_system(
            pv_output, load_data, bat_opt, soc_min, soc_initial, battery_rt_efficiency
        )
        lpsp_opt = compute_lpsp(deficit, load_data)
        cost_opt = compute_annualized_cost(
            pv_opt, cam_pv_unit_cost, cam_pv_lifetime,
            bat_opt, cam_battery_unit_cost, cam_battery_lifetime,
            cam_project_lifetime, cam_discount_rate, cam_maintenance_ratio
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

    return pd.DataFrame(results_list)


# ------------------------------------------------------------------
# Comparison Pareto chart
# ------------------------------------------------------------------
def plot_compare_pareto(gan_results, cam_results):
    """Side-by-side Pareto front comparison."""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    for ax, df, region in [
        (ax1, gan_results, "Gansu, China"),
        (ax2, cam_results, "Cameroon")
    ]:
        ax.scatter(df["acs_usd_per_year"] / 1000, df["lpsp_pct"],
                   c="steelblue", s=60, zorder=3, edgecolors="white", linewidth=0.5)

        pareto_df = df.sort_values("acs_usd_per_year")
        ax.plot(pareto_df["acs_usd_per_year"] / 1000, pareto_df["lpsp_pct"],
                "o-", color="darkorange", linewidth=1.5, markersize=8, zorder=4,
                label="Pareto front")

        ax.axhline(y=5.0, color="red", linestyle="--", linewidth=0.8, alpha=0.6,
                   label="Target: LPSP = 5%")

        ax.set_xlabel("Annualized System Cost (k$ / year)")
        ax.set_ylabel("Loss of Power Supply Probability (%)")
        ax.set_title(f"{region}")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

    fig.suptitle("Pareto Front Comparison: Gansu (China) vs Cameroon", fontsize=14, y=1.02)
    plt.tight_layout()

    for fmt in ["png", "pdf"]:
        fig.savefig(f"figures/fig_compare_pareto.{fmt}")
    print("  Saved: figures/fig_compare_pareto.png / .pdf")
    plt.close(fig)


# ------------------------------------------------------------------
# Comparison bar chart
# ------------------------------------------------------------------
def plot_compare_bar(gan_results, cam_results):
    """Bar chart: optimal capacity and cost."""

    gan_best = gan_results.iloc[0]
    cam_best = cam_results.iloc[0]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    labels = ["Gansu", "Cameroon"]
    colors = ["#2c7fb8", "#e68a2e"]

    # PV capacity
    axes[0].bar(labels, [gan_best["pv_kwp"], cam_best["pv_kwp"]], color=colors, alpha=0.85)
    axes[0].set_ylabel("PV Capacity (kWp)")
    axes[0].set_title("Optimal PV Capacity")
    for i, v in enumerate([gan_best["pv_kwp"], cam_best["pv_kwp"]]):
        axes[0].text(i, v + 5, f"{v:.0f}", ha="center", fontsize=12, fontweight="bold")

    # Battery capacity
    axes[1].bar(labels, [gan_best["bat_kwh"], cam_best["bat_kwh"]], color=colors, alpha=0.85)
    axes[1].set_ylabel("Battery Capacity (kWh)")
    axes[1].set_title("Optimal Battery Capacity")
    for i, v in enumerate([gan_best["bat_kwh"], cam_best["bat_kwh"]]):
        axes[1].text(i, v + 5, f"{v:.0f}", ha="center", fontsize=12, fontweight="bold")

    # Cost per household
    gan_per_hh = gan_best["acs_usd_per_year"] / 200
    cam_per_hh = cam_best["acs_usd_per_year"] / 200
    axes[2].bar(labels, [gan_per_hh, cam_per_hh], color=colors, alpha=0.85)
    axes[2].set_ylabel("Cost ($/household/year)")
    axes[2].set_title("Cost per Household")
    for i, v in enumerate([gan_per_hh, cam_per_hh]):
        axes[2].text(i, v + 3, f"${v:.0f}", ha="center", fontsize=12, fontweight="bold")

    fig.suptitle("Optimal Configuration Comparison: Gansu vs Cameroon (LPSP ~5%)", fontsize=14, y=1.02)
    plt.tight_layout()

    for fmt in ["png", "pdf"]:
        fig.savefig(f"figures/fig_compare_bar.{fmt}")
    print("  Saved: figures/fig_compare_bar.png / .pdf")
    plt.close(fig)


# ------------------------------------------------------------------
# Climate comparison
# ------------------------------------------------------------------
def plot_climate_compare(gan_weather, cam_weather):
    """Compare solar irradiance and temperature."""

    gan_monthly_ghi = gan_weather["GHI"].resample("ME").mean()
    cam_monthly_ghi = cam_weather["GHI"].resample("ME").mean()
    gan_monthly_temp = gan_weather["T_amb"].resample("ME").mean()
    cam_monthly_temp = cam_weather["T_amb"].resample("ME").mean()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    month_labels = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"]
    month_nums = range(1, 13)

    # Monthly mean GHI
    ax1.plot(month_nums, gan_monthly_ghi.values, "o-", color="#2c7fb8", linewidth=2, markersize=6, label="Gansu")
    ax1.plot(month_nums, cam_monthly_ghi.values, "s-", color="#e68a2e", linewidth=2, markersize=6, label="Cameroon")
    ax1.axhline(y=gan_weather["GHI"].mean(), color="#2c7fb8", linestyle=":", alpha=0.5)
    ax1.axhline(y=cam_weather["GHI"].mean(), color="#e68a2e", linestyle=":", alpha=0.5)
    ax1.set_xlabel("Month")
    ax1.set_ylabel("GHI (W/m^2)")
    ax1.set_title("Monthly Mean Solar Irradiance")
    ax1.set_xticks(month_nums)
    ax1.set_xticklabels(month_labels)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Monthly mean temperature
    ax2.plot(month_nums, gan_monthly_temp.values, "o-", color="#2c7fb8", linewidth=2, markersize=6, label="Gansu")
    ax2.plot(month_nums, cam_monthly_temp.values, "s-", color="#e68a2e", linewidth=2, markersize=6, label="Cameroon")
    ax2.axhline(y=0, color="gray", linestyle="-", alpha=0.3)
    ax2.set_xlabel("Month")
    ax2.set_ylabel("Temperature (°C)")
    ax2.set_title("Monthly Mean Temperature")
    ax2.set_xticks(month_nums)
    ax2.set_xticklabels(month_labels)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Solar Irradiance and Temperature: Gansu vs Cameroon", fontsize=14, y=1.02)
    plt.tight_layout()

    for fmt in ["png", "pdf"]:
        fig.savefig(f"figures/fig_compare_climate.{fmt}")
    print("  Saved: figures/fig_compare_climate.png / .pdf")
    plt.close(fig)


# ------------------------------------------------------------------
# Load profile comparison
# ------------------------------------------------------------------
def plot_load_compare(gan_load, cam_load):
    """Compare daily load shapes."""

    gan_daily = np.zeros(24)
    cam_daily = np.zeros(24)
    for h in range(24):
        gan_daily[h] = gan_load.reshape(-1, 24)[:, h].mean()
        cam_daily[h] = cam_load.reshape(-1, 24)[:, h].mean()

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(range(24), gan_daily, "o-", color="#2c7fb8", linewidth=2, markersize=6, label="Gansu (China)")
    ax.plot(range(24), cam_daily, "s-", color="#e68a2e", linewidth=2, markersize=6, label="Cameroon")
    ax.fill_between(range(24), 0, gan_daily, alpha=0.08, color="#2c7fb8")
    ax.fill_between(range(24), 0, cam_daily, alpha=0.08, color="#e68a2e")

    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Average Power (kW)")
    ax.set_title("Typical Daily Load Profile (200 households)")
    ax.set_xticks(range(0, 24, 2))
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    for fmt in ["png", "pdf"]:
        fig.savefig(f"figures/fig_compare_load.{fmt}")
    print("  Saved: figures/fig_compare_load.png / .pdf")
    plt.close(fig)


# ------------------------------------------------------------------
# Print comparison table
# ------------------------------------------------------------------
def print_comparison_table(gan_results, cam_results):
    """Print side-by-side comparison."""

    gan_best = gan_results.iloc[0]
    cam_best = cam_results.iloc[0]

    print("\n" + "=" * 70)
    print("Gansu (China) vs Cameroon -- Optimal Design Comparison")
    print("=" * 70)
    print(f"{'Metric':<30s} {'Gansu':>15s} {'Cameroon':>15s} {'Diff':>10s}")
    print("-" * 70)
    print(f"{'PV capacity (kWp)':<30s} {gan_best['pv_kwp']:>15.0f} {cam_best['pv_kwp']:>15.0f} {cam_best['pv_kwp']-gan_best['pv_kwp']:>+10.0f}")
    print(f"{'Battery capacity (kWh)':<30s} {gan_best['bat_kwh']:>15.0f} {cam_best['bat_kwh']:>15.0f} {cam_best['bat_kwh']-gan_best['bat_kwh']:>+10.0f}")
    print(f"{'ACS ($/yr)':<30s} {gan_best['acs_usd_per_year']:>15,.0f} {cam_best['acs_usd_per_year']:>15,.0f} {cam_best['acs_usd_per_year']-gan_best['acs_usd_per_year']:>+10,.0f}")
    print(f"{'Cost/household ($/yr)':<30s} {gan_best['acs_usd_per_year']/200:>15.0f} {cam_best['acs_usd_per_year']/200:>15.0f} {cam_best['acs_usd_per_year']/200-gan_best['acs_usd_per_year']/200:>+10.0f}")
    print(f"{'LPSP (%)':<30s} {gan_best['lpsp_pct']:>15.4f} {cam_best['lpsp_pct']:>15.4f} {cam_best['lpsp_pct']-gan_best['lpsp_pct']:>+10.4f}")
    print("=" * 70)

    print("\n--- Analysis ---")
    if cam_best["pv_kwp"] < gan_best["pv_kwp"]:
        print("Cameroon requires less PV capacity because tropical solar irradiance is more stable,")
        print("with higher annual mean GHI and smaller seasonal variation, yielding more energy per kWp.")
    else:
        print("Cameroon requires more PV, possibly due to different load patterns or temperature effects.")

    if cam_best["acs_usd_per_year"] > gan_best["acs_usd_per_year"]:
        print("Cameroon's total cost is higher mainly due to higher equipment logistics and installation costs,")
        print("as well as higher capital cost (discount rate 8% vs 6%). However, per-unit cost differences")
        print("should be considered together with lower per-household consumption.")
    print()


# ==================================================================
# Main
# ==================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("Cross-Regional Comparison: Gansu (China) vs Cameroon")
    print("=" * 70)
    print()

    cam_weather = fetch_cam_weather()
    cam_ghi = cam_weather["GHI"].values
    cam_temp = cam_weather["T_amb"].values

    print("Loading Gansu data...")
    gan_weather = pd.read_csv("data/nasa_power_data.csv", index_col=0, parse_dates=True)
    gan_ghi = gan_weather["GHI"].values
    gan_temp = gan_weather["T_amb"].values
    print(f"  Gansu annual mean GHI: {gan_ghi.mean():.1f} W/m^2")
    print(f"  Gansu annual mean temperature: {gan_temp.mean():.1f} °C\n")

    gan_load = generate_load_profile(200, 3.0, 1.8)
    cam_load = generate_cam_load_profile(cam_households, cam_daily_demand, cam_peak_factor)

    print(f"Gansu load: total {gan_load.sum():,.0f} kWh, mean {gan_load.mean():.1f} kW")
    print(f"Cameroon load: total {cam_load.sum():,.0f} kWh, mean {cam_load.mean():.1f} kW\n")

    try:
        gan_results = pd.read_csv("data/optimization_results.csv")
        print("Loaded Gansu optimization results (data/optimization_results.csv)\n")
    except FileNotFoundError:
        print("No Gansu optimization results found, running with step3 settings...")
        from step2_model import (
            n_households, daily_demand_per_household, peak_factor,
            pv_unit_cost, pv_lifetime,
            battery_unit_cost, battery_lifetime,
        )

        gan_load2 = generate_load_profile(n_households, daily_demand_per_household, peak_factor)

        def objective_gan(x, cost_w, lpsp_w):
            pv_cap, bat_cap = x
            if pv_cap < 0 or bat_cap < 0:
                return 1e12
            pv_out = compute_pv_output(gan_ghi, gan_temp, pv_cap, temp_coeff, derating_factor)
            soc, deficit, curtailment = simulate_system(pv_out, gan_load2, bat_cap, soc_min, soc_initial, battery_rt_efficiency)
            lpsp_val = compute_lpsp(deficit, gan_load2)
            cost_val = compute_annualized_cost(pv_cap, pv_unit_cost, pv_lifetime, bat_cap, battery_unit_cost, battery_lifetime, project_lifetime, discount_rate, maintenance_ratio)
            penalty = 0
            if lpsp_val > 0.05:
                penalty = (lpsp_val - 0.05) * 100
            return cost_w * cost_val / 100000.0 + lpsp_w * lpsp_val + penalty

        gan_list = []
        for wc, wl in [(1.0, 0.0), (0.8, 0.2), (0.5, 0.5), (0.2, 0.8), (0.0, 1.0)]:
            res = pso(lambda x: objective_gan(x, wc, wl), lb=[20, 50], ub=[300, 800], swarmsize=30, maxiter=50, debug=False)
            pv_opt, bat_opt = res.x
            pv_out = compute_pv_output(gan_ghi, gan_temp, pv_opt, temp_coeff, derating_factor)
            soc, deficit, curtailment = simulate_system(pv_out, gan_load2, bat_opt, soc_min, soc_initial, battery_rt_efficiency)
            lpsp_opt = compute_lpsp(deficit, gan_load2)
            cost_opt = compute_annualized_cost(pv_opt, pv_unit_cost, pv_lifetime, bat_opt, battery_unit_cost, battery_lifetime, project_lifetime, discount_rate, maintenance_ratio)
            gan_list.append({"w_cost": wc, "w_lpsp": wl, "pv_kwp": pv_opt, "bat_kwh": bat_opt, "acs_usd_per_year": cost_opt, "lpsp_pct": lpsp_opt * 100})
        gan_results = pd.DataFrame(gan_list)
        print("Gansu optimization completed.\n")

    cam_results = optimize_cam(cam_ghi, cam_temp, cam_load)

    cam_results.to_csv("data/optimization_results_cameroon.csv", index=False)
    print(f"\nCameroon results saved to data/optimization_results_cameroon.csv")

    print("\nGenerating comparison charts...")
    plot_compare_pareto(gan_results, cam_results)
    plot_compare_bar(gan_results, cam_results)
    plot_climate_compare(gan_weather, cam_weather)
    plot_load_compare(gan_load, cam_load)

    print_comparison_table(gan_results, cam_results)

    print("Cross-regional comparison complete.")
