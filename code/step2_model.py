import numpy as np
import pandas as pd

# Village parameters
n_households = 200
daily_demand_per_household = 3.0      # kWh per household per day
peak_factor = 1.8                      # peak-to-mean ratio

# PV panel parameters
pv_capacity = 150.0                    # installed PV capacity (kWp)
pv_unit_cost = 600                     # USD per kWp
pv_lifetime = 25                       # years
pv_efficiency = 0.20                   # module efficiency (informational)
temp_coeff = -0.004                    # power temp coefficient (per °C)
derating_factor = 0.85                 # inverter, wiring, soiling, mismatch

# Battery parameters
battery_capacity = 600.0               # battery energy capacity (kWh)
battery_unit_cost = 250                # USD per kWh
battery_lifetime = 10                  # years
battery_rt_efficiency = 0.92           # round-trip efficiency
depth_of_discharge = 0.80              # max allowable DoD
soc_min = 1 - depth_of_discharge       # minimum SOC = 20%
soc_initial = 0.50                     # initial SOC

# Financial parameters
project_lifetime = 20                  # years
discount_rate = 0.06                   # annual discount rate
maintenance_ratio = 0.02               # annual O&M as fraction of capex

# ------------------------------------------------------------------
# Load profile generator
# ------------------------------------------------------------------
def generate_load_profile(n_households, daily_demand_per_household, peak_factor):
    """Generate an 8760-hour rural load profile."""

    hour_index = np.arange(8760)
    hour_of_day = hour_index % 24

    avg_hourly_per_household = daily_demand_per_household / 24

    # Rural daily load shape (relative to mean)
    shape_factor = np.array([
        0.35, 0.30, 0.28, 0.26, 0.28, 0.35,   # 0-5
        0.55, 0.80, 0.95, 0.85, 0.75, 0.72,   # 6-11
        0.70, 0.72, 0.74, 0.75, 0.78, 0.85,   # 12-17
        1.10, 1.50, 1.60, 1.40, 1.10, 0.90,   # 18-23
    ])

    # Normalize so daily mean = 1.0
    shape_factor = shape_factor / shape_factor.mean()

    hourly_load = np.zeros(8760)
    for h in range(8760):
        base = avg_hourly_per_household * n_households
        h_idx = hour_of_day[h]
        hourly_load[h] = base * shape_factor[h_idx]

    # Seasonal variation: higher in winter and summer
    day_of_year = (hour_index / 24).astype(int)

    seasonal_variation = 1.0 + 0.12 * np.sin(2 * np.pi * (day_of_year - 15) / 365)
    seasonal_variation = seasonal_variation + 0.15 * np.sin(2 * np.pi * (day_of_year - 195) / 365)

    hourly_load = hourly_load * seasonal_variation

    # Scale to match target annual total
    target_annual = n_households * daily_demand_per_household * 365
    hourly_load = hourly_load * (target_annual / hourly_load.sum())

    return hourly_load


# ------------------------------------------------------------------
# PV output model
# ------------------------------------------------------------------
def compute_pv_output(ghi, ambient_temp, pv_capacity, temp_coeff, derating_factor):
    """Compute hourly PV power output (kW)."""

    # Cell temperature (Ross formula)
    cell_temp = ambient_temp + 0.03 * ghi

    # Temperature correction
    temp_correction = 1 + temp_coeff * (cell_temp - 25.0)
    temp_correction = np.clip(temp_correction, 0, 1.2)

    # PV output (kW)
    pv_output = pv_capacity * (ghi / 1000.0) * temp_correction * derating_factor

    return pv_output


# ------------------------------------------------------------------
# System simulation (hourly energy balance)
# ------------------------------------------------------------------
def simulate_system(pv_output, load, battery_capacity, soc_min, soc_initial,
                    rt_efficiency):
    """Simulate one full year (8760 h) of microgrid operation."""

    n_hours = len(pv_output)
    one_way_efficiency = np.sqrt(rt_efficiency)

    soc = np.zeros(n_hours)          # state of charge, 0 to 1
    deficit = np.zeros(n_hours)      # unmet load (kWh)
    curtailment = np.zeros(n_hours)  # wasted energy (kWh)
    battery_energy = np.zeros(n_hours)  # energy stored (kWh)

    # Initial state
    soc[0] = soc_initial
    max_battery_energy = battery_capacity
    battery_energy[0] = soc[0] * max_battery_energy

    for h in range(n_hours):
        battery_energy[h] = soc[h] * max_battery_energy

        net_power = pv_output[h] - load[h]

        if net_power > 0:
            # Surplus: charge battery
            charge_headroom = (1.0 - soc[h]) * max_battery_energy / one_way_efficiency
            actual_charge = min(net_power, charge_headroom)
            battery_energy[h] = battery_energy[h] + actual_charge * one_way_efficiency
            curtailment[h] = net_power - actual_charge

        elif net_power < 0:
            # Deficit: discharge battery
            shortage = -net_power
            discharge_limit = (soc[h] - soc_min) * max_battery_energy * one_way_efficiency
            actual_discharge = min(shortage, discharge_limit)

            battery_energy[h] = battery_energy[h] - actual_discharge / one_way_efficiency
            deficit[h] = shortage - actual_discharge

        # Clamp battery energy
        battery_energy[h] = np.clip(battery_energy[h], 0, max_battery_energy)

        # Compute SOC for next hour
        if h < n_hours - 1:
            soc[h + 1] = battery_energy[h] / max_battery_energy

    # Recompute SOC array
    soc = battery_energy / max_battery_energy
    soc = np.clip(soc, 0, 1)

    return soc, deficit, curtailment


# ------------------------------------------------------------------
# Performance metrics
# ------------------------------------------------------------------
def compute_lpsp(deficit, load):
    """Loss of Power Supply Probability = total deficit / total demand."""
    total_deficit = deficit.sum()
    total_load = load.sum()
    if total_load > 0:
        return total_deficit / total_load
    else:
        return 0.0


def compute_annualized_cost(pv_capacity, pv_unit_cost, pv_lifetime,
                            battery_capacity, battery_unit_cost, battery_lifetime,
                            project_lifetime, discount_rate, maintenance_ratio):
    """Compute the annualized system cost (ACS) using CRF."""

    def crf(rate, years):
        if rate == 0:
            return 1 / years
        return rate * (1 + rate)**years / ((1 + rate)**years - 1)

    # PV capital cost
    pv_capital = pv_capacity * pv_unit_cost

    # Battery capital cost (with mid-life replacement)
    battery_capital = battery_capacity * battery_unit_cost
    n_replacements = int(project_lifetime / battery_lifetime) - 1
    battery_replacement_total = 0
    for k in range(1, n_replacements + 1):
        year_k = k * battery_lifetime
        battery_replacement_total = battery_replacement_total + battery_capital / ((1 + discount_rate) ** year_k)

    total_capital = pv_capital + battery_capital + battery_replacement_total

    # Annual maintenance
    maintenance_cost = (pv_capital + battery_capital) * maintenance_ratio

    # Annualize
    annualized = total_capital * crf(discount_rate, project_lifetime) + maintenance_cost

    return annualized


# ------------------------------------------------------------------
# Quick test run
# ------------------------------------------------------------------
if __name__ == "__main__":
    # Load NASA data
    try:
        weather = pd.read_csv("data/nasa_power_data.csv", index_col=0, parse_dates=True)
    except FileNotFoundError:
        print("Please run step1_fetch_data.py first to download weather data")
        exit(1)

    ghi_data = weather["GHI"].values
    temp_data = weather["T_amb"].values

    # Generate load profile
    load_data = generate_load_profile(n_households, daily_demand_per_household, peak_factor)

    # Compute PV output
    pv_data = compute_pv_output(ghi_data, temp_data, pv_capacity, temp_coeff, derating_factor)

    # Simulate system
    soc, deficit, curtailment = simulate_system(
        pv_data, load_data, battery_capacity, soc_min,
        soc_initial, battery_rt_efficiency
    )

    lpsp_value = compute_lpsp(deficit, load_data)
    annual_cost = compute_annualized_cost(
        pv_capacity, pv_unit_cost, pv_lifetime,
        battery_capacity, battery_unit_cost, battery_lifetime,
        project_lifetime, discount_rate, maintenance_ratio
    )

    total_pv = pv_data.sum()
    total_load = load_data.sum()
    total_deficit = deficit.sum()
    total_curtailment = curtailment.sum()
    renewable_fraction = (total_load - total_deficit) / total_load * 100

    print("=" * 55)
    print("Single Run Results")
    print("=" * 55)
    print(f"  PV capacity:              {pv_capacity:.0f} kWp")
    print(f"  Battery capacity:         {battery_capacity:.0f} kWh")
    print(f"  Annual PV generation:     {total_pv:,.0f} kWh")
    print(f"  Annual load demand:       {total_load:,.0f} kWh")
    print(f"  Annual deficit:           {total_deficit:,.0f} kWh")
    print(f"  Annual curtailment:       {total_curtailment:,.0f} kWh")
    print(f"  LPSP:                     {lpsp_value*100:.4f}%")
    print(f"  Renewable fraction:       {renewable_fraction:.1f}%")
    print(f"  Annualized system cost:   ${annual_cost:,.0f}/year")
    print(f"  Cost per household:       ${annual_cost/n_households:,.0f}/year/household")
    print("=" * 55)
