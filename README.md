# Optimal Sizing of Battery Energy Storage for Solar-Powered Rural Microgrids

> **CIS Research Fair 2026 — SDG 7: Affordable and Clean Energy**
>
> Multi-objective Particle Swarm Optimization (PSO) for cost-optimal PV + battery sizing in off-grid rural electrification.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)

---

## Overview

About 760 million people worldwide still lack access to electricity, most in rural areas of developing countries. Solar microgrids with battery storage offer a viable solution, but sizing the battery is critical — undersizing leads to blackouts, oversizing wastes capital.

This project provides a reproducible, data-driven framework that:
- Downloads free satellite weather data from **NASA POWER API**
- Simulates hourly PV + battery + load over a full year (8760 h)
- Runs **multi-objective PSO** to trace the cost–reliability Pareto front
- Compares two climatically distinct regions (Gansu, China vs Cameroon)
- Uses 100% open-source tools (Python, numpy, pyswarm, matplotlib)

## Key Results

### Gansu, China (35.0°N, 104.0°E) — 200 households

| Configuration | PV (kWp) | Battery (kWh) | ACS ($/yr) | LPSP | $/household/yr |
|--------------|----------|---------------|------------|------|----------------|
| **Cost-optimal** | **271** | **483** | **$36,259** | **5.00%** | **$181 (~1,300 CNY)** |
| Balanced | 300 | 527 | $39,820 | 3.27% | $199 |
| High-reliability | 300 | 800 | $50,467 | 1.47% | $252 |

### Cameroon (5.0°N, 12.0°E) — 200 households

| Configuration | PV (kWp) | Battery (kWh) | ACS ($/yr) | LPSP | $/household/yr |
|--------------|----------|---------------|------------|------|----------------|
| **Cost-optimal** | **154** | **390** | **$30,325** | **5.00%** | **$152** |

### Cross-Regional Comparison

Cameroon requires 43% less PV (154 vs 271 kWp) and 19% less battery (390 vs 483 kWh), with a 16% lower cost per household ($152 vs $181). The stronger, more stable tropical solar resource (213 vs 186 W/m² annual mean GHI) drives this difference. Sensitivity analysis confirms that solar irradiance — not battery price — is the dominant sizing factor.

## Project Structure

```
├── code/
│   ├── step1_fetch_data.py              # Download NASA POWER hourly data
│   ├── step2_model.py                   # PV + Battery + Load simulation
│   ├── step3_optimize.py                # Multi-objective PSO optimization
│   ├── step4_visualize.py               # Generate figures
│   ├── step6_compare_cameroon.py        # Cross-regional comparison
│   ├── gen_flowchart.py                 # Methodology pipeline diagram
│   ├── gen_qrcode.py                    # QR code generator
│   ├── gen_qrcode_data.py               # Data QR code generator
│   ├── requirements.txt                 # Python dependencies
│   └── figures/                         # Generated figures (PNG + PDF)
├── data/
│   ├── nasa_power_data.csv              # Gansu weather (8760 h)
│   ├── optimization_results.csv         # PSO results (Gansu)
│   └── optimization_results_cameroon.csv # PSO results (Cameroon)
└── README.md
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r code/requirements.txt

# 2. Download NASA weather data
python code/step1_fetch_data.py

# 3. Test with a single configuration
python code/step2_model.py

# 4. Run PSO optimization
python code/step3_optimize.py

# 5. Generate figures
python code/step4_visualize.py

# 6. Cross-regional comparison (Cameroon)
python code/step6_compare_cameroon.py
```

## Methodology

### System Model
- **PV output:** P = P_rated × (GHI / 1000) × [1 + α_T × (T_cell − 25°C)] × η_derate
- **Battery:** hourly SOC tracking, max 80% DoD, 92% round-trip efficiency
- **Load:** synthetic 200-household profile, diurnal peak (~1.6× at 19:00–20:00), seasonal variation ±15%
- **Reliability:** LPSP = total annual deficit / total annual demand (target ≤ 5%)

### Optimization
- **Algorithm:** Particle Swarm Optimization (PSO) via `pyswarm`
- **Decision variables:** PV capacity [20, 300] kWp, Battery capacity [50, 800] kWh
- **Objectives:** minimize ACS + minimize LPSP
- **Constraint:** LPSP ≤ 5%

## Tools & Data

| Component | Stack |
|-----------|-------|
| Language | Python 3.10+ |
| Weather data | NASA POWER API (GHI + temperature, free, no key) |
| Optimization | `pyswarm` (PSO, 30 particles × 50 iterations) |
| Visualization | `matplotlib` |
| Computation | `numpy`, `pandas` |

All tools are free and open-source. All data is publicly available.

## References

1. Upadhyay & Sharma (2014). *Renewable and Sustainable Energy Reviews*, 38, 47–63.
2. Maleki & Askarzadeh (2014). *Solar Energy*, 99, 272–282.
3. Bhattacharyya & Palit (2016). *Energy for Sustainable Development*, 31, 1–13.
4. NASA POWER: [power.larc.nasa.gov](https://power.larc.nasa.gov/)
5. IRENA (2024). *Renewable Power Generation Costs in 2023*.
6. Kennedy & Eberhart (1995). PSO. *Proceedings of ICNN'95*.
