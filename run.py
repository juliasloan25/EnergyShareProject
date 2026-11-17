# Main file: calls functions from resources.py, dispatch.py, and optimization.py
# to find the optimal investment fractions for nuclear, wind, solar, and storage.
# Also calls plotting functions from analysis.py.

import time

import matplotlib.pyplot as plt

from constants import (
    TECHS,
    INV_PER_CAPITA,
    COSTS,
    CITY,
    EMISSION_FACTORS,
)
from resources import (
    load_resource_data, build_profiles, demand_profile, WIND_MULTIPLIER,
)
from optimization import optimize_mix, plot_optimization_convergence
from dispatch import validate_dispatch
from analysis import co2_per_kwh, cc_stats, spared_pct, plot_pie, plot_co2_pie


# Input data files (as saved in the project folder, see Annex II / V)
NASA_FILE = "POWER_Point_Hourly_20250101_20251231_038d03S_057d55W_UTC.csv"
RIVER_FILE = "river_flow_norm.csv"
DEMAND_FILE = "demand_data2026.dat"


if __name__ == "__main__":
    print("=" * 60)
    print(f"Energy mix optimization for {CITY['name']}")
    print(f"  population: {CITY['pop']:,}   area: {CITY['area_km2']:,} km^2")
    budget = CITY["pop"] * INV_PER_CAPITA
    print(f"  budget:     {budget / 1e9:.2f} B EUR  "
          f"({CITY['pop']:,} people x {INV_PER_CAPITA} EUR/person)")
    print("=" * 60)

    # ---------------------------------------------------------------
    # [1] Load data and build renewable profiles
    # ---------------------------------------------------------------
    print("\n[1] Loading data and building profiles...")
    data = load_resource_data(NASA_FILE, RIVER_FILE, DEMAND_FILE)
    profiles = build_profiles(data)
    demand = demand_profile(data["demand_per_capita"], CITY["pop"])

    print(f"  loaded {len(demand)} hourly values")
    print(f"  solar CF: {profiles['solar'].mean() * 100:.1f} %  "
          f"(max {profiles['solar'].max():.2f} kW/kW)")
    print(f"  wind CF:  {profiles['wind'].mean() * 100:.1f} %  "
          f"(hub-height multiplier {WIND_MULTIPLIER:.3f})")
    print(f"  hydro:    max {profiles['hydro'].max() / 1000:.1f} MW,  "
          f"CF {profiles['hydro'].mean() / profiles['hydro'].max() * 100:.1f} %")
    print(f"  demand:   {demand.sum() / 1e9:.2f} TWh/year,  "
          f"peak {demand.max() / 1e6:.2f} GW")

    # ---------------------------------------------------------------
    # [2] Optimize the investment mix
    # ---------------------------------------------------------------
    print("\n[2] Running optimizer...")
    t0 = time.time()
    result = optimize_mix(profiles, demand, COSTS)
    print(f"  done in {time.time() - t0:.1f} s")

    # ---------------------------------------------------------------
    # [3] Report the best investment mix and installed capacities
    # ---------------------------------------------------------------
    print("\n[3] Best investment mix:")
    for tech in TECHS:
        share = result["fractions"][tech]
        cap = result["capacities"][tech]
        unit = "MWh" if tech == "storage" else "MW"
        print(f"  {tech:>8}: {share * 100:5.1f} %  "
              f"({cap / 1000:>7.1f} {unit})")
    print(f"  gas:    {result['gas_kwh'] / 1e9:.2f} TWh/year")

    df = result["df"]

    # ---------------------------------------------------------------
    # [4] Validate the best-case hourly dispatch
    # ---------------------------------------------------------------
    print("\n[4] Validating best-case dispatch:")
    validation = validate_dispatch(df, demand)
    for check, ok in validation.items():
        print(f"  {check}: {ok}")

    # ---------------------------------------------------------------
    # [5] Analysis metrics (CO2, gas plant, curtailment)
    # ---------------------------------------------------------------
    print("\n[5] Analysis metrics:")
    co2 = co2_per_kwh(df, EMISSION_FACTORS)
    gas_stats = cc_stats(df)
    spared = spared_pct(df)
    print(f"  CO2 intensity:       {co2:.1f} gCO2/kWh")
    print(f"  Gas plant capacity:  {gas_stats['installed_gas_capacity_MW']:.1f} MW")
    print(f"  Gas load factor:     {gas_stats['gas_load_factor'] * 100:.1f} %")
    print(f"  Spared (curtailed):  {spared:.1f} % of total production")

    # ---------------------------------------------------------------
    # [6] Plots
    # ---------------------------------------------------------------
    print("\n[6] Rendering plots...")

    # Convergence plot (saved to disk, not shown)
    plot_optimization_convergence(result["result"], "convergence.png")
    plt.close("all")  # drop the convergence figure so plot_pie's show() is clean
    print("  convergence plot -> convergence.png")

    # Annual generation mix pie and CO2 emissions pie (both saved to disk
    # and shown interactively; close each window to continue)
    print("  generation mix pie   -> generation_mix.png")
    print("  CO2 emissions pie    -> co2_emissions.png")
    plot_pie(df, save_path="generation_mix.png")
    plot_co2_pie(df, EMISSION_FACTORS, save_path="co2_emissions.png")
