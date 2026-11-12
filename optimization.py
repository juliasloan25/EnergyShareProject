# This file defines functions needed to run the optimization.
# They answer the question: For a given split of the budget,
#  how much gas is used over the year?
# This includes the objective (total gas usage), solving the
# optimization, and helper functions.
#
# Structure follows Annex VI of the project description:
# `gp_minimize` searches over 4 investment shares (nuclear, wind,
# solar, storage). For each candidate split, calculate_gas
# normalizes it, converts it into installed capacities, runs
# the full-year hourly dispatch, and returns the total gas use.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skopt import gp_minimize
from skopt.plots import plot_convergence

from dispatch import dispatch
from constants import TECHS, CITY, INV_PER_CAPITA


def capacities_from_investment(fractions, costs, pop=CITY["pop"],
                               inv_per_capita=INV_PER_CAPITA) -> dict:
    """Convert investment fractions into installed capacities
    for nuclear, wind, solar, and storage.
    Inputs:
    - fractions: dict of investment fractions for nuclear, wind, solar, and storage
    - costs: dict of costs for nuclear, wind, solar, and storage
      (EUR/kW for nuclear/wind/solar, EUR/kWh for storage)
    - pop: population of the country
    - inv_per_capita: investment per capita (EUR/person)
    Outputs:
    - dict of installed capacities for nuclear, wind, solar (kW), and storage (kWh)
    """

    # Normalize so the shares sum to 1. The optimizer may hand us any
    # 4 numbers in [0, 1], not shares that already add up to 1.
    total_share = sum(fractions[t] for t in TECHS)
    if total_share <= 0:
        # Degenerate all-zero case: fall back to an equal split so we
        # never divide by zero.
        normalized = {t: 0.25 for t in TECHS}
    else:
        normalized = {t: fractions[t] / total_share for t in TECHS}

    total_investment_eur = pop * inv_per_capita

    return {
        "nuclear": normalized["nuclear"] * total_investment_eur / costs["nuclear"],  # kW
        "wind":    normalized["wind"]    * total_investment_eur / costs["wind"],     # kW
        "solar":   normalized["solar"]   * total_investment_eur / costs["solar"],    # kW
        "storage": normalized["storage"] * total_investment_eur / costs["storage"],  # kWh
    }


def simulate_year(capacities, profiles, demand) -> pd.DataFrame:
    """Simulate a year of production using the given setup,
    by running the hourly dispatch simulation.
    Inputs:
    - capacities: dict of installed capacities for nuclear, wind, solar, and storage
    - profiles: dict of resource profiles for wind (per kW installed),
      solar (per kW installed), and hydro (kW)
    - demand: array of hourly demand in kW
    Outputs:
    - prod_by_source: DataFrame of production by energy source and hour
    """

    wind_profile = np.asarray(profiles["wind"], dtype=float)
    solar_profile = np.asarray(profiles["solar"], dtype=float)

    return dispatch(
        demand,
        nuclear = capacities["nuclear"],
        hydro = profiles["hydro"],
        wind = capacities["wind"] * wind_profile,
        solar = capacities["solar"] * solar_profile,
        C_storage = capacities["storage"],
    )


def gas_total(prod_by_source) -> float:
    """Return total annual gas generation; optimization objective.
    Inputs:
    - prod_by_source: DataFrame of production by energy source and hour
    Outputs:
    - gas_total: total annual gas generation in kWh
    """
    return float(prod_by_source["gas"].sum())


def optimize_mix(profiles, demand, costs, pop=CITY["pop"]) -> dict:
    """Find an investment allocation that minimizes annual
    gas generation, subject to the investment constraints.
    Inputs:
    - profiles: dict of resource profiles for wind (per kW installed),
      solar (per kW installed), and hydro (kW)
    - demand: array of hourly demand in kW
    - costs: dict of costs for nuclear, wind, solar, and storage
    - pop: population of the country
    Outputs:
    - dictionary containing optimized fractions, installed capacities, total gas (kWh),
      the full dispatch DataFrame for the best split, and the raw
      scikit-optimize result (for plot_convergence / sensitivity analysis)
    """

    def calculate_gas(resource_fractions):
        """Score one candidate budget split by total annual gas use (kWh).

        gp_minimize passes a list of 4 numbers in [0, 1]. We treat them
        as shares for nuclear, wind, solar, and storage (in that order).
        """
        fractions = dict(zip(TECHS, resource_fractions))
        capacities = capacities_from_investment(fractions, costs, pop)
        prod_by_source = simulate_year(capacities, profiles, demand)
        return gas_total(prod_by_source)

    best = gp_minimize(
        calculate_gas,                  # objective function to minimize
        [(0.0, 1.0)] * 4,               # search bounds for each resource
        acq_func = "EI",                # expected improvement
        n_calls = 110,                  # total evaluations of f
        n_initial_points = 100,         # random initialization points
        x0 = [0.25, 0.25, 0.25, 0.25],  # start from an equal split
        random_state = 1234,
    )

    # Reconstruct the best solution as (normalized) fractions and capacities,
    # and re-run the simulation once so callers get the full hourly DataFrame.
    total_share = sum(best.x)
    opt_fractions = dict(zip(TECHS, [s / total_share for s in best.x]))
    opt_capacities = capacities_from_investment(opt_fractions, costs, pop)
    opt_df = simulate_year(opt_capacities, profiles, demand)

    return {
        "fractions": opt_fractions,
        "capacities": opt_capacities,
        "gas_kwh": float(best.fun),
        "df": opt_df,
        "result": best,
    }


def plot_optimization_convergence(result, path="convergence.png"):
    """Save a convergence plot for a gp_minimize result.
    Inputs:
    - result: raw OptimizeResult returned by gp_minimize (the "result"
      key of optimize_mix's output)
    - path: file path to save the plot to
    """

    plot_convergence(result)
    plt.savefig(path, bbox_inches="tight", pad_inches=0)


if __name__ == "__main__":
    # Unit test with small synthetic profiles.
    # Keep n_calls small so the test finishes quickly.
    from dispatch import validate_dispatch

    # 24 hours of flat demand and simple day/night solar + steady wind + hydro
    hours = 24
    demand_test = np.full(hours, 1000.0)  # kW
    profiles_test = {
        "solar": np.where(
            (np.arange(hours) >= 6) & (np.arange(hours) < 18), 0.6, 0.0
        ),  # per kW
        "wind":  np.full(hours, 0.3),     # per kW
        "hydro": np.full(hours, 100.0),   # kW
    }
    costs_test = {
        "nuclear": 47000,  # EUR/kW
        "wind":    11000,  # EUR/kW
        "solar":    6300,  # EUR/kW
        "storage":  3000,  # EUR/kWh
    }
    pop_test = 1000

    # 1. Capacities should spend exactly the budget.
    fractions = {"nuclear": 0.2, "wind": 0.3, "solar": 0.3, "storage": 0.2}
    caps = capacities_from_investment(fractions, costs_test, pop=pop_test)
    budget = pop_test * 20000
    spent = (
        caps["nuclear"] * costs_test["nuclear"]
        + caps["wind"]    * costs_test["wind"]
        + caps["solar"]   * costs_test["solar"]
        + caps["storage"] * costs_test["storage"]
    )
    print(f"Budget check: spent {spent:.2f} EUR vs budget {budget} EUR  "
          f"(match: {np.isclose(spent, budget)})")

    # 2. 100%-nuclear split should run and use very little / no gas when
    #    the nuclear block alone covers demand.
    all_nuclear = {"nuclear": 1.0, "wind": 0.0, "solar": 0.0, "storage": 0.0}
    caps_nuc = capacities_from_investment(all_nuclear, costs_test, pop=pop_test)
    df_nuc = simulate_year(caps_nuc, profiles_test, demand_test)
    print(f"All-nuclear capacity: {caps_nuc['nuclear']:.1f} kW, "
          f"gas: {gas_total(df_nuc):.1f} kWh")

    # 3. optimize_mix runs end-to-end with a small call budget.
    result = optimize_mix(
        profiles_test, demand_test, costs_test,
        pop=pop_test)

    print("\nOptimization result:")
    for tech, share in result["fractions"].items():
        print(f"  {tech:>8}: {share*100:5.1f}%  "
              f"({result['capacities'][tech]:.1f} "
              f"{'kWh' if tech == 'storage' else 'kW'})")
    print(f"  gas: {result['gas_kwh']:.1f} kWh")

    # Validate the best-result dispatch.
    validation = validate_dispatch(result["df"], demand_test)
    print("\nDispatch validation of best result:")
    for check, ok in validation.items():
        print(f"  {check}: {ok}")
