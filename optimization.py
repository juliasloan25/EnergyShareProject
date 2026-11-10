# This file defines functions needed to run the optimization.
# This includes the objective (total gas usage), solving the
# optimization, and helper functions.

def capacities_from_investment(fractions, pop, costs,
                               per_capita=20000) -> dict:
    """Convert investment fractions into installed capacities
    for nuclear, wind, solar, and storage."""

def calc(caps, profiles, demand) -> pd.DataFrame:
    """Scale resource profiles using installed capacities and
    run the hourly dispatch simulation."""

def gas_total(df) -> float:
    """Return total annual gas generation; optimization objective."""

def optimize_mix(profiles, demand, pop, costs) -> dict:
    """Find an investment allocation that minimizes annual
    gas generation, subject to the investment constraints."""

