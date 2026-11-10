# This file defines functions related to analyzing and
# communicating the results of the optimization, such as
# sensitivity analysis and plotting.

def co2_per_kwh(df, emission_factors) -> float:
    """Calculate annual CO₂ emissions per kWh generated."""

def cc_stats(df) -> dict:
    """Calculate installed gas capacity and its load factor."""

def spared_pct(df) -> float:
    """Calculate surplus electricity as a percentage
    of total production."""

def plot_week(df, demand, start_hour):
    """Plot generation, demand, and storage over one week."""

def plot_pie(df):
    """Plot the annual generation mix."""

def sensitivity_analysis(profiles, demand, pop, costs, scenarios) -> pd.DataFrame:
    """Evaluate how changes in assumptions affect gas use
    and other key results."""

