# This file contains functions to get environmental data,
# compute hourly output for each renewable energy type,
# and determine the total hourly demand profile.

def load_resource_data(nasa_file, river_file) -> dict:
    """Load and clean NASA weather data and river-flow data."""

def solar_profile(dni, dhi, params) -> np.ndarray:
    """Return hourly solar AC output per kW installed."""

def wind_profile(v50, multiplier, power_curve) -> np.ndarray:
    """Return hourly wind output per kW installed."""

def hydro_profile(river_flow_norm, area) -> np.ndarray:
    """Return hourly hydro generation in kW, based on the area."""

def demand_profile(per_capita_kw, population) -> np.ndarray:
    """Return total hourly electricity demand in kW."""
