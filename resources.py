# This file contains functions to get environmental data,
# compute hourly output for each renewable energy type,
# and determine the total hourly demand profile.
#
# All hourly data are used in UTC, as downloaded from NASA POWER
# (the demand file has no time zone, so it is used as it is).
# Every profile returned has 8760 values (one per hour of 2025).

import numpy as np
import pandas as pd
import pvlib

from constants import HYDRO_PER_AREA, CITY


# ---------------------------------------------------------------
# Solar panel: First Solar Series 7 TR1, FS-7550A-TR1 (datasheet)
# Installation: 2-axis tracking, inverter 96 %, shadow below 10 deg
# (handout, Annex IV)
# ---------------------------------------------------------------
SOLAR_PARAMS = {
    "lat": CITY["lat"],
    "lon": CITY["lon"],
    "p_nom": 550.0,        # W, nominal power at STC
    "area": 2.80,          # m2, module area
    "eff_stc": 0.197,      # module efficiency at STC
    "gamma": -0.0032,      # 1/degC, temperature coefficient of Pmax
    "t_noct": 43.0,        # degC, nominal operating cell temperature
    "inv_eff": 0.96,       # inverter efficiency
    "min_elevation": 10.0, # deg, below this the panel is fully shaded
}

# ---------------------------------------------------------------
# Wind turbine: Gamesa G132-5.0 MW (handout, Annex III)
# Power curve read from en.wind-turbine-models.com
# Cut-in 1.5 m/s, rated 13 m/s, cut-out 27 m/s
# ---------------------------------------------------------------
G132_POWER_CURVE = {
    "rated_kw": 5000.0,
    "cut_out": 27.0,  # m/s, above this the turbine stops
    "speed": np.array([1.5, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                       16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]),  # m/s
    "power": np.array([0, 70, 200, 430, 830, 1350, 2030, 2850, 3680, 4340,
                       4740, 4940, 4980, 5000, 5000, 5000, 4980, 4940, 4850,
                       4680, 4470, 4260, 4020, 3780, 3540, 3310]),        # kW
}

# Wind speed from 50 m (NASA) to 140 m (hub height), logarithmic profile
# with roughness class 4 (z0 = 1.6 m):  v2 = v1 * ln(h2/z0) / ln(h1/z0)
HUB_HEIGHT = 140.0  # m
NASA_HEIGHT = 50.0  # m
Z0 = 1.6            # m
WIND_MULTIPLIER = np.log(HUB_HEIGHT / Z0) / np.log(NASA_HEIGHT / Z0)  # = 1.299


def load_resource_data(nasa_file, river_file, demand_file=None) -> dict:
    """Load and clean NASA weather data and river-flow data.

    Returns a dict of numpy arrays (8760 values each):
    - times: hourly UTC timestamps
    - dni, dhi: direct normal and diffuse horizontal irradiance [W/m2]
    - v50: wind speed at 50 m [m/s]
    - river_flow_norm: normalized river flow [-]
    - demand_per_capita: hourly demand per person [kW] (if demand_file given)
    """

    # NASA file: skip the text header (it ends with the line "-END HEADER-")
    with open(nasa_file) as f:
        lines = f.readlines()
    n_header = next(i for i, line in enumerate(lines) if "-END HEADER-" in line) + 1
    nasa = pd.read_csv(nasa_file, skiprows=n_header)

    # NASA uses -999 for missing values: replace them by linear interpolation
    nasa = nasa.replace(-999, np.nan).interpolate(limit_direction="both")

    times = pd.to_datetime(
        dict(year=nasa["YEAR"], month=nasa["MO"], day=nasa["DY"], hour=nasa["HR"])
    ).dt.tz_localize("UTC")

    river = pd.read_csv(river_file)

    data = {
        "times": pd.DatetimeIndex(times),
        "dni": nasa["ALLSKY_SFC_SW_DNI"].clip(lower=0).to_numpy(),
        "dhi": nasa["ALLSKY_SFC_SW_DIFF"].clip(lower=0).to_numpy(),
        "v50": nasa["WS50M"].clip(lower=0).to_numpy(),
        "river_flow_norm": river["normalized_flow"].to_numpy(),
    }

    if demand_file is not None:
        demand = pd.read_csv(demand_file)
        # Hourly energy per capita [kWh] = average power per capita [kW]
        data["demand_per_capita"] = demand.iloc[:, 1].to_numpy()

    # All series must cover the full year (8760 h, 2025 is not a leap year)
    for name, values in data.items():
        if len(values) != 8760:
            raise ValueError(f"{name} has {len(values)} values, expected 8760.")

    return data


def solar_profile(dni, dhi, params) -> np.ndarray:
    """Return hourly solar AC output per kW installed.

    params must include "times" (UTC timestamps) plus the keys of SOLAR_PARAMS.
    """

    # 1) Sun position at the middle of each hour (NASA values are hourly averages)
    times_mid = params["times"] + pd.Timedelta(minutes=30)
    solpos = pvlib.solarposition.get_solarposition(times_mid, params["lat"], params["lon"])
    zenith = solpos["apparent_zenith"].to_numpy()
    elevation = solpos["apparent_elevation"].to_numpy()

    # 2) Irradiance on the panel with 2-axis tracking (panel always faces the sun).
    #    Monofacial panel: only the front side receives light.
    #    - Beam: the panel is perpendicular to the sun, so it receives all the DNI.
    #    - Diffuse: isotropic sky model, panel tilt = sun zenith angle.
    tilt = np.radians(zenith)
    poa = dni + dhi * (1 + np.cos(tilt)) / 2

    # 3) Landscape shadow: no production at all below 10 deg of sun elevation
    poa = np.where(elevation < params["min_elevation"], 0.0, poa)

    # 4) Module efficiency at NOCT (constant):  eff = eff_stc * (1 + gamma * (T_noct - 25))
    eff_noct = params["eff_stc"] * (1 + params["gamma"] * (params["t_noct"] - 25))

    # 5) AC power of one module [W], then divide by its nominal power -> kW per kW installed
    p_ac = poa * params["area"] * eff_noct * params["inv_eff"]
    return p_ac / params["p_nom"]


def wind_profile(v50, multiplier, power_curve) -> np.ndarray:
    """Return hourly wind output per kW installed."""

    # Wind speed at hub height
    v_hub = np.asarray(v50, dtype=float) * multiplier

    # Power from the power curve (linear interpolation between points).
    # Below cut-in -> 0; above cut-out -> 0.
    power = np.interp(v_hub, power_curve["speed"], power_curve["power"], left=0.0)
    power = np.where(v_hub > power_curve["cut_out"], 0.0, power)

    return power / power_curve["rated_kw"]


def hydro_profile(river_flow_norm, area) -> np.ndarray:
    """Return hourly hydro generation in kW, based on the area.

    1.0 = nominal hydro power = maximum output; flows above 1.0 are spilled
    (handout, Annex V). Nominal power = 20 kW/km2 * area.
    """
    p_nominal = HYDRO_PER_AREA * area  # kW
    return np.minimum(np.asarray(river_flow_norm, dtype=float), 1.0) * p_nominal


def demand_profile(per_capita_kw, population) -> np.ndarray:
    """Return total hourly electricity demand in kW."""
    return np.asarray(per_capita_kw, dtype=float) * population


def build_profiles(data) -> dict:
    """Compute the {solar, wind, hydro} profile dict from loaded NASA + river data.

    This is the format expected by optimization.simulate_year / optimize_mix:
    - "solar": AC output per kW installed  (unitless)
    - "wind":  AC output per kW installed  (unitless)
    - "hydro": actual output in kW (already scaled by area and river flow)
    """
    return {
        "solar": solar_profile(
            data["dni"], data["dhi"], {**SOLAR_PARAMS, "times": data["times"]},
        ),
        "wind":  wind_profile(data["v50"], WIND_MULTIPLIER, G132_POWER_CURVE),
        "hydro": hydro_profile(data["river_flow_norm"], CITY["area_km2"]),
    }


if __name__ == "__main__":
    # Quick check of all profiles with the real data files
    data = load_resource_data(
        "POWER_Point_Hourly_20250101_20251231_038d03S_057d55W_UTC.csv",
        "river_flow_norm.csv",
        "demand_data2026.dat",
    )

    profiles = build_profiles(data)
    demand = demand_profile(data["demand_per_capita"], CITY["pop"])

    print(f"Wind multiplier (50 m -> 140 m): {WIND_MULTIPLIER:.3f}")
    print(f"Solar capacity factor: {profiles['solar'].mean() * 100:.1f} %  "
          f"(max {profiles['solar'].max():.2f} kW/kW)")
    print(f"Wind capacity factor:  {profiles['wind'].mean() * 100:.1f} %")
    print(f"Hydro: max {profiles['hydro'].max() / 1000:.1f} MW, capacity factor "
          f"{profiles['hydro'].mean() / profiles['hydro'].max() * 100:.1f} %")
    print(f"Demand: {demand.sum() / 1e9:.2f} TWh/year, peak {demand.max() / 1e6:.2f} GW")

    for name, p in {**profiles, "demand": demand}.items():
        ok = len(p) == 8760 and np.all(np.isfinite(p)) and np.all(p >= 0)
        print(f"{name}: 8760 values, finite, non-negative -> {ok}")
