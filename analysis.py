# This file defines functions related to analyzing and
# communicating the results of the optimization, such as
# sensitivity analysis and plotting.

def emissions_by_tech(df, emission_factors) -> dict:
    """Annual CO2 emissions per technology, in gCO2.

    Returns a dict keyed by technology name ("nuclear", "hydro", "wind",
    "solar", "gas"). Used by both co2_per_kwh and plot_co2_pie.
    """

    technologies = ["nuclear", "hydro", "wind", "solar", "gas"]

    # Check that all required columns exist
    for technology in technologies:
        if technology not in df.columns:
            raise ValueError(f"Missing column: {technology}")

        if technology not in emission_factors:
            raise ValueError(f"Missing emission factor: {technology}")

    # kWh generated * gCO2/kWh = gCO2
    return {
        technology: float(df[technology].sum() * emission_factors[technology])
        for technology in technologies
    }


def co2_per_kwh(df, emission_factors) -> float:
    """Calculate annual CO2 emissions per kWh generated."""

    # Save per-technology emissions for reuse by plot_co2_pie
    emissions = emissions_by_tech(df, emission_factors)

    # Calculate total CO2 emissions and total annual electricity generation
    total_emissions = sum(emissions.values())
    total_generation = df[list(emissions.keys())].sum().sum()

    # Avoid division by zero
    if total_generation == 0:
        raise ValueError("Total electricity generation cannot be zero.")

    # Calculate average CO2 emissions per kWh
    co2_intensity = total_emissions / total_generation

    return float(co2_intensity)


def cc_stats(df) -> dict:
    """Calculate installed gas capacity and its load factor."""

    # Check that the gas column exists
    if "gas" not in df.columns:
        raise ValueError("Missing column: gas")

    # Calculate maximum gas generation (installed capacity) in MW
    installed_capacity = df["gas"].max() / 1000

    # Calculate average gas generation in kW
    average_generation = df["gas"].mean()

    # Calculate load factor
    if installed_capacity > 0:
        load_factor = average_generation / (installed_capacity * 1000)
    else:
        load_factor = 0.0

    # Return results as a dictionary
    return {
        "installed_gas_capacity_MW": float(installed_capacity),
        "gas_load_factor": float(load_factor)
    }


def spared_pct(df) -> float:
    """Calculate surplus electricity as a percentage of total production."""

    # Check that the required columns exist
    required_columns = ["curtailment", "total_generation"]

    for column in required_columns:
        if column not in df.columns:
            raise ValueError(f"Missing column: {column}")

    # Calculate total curtailed energy
    total_surplus = df["curtailment"].sum()

    # Calculate total electricity generated
    total_generation = df["total_generation"].sum()

    # Avoid division by zero
    if total_generation == 0:
        raise ValueError("Total generation cannot be zero.")

    # Calculate surplus percentage
    surplus_percentage = (total_surplus / total_generation) * 100

    return float(surplus_percentage)

def plot_week(df, demand, start_hour):
    """Plot generation, demand, and storage over one week."""


import matplotlib.pyplot as plt

def plot_pie(df, save_path=None):
    """Plot the annual contribution of each technology and storage.

    If save_path is given, save the figure to that path (in addition to
    showing it interactively).
    """

    # Define the columns and labels for the pie chart
    columns = [
        "nuclear",
        "hydro",
        "wind",
        "solar",
        "gas",
        "battery_discharge"
    ]

    labels = [
        "Nuclear",
        "Hydro",
        "Wind",
        "Solar",
        "Gas",
        "Storage"
    ]

    # Check that all required columns exist
    for column in columns:
        if column not in df.columns:
            raise ValueError(f"Missing column: {column}")

    # Calculate annual energy supplied by each source
    annual_energy = df[columns].sum()

    # Calculate total electricity supplied
    total_energy = annual_energy.sum()

    if total_energy == 0:
        raise ValueError("Total energy supplied cannot be zero.")

    # Create pie chart
    plt.figure(figsize=(9, 7))

    plt.pie(
        annual_energy,
        labels=labels,
        autopct="%1.1f%%",
        startangle=90
    )

    plt.title("Annual Electricity Supply Mix")
    plt.axis("equal")
    plt.tight_layout()

    # Save before show(): with some backends the figure is closed when
    # the user dismisses the window.
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")

    plt.show()


def plot_co2_pie(df, emission_factors, save_path=None):
    """Plot the annual CO2 emissions by technology as a pie chart.

    If save_path is given, save the figure to that path (in addition to
    showing it interactively).
    """

    # Reuse the per-technology emissions computation from emissions_by_tech
    emissions = emissions_by_tech(df, emission_factors)

    total_emissions = sum(emissions.values())

    if total_emissions == 0:
        raise ValueError("Total emissions cannot be zero.")

    labels = [technology.capitalize() for technology in emissions.keys()]

    # Create pie chart
    plt.figure(figsize=(9, 7))

    plt.pie(
        list(emissions.values()),
        labels=labels,
        autopct="%1.1f%%",
        startangle=90,
    )

    # 1 Mt CO2 = 1e12 gCO2. Report in Mt because annual emissions at this
    # scale are typically 0.5-few Mt.
    plt.title(
        "Annual CO2 Emissions by Technology\n"
        f"Total: {total_emissions / 1e12:.2f} Mt CO2/year"
    )
    plt.axis("equal")
    plt.tight_layout()

    # Save before show(): with some backends the figure is closed when
    # the user dismisses the window.
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")

    plt.show()


def sensitivity_analysis(profiles, demand, pop, costs, scenarios) -> pd.DataFrame:
    """Evaluate how changes in assumptions affect gas use
    and other key results."""


if __name__ == "__main__":
    # Integration smoke test: run each analysis function on a real,
    # full-year scenario. Uses a fixed (non-optimized) budget split so
    # the test is fast — the optimizer is tested in optimization.py.
    import matplotlib
    matplotlib.use("Agg", force=True)  # make plt.show() a no-op for the test

    from resources import load_resource_data, build_profiles, demand_profile
    from optimization import capacities_from_investment, simulate_year
    from constants import CITY, COSTS, EMISSION_FACTORS

    print("=" * 60)
    print("analysis.py smoke test")
    print("=" * 60)

    # [1] Build a real scenario.
    print("\n[1] Setting up scenario")
    data = load_resource_data(
        "POWER_Point_Hourly_20250101_20251231_038d03S_057d55W_UTC.csv",
        "river_flow_norm.csv",
        "demand_data2026.dat",
    )
    profiles = build_profiles(data)
    demand = demand_profile(data["demand_per_capita"], CITY["pop"])

    fractions = {"nuclear": 0.4, "wind": 0.3, "solar": 0.2, "storage": 0.1}
    caps = capacities_from_investment(fractions, COSTS)
    df = simulate_year(caps, profiles, demand)

    print(f"  fractions: {fractions}")
    print(f"  gas: {df['gas'].sum() / 1e9:.2f} TWh/year")

    # [2] co2_per_kwh
    print("\n[2] co2_per_kwh")
    co2 = co2_per_kwh(df, EMISSION_FACTORS)
    print(f"  CO2 intensity: {co2:.1f} gCO2/kWh")

    # [3] cc_stats
    print("\n[3] cc_stats")
    stats = cc_stats(df)
    print(f"  installed gas capacity: {stats['installed_gas_capacity_MW']:.1f} MW")
    print(f"  gas load factor:        {stats['gas_load_factor'] * 100:.1f} %")

    # [4] spared_pct
    print("\n[4] spared_pct")
    spared = spared_pct(df)
    print(f"  spared (curtailed): {spared:.1f} % of total production")

    # [5] plot_pie and plot_co2_pie: just check they run end-to-end without
    #     raising. With the Agg backend, plt.show() warns but is a no-op;
    #     suppress it.
    print("\n[5] plot_pie / plot_co2_pie")
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        plot_pie(df)
        plot_co2_pie(df, EMISSION_FACTORS)
    plt.close("all")
    print("  ran without error")
