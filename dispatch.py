# This file contains functions to compute and validate
# the amount of energy produced by each energy source at each
# hour of the year.

import numpy as np
import pandas as pd


def dispatch(demand, nuclear, hydro, wind, solar,
             C_storage, eta=0.85, dispatch_h=4, soc0=0.5) -> pd.DataFrame:
    """Simulate hourly electricity dispatch."""

    # Convert inputs to numpy arrays
    demand = np.asarray(demand, dtype=float)
    hydro = np.asarray(hydro, dtype=float)
    wind = np.asarray(wind, dtype=float)
    solar = np.asarray(solar, dtype=float)

    # Nuclear generation is constant throughout the year
    nuclear = np.full(len(demand), nuclear, dtype=float)

    # Calculate total generation
    total_generation = nuclear + hydro + wind + solar

    # Calculate difference between generation and demand
    net_power = total_generation - demand
    
    # Battery parameters
    dt = 1.0  # Each time step is 1 hour
    P_storage_max = C_storage / dispatch_h
    E_battery = C_storage * soc0

    # Arrays for hourly results
    battery_charge = np.zeros(len(demand))
    battery_discharge = np.zeros(len(demand))
    gas = np.zeros(len(demand))
    curtailment = np.zeros(len(demand))
    soc = np.zeros(len(demand))

    # Hourly dispatch simulation
    for t in range(len(demand)):

        if net_power[t] > 0:
            # Surplus electricity: charge battery
            available_capacity = C_storage - E_battery

            charge = min(
                net_power[t],
                P_storage_max,
                available_capacity / (eta * dt)
            )

            battery_charge[t] = charge
            E_battery += charge * eta * dt

            # Remaining surplus cannot be used
            curtailment[t] = net_power[t] - charge

        elif net_power[t] < 0:
            # Electricity deficit: discharge battery
            deficit = -net_power[t]

            discharge = min(
                deficit,
                P_storage_max,
                E_battery / dt
            )

            battery_discharge[t] = discharge
            E_battery -= discharge * dt

            # Remaining deficit supplied by gas
            gas[t] = deficit - discharge

        # Store battery state of charge
        soc[t] = E_battery / C_storage if C_storage > 0 else 0

               
    df = pd.DataFrame({
        "demand": demand,
        "nuclear": nuclear,
        "hydro": hydro,
        "wind": wind,
        "solar": solar,
        "total_generation": total_generation,
        "net_power": net_power,
        "battery_charge": battery_charge,
        "battery_discharge": battery_discharge,
        "gas": gas,
        "curtailment": curtailment,
        "soc": soc
    })

    return df


def validate_dispatch(df, demand) -> dict:
    """Validate dispatch constraints."""
    pass
if __name__ == "__main__":

    demand = np.array([100, 100, 100, 100, 100])
    nuclear = 40
    hydro = np.array([10, 10, 10, 10, 10])
    wind = np.array([20, 30, 10, 40, 20])
    solar = np.array([50, 30, 0, 20, 0])

    df = dispatch(
        demand, nuclear, hydro, wind, solar,
        C_storage=100,
        eta=0.85,
        dispatch_h=4,
        soc0=0.5
    )

    print(df[[
        "net_power",
        "battery_charge",
        "battery_discharge",
        "gas",
        "soc"
    ]])
