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
    df.attrs["C_storage"] = C_storage
    df.attrs["dispatch_h"] = dispatch_h
    df.attrs["eta"] = eta
    df.attrs["soc0"] = soc0
    return df


def validate_dispatch(df, demand) -> dict:
    """Validate hourly dispatch results."""

    tolerance = 1e-6

    # 1. Check electricity balance
    supply = (
        df["total_generation"]
        + df["battery_discharge"]
        + df["gas"]
    )

    consumption = (
        df["demand"]
        + df["battery_charge"]
        + df["curtailment"]
    )

    balance_error = np.abs(supply - consumption)
    balance_ok = bool((balance_error < tolerance).all())

    # 2. Check battery SOC limits
    soc_ok = bool(
        ((df["soc"] >= -tolerance) &
         (df["soc"] <= 1 + tolerance)).all()
    )

    # 3. Check non-negative values
    non_negative_ok = bool(
        (df[[
            "battery_charge",
            "battery_discharge",
            "gas",
            "curtailment"
        ]] >= -tolerance).all().all()
    )

    # 4. Check no simultaneous charging/discharging
    simultaneous_ok = bool(
        ~((df["battery_charge"] > tolerance) &
          (df["battery_discharge"] > tolerance)).any()
    )

    # 5. Check demand matches input
    demand_ok = bool(
        np.allclose(df["demand"], demand, atol=tolerance)
    )

    # 6. Check battery power limits
    C_storage = df.attrs["C_storage"]
    dispatch_h = df.attrs["dispatch_h"]
    eta = df.attrs["eta"]
    soc0 = df.attrs["soc0"]

    P_storage_max = C_storage / dispatch_h

    power_limit_ok = bool(
        (df["battery_charge"] <= P_storage_max + tolerance).all()
        and
        (df["battery_discharge"] <= P_storage_max + tolerance).all()
    )

    # 7. Check battery SOC evolution
    expected_energy = C_storage * soc0
    soc_evolution_ok = True

    for t in range(len(df)):
        expected_energy += (
            eta * df["battery_charge"].iloc[t]
            - df["battery_discharge"].iloc[t]
        )

        actual_energy = df["soc"].iloc[t] * C_storage

        if not np.isclose(
            expected_energy, actual_energy, atol=tolerance
        ):
            soc_evolution_ok = False
            break
     # 8. Check battery SOC e    
    results = {
        "energy_balance": balance_ok,
        "soc_limits": soc_ok,
        "non_negative_values": non_negative_ok,
        "no_simultaneous_charge_discharge": simultaneous_ok,
        "demand_match": demand_ok,
        "battery_power_limits": power_limit_ok,
        "soc_evolution": soc_evolution_ok
    }

    results["all_valid"] = all(results.values())

    return results

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
    validation = validate_dispatch(df, demand)

    print("\nDispatch Validation Results:")
    for check, result in validation.items():
        print(f"{check}: {result}")

