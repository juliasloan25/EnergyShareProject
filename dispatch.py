# This file contains functions to compute and validate
# the amount of energy produced by each energy source at each
# hour of the year.

def dispatch(demand, nuclear, hydro, wind, solar,
             C_storage, eta=0.85, dispatch_h=4, soc0=0.5) -> pd.DataFrame:
    """Simulate hourly dispatch (energy production of each energy source)
    and return generation, storage, gas generation, state of charge,
    and surplus electricity."""

def validate_dispatch(df, demand) -> dict:
    """Check hourly energy balance, storage limits, and other
    dispatch constraints; return validation results."""

