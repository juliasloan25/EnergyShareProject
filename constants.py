# This file defines shared constants used across the project:
# technology names, cost figures from Table 1, storage parameters,
# and the Mar del Plata city data from Annex I.
#
# Note: Keep this file free of imports from the rest of the project so
# every module can safely import from it without circular imports.


# Technology order used when converting between lists and dicts.
# Nuclear, wind, and solar capacities are in kW; storage is in kWh.
TECHS = ["nuclear", "wind", "solar", "storage"]

# Investment per capita (Table 1)
INV_PER_CAPITA = 20000  # EUR/person

# Hydro nominal power per unit of country area (Table 1)
HYDRO_PER_AREA = 20  # kW/km^2

# Levelized cost of installed capacity for 60 years (Table 1)
COSTS = {
    "nuclear": 47000,  # EUR/kW
    "wind":    11000,  # EUR/kW
    "solar":    6300,  # EUR/kW
    "storage":  3000,  # EUR/kWh
}

# Storage parameters (Table 1)
STORAGE = {
    "dispatch_time": 4,     # hours
    "eff_rt":        0.85,  # round-trip efficiency (applied when charging)
    "soc0":          0.5,   # initial state of charge
}

# Mar del Plata (Annex I)
CITY = {
    "name":     "MardelPlata",
    "pop":      2_001_400,  # people (2.0014 million)
    "area_km2": 5639,
    "lat":      -38.03,
    "lon":      -57.55,
}
