# Main file: calls functions from resources.py, dispatch.py, and optimization.py
# to find the optimal investment fractions for nuclear, wind, solar, and storage.
# Also calls plotting functions from analysis.py.

from constants import (
    TECHS,
    INV_PER_CAPITA,
    HYDRO_PER_AREA,
    COSTS,
    STORAGE,
    CITY,
)
