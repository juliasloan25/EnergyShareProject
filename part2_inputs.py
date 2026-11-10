# This file defines the functions that take the calculations
# from part 1 of the project and provide them as inputs to
# part 2. So far, this includes a helper to get the generation
# and demand for a specific hour of the year. 

def snapshot(df, demand, hour) -> dict:
    """Extract generation and demand for one selected hour,
    converting kW to MW for the Part 2 network model."""

