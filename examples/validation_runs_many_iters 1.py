"""Analysis script for the ORE-Catapult validation work."""

from time import perf_counter
from pathlib import Path

import numpy as np
import pandas as pd
import stopit
from wombat import Simulation

from additional_metrics import gather_results, vessel_summary


ANALYSIS_LIBRARY = Path("../library").resolve()

# Dictionary to determine what the primary results will be
RESULTS_BASE = {
    "opex_total": 0,
    "opex_levelized": 0,
    "vessel_annual": 0,
    "vessel_levelized": 0,
    "ctv_annual": 0,
    "rov_annual": 0,
    "cab_annual": 0,
    "ahv_annual": 0,
    "jackup_annual": 0,
    "tugboat_annual": 0,
    "cab_mobilization": 0,
    "ahv_mobilization": 0,
    "jackup_mobilization": 0,
    "activities_annual": 0,
    "labor_annual": 0,
    "activities_levelized": 0,
    "labor_levelized": 0,
    "energy": 0,
    "opex_energy": 0,
    "time_availability": 0,
    "energy_availability": 0,
    "planned_visits": 0,
    "unplanned_visits": 0,
    "time_to_repair_start": 0,
    "quayside_annual": 0,
    "gcf": 0,
    "ncf": 0,
}
results_columns = [*RESULTS_BASE]

# Determine a consistent ordering of the results
results_order = [
    "opex_total",
    "opex_levelized",
    "vessel_annual",
    "vessel_levelized",
    "quayside_annual",
    "ctv_annual",
    "rov_annual",
    "cab_annual",
    "ahv_annual",
    "jackup_annual",
    "tugboat_annual",
    "cab_mobilization",
    "ahv_mobilization",
    "jackup_mobilization",
    "activities_annual",
    "labor_annual",
    "activities_levelized",
    "labor_levelized",
    "energy",
    "opex_energy",
    "time_availability",
    "energy_availability",
    "planned_visits",
    "unplanned_visits",
    "time_to_repair_start",
    "gcf",
    "ncf",
]
vessel_results_order = ["Weather Delay", "Vessel Usage", "Mobilization", "Vessel Inactive", "Vessel Cost"]


def load_and_run(name: str, i: int, rng) -> Simulation:
    """Run a given configuration, and output the summary runtime stats and primary results."""
    print(f"{name.rjust(40)} | {i:>3.0f}", end=" | ")
    start = perf_counter()
    sim = Simulation(ANALYSIS_LIBRARY, f"{name}.yaml", random_generator=rng)
    end = perf_counter()
    print(f"Load: {(end - start) / 60:.2f} m", end=" | ")

    start = perf_counter()
    sim.run()
    end = perf_counter()
    print(f"Run: {(end - start) / 60:5.2f} m", end=" | ")

    avail = sim.metrics.production_based_availability(
        frequency="project", by="windfarm"
    ).values[0][0]
    opex = sim.metrics.opex("project").values[0][0] / sim.windfarm.capacity
    print(f"Avail: {avail:6.2%} | Opex ($/kw): {opex: 8,.2f}")
    sim.env.cleanup_log_files()
    return sim

def compile_results(sim: Simulation) -> dict:
    """Gathers the primary results, vessel usage summaries, and activities summaries."""
    results = gather_results(sim)
    
    # Calculate the vessel breakdown
    vessel_results = vessel_summary(sim.metrics)

    # Summarize the repair activities
    materials = sim.metrics.events.loc[
        sim.metrics.events.materials_cost > 0,
        ["part_name", "reason", "materials_cost"]
    ]
    materials.loc[materials.part_name.str.startswith("CB"), "part_name"] = "cable"
    materials_results = materials.groupby(["part_name", "reason"]).agg(["count", "sum"]).sort_index().T.droplevel(0).T
    
    return results, vessel_results, materials_results


if __name__ == "__main__":
    
    # Determine the configurations that will be used
    configs = [
        "layout_1_in_situ_base",
        "layout_1_in_situ_base_24hr",
        "layout_1_in_situ_no_major",
        "layout_1_in_situ_increase_maintenance",
        "layout_1_in_situ_increase_failure",
        "layout_1_in_situ_increase_failure_24hr",
        "layout_1_in_situ_increase_wave",
        "layout_1_tow_base",
        "layout_1_tow_base_24hr",
        "layout_1_tow_increase_wave",
        "layout_1_tow_increase_failure",
        "layout_1_tow_increase_maintenance",
        "layout_2_tow_base",
        "layout_2_in_situ_base",
    ]

    results = {}
    for config in configs:

        # Initialize results lists for later concatenation
        results_list = []
        vessel_results_list = []
        activities_results_list = []

        # Reset the random generator for the scenario, and run each scenario 50 times
        rng = np.random.default_rng(seed=34)
        for i in range(50):

            # Only needed as a safety net for stuck simulations
            with stopit.ThreadingTimeout(15 * 60) as time_manager:
                sim = load_and_run(config, i, rng)
            if time_manager.state != time_manager.EXECUTED:
                with stopit.ThreadingTimeout(20 * 60) as time_manager:
                    sim = load_and_run(config, i, rng)
            
            # Gather the results
            results, vessel_results, materials_results = compile_results(sim)
            
            # Create the primary results data frame and append each to the list of results
            results = pd.DataFrame.from_dict(results, orient="index", columns=[config]).loc[results_order]
            results.index.name = "metric"
            results_list.append(results)
            vessel_results_list.append(vessel_results)
            activities_results_list.append(materials_results)

            # Delete the logging files
            sim.env.cleanup_log_files()
        
        # Concatenate each set of results into a single dataframe
        results = pd.concat(results_list).groupby(["metric"]).agg(["mean", "std"]).loc[results_order]
        vessel_results = pd.concat(vessel_results_list).groupby(["Name"]).agg(["mean", "std"]).loc[vessel_results_order]
        activities_results = pd.concat(activities_results_list).groupby(["part_name", "reason"]).agg(["mean", "std"])

        # Set the main attributes to make the saved results legible
        vessel_results.columns.names = ["Vessel", "Value"]
        vessel_results = vessel_results.unstack().to_frame()
        vessel_results.index = vessel_results.index.swaplevel("Vessel", "Name").swaplevel("Value", "Vessel").set_names(["Metric", "Vessel", "Value"])
        vessel_results = vessel_results.unstack()
        vessel_results.columns = vessel_results.columns.droplevel(0).set_names([None])

        # Save to file
        results.to_csv(f"{config}_summary.csv")
        vessel_results.to_csv(f"{config}_vessel_summary.csv")
        activities_results.to_csv(f"{config}_activities.csv")
