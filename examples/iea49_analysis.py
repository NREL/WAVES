"""Script to generate results for the IEA Wind Task 49 O&M model comparison between NREL's WOMBAT
model (with WAVES supplement for additonal results calculation) and TNO's UWISE model.
"""

# /// script
# requires-python = "==3.11"
# dependencies = [
#   "waves>=0.5.3",
#   "wombat>=0.11.3",
#   "rich",
#   "openpyxl",
# ]
# ///


from time import perf_counter
from pathlib import Path

import numpy as np
import pandas as pd
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn, SpinnerColumn
from pandas.api.types import CategoricalDtype

from waves import Project
from waves.utilities import load_yaml


# Update core Pandas display settings
pd.options.display.float_format = "{:,.2f}".format
pd.options.display.max_columns = 100
pd.options.display.max_rows = 100


metrics_configuration = {
    "Time_Availability": {
        "metric": "availability",
        "kwargs": {"which": "time"},
    },
    "Pro_Availability": {
        "metric": "availability",
        "kwargs": {"which": "energy"},
    },
    "OM_Total": {
        "metric": "opex",
        "kwargs": {"frequency": "project"},
    },
}

subassembly_order = [
    "Power Electrical System",
    "Power Converter",
    "Pitch System",
    "Ballast Pump",
    "Yaw System",
    "Rotor Blades",
    "Direct Drive Generator",
    "Main Shaft",
    "Mooring Line",
    "Anchor",
    "Buoyancy Module",
    "Wind Turbine",
    "Array Cable",
    "Export Cable",
    "Offshore Substation",
]
repair_order = [
    "Minor Repair",
    "Major Repair",
    "Replacement",
    "Annual Turbine Inspection",
    "Structural Annual Inspection",
    "Structural Subsea Inspection",
    "Export Cable Subsea Inspection",
    "OSS Annual Inspection",
]
SubassemblyCategory = CategoricalDtype(subassembly_order, ordered=True)
MaintenanceCategory = CategoricalDtype(repair_order, ordered=True)

invalid_corrective_combos = [
    "BallastPump_MajorRepair",
    "BallastPump_Replacement",
    "Anchor_MinorRepair",
    "BuoyancyModule_MinorRepair",
    "BuoyancyModule_MajorRepair",
    "ArrayCable_MinorRepair",
    "ExportCable_MinorRepair",
    "ExportCable_Replacement",
    "OffshoreSubstation_Replacement",
    "WindTurbine_MinorRepair",
    "WindTurbine_MajorRepair",
    "WindTurbine_Replacement",
]
valid_scheduled_combos = [
    "WindTurbine_AnnualTurbineInspection",
    "WindTurbine_StructuralAnnualInspection",
    "WindTurbine_StructuralSubseaInspection",
    "ExportCable_ExportCableSubseaInspection",
    "OffshoreSubstation_OSSAnnualInspection",
]

ctv_cols = [f"Crew Transfer Vessel {i}" for i in range(1, 4)]
dsv_cols = [f"Diving Support Vessel {i}" for i in range(1, 3)]


def run_analysis(
    seed: int,
    rng: np.random._generator.Generator,
    base_config: dict,
    wombat_config: dict,
    project: Project | None,
) -> tuple[Project, float]:
    """Run a single analysis, modifying the configuration as needed for WOMBAT's random generator.

    Parameters
    ----------
    seed : int
        The iteration number for the scenario.
    rng : np.random._generator.Generator
        Preset NumPy random generator to be recycled for each scenario.
    base_config : dict
        The WAVES configuration dictionary.
    wombat_config : dict
        The WOMBAT configuration dictionary.
    project : waves.Project
        WAVES Simulation object, if one has already been initialized with ORBIT and FLORIS
        simulation results prepared.

    Returns
    -------
        waves.Project : Complete simulation object.
    """
    config_wombat["random_generator"] = rng
    config["wombat_config"] = config_wombat
    start = perf_counter()
    if seed == 0:
        project = Project.from_dict(config)
        assert isinstance(project, Project)
        project.run()
    else:
        om_project = Project.from_dict(config)
        assert isinstance(project, Project)
        om_project.run(skip=["orbit", "floris"])
        project.wombat = om_project.wombat
    project.wombat.env.cleanup_log_files()
    end = perf_counter()
    run_time = end - start
    return project, run_time


def calculate_monthly_work_orders(seed: int, project: Project) -> pd.DataFrame:
    """Calculate the cumulative remaining work orders, and the number of submitted, canceled,
    and completed worker orders by simulation month.

    Parameters
    ----------
    seed : int
        The iteration number for the a scenario.
    project : waves.Project
        The simulation object.

    Returns
    -------
    pd.DataFrame : A monthly data frame of work order statuses.
    """
    repair_order = ["Minor Repair", "Major Repair", "Replacement"]
    total_order = repair_order + ["Scheduled"]
    ev = project.wombat.metrics.events
    request_map = dict(ev.loc[ev.action.eq("repair request"), ["request_id", "reason"]].values)
    base_df = ev[["year", "month", "duration"]].groupby(["year", "month"]).count()
    base_df[total_order] = 0
    base_df = base_df[total_order]

    repair_requests = (
        ev.loc[
            ev.action.eq("repair request"),
            ["year", "month", "request_id", "reason"],
        ]
        .drop_duplicates()
        .groupby(["year", "month", "reason"])
        .count()
        .unstack()
        .fillna(0)
    )
    repair_requests.columns = repair_requests.columns.droplevel(level=0)

    maintenance_requests = (
        ev.loc[
            ev.action.eq("maintenance request"),
            ["year", "month", "request_id"],
        ]
        .drop_duplicates()
        .groupby(["year", "month"])
        .count()
        .rename(columns={"request_id": "Scheduled"})
    )

    canceled_repairs = (
        ev.loc[
            ev.action.eq("repair canceled"),
            ["year", "month", "request_id", "duration"],
        ]
        .drop_duplicates()
        .replace(request_map)
        .groupby(["year", "month", "request_id"])
        .count()
        .unstack()
        .fillna(0)
    )
    canceled_repairs.columns = canceled_repairs.columns.droplevel(level=0)

    canceled_maintenance = (
        ev.loc[
            ev.action.eq("maintenance canceled"),
            ["year", "month", "request_id"],
        ]
        .drop_duplicates()
        .groupby(["year", "month"])
        .count()
        .rename(columns={"request_id": "Scheduled"})
    )

    complete_maintenance = (
        ev.loc[
            ev.action.eq("maintenance complete"),
            ["year", "month", "request_id"],
        ]
        .drop_duplicates()
        .groupby(["year", "month"])
        .count()
        .rename(columns={"request_id": "Scheduled"})
    )

    complete_repairs = (
        ev.loc[
            ev.action.eq("repair complete"),
            ["year", "month", "request_id", "duration"],
        ]
        .drop_duplicates()
        .replace(request_map)
        .groupby(["year", "month", "request_id"])
        .count()
        .unstack()
        .fillna(0)
    )
    complete_repairs.columns = complete_repairs.columns.droplevel(level=0)

    requests = (
        repair_requests.join(maintenance_requests).reindex_like(base_df).fillna(0).astype(int)
    )
    canceled = (
        canceled_repairs.join(canceled_maintenance).reindex_like(base_df).fillna(0).astype(int)
    )
    complete = (
        complete_repairs.join(complete_maintenance).reindex_like(base_df).fillna(0).astype(int)
    )
    total = requests.cumsum() - canceled.cumsum() - complete.cumsum()
    total.insert(0, "NumberOfWO", total.sum(axis=1))
    requests.insert(0, "NumberOfSubmittedWO", requests.sum(axis=1))
    canceled.insert(0, "NumberOfCanceledWO", canceled.sum(axis=1))
    complete.insert(0, "NumberOfCompletedWO", complete.sum(axis=1))
    work_orders = (
        total.rename(columns={c: f"NumberOfWO_{c.replace(' ', '')}" for c in total_order})
        .join(
            requests.cumsum().rename(
                columns={c: f"NumberOfSubmittedWO_{c.replace(' ', '')}" for c in total_order}
            )
        )
        .join(
            canceled.cumsum().rename(
                columns={c: f"NumberOfCanceledWO_{c.replace(' ', '')}" for c in total_order}
            )
        )
        .join(
            complete.cumsum().rename(
                columns={c: f"NumberOfCompletedWO_{c.replace(' ', '')}" for c in total_order}
            )
        )
    )
    return work_orders


def gather_project_results(seed: int, project: Project) -> pd.DataFrame:
    """Calculate the availability, revenue, O&M cost breakdown, failure rate, vessel hours at sea,
    utilization rate, and dispatch summaries; and combine them as a single row for concatenation.

    Parameters
    ----------
    seed : int
        The iteration number for the a scenario.
    project : waves.Project
        The simulation object.

    Returns
    -------
    pd.DataFrame : A single row data frame, with index :py:attr:`seed`, with all project-level
        results.
    """
    years = project.wombat.env.simulation_years
    metrics = project.wombat.metrics
    vessels = [*project.wombat.service_equipment]

    results = project.generate_report(metrics_configuration, simulation_name=seed)  # type: ignore
    fixed_costs = metrics.project_fixed_costs(frequency="project", resolution="medium")
    vessel_costs = metrics.equipment_costs(
        frequency="project", by_equipment=True
    )  # TODO: This value is far too high, is there double/triple counting?

    # Availability-only revenue and downtime losses
    potential = project.wombat.metrics.potential.windfarm.values.sum() / 1000
    production = project.wombat.metrics.production.windfarm.values.sum() / 1000
    results["Revenue"] = production * project.offtake_price
    results["Downtime_Loss"] = (potential - production) * project.offtake_price

    # Fixed cost breakdown
    base_cols = ["operations_management_administration", "operating_facilities"]
    results["OM_OMBase"] = fixed_costs[base_cols].values.sum()
    results["OM_Tech"] = fixed_costs["labor"].squeeze()
    results["OM_Crane"] = metrics.port_fees(frequency="project").squeeze()
    results["OM_Tug"] = fixed_costs["onshore_electrical_maintenance"].squeeze()
    results["OM_CTV"] = vessel_costs[ctv_cols].values.sum()

    # Vessel and materials cost breakdown
    results["OM_CLV"] = vessel_costs["Cable Lay Vessel"].squeeze()
    results["OM_DSV"] = vessel_costs[dsv_cols].values.sum()
    results["OM_AHV"] = vessel_costs["Anchor Handling Tug"].squeeze()
    results["OM_Materials"] = metrics.component_costs("project").sum().squeeze()

    # Normalize costs to k€/MW/yr
    cost_cols = [c for c in results if c.startswith("OM_")]
    cost_cols.extend(["Revenue", "Downtime_Loss"])
    results[cost_cols] = results[cost_cols] / years / 1000 / project.capacity(units="mw")
    results.index.name = "seed"

    # Corrective and scheduled maintenance summary
    failures = metrics.request_summary()
    _failures = (
        failures.total_requests.to_frame().rename({"OSS": "Offshore Substation"}).reset_index()
    )
    _failures = _failures.astype({"subassembly": SubassemblyCategory, "task": MaintenanceCategory})
    _failures = _failures.groupby(["subassembly", "task"], observed=False).sum().reset_index()
    is_corrective = _failures.task.str.contains("Repair|Replacement")

    corrective = _failures.loc[is_corrective].sort_values(["subassembly", "task"])
    corrective["name"] = (
        corrective.subassembly.str.replace(" ", "") + "_" + corrective.task.str.replace(" ", "")
    )
    corrective = (
        corrective.loc[~corrective.name.isin(invalid_corrective_combos)]
        .set_index("name")[["total_requests"]]
        .T.rename({"total_requests": seed})
    )
    corrective.index.name = "seed"

    scheduled = _failures.loc[~is_corrective].sort_values(["subassembly", "task"])
    scheduled["name"] = (
        scheduled.subassembly.str.replace(" ", "") + "_" + scheduled.task.str.replace(" ", "")
    )
    scheduled = (
        scheduled.loc[scheduled.name.isin(valid_scheduled_combos)]
        .set_index("name")[["total_requests"]]
        .T.rename({"total_requests": seed})
    )
    scheduled.columns = scheduled.T.index.str.split("_").str[1]
    scheduled.index.name = "seed"

    # Vessel hours at sea
    vessel_hours = metrics.vessel_crew_hours_at_sea(
        frequency="project", by_equipment=True, vessel_crew_assumption={}
    )[vessels].rename({0: seed})
    vessel_hours.columns = [f"SeaHours_{col.replace(' ', '')}" for col in vessel_hours.columns]
    vessel_hours.index.name = "seed"

    # Unscheduled, scheduled, and combined utilization rates
    utilization = metrics.service_equipment_utilization(frequency="project")[vessels].rename(
        {0: seed}
    )  # noqa: E501
    utilization.columns = [f"UtilizationRate_{col.replace(' ', '')}" for col in utilization.columns]
    utilization.index.name = "seed"

    # Mobilization and chartering time summary
    mobilization_summary = metrics.dispatch_summary(frequency="project")
    mobilization_summary = mobilization_summary.loc[mobilization_summary["N Mobilizations"] > 1]

    mobilizations = mobilization_summary[["N Mobilizations"]].T.rename({"N Mobilizations": seed})[
        [v for v in vessels if v in mobilization_summary.index]
    ]
    charter_period = mobilization_summary[["Average Charter Days"]].T.rename(
        {"Average Charter Days": seed}
    )[[v for v in vessels if v in mobilization_summary.index]]

    mobilizations.columns = [f"NumMob_{col.replace(' ', '')}" for col in mobilizations.columns]
    charter_period.columns = [
        f"CharterDay_{col.replace(' ', '')}" for col in charter_period.columns
    ]
    mobilizations.index.name = "seed"
    charter_period.index.name = "seed"

    # Combine all seed results
    results = pd.concat(
        (
            results,
            corrective / years,
            scheduled / years,
            vessel_hours / years,
            utilization,
            mobilizations / years,
            charter_period / years,
        ),
        axis=1,
    )
    return results


def gather_timeline_results(seed: int, project: Project) -> pd.DataFrame:
    """Calculate the monthly availability and task completion rates as columns of a dataframe.

    Parameters
    ----------
    seed : int
        The iteration number for the a scenario.
    project : waves.Project
        The simulation object.

    Returns
    -------
    pd.DataFrame : A 240-row data frame with indices :py:attr:`seed`, "year", and "month".
    """
    metrics = project.wombat.metrics
    results = [
        metrics.time_based_availability("month-year", "windfarm").rename(
            columns={"windfarm": "Time_Availability"}
        ),
        metrics.production_based_availability("month-year", "windfarm").rename(
            columns={"windfarm": "Pro_Availability"}
        ),
        metrics.task_completion_rate("unscheduled", "month-year").rename(
            columns={"Completion Rate": "CompletionRate_Unscheduled"}
        ),
        metrics.task_completion_rate("scheduled", "month-year").rename(
            columns={"Completion Rate": "CompletionRate_Scheduled"}
        ),
        metrics.task_completion_rate("both", "month-year").rename(
            columns={"Completion Rate": "CompletionRate_Both"}
        ),
        calculate_monthly_work_orders(seed, project),
    ]
    results = pd.concat(results, axis=1)
    assert isinstance(results, pd.DataFrame)
    results = results.set_index(
        pd.Index(np.full(results.shape[0], seed, dtype="int"), name="seed"), append=True
    )
    results.index = results.index.swaplevel(0, 2).swaplevel(1, 2)  # type: ignore
    return results


if __name__ == "__main__":
    N = 20
    scenarios = [
        "base",
    ]

    library_path = Path(__file__).parents[1] / "library/IEA_49/"
    print(f"Using library: {library_path}")

    job_progress = Progress(
        "{task.description}",
        SpinnerColumn(),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
    )
    jobs = {scenario: job_progress.add_task(scenario, total=N) for scenario in scenarios}

    total = sum(task.total for task in job_progress.tasks)
    overall_progress = Progress()
    overall_task = overall_progress.add_task("Scenarios", total=int(total))

    progress_table = Table.grid()
    progress_table.add_row(
        Panel.fit(overall_progress, title="Overall Progress", border_style="green", padding=(2, 2)),
        Panel.fit(job_progress, title="[b]Jobs", border_style="red", padding=(1, 2)),
    )

    with Live(progress_table, refresh_per_second=1):
        while not overall_progress.finished:
            for scenario in scenarios:
                rng = np.random.default_rng(seed=834)

                config = load_yaml(
                    library_path / "project/config", f"{scenario}_floating_deep.yaml"
                )
                config["floris_config"] = load_yaml(
                    library_path / "project/config", config["floris_config"]
                )
                config["floris_config"]["farm"]["turbine_library_path"] = library_path / "turbines"
                config.update({"library_path": library_path})

                config_wombat = load_yaml(
                    library_path / "project/config", config["wombat_config"]
                )  # TODO: make this file name scenario based

                scenario_project_results = []
                scenario_timeline_results = []
                job, *_ = (j for j in job_progress.tasks if j.description == scenario)
                for seed in range(N):
                    if seed == 0:
                        project = None
                    project, timing = run_analysis(
                        seed, rng, config, config_wombat, project=project
                    )
                    scenario_project_results.append(gather_project_results(seed, project))
                    scenario_timeline_results.append(gather_timeline_results(seed, project))

                    job_progress.advance(job.id)
                    completed = sum(task.completed for task in job_progress.tasks)
                    overall_progress.update(overall_task, completed=completed)

                project_results = pd.concat(scenario_project_results, axis=0)
                timeline_results = pd.concat(scenario_timeline_results, axis=0)

                results_file = library_path / "results" / f"wombat_{scenario}.xlsx"
                with pd.ExcelWriter(results_file) as writer:
                    project_results.to_excel(writer, sheet_name="Project", merge_cells=False)
                    timeline_results.to_excel(writer, sheet_name="MonthYear", merge_cells=False)

                print(f"Results saved to: {results_file}")
            overall_progress.update(overall_task, completed=completed)
            completed = sum(task.completed for task in job_progress.tasks)
