from itertools import product

import numpy as np
import pandas as pd

from wombat import Simulation
from wombat.core.post_processor import _check_frequency, Metrics


def equipment_breakdowns(
    metrics: Metrics, frequency: str, by_category: bool = False
) -> pd.DataFrame:
    """Calculates the producitivty cost breakdowns for the simulation at a project,
    annual, or monthly level that can be broken out to include the equipment and
    labor components.

    .. note:: Doesn't produce a value if there's no cost associated with a "reason".

    Parameters
    ----------
    frequency : str
        One of "project", "annual", "monthly", or "month-year".
    by_category : bool, optional
        Indicates whether to include the equipment and labor categories (True) or
        not (False), by default False.

    Returns
    -------
    pd.DataFrame
        Returns pandas ``DataFrame`` with columns:
            - year (if appropriate for frequency)
            - month (if appropriate for frequency)
            - reason
            - time (hours)
            - equipment_cost (if by_category == ``True``)

    Raises
    ------
    ValueError
        If ``frequency`` is not one of "project", "annual", "monthly", or
        "month-year".
    ValueError
        If ``by_category`` is not one of ``True`` or ``False``.
    """
    frequency = _check_frequency(frequency, which="all")
    if not isinstance(by_category, bool):
        raise ValueError("``by_equipment`` must be one of ``True`` or ``False``")

    group_filter = ["action", "reason", "additional", "agent"]
    if frequency in ("annual", "month-year"):
        group_filter.insert(0, "year")
    elif frequency == "monthly":
        group_filter.insert(0, "month")
    if frequency == "month-year":
        group_filter.insert(1, "month")

    action_list = [
        "delay",
        "repair",
        "maintenance",
        "mobilization",
        "transferring crew",
        "traveling",
        "towing",
        "mooring reconnection",
        "unmooring",
    ]
    equipment = metrics.events[metrics.events[metrics._equipment_cost] > 0].agent.unique()
    time_costs = (
        metrics.events.loc[
            metrics.events.agent.isin(equipment)
            & metrics.events.action.isin(action_list)
            & ~metrics.events.additional.isin(["work is complete"]),
            group_filter + [metrics._equipment_cost, "duration"],
        ]
        .groupby(group_filter)
        .sum()
        .reset_index()
    )
    time_costs["display_reason"] = [""] * time_costs.shape[0]

    non_shift_hours = (
        "not in working hours",
        "work shift has ended; waiting for next shift to start",
        "no more return visits will be made",
        "will return next year",
        "waiting for next operational period",
    )
    weather_hours = ("weather delay", "weather unsuitable to transfer crew")
    time_costs.loc[
        (time_costs.action == "delay") & (time_costs.additional.isin(non_shift_hours)),
        "display_reason",
    ] = "Not in Shift"
    time_costs.loc[time_costs.action == "repair", "display_reason"] = "Repair"
    time_costs.loc[time_costs.action == "maintenance", "display_reason"] = "Maintenance"
    time_costs.loc[
        time_costs.action == "transferring crew", "display_reason"
    ] = "Crew Transfer"
    time_costs.loc[time_costs.action == "traveling", "display_reason"] = "Site Travel"
    time_costs.loc[time_costs.action == "mobilization", "display_reason"] = "Mobilization"
    time_costs.loc[
        time_costs.additional.isin(weather_hours), "display_reason"
    ] = "Weather Delay"
    time_costs.loc[time_costs.reason == "no requests", "display_reason"] = "No Requests"

    time_costs.reason = time_costs.display_reason

    drop_columns = ["display_reason", "additional", "action"]
    group_filter.pop(group_filter.index("additional"))
    group_filter.pop(group_filter.index("action"))
    time_costs = time_costs.drop(columns=drop_columns)
    time_costs = time_costs.groupby(group_filter).sum().reset_index()

    month_year = frequency == "month-year"
    if frequency in ("annual", "month-year"):
        years = time_costs.year.unique()
        reasons = time_costs.reason.unique()
        comparison_values = product(years, reasons)
        if month_year:
            months = time_costs.month.unique()
            comparison_values = product(years, months, reasons)

        zeros = np.zeros(time_costs.shape[1] - 2).tolist()
        for _year, *_month, _reason in comparison_values:
            row_filter = time_costs.year.values == _year
            row = [_year, _reason] + zeros
            if month_year:
                _month = _month[0]
                row_filter &= time_costs.month.values == _month
                row = [_year, _month, _reason] + zeros[:-1]

            row_filter &= time_costs.reason.values == _reason
            if time_costs.loc[row_filter].size > 0:
                continue
            time_costs.loc[time_costs.shape[0]] = row
    elif frequency == "monthly":
        months = time_costs.month.unique()
        reasons = time_costs.reason.unique()
        comparison_values = product(months, reasons)
        zeros = np.zeros(time_costs.shape[1] - 2).tolist()
        for _month, _reason in comparison_values:
            row_filter = time_costs.month.values == _month
            row_filter &= time_costs.reason.values == _reason
            row = [_month, _reason] + zeros
            if time_costs.loc[row_filter].size > 0:
                continue
            time_costs.loc[time_costs.shape[0]] = row

    new_sort = [
        "Maintenance",
        "Repair",
        "Crew Transfer",
        "Site Travel",
        "Mobilization",
        "Weather Delay",
        "No Requests",
        "Not in Shift",
    ]
    time_costs.reason = pd.Categorical(time_costs.reason, new_sort)
    time_costs = time_costs.set_index(group_filter)
    if frequency == "project":
        return time_costs.sort_values(by="reason")
    if frequency == "annual":
        return time_costs.sort_values(by=["year", "reason"])
    if frequency == "monthly":
        return time_costs.sort_values(by=["month", "reason"])
    return time_costs.sort_values(by=["year", "month", "reason"])


def vessel_summary(metrics: Metrics, frequency="project") -> pd.DataFrame:
    keep = ["Weather Delay", "Vessel Usage", "Mobilization", "Vessel Inactive"]
    order = [
        "Crew Transfer Vessel 1",
        "Crew Transfer Vessel 2",
        "Crew Transfer Vessel 3",
        "ROV Support Vessel",
        "Cable Laying Vessel",
        "Anchor Handling Vessel",
        "Heavy Lift Vessel",
        "Tugboat 1",
    ]
    vessel_results = metrics.equipment_labor_cost_breakdowns(
        frequency, by_category=True, by_equipment=True
    )
    vessel_results = vessel_results.unstack(level="equipment_name").fillna(0)

    if "Towing" not in vessel_results.index:
        vessel_results.loc["Towing"] = 0

    vessel_results.loc["Vessel Usage"] = 0
    vessel_results.loc["Vessel Inactive"] = 0
    vessel_results.loc["Vessel Usage"] = (
        vessel_results.loc[["Maintenance", "Repair", "Crew Transfer", "Site Travel", "Towing"], :]
        .sum(axis=0)
        .to_frame(name="Vessel Usage")
        .T.values
    )
    vessel_results.loc["Vessel Inactive"] = (
        vessel_results.loc[["Not in Shift", "No Requests"], :]
        .sum(axis=0)
        .to_frame(name="Vessel Inactive")
        .T.values
    )
    vessel_results = vessel_results.loc[keep]

    vessel_results = pd.concat(
        [
            vessel_results.loc[:, "total_hours"],
            vessel_results.loc[:, "equipment_cost"]
            .sum()
            .to_frame(name="Vessel Cost")
            .T,
        ]
    )
    vessel_results.columns.name = None
    vessel_results.index.name = "Name"
    return vessel_results[[el for el in order if el in vessel_results.columns]]


def gather_results(sim: Simulation) -> dict:
    """Calculates the primary set of results without granular breakdowns."""
    project_years = 20
    n_turbines = len(sim.windfarm.turbine_id)
    metrics = sim.metrics
    events = metrics.events
    capacity_kw = sim.windfarm.capacity
    results = {}

    opex = metrics.opex("project").values[0][0]
    opex_levelized = opex / capacity_kw / project_years

    port_annual = metrics.port_fees("project").values[0][0] / project_years

    mobilization = (
        events.loc[events.action == "mobilization", ["agent", "total_cost"]]
        .groupby("agent")
        .sum()
    )
    cab_mobilization = (
        mobilization.loc[
            [el for el in mobilization.index if "Cable Laying Vessel" in el]
        ].values.sum()
        / project_years
    )
    ahv_mobilization = (
        mobilization.loc[
            [el for el in mobilization.index if "Anchor Handling Vessel" in el]
        ].values.sum()
        / project_years
    )
    jackup_mobilization = (
        mobilization.loc[
            [el for el in mobilization.index if "Heavy Lift Vessel" in el]
        ].values.sum()
        / project_years
    )

    vessels = metrics.equipment_costs(frequency="project", by_equipment=True)
    vessel_annual = vessels.values.sum() / project_years
    vessel_levelized = vessel_annual / capacity_kw
    ctv_annual = (
        vessels[
            [el for el in vessels.columns if "Crew Transfer Vessel" in el]
        ].values.sum()
        / project_years
    )
    rov_annual = (
        vessels[
            [el for el in vessels.columns if "ROV Support Vessel" in el]
        ].values.sum()
        / project_years
    )
    cab_annual = (
        vessels[
            [el for el in vessels.columns if "Cable Laying Vessel" in el]
        ].values.sum()
        / project_years
        - cab_mobilization
    )
    ahv_annual = (
        vessels[
            [el for el in vessels.columns if "Anchor Handling Vessel" in el]
        ].values.sum()
        / project_years
        - ahv_mobilization
    )
    jackup_annual = (
        vessels[
            [el for el in vessels.columns if "Heavy Lift Vessel" in el]
        ].values.sum()
        / project_years
        - jackup_mobilization
    )
    tugboat_annual = (
        vessels[
            [el for el in vessels.columns if "Tugboat" in el]
        ].values.sum()
        / project_years
    )

    materials_cost = metrics.events.materials_cost.sum()
    materials_annual = materials_cost / project_years
    materials_levelized = materials_annual / capacity_kw

    labor_annual = (
        metrics.project_fixed_costs("project", "high").labor.values[0]
        + metrics.labor_costs("project", False).total_labor_cost.values[0]
    ) / project_years
    labor_levelized = labor_annual / capacity_kw

    energy = metrics.power_production("project", "windfarm", "kwh").values[0][0]
    opex_energy = (
        opex / metrics.power_production("project", "windfarm", "mwh").values[0][0]
    )
    time_availability = metrics.time_based_availability(
        frequency="project", by="windfarm"
    ).values[0][0]
    energy_availability = metrics.production_based_availability(
        frequency="project", by="windfarm"
    ).values[0][0]

    serviced_requests = pd.Series(
        events.loc[
            events.agent.isin(metrics.service_equipment_names)
        ].request_id.unique(),
        name="request_id",
    )
    n_maintenance = (
        serviced_requests.str.startswith("MNT").sum() / n_turbines / project_years
    )
    n_repair = (
        serviced_requests.str.startswith("RPR").sum() / n_turbines / project_years
    )

    timing = metrics.process_times()
    average_time_to_start = timing.sum().T.time_to_start / timing.sum().T.N

    gcf = metrics.capacity_factor("gross", "project", "windfarm").values[0][0]
    ncf = metrics.capacity_factor("net", "project", "windfarm").values[0][0]

    results["opex_total"] = opex
    results["opex_levelized"] = opex_levelized
    results["vessel_annual"] = vessel_annual
    results["vessel_levelized"] = vessel_levelized
    results["quayside_annual"] = port_annual
    results["ctv_annual"] = ctv_annual
    results["rov_annual"] = rov_annual
    results["cab_annual"] = cab_annual
    results["ahv_annual"] = ahv_annual
    results["jackup_annual"] = jackup_annual
    results["tugboat_annual"] = tugboat_annual
    results["cab_mobilization"] = cab_mobilization
    results["ahv_mobilization"] = ahv_mobilization
    results["jackup_mobilization"] = jackup_mobilization
    results["activities_annual"] = materials_annual
    results["labor_annual"] = labor_annual
    results["activities_levelized"] = materials_levelized
    results["labor_levelized"] = labor_levelized
    results["energy"] = energy
    results["opex_energy"] = opex_energy
    results["time_availability"] = time_availability
    results["energy_availability"] = energy_availability
    results["planned_visits"] = n_maintenance
    results["unplanned_visits"] = n_repair
    results["time_to_repair_start"] = average_time_to_start
    results["gcf"] = gcf
    results["ncf"] = ncf

    return results