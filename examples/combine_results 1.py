from pathlib import Path

import numpy as np
import pandas as pd


column_order = [
    "layout_1_in_situ_base",
    "layout_1_in_situ_no_major",
    "layout_1_in_situ_increase_failure",
    "layout_1_in_situ_increase_maintenance",
    "layout_1_in_situ_increase_wave",
    "layout_1_tow_base",
    "layout_1_tow_increase_failure",
    "layout_1_tow_increase_maintenance",
    "layout_1_tow_increase_wave",
    "layout_2_tow_base",
    "layout_2_in_situ_base",
    "layout_1_in_situ_base_24hr",
    "layout_1_in_situ_increase_failure_24hr",
    "layout_1_tow_base_24hr",
]


def read_summary_csv(fn):
    df = pd.read_csv(fn, skiprows=[1, 2]).rename(columns={"Unnamed: 0": "metric"})
    mean = df.iloc[:, [0, 1]].set_index("metric")
    std = df.iloc[:, [0, 2]].set_index("metric")
    std.columns = [el.replace(".1", "") for el in std.columns]
    return mean, std


def read_vessel_csv(fn):
    name = fn.name.replace("_vessel_summary.csv", "")
    df = (
        pd.read_csv(fn)
        .rename(columns={"Unnamed: 0": "metric"})
        .set_index(["Metric", "Vessel"])
    )
    return df.drop(columns="std").rename(columns={"mean": name}), df.drop(
        columns="mean"
    ).rename(columns={"std": name})


def read_activities_csv(fn):
    name = fn.name.replace("_activities.csv", "")
    df = (
        pd.read_csv(fn, skiprows=[2])
        .rename(columns={"Unnamed: 0": "part_name", "Unnamed: 1": "reason"})
        .set_index(["part_name", "reason"])
    )
    ix_drop = ~df.index.get_level_values("part_name").isna()
    count_mean = df.loc[ix_drop, ["count"]].rename(columns={"count": name})
    count_std = df.loc[ix_drop, ["count.1"]].rename(columns={"count.1": name})
    sum_mean = df.loc[ix_drop, ["sum"]].rename(columns={"sum": name})
    sum_std = df.loc[ix_drop, ["sum.1"]].rename(columns={"sum.1": name})
    return count_mean, count_std, sum_mean, sum_std


def main():
    # Load in the data
    summary_list = [
        el
        for el in Path(".").resolve().iterdir()
        if el.name.startswith("layout")
        and el.name.endswith("summary.csv")
        and "vessel" not in el.name
    ]
    vessel_list = [
        el
        for el in Path(".").resolve().iterdir()
        if el.name.startswith("layout")
        and el.name.endswith("summary.csv")
        and "vessel" in el.name
    ]
    activities_list = [
        el
        for el in Path(".").resolve().iterdir()
        if el.name.startswith("layout") and el.name.endswith("activities.csv")
    ]

    # Split between mean and standard deviation data

    # Overall results
    summary_data = [read_summary_csv(el) for el in summary_list]

    summary_means = pd.concat(
        [el[0] for el in summary_data], axis=1, join="outer"
    ).fillna(0.0)
    summary_stds = pd.concat(
        [el[1] for el in summary_data], axis=1, join="outer"
    ).fillna(0.0)

    # Vessels
    vessel_data = [read_vessel_csv(el) for el in vessel_list]

    vessel_means = (
        pd.concat([el[0] for el in vessel_data], axis=1, join="outer").fillna(0.0)
        / 20.0
    )
    vessel_stds = (
        pd.concat([el[1] for el in vessel_data], axis=1, join="outer").fillna(0.0)
        / 20.0
    )

    vessel_means = vessel_means.sort_index().reset_index(drop=False)
    vessel_means.Metric = pd.Categorical(
        vessel_means["Metric"],
        ordered=True,
        categories=[
            "Weather Delay",
            "Mobilization",
            "Vessel Usage",
            "Vessel Inactive",
            "Vessel Cost",
        ],
    )
    vessel_means.Vessel = pd.Categorical(
        vessel_means["Vessel"],
        ordered=True,
        categories=[
            "Crew Transfer Vessel 1",
            "Crew Transfer Vessel 2",
            "Crew Transfer Vessel 3",
            "ROV Support Vessel",
            "Cable Laying Vessel",
            "Anchor Handling Vessel",
            "Heavy Lift Vessel",
            "Tugboat 1",
        ],
    )
    vessel_means = vessel_means.sort_values(["Metric", "Vessel"]).set_index(
        ["Metric", "Vessel"]
    )

    vessel_stds = vessel_stds.sort_index().reset_index(drop=False)
    vessel_stds.Metric = pd.Categorical(
        vessel_stds["Metric"],
        ordered=True,
        categories=[
            "Weather Delay",
            "Mobilization",
            "Vessel Usage",
            "Vessel Inactive",
            "Vessel Cost",
        ],
    )
    vessel_stds.Vessel = pd.Categorical(
        vessel_stds["Vessel"],
        ordered=True,
        categories=[
            "Crew Transfer Vessel 1",
            "Crew Transfer Vessel 2",
            "Crew Transfer Vessel 3",
            "ROV Support Vessel",
            "Cable Laying Vessel",
            "Anchor Handling Vessel",
            "Heavy Lift Vessel",
            "Tugboat 1",
        ],
    )
    vessel_stds = vessel_stds.sort_values(["Metric", "Vessel"]).set_index(
        ["Metric", "Vessel"]
    )

    # Activities summary
    activities_data = [read_activities_csv(el) for el in activities_list]

    count_means = pd.concat(
        [el[0] for el in activities_data], axis=1, join="outer"
    ).fillna(0.0)[column_order]
    count_stds = pd.concat(
        [el[1] for el in activities_data], axis=1, join="outer"
    ).fillna(0.0)[column_order]
    sum_means = pd.concat(
        [el[2] for el in activities_data], axis=1, join="outer"
    ).fillna(0.0)[column_order]
    sum_stds = pd.concat(
        [el[3] for el in activities_data], axis=1, join="outer"
    ).fillna(0.0)[column_order]

    # Save the data
    summary_means[column_order].to_csv("results_summary_means.csv")
    summary_stds[column_order].to_csv("results_summary_std.csv")

    vessel_means[column_order].to_csv("vessel_summary_means.csv")
    vessel_stds[column_order].to_csv("vessel_summary_std.csv")

    count_means.to_csv("activities_count_mean.csv")
    count_stds.to_csv("activities_count_std.csv")
    sum_means.to_csv("activities_sum_mean.csv")
    sum_stds.to_csv("activities_sum_std.csv")


if __name__ == "__main__":
    main()
