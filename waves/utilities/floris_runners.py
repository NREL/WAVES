"""Provides the FLORIS-based methods for pre-processing data and post-processing results."""


from __future__ import annotations

import numpy as np
import pandas as pd
from floris import WindRose, TimeSeries


def create_single_month_wind_rose(weather_df: pd.DataFrame, month: int) -> tuple[int, WindRose]:
    """Creates the FLORIS ``WindRose`` object for a given :py:attr:`month` based on the
    :py:attr:`weather_df`'s ``DatetimeIndex``.

    Parameters
    ----------
    weather_df : pd.DataFrame
        The weather profile used to create long-term, month-based ``WindRose`` objects
    month : int
        The month of the year to create a ``WindRose`` object.

    Returns
    -------
    WindRose
        Month-specific ``WindRose`` object.
    """
    wd, ws, ti = weather_df.loc[weather_df.index.month == month].values.T
    wind_rose = TimeSeries(
        wind_directions=wd, wind_speeds=ws, turbulence_intensities=ti
    ).to_WindRose()
    return wind_rose


def create_monthly_wind_rose(weather_df: pd.DataFrame) -> dict[int, WindRose]:
    """Create a dictionary of month and a long-term ``WindRose`` object based on all the
    wind condition data for that month.

    Parameters
    ----------
    weather_df : pd.DataFrame
        The weather profile used to create long-term, month-based ``WindRose`` objects

    Returns
    -------
    dict[int, WindRose]
        A dictionary of the integer month and the long-term ``WindRose`` object associated
        with all the wind conditions during that month.
    """
    monthly_wr = {
        month: create_single_month_wind_rose(weather_df=weather_df, month=month)
        for month in range(1, 13)
    }
    return monthly_wr


def check_monthly_wind_rose(
    project_wind_rose: WindRose, monthly_wind_rose: dict[int, WindRose]
) -> dict[int, WindRose]:
    """Checks the monthly wind rose parameterizations to ensure the DataFrames are the
    correct shape, so that when the frequency column is extracted, the compared data
    is the same.

    Parameters
    ----------
    project_wind_rose : WindRose
        The ``WindRose`` created using the long term reanalysis weather profile.
    monthly_wind_rose : dict[int, WindRose]
        A dictionary of the month as an ``int`` and ``WindRose`` created from the long
        term project reanalysis weather profile that was filtered on weather data for
        the focal month.

    Returns
    -------
    dict[int, WindRose]
        The :py:attr:`monthly_wind_rose` but with an missing wind conditions added into
        the ``WindRose`` with 0 frequency.
    """
    project_df = pd.DataFrame(
        project_wind_rose.freq_table,
        columns=project_wind_rose.wind_speeds,
        index=project_wind_rose.wind_directions,
    )
    project_shape = project_df.shape
    project_wd = project_wind_rose.wind_directions
    project_ws = project_wind_rose.wind_speeds
    for month, wind_rose in monthly_wind_rose.items():
        if wind_rose.freq_table.shape == project_shape:
            continue

        wr_df = pd.DataFrame(
            wind_rose.freq_table, columns=wind_rose.wind_speeds, index=wind_rose.wind_directions
        )
        ti_df = pd.DataFrame(
            wind_rose.ti_table, columns=wind_rose.wind_speeds, index=wind_rose.wind_directions
        )

        # Find the missing combinations, add them to the wind rose DataFrame, and resort
        missing_wd = list(set(project_wd).difference(wind_rose.wind_directions))
        missing_ws = list(set(project_ws).difference(wind_rose.wind_speeds))
        wr_df.loc[missing_wd] = 0
        wr_df.loc[:, missing_ws] = 0
        wr_df = wr_df.sort_index().T.sort_index().T
        if wr_df.shape != project_shape:
            raise ValueError("The monthly wind rose was unable to be resampled.")

        ti_df.loc[missing_wd] = 0
        ti_df.loc[:, missing_ws] = 0
        ti_df = ti_df.sort_index().T.sort_index().T
        if ti_df.shape != project_shape:
            raise ValueError("The monthly wind rose was unable to be resampled.")

        # Recreate the WindRose object with the missing wd/ws values added back in
        monthly_wind_rose[month] = WindRose(
            wind_directions=wr_df.index.values,
            wind_speeds=wr_df.columns.values,
            freq_table=wr_df.values,
            ti_table=ti_df.values,
        )

    return monthly_wind_rose


def calculate_monthly_wind_rose_results(
    turbine_power: np.ndarray,
    wind_rose_monthly: dict[int, WindRose],
) -> pd.DataFrame:
    """Calculate the turbine AEP contribution for each month of a year, in MWh.

    Parameters
    ----------
    turbine_power : np.ndarray
        The array of turbine powers, with shape (num wd x num ws x num turbines), calculated from
        the possible wind conditions at the site given the turbine layout.
    wind_rose_monthly : dict[int, WindRose]
        The dictionary of integer months (i.e., 1 for January) and array of frequences, with
        ``WindRose`` objects created by the long term wind conditions filtered on the month.

    Returns
    -------
    pd.DataFrame, pd.DataFrame
        A DataFrame of each month's contribution to the AEP for each turbine in the wind farm, with
        shape (12 x num turbines).
    """
    month_day_map = pd.DataFrame(
        [
            [1, 31],
            [2, 28],
            [3, 31],
            [4, 30],
            [5, 31],
            [6, 30],
            [7, 31],
            [8, 31],
            [9, 30],
            [10, 31],
            [11, 30],
            [12, 31],
        ],
        columns=["month", "n_days"],
    ).set_index("month")

    # Calculate the monthly turbine energy and sum over the turbines to get the farm energy
    turbine_energy = pd.DataFrame.from_dict(
        {
            m: np.nansum(mwr[:, :, np.newaxis] * turbine_power, axis=(0, 1))
            for m, mwr in wind_rose_monthly.items()
        },
        orient="index",
    ).sort_index()
    turbine_energy *= month_day_map.n_days.values.reshape(12, 1) * 24 / 1e6
    return turbine_energy
