"""Provides the FLORIS-based methods for pre-processing data and post-processing results."""


from __future__ import annotations

import multiprocessing as mp

import numpy as np
import pandas as pd
from tqdm import tqdm
from floris import WindRose, TimeSeries


# **************************************************************************************************
# Time series methods
# **************************************************************************************************


def run_chunked_time_series_floris(
    args: tuple,
) -> tuple[tuple[int, int], FlorisInterface, pd.DataFrame, pd.DataFrame]:
    """Runs ``fi.calculate_wake()`` over a chunk of a larger time series analysis and
    returns the individual turbine powers for each corresponding time.

    Parameters
    ----------
    fi : FlorisInterface
        A copy of the base ``FlorisInterface`` object.
    weather : pd.DataFrame
        A subset of the full weather profile, with only the datetime index and columns:
        "windspeed" and "wind_direction".
    chunk_id : tuple[int, int]
        A tuple of the year and month for the data being processed.
    reinit_kwargs : dict, optional
        Any additional reinitialization keyword arguments. Defaults to {}.
    run_kwargs : dict, optional
        Any additional calculate_wake keyword arguments. Defaults to {}.

    Returns
    -------
    tuple[tuple[int, int], FlorisInterface, pd.DataFrame, pd.DataFrame]
        The ``chunk_id``, a reinitialized ``fi`` using the appropriate wind parameters
        that can be used for further post-processing, and the resulting turbine potential and
        production powers.
    """
    fi: FlorisInterface = args[0]
    weather: pd.DataFrame = args[1]
    chunk_id: tuple[int, int] = args[2]
    reinit_kwargs: dict = args[3]
    run_kwargs: dict = args[4]

    # Reinitialize the FLORIS interface
    reinit_kwargs["wind_directions"] = weather.wind_direction.values
    reinit_kwargs["wind_speeds"] = weather.windspeed.values
    fi.reinitialize(time_series=True, **reinit_kwargs)

    # Calculate energy potential
    fi.calculate_no_wake(**run_kwargs)
    potential_df = pd.DataFrame(fi.get_turbine_powers()[:, 0, :], index=weather.index)

    # Calculate waked energy production
    fi.calculate_wake(**run_kwargs)
    production_df = pd.DataFrame(fi.get_turbine_powers()[:, 0, :], index=weather.index)
    return chunk_id, fi, potential_df, production_df


def run_parallel_time_series_floris(
    args_list: list[tuple[FlorisInterface, pd.DataFrame, tuple[int, int], dict, dict]],
    nodes: int = -1,
) -> tuple[dict[tuple[int, int], FlorisInterface], pd.DataFrame, pd.DataFrame]:
    """Runs the time series floris calculations in parallel.

    Parameters
    ----------
    args_list : list[tuple[FlorisInterface, pd.DataFrame, tuple[int, int], dict, dict]])
        A list of the chunked by month arguments that get passed to
        ``run_chunked_time_series_floris``.
    nodes : int, optional
        The number of nodes to parallelize over. If -1, then it will use the floor of 80% of the
        available CPUs on the computer. Defaults to -1.

    Returns
    -------
    tuple[dict[tuple[int, int], FlorisInterface], pd.DataFrame]
        A dictionary of the ``chunk_id`` and ``FlorisInterface`` object, and the full turbine power
        potential and production dataframes (without renamed columns).
    """
    nodes = int(mp.cpu_count() * 0.7) if nodes == -1 else nodes
    with mp.Pool(nodes) as pool:
        with tqdm(total=len(args_list), desc="Time series energy calculation") as pbar:
            df_potential_list = []
            df_production_list = []
            fi_dict = {}
            for i, fi, df1, df2 in pool.imap_unordered(run_chunked_time_series_floris, args_list):
                df_potential_list.append(df1)
                df_production_list.append(df2)
                fi_dict[i] = fi
                pbar.update()

    fi_dict = dict(sorted(fi_dict.items()))
    turbine_potential_df = pd.concat(df_potential_list).sort_index()
    turbine_production_df = pd.concat(df_production_list).sort_index()
    return fi_dict, turbine_potential_df, turbine_production_df


# **************************************************************************************************
# Wind rose methods
# **************************************************************************************************


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
    tuple[int, WindRose]
        A tuple of the :py:attr:`month` passed and the final ``WindRose`` object.
    """
    wd, ws, ti = weather_df.loc[weather_df.index.month == month].values.T
    wind_rose = TimeSeries(
        wind_directions=wd, wind_speeds=ws, turbulence_intensities=ti
    ).to_WindRose()
    return (month, wind_rose)


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

        # Find the missing combinations, add them to the wind rose DataFrame, and resort
        wr_df = pd.DataFrame(
            wind_rose.freq_table, columns=wind_rose.wind_speeds, index=wind_rose.wind_directions
        )
        missing_wd = list(set(project_wd).difference(wind_rose.wind_directions))
        missing_ws = list(set(project_ws).difference(wind_rose.wind_speeds))
        wr_df.loc[missing_wd] = 0
        wr_df.loc[:, missing_ws] = 0
        wr_df = wr_df.sort_index().T.sort_index().T
        if wr_df.shape != project_shape:
            raise ValueError("The monthly wind rose was unable to be resampled.")

        # Recreate the WindRose object with the missing wd/ws values added back in
        monthly_wind_rose[month] = WindRose(
            wind_directions=wr_df.index.values,
            wind_speeds=wr_df.columns.values,
            freq_table=wr_df.values,
            ti_table=wind_rose.ti_table,
        )

    return monthly_wind_rose


def calculate_monthly_wind_rose_results(
    turbine_power: np.ndarray,
    freq_monthly: dict[int, np.ndarray],
) -> pd.DataFrame:
    """Calculate the turbine AEP contribution for each month of a year, in MWh.

    Parameters
    ----------
    turbine_power : np.ndarray
        The array of turbine powers, with shape (num wd x num ws x num turbines), calculated from
        the possible wind conditions at the site given the turbine layout.
    freq_monthly : dict[int, np.ndarray]
        The dictionary of integer months (i.e., 1 for January) and array of frequences, with shape
        (num wd x num ws), created by the long term wind conditions filtered on the month.

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
            m: np.sum(
                freq_monthly[m].reshape((*freq_monthly[m].shape, 1)) * turbine_power, axis=(0, 1)
            )
            for m in freq_monthly
        },
        orient="index",
    ).sort_index()
    turbine_energy *= month_day_map.n_days.values.reshape(12, 1) * 24 / 1e6
    return turbine_energy
