"""Provides the FLORIS-based methods for pre-processing data and post-processing results."""


from __future__ import annotations

import multiprocessing as mp

import numpy as np
import pandas as pd
from tqdm import tqdm
from floris.tools import FlorisInterface
from floris.tools.wind_rose import WindRose


# **************************************************************************************************
# Time series methods
# **************************************************************************************************


def run_chunked_time_series_floris(
    args: tuple,
) -> tuple[tuple[int, int], FlorisInterface, pd.DataFrame]:
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
    tuple[tuple[int, int], FlorisInterface, pd.DataFrame]
        The ``chunk_id``, a reinitialized ``fi`` using the appropriate wind parameters
        that can be used for further post-processing, and the resulting turbine powers.
    """
    fi: FlorisInterface = args[0]
    weather: pd.DataFrame = args[1]
    chunk_id: tuple[int, int] = args[2]
    reinit_kwargs: dict = args[3]
    run_kwargs: dict = args[4]

    reinit_kwargs["wind_directions"] = weather.wind_direction.values
    reinit_kwargs["wind_speeds"] = weather.windspeed.values
    fi.reinitialize(time_series=True, **reinit_kwargs)
    fi.calculate_wake(**run_kwargs)
    power_df = pd.DataFrame(fi.get_turbine_powers()[:, 0, :], index=weather.index)
    return chunk_id, fi, power_df


def run_parallel_time_series_floris(
    args_list: list[tuple[FlorisInterface, pd.DataFrame, tuple[int, int], dict, dict]],
    nodes: int = -1,
) -> tuple[dict[tuple[int, int], FlorisInterface], pd.DataFrame]:
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
        dataframe (without renamed columns).
    """
    nodes = int(mp.cpu_count() * 0.7) if nodes == -1 else nodes
    with mp.Pool(nodes) as pool:
        with tqdm(total=len(args_list), desc="Time series energy calculation") as pbar:
            df_list = []
            fi_dict = {}
            for chunk_id, fi, df in pool.imap_unordered(run_chunked_time_series_floris, args_list):
                df_list.append(df)
                fi_dict[chunk_id] = fi
                pbar.update()

    fi_dict = dict(sorted(fi_dict.items()))
    turbine_power_df = pd.concat(df_list).sort_index()
    return fi_dict, turbine_power_df


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
    wd, ws = weather_df.loc[weather_df.index.month == month, ["wd", "ws"]].values.T
    wind_rose = WindRose()
    _ = wind_rose.make_wind_rose_from_user_data(wd, ws)
    return (month, wind_rose)


def create_monthly_wind_rose(weather_df: pd.DataFrame) -> dict[int, WindRose]:
    """Create a dictionary of month and a long-term ``WindRose`` object based on all the
    wind condition data for that month.

    Parameters
    ----------
    weather_df : pd.DataFrame
        The weather profile used to create long-term, month-based ``WindRose`` objects
    month : int
        The month of the year to create a ``WindRose`` object.

    Returns
    -------
    dict[int, WindRose]
        A dictionary of the integer month and the long-term ``WindRose`` object associated
        with all the wind conditions during that month.
    """
    return dict(create_single_month_wind_rose(weather_df, month) for month in range(1, 13))


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
    project_df = project_wind_rose.df
    wr_combinations = list(map(tuple, project_df[["wd", "ws"]].values))
    for month, wind_rose in monthly_wind_rose.items():
        if wind_rose.df.shape == project_df.shape:
            continue

        # Find the missing combinations, add them to the wind rose DataFrame, and resort
        wr_df = wind_rose.df
        missing = set(wr_combinations).difference(list(map(tuple, wr_df[["wd", "ws"]].values)))
        missing_df = pd.DataFrame([], columns=wr_df.columns)
        missing_df[["wd", "ws"]] = list(missing)
        missing_df.freq_val = 0.0
        wr_df = pd.concat([wr_df, missing_df]).sort_values(["wd", "ws"])

        # Tidy up the WindRose object itself to ensure it can be used correctly
        # Note: taken from the WindRose.read_wind_rose_csv() method without renormalizing the
        # frequency data because we're only adding in missing values with 0 frequency
        wind_rose.df = wr_df

        # Call the resample function in order to set all the internal variables
        wind_rose.internal_resample_wind_speed(ws=wr_df.ws.unique())
        wind_rose.internal_resample_wind_direction(wd=wr_df.wd.unique())
        monthly_wind_rose[month] = wind_rose

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
