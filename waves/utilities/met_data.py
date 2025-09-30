"""Provides a series of functions for processing meteorological data."""

import warnings

import numpy as np
import pandas as pd
from scipy import stats


def fit_weibull_distribution(wind_speed_data, random_seed=1):
    """Fits a Weibull distribution to wind speed data and returns the shape parameter.

    This function fits a Weibull distribution to the provided wind speed data, assuming
    that wind speeds are non-negative by fixing the location parameter of the Weibull
    distribution to 0. It returns the shape parameter of the fitted Weibull distribution.
    The function also allows for reproducibility by setting a random seed for the fitting
    process. If the required wind speed data is not available or the data is invalid, an
    error message is printed.

    Parameters
    ----------
    wind_speed_data : np.ndarray
        The array of wind speed data to fit the Weibull distribution to.
    random_seed : int, optional
        A random seed for reproducibility. Defaults to 1. This seed initializes the random
        number generator used in the fitting procedure.

    Returns
    -------
    float | str
        The shape parameter of the fitted Weibull distribution if the wind speed data is
        valid.
        If the data is missing or invalid, it returns an error message indicating the issue.
    """
    # Set random seed for reproducibility, fit a Weibull distribution to the wind speed data
    np.random.seed(random_seed)

    shape, loc, scale = stats.weibull_min.fit(
        wind_speed_data, floc=0
    )  # fixing location to 0 (assuming windspeed is non-negative)

    # Return the fitted parameters
    return shape


def compute_shear(
    data: pd.DataFrame,
    ws_heights: dict[str, float],
    return_reference_values: bool = False,
) -> pd.Series | tuple[pd.Series, float, pd.Series]:
    """Computes the shear coefficient between wind speed measurements using the power law.

    The shear coefficient is calculated by evaluating the expression for an Ordinary Least
    Squares (OLS) regression coefficient. The power law is used to model the relationship
    between wind speed and sensor height. This function is based on the implementation
    provided by OpenOA (https://github.com/NREL/OpenOA)

    Parameters
    ----------
    data : pandas.DataFrame
        A DataFrame containing wind speed columns that correspond to the keys of `ws_heights`.
        Each column should represent wind speed measurements at a specific height.

    ws_heights : dict[str, float]
        A dictionary where the keys are the names of the wind speed columns in `data`, and the
        values are the respective sensor heights (in meters) for those columns.

    return_reference_values : bool, optional
        If True, the function will return a tuple containing the shear exponents, the reference
        height (m), and the reference wind speed (m/s). Defaults to False.

    Returns
    -------
    pd.Series | tuple[pd.Series, float, pd.Series]
        If `return_reference_values` is False, the function returns the shear coefficient
        (unitless) as a pandas Series.
        If `return_reference_values` is True, the function returns a tuple:
            - The shear coefficient (unitless) as a pandas Series,
            - The reference height (float) in meters,
            - The reference wind speed (pandas Series) in meters per second.
    """
    # Extract the wind speed columns from `data` and create "u" 2-D array; where element
    # [i,j] is the wind speed measurement at the ith timestep and jth sensor height

    u: np.ndarray = np.column_stack(
        tuple(data.loc[:, col].copy() if col is not None else col for col in ws_heights)
    )

    # create "z" 2_D array; columns are filled with the sensor height
    z: np.ndarray = np.repeat([[*ws_heights.values()]], len(data), axis=0)

    # take log of z & u
    with warnings.catch_warnings():  # suppress log division by zero warning.
        warnings.filterwarnings("ignore", r"divide by zero encountered in log")
        u = np.log(u)
        z = np.log(z)

    # correct -inf or NaN if any.
    nan_or_ninf = np.logical_or(np.isneginf(u), np.isnan(u))
    if np.any(nan_or_ninf):
        # replace -inf or NaN with zero or NaN in u and corresponding location in
        # z such that these
        # elements are excluded from the regression.
        u[nan_or_ninf] = 0
        z[nan_or_ninf] = np.nan

    # shift rows of z by the mean of z to simplify shear calculation
    z = z - (np.nanmean(z, axis=1))[:, None]

    # finally, replace NaN's in z by zero so those points are effectively excluded
    # from the regression
    z[np.isnan(z)] = 0

    # compute shear based on simple linear regression
    alpha = (z * u).sum(axis=1) / (z * z).sum(axis=1)

    if not return_reference_values:
        return alpha

    # compute reference height
    z_ref: float = np.exp(np.mean(np.log(np.array(list(ws_heights.values())))))

    # replace zeros in u (if any) with NaN
    u[u == 0] = np.nan

    # compute reference wind speed
    u_ref = np.exp(np.nanmean(u, axis=1))

    return alpha, z_ref, u_ref


def extrapolate_windspeed(
    v1: pd.Series | np.ndarray | float, z1: float, z2: float, shear: pd.Series | np.ndarray | float
) -> pd.Series | np.ndarray | float:
    """
    Extrapolates wind speed vertically using the Power Law. This function is based on the
    implementation provided by OpenOA (https://github.com/NREL/OpenOA).

    Parameters
    ----------
    v1 : :obj: `pandas.Series` | :obj:`np.ndarray` | :obj:`float`
        A pandas ``Series`` of the wind speed measurements at the reference height, or the name of
        the column in :py:attr:`data`.
    z1 : :obj:`float`
        Height of reference wind speed measurements; units in meters.
    z2 : :obj:`float`
        Target extrapolation height; units in meters.
    shear : :obj: `pandas.Series` | :obj:`np.ndarray` | `float`
        A pandas ``Series`` of the shear values, or a single shear value.

    Returns
    -------
    :obj: `pandas.Series` | :obj:`np.ndarray` | :obj:`float`
        Wind speed extrapolated to target height, :py:arg:`z2`.
    """
    return v1 * (z2 / z1) ** shear
