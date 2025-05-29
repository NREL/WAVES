"""
Provides a series of validation functions for commonly used keyword arguments in the
:py:class:`Project` results methods.
"""

import inspect
from typing import Any
from functools import wraps
from collections.abc import Callable


def validate_frequency(frequency: str):
    """Checks that the :py:attr:`frequency` input is valid.

    Parameters
    ----------
    frequency : str
        Date frequency for the timeseries aggregation. Must be one of "project", "annual",
        or "month-year".

    Raises
    ------
    ValueError
        Raised if the input to :py:attr:`frequency` is not one of "project", "annual",
        or "month-year".
    """
    opts = ("project", "annual", "month-year")
    if frequency not in opts:
        raise ValueError(f"`frequency` must be one of {opts}. Provided: {frequency}.")


def validate_units(units: str):
    """Checks that the :py:attr:`units` input is valid.

    Parameters
    ----------
    units : str
        Power-basis for the energy calculation. Must be one of "gw", "mw", or "kw".

    Raises
    ------
    ValueError
        Raised if the input to :py:attr:`units` is not one of "gw", "mw", or "kw".
    """
    opts = ("gw", "mw", "kw")
    if units not in opts:
        raise ValueError(f"`units` must be one of {opts}. Provided: {units}.")


def validate_per_capacity(per_capacity: str | None):
    """Checks that the :py:attr:`per_capacity` input is valid.

    Parameters
    ----------
    per_capacity : str
        Power-basis for the energy calculation. Must be one of "gw", "mw", "kw", or None.

    Raises
    ------
    ValueError
        Raised if the input to :py:attr:`per_capacity` is not one of "gw", "mw", or "kw".
    """
    opts = ("gw", "mw", "kw", None)
    if per_capacity not in opts:
        raise ValueError(f"`per_capacity` must be one of {opts}. Provided: {per_capacity}.")


def validate_by(by: str):
    """Checks that the :py:attr:`by` input is valid.

    Parameters
    ----------
    by : str
        Wind farm aggregation basis. Must be one of "windfarm" or "turbine".

    Raises
    ------
    ValueError
        Raised if the input to :py:attr:`by` is not one of "windfarm" or "turbine".
    """
    opts = ("windfarm", "turbine")
    if by not in opts:
        raise ValueError(f"`by` must be one of {opts}. Provided: {by}.")


def validate_common_inputs(which: list[str] | None = None):
    """Validates the standard inputs to many of :py:class:`Project`'s results methods. This is a
    convenience wrapper to writing boilerplate checks for each aggregation method.

    Parameters
    ----------
    which : list[str], optional
        The names of the method arguments that should be validated. Can only be frequency, units, or
        by.
    """

    def decorator(func: Callable):
        """Gathes the arg indices from :py:attr:`data_cols` to be used in ``wrapper``."""
        argspec = inspect.getfullargspec(func)
        signature = inspect.signature(func)
        if which is None:
            raise ValueError("No arguments provided for validation")
        arg_ix_list = [argspec.args.index(name) for name in which]

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any):
            # Fetch the user inputs, convert them to the Series and None values, as appropriate, and
            # update the args and kwargs for the new values
            inputs = {}
            for ix, name in zip(arg_ix_list, which):
                try:
                    inputs[name] = args[ix]
                except IndexError:
                    # Check that the argument isn't in fact an optional keyword argument
                    # and get the default if being used.
                    try:
                        inputs[name] = kwargs[name]
                    except KeyError:
                        inputs[name] = signature.parameters[name].default

            if "frequency" in inputs:
                validate_frequency(inputs["frequency"])
            if "units" in inputs:
                validate_units(inputs["units"])
            if "per_capacity" in inputs:
                validate_per_capacity(inputs["per_capacity"])
            if "by" in inputs:
                validate_by(inputs["by"])

            return func(*args, **kwargs)

        return wrapper

    return decorator
