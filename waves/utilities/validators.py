"""Utility functions to validating model inputs for each of the models that can be called by
WAVES.
"""

from typing import Any


def check_dict_consistent(dict_values: dict, output_name: str) -> Any:
    """Check a dictionary to ensure all values are equal to each other (or None).

    Parameters
    ----------
    dict_values : dict
        Dictionary containing values to checks.
    output_name : str
        Name of the output (for error messages only).

    Returns
    -------
    Any
        If the dictionary values are consistent

    Raises
    ------
    ValueError
        Raised if none of the items in the dict contain a non-None value.

    ValueError
        Raised if there is more than one distinct non-None value in the dict.
    """
    values = set(dict_values.values()).difference({None})
    num_unique_values = len(values)
    if num_unique_values == 0:
        raise ValueError(f"No models produce an output for ``{output_name:s}``")
    elif num_unique_values > 1:
        raise ValueError(
            f"Conflicting value of ``{output_name:s}`` between models: {str(dict_values):s}"
        )
