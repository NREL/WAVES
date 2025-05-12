"""Provides a consistent way to read and write YAML data, largely based on WOMBAT's
library module.

All library data should adhere to the following directory structure within a library path:

::

  ├── project
    ├── config     <- Project-level configuration files
    ├── port       <- Port configuration files
    ├── plant      <- Wind farm layout files
  ├── cables       <- Export and Array cable configuration files
  ├── substations  <- Substation configuration files
  ├── turbines     <- Turbine configuration and power curve files
  ├── vessels      <- Land-based and offshore servicing equipment configuration files
  ├── weather      <- Weather profiles
  ├── results      <- The analysis log files and any saved output data
"""

from __future__ import annotations

import re
from typing import Any
from pathlib import Path

import yaml


# YAML SafeLoader that is able to read scientific notation and Python Tuples
class CustomSafeLoader(yaml.SafeLoader):
    """Customized ``yaml.SafeLoader`` that adds custom constructors for consistent
    data loading in safe mode.
    """

    def construct_python_tuple(self, node):
        """Loads a YAML object to a Pytho Tuple.s."""
        return tuple(self.construct_sequence(node))


custom_loader = yaml.SafeLoader
custom_loader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(
        """^(?:
     [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
    |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
    |\\.[0-9_]+(?:[eE][-+][0-9]+)?
    |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
    |[-+]?\\.(?:inf|Inf|INF)
    |\\.(?:nan|NaN|NAN))$""",
        re.X,
    ),
    list("-+0123456789."),
)
custom_loader.add_constructor(
    "tag:yaml.org,2002:python/tuple", CustomSafeLoader.construct_python_tuple
)


def resolve_path(value: str | Path) -> Path:
    """Converts a user-input string to a ``Path`` object and resolves it.

    Parameters
    ----------
    value : str | Path
        A string or Path to a configuration library.

    Raises
    ------
    TypeError: Raised if the input to :py:attr:`value` is not either a ``str`` or ``pathlib.Path``.

    Returns
    -------
    Path
        The resolved Path object version of the input library path.
    """
    if isinstance(value, str):
        value = Path(value)
    if isinstance(value, Path):
        return value.resolve()

    raise TypeError(f"The input path: {value}, must be of type `str` or `pathlib.Path`.")


def load_yaml(path: str | Path, fname: str | Path) -> Any:
    """Loads and returns the contents of the YAML file.

    Parameters
    ----------
    path : str | Path
        Path to the file to be loaded.
    fname : str | Path
        Name of the file (ending in .yaml) to be loaded.

    Returns
    -------
    Any
        Whatever content is in the YAML file.
    """
    path = Path(path).resolve()

    with (path / fname).open() as f:
        return yaml.load(f, Loader=custom_loader)


def write_yaml(path: str | Path, fname: str | Path, data: dict):
    """Writes a yaml file based on a dictionary input.

    Parameters
    ----------
    path : str | Path
        Path to the write the new file
    fname: str | Path
        Name of the file (ending in .yaml) to be written.
    data: dict
        Dictionary of data to become a .yaml file.

    Returns
    -------
    None
    """
    path = Path(path).resolve()

    with (path / fname).open() as f:
        yaml.dump(data, f, sort_keys=False, default_flow_style=False)
