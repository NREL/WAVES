"""Provides the ``Project`` class that ties to together ORBIT (CapEx), WOMBAT (OpEx), and
FLORIS (AEP) simulation libraries for a simplified modeling workflow.
"""

from __future__ import annotations

import json
from copy import deepcopy
from typing import TYPE_CHECKING
from pathlib import Path
from functools import reduce, partial
from itertools import product

import yaml
import attrs
import numpy as np
import pandas as pd
import pyarrow as pa
import networkx as nx
import pyarrow.csv  # pylint: disable=W0611
import numpy_financial as npf
import matplotlib.pyplot as plt
from attrs import field, define
from ORBIT import ProjectManager, load_config
from wombat.core import Simulation
from floris.tools import FlorisInterface
from floris.tools.wind_rose import WindRose
from wombat.core.data_classes import FromDictMixin

from waves.utilities import (
    load_yaml,
    resolve_path,
    check_monthly_wind_rose,
    create_monthly_wind_rose,
    run_parallel_time_series_floris,
    calculate_monthly_wind_rose_results,
)


def convert_to_multi_index(
    date_tuples: tuple[int, int] | list[tuple[int, int]] | pd.MultiIndex | None, name: str
) -> pd.MultiIndex:
    """Converts year and month tuple(s) into a pandas MultiIndex.

    Parameters
    ----------
    date_tuples : tuple[int, int] | list[tuple[int, int]] | pd.MultiIndex | None
        A single (``tuple``), or many combinations (``list`` of ``tuple``s) of ``int`` year and
        ``int`` month. If a ``MultiIndex`` or ``None`` is passed, it will be returned as-is.
    name : str
        The name of the variable to ensure that a helpful error is raised in case of invalid inputs.

    Returns
    -------
    pd.MultiIndex
        A pandas MultIndex with index columns: "year" and "month", or None, if None is passed.

    Raises
    ------
    ValueError
        Raised if the year, month combinations are not length 2 and are not tuples
    """
    if date_tuples is None or isinstance(date_tuples, pd.MultiIndex):
        return date_tuples
    if isinstance(date_tuples, tuple):
        date_tuples = [date_tuples]

    for date_tuple in date_tuples:
        if not (isinstance(date_tuple, tuple) and len(date_tuple) == 2):
            raise ValueError(
                f"The input to `{name}` must contain tuple(s) of length 2 for"
                " (year, month) combination(s)."
            )

    return pd.MultiIndex.from_tuples(date_tuples, names=["year", "month"])


def load_weather(value: str | Path | pd.DataFrame) -> pd.DataFrame:
    """Loads in the weather file using PyArrow, but returing a ``pandas.DataFrame``
    object. Must have the column "datetime", which can be converted to a
    ``pandas.DatetimeIndex``.

    Args:
        value : str | Path | pd.DataFrame
            The input file name and path, or a ``pandas.DataFrame`` (gets passed back
            without modification).

    Returns
    -------
        pd.DataFrame
            The full weather profile with the column "datetime" as a ``pandas.DatetimeIndex``.
    """
    if isinstance(value, pd.DataFrame):
        return value

    value = resolve_path(value)
    convert_options = pa.csv.ConvertOptions(
        timestamp_parsers=[
            "%m/%d/%y %H:%M",
            "%m/%d/%y %I:%M",
            "%m/%d/%Y %H:%M",
            "%m/%d/%Y %I:%M",
            "%m-%d-%y %H:%M",
            "%m-%d-%y %I:%M",
            "%m-%d-%Y %H:%M",
            "%m-%d-%Y %I:%M",
        ]
    )
    weather = (
        pa.csv.read_csv(value, convert_options=convert_options)
        .to_pandas()
        .set_index("datetime")
        .fillna(0.0)
        .resample("H")
        .interpolate(limit_direction="both", limit=5)
    )
    return weather


@define(auto_attribs=True)
class Project(FromDictMixin):
    """The unified interface for creating, running, and assessing analyses that combine
    ORBIT, WOMBAT, and FLORIS.

    Parameters
    ----------
    library_path : str | pathlib.Path
        The file path where the configuration data for ORBIT, WOMBAT, and FLORIS can be found.
    weather_profile : str | pathlib.Path
        The file path where the weather profile data is located, with the following column
        requirements:

            - datetime: The timestamp column
            - orbit_weather_cols: see ``orbit_weather_cols``
            - floris_windspeed: see ``floris_windspeed``
            - floris_wind_direction: see ``floris_wind_direction``
    orbit_weather_cols : list[str]
        The windspeed and wave height column names in ``weather`` to use for
        running ORBIT. Defaults to ``["windspeed", "wave_height"]``.
    floris_windspeed : str
        The windspeed column in ``weather`` that will be used for the FLORIS
        wake analysis. Defaults to "windspeed_100m".
    floris_wind_direction : str
        The wind direction column in ``weather`` that will be used for the FLORIS
        wake analysis. Defaults to "wind_direction_100m".
    floris_x_col : str
        The column of x-coordinates in the WOMBAT layout file that corresponds to
        the ``Floris.farm.layout_x`` Defaults to "floris_x".
    floris_y_col : str
        The column of x-coordinates in the WOMBAT layout file that corresponds to
        the ``Floris.farm.layout_y`` Defaults to "floris_y".
    orbit_config : str | pathlib.Path | dict | None
        The ORBIT configuration file name or dictionary. If None, will not set up
        the ORBIT simulation component.
    wombat_config : str | pathlib.Path | dict | None
        The WOMBAT configuration file name or dictionary. If None, will not set up
        the WOMBAT simulation component.
    floris_config : str | pathlib.Path | dict | None
        The FLORIS configuration file name or dictionary. If None, will not set up
        the FLORIS simulation component.
    connect_floris_to_layout : bool, optional
        If True, automatically connect the FLORIS and WOMBAT layout files, so that
        the simulation results can be linked. If False, don't connec the two models.
        Defaults to True.

        .. note:: This should only be set to False if the FLORIS and WOMBAT layouts
            need to be connected in an additional step

    connect_orbit_array_design : bool, optional
        If True, the ORBIT array cable lengths will be calculated on initialization
        and added into the primary layout file.
    offtake_price : float, optional
        The price paid per MWh of energy produced. Defaults to None.
    fixed_charge_rate : float, optional
        Revenue per amount of investment required to cover the investment cost, with the default
        provided through the NREL 2021 Cost of Energy report [1]_. Defaults to 0.0582.
    discount_rate : float, optional
        The minimum acceptable rate of return, or the assumed return on an alternative
        investment of comparable risk. Defaults to None.
    finance_rate : float, optional
        Interest rate paid on the cash flows. Defaults to None.
    reinvestment_rate : float, optional
        Interest rate paid on the cash flows upon reinvestment. Defaults to None.
    loss_ratio : float, optional
        Additional non-wake losses to deduct from the total energy production. Should be
        represented as a decimal in the range of [0, 1]. Defaults to None.
    orbit_start_date : str | None
        The date to use for installation phase start timings that are set to "0" in the
        ``install_phases`` configuration. If None the raw configuration data will be used.
        Defaults to None.
    soft_capex_date : tuple[int, int] | list[tuple[int, int]] | None, optional
        The date(s) where the ORBIT soft CapEx costs should be applied as a tuple of year and
        month, for instance: ``(2020, 1)`` for January 2020. Alternatively multiple dates can be
        set, which evenly divides the cost over all the dates, by providing a list of year and
        month combinations, for instance, a semi-annual 2 year cost starting in 2020 would look
        like: ``[(2020, 1), (2020, 7), (2021, 1), (2021, 7)]``. If None is provided, then the
        CapEx date will be the same as the start of the installation. Defaults to None
    project_capex_date : tuple[int, int] | list[tuple[int, int]] | None, optional
        The date(s) where the ORBIT project CapEx costs should be applied as a tuple of year and
        month, for instance: ``(2020, 1)`` for January 2020. Alternatively multiple dates can be
        set, which evenly divides the cost over all the dates, by providing a list of year and
        month combinations, for instance, a semi-annual 2 year cost starting in 2020 would look
        like: ``[(2020, 1), (2020, 7), (2021, 1), (2021, 7)]``. If None is provided, then the
        CapEx date will be the same as the start of the installation. Defaults to None
    system_capex_date : tuple[int, int] | list[tuple[int, int]] | None, optional
        The date(s) where the ORBIT system CapEx costs should be applied as a tuple of year and
        month, for instance: ``(2020, 1)`` for January 2020. Alternatively multiple dates can be
        set, which evenly divides the cost over all the dates, by providing a list of year and
        month combinations, for instance, a semi-annual 2 year cost starting in 2020 would look
        like: ``[(2020, 1), (2020, 7), (2021, 1), (2021, 7)]``. If None is provided, then the
        CapEx date will be the same as the start of the installation. Defaults to None.
    turbine_capex_date : tuple[int, int] | list[tuple[int, int]] | None, optional
        The date(s) where the ORBIT turbine CapEx costs should be applied as a tuple of year and
        month, for instance: ``(2020, 1)`` for January 2020. Alternatively multiple dates can be
        set, which evenly divides the cost over all the dates, by providing a list of year and
        month combinations, for instance, a semi-annual 2 year cost starting in 2020 would look
        like: ``[(2020, 1), (2020, 7), (2021, 1), (2021, 7)]``. If None is provided, then the
        CapEx date will be the same as the start of the installation. Defaults to None.
    report_config : dict[str, dict], optional
        A dictionary that can be passed to :py:meth:`generate_report`, and be used as the
        ``metrics_configuration`` dictionary. An additional field of ``name`` is required
        as input, which will be passed to ``simulation_name``. Defaults to None.

    References
    ----------
    .. [1] Stehly, Tyler, and Duffy, Patrick. 2021 Cost of Wind Energy Review. United States: N. p.,
        2022. Web. doi:10.2172/1907623.
    """

    library_path: Path = field(converter=resolve_path)
    weather_profile: str = field(converter=str)
    orbit_config: str | Path | dict | None = field(
        default=None, validator=attrs.validators.instance_of((str, Path, dict, type(None)))
    )
    wombat_config: str | Path | dict | None = field(
        default=None, validator=attrs.validators.instance_of((str, Path, dict, type(None)))
    )
    floris_config: str | Path | dict | None = field(
        default=None, validator=attrs.validators.instance_of((str, Path, dict, type(None)))
    )
    orbit_start_date: str | None = field(default=None)
    orbit_weather_cols: list[str] = field(
        default=["windspeed", "wave_height"],
        validator=attrs.validators.deep_iterable(
            member_validator=attrs.validators.instance_of(str),
            iterable_validator=attrs.validators.instance_of(list),
        ),
    )
    floris_windspeed: str = field(default="windspeed", converter=str)
    floris_wind_direction: str = field(default="wind_direction", converter=str)
    floris_x_col: str = field(default="floris_x", converter=str)
    floris_y_col: str = field(default="floris_y", converter=str)
    connect_floris_to_layout: bool = field(
        default=True, validator=attrs.validators.instance_of(bool)
    )
    connect_orbit_array_design: bool = field(
        default=True, validator=attrs.validators.instance_of(bool)
    )
    offtake_price: float | int = field(
        default=None, validator=attrs.validators.instance_of((float, int, type(None)))
    )
    fixed_charge_rate: float = field(
        default=0.0582, validator=attrs.validators.instance_of((float, type(None)))
    )
    discount_rate: float = field(
        default=None, validator=attrs.validators.instance_of((float, type(None)))
    )
    finance_rate: float = field(
        default=None, validator=attrs.validators.instance_of((float, type(None)))
    )
    reinvestment_rate: float = field(
        default=None, validator=attrs.validators.instance_of((float, type(None)))
    )
    loss_ratio: float = field(
        default=None, validator=attrs.validators.instance_of((float, type(None)))
    )
    soft_capex_date: tuple[int, int] | list[tuple[int, int]] | None = field(
        default=None, converter=partial(convert_to_multi_index, name="soft_capex_date")
    )
    project_capex_date: tuple[int, int] | list[tuple[int, int]] | None = field(
        default=None, converter=partial(convert_to_multi_index, name="project_capex_date")
    )
    system_capex_date: tuple[int, int] | list[tuple[int, int]] | None = field(
        default=None, converter=partial(convert_to_multi_index, name="system_capex_date")
    )
    turbine_capex_date: tuple[int, int] | list[tuple[int, int]] | None = field(
        default=None, converter=partial(convert_to_multi_index, name="turbine_capex_date")
    )
    report_config: dict[str, dict] | None = field(
        default=None, validator=attrs.validators.instance_of((dict, type(None)))
    )

    # Internally created attributes, aka, no user inputs to these
    weather: pd.DataFrame = field(init=False)
    orbit_config_dict: dict = field(factory=dict, init=False)
    wombat_config_dict: dict = field(factory=dict, init=False)
    floris_config_dict: dict = field(factory=dict, init=False)
    wombat: Simulation = field(init=False)
    orbit: ProjectManager = field(init=False)
    floris: FlorisInterface = field(init=False)
    project_wind_rose: WindRose = field(init=False)
    monthly_wind_rose: WindRose = field(init=False)
    floris_turbine_order: list[str] = field(init=False, factory=list)
    turbine_potential_energy: pd.DataFrame = field(init=False)
    turbine_production_energy: pd.DataFrame = field(init=False)
    project_potential_energy: pd.DataFrame = field(init=False)
    project_production_energy: pd.DataFrame = field(init=False)
    _fi_dict: dict[tuple[int, int], FlorisInterface] = field(init=False, factory=dict)
    floris_results_type: str = field(init=False)
    operations_start: pd.Timestamp = field(init=False)
    operations_end: pd.Timestamp = field(init=False)
    operations_years: int = field(init=False)

    def __attrs_post_init__(self) -> None:
        """Post-initialization hook to complete the setup."""
        if isinstance(self.weather_profile, str | Path):
            weather_path = self.library_path / "weather" / self.weather_profile
            self.weather = load_weather(weather_path)
        self.setup_orbit()
        self.setup_wombat()
        self.setup_floris()
        if self.connect_floris_to_layout:
            self.connect_floris_to_turbines()
        if self.connect_orbit_array_design:
            self.connect_orbit_cable_lengths()

    # **********************************************************************************************
    # Input validation methods
    # **********************************************************************************************

    @library_path.validator  # type: ignore
    def library_exists(self, attribute: attrs.Attribute, value: Path) -> None:
        """Validates that the user input to :py:attr:`library_path` is a valid directory.

        Parameters
        ----------
        attribute : attrs.Attribute
            The attrs Attribute information/metadata/configuration.
        value : Path
            The user input.

        Raises
        ------
        FileNotFoundError
            Raised if :py:attr:`value` does not exist.
        ValueError
            Raised if the :py:attr:`value` exists, but is not a directory.
        """
        if not value.exists():
            raise FileNotFoundError(f"The input path to {attribute.name} cannot be found: {value}")
        if not value.is_dir():
            raise ValueError(f"The input path to {attribute.name}: {value} is not a directory.")

    @report_config.validator  # type: ignore
    def validate_report_config(self, attribute: attrs.Attribute, value: dict | None) -> None:
        """Validates the user input for :py:attr:`report_config`.

        Parameters
        ----------
        attribute : attrs.Attribute
            The attrs Attribute information/metadata/configuration.
        value : dict | None
            _description_

        Raises
        ------
        ValueError
            Raised if :py:attr:`report_config` is not a dictionary.
        KeyError
            Raised if :py:attr:`report_config` does not contain a key, value pair for "name".
        """
        if value is None:
            return

        if not isinstance(value, dict):
            raise ValueError("`report_config` must be a dictionary, if provided")

        if "name" not in value:
            raise KeyError("A key, value pair for `name` must be provided.")

    # **********************************************************************************************
    # Configuration methods
    # **********************************************************************************************

    @classmethod
    def from_file(cls, library_path: str | Path, config_file: str | Path) -> Project:
        """Creates a ``Project`` object from either a JSON or YAML file. See
        :py:class:`Project` for configuration requirements.

        Parameters
        ----------
        library_path : str | Path
            The library path to be used in the simulation.
        config_file : str | Path
            The configuration file to create a :py:class:`Project` object from, which should be
            located at: ``library_path`` / project / config / ``config_file``.

        Raises
        ------
        FileExistsError
            Raised if :py:attr:`library_path` is not a valid directory.
        ValueError
            Raised if :py:attr:`config_file` is not a JSON or YAML file.

        Returns
        -------
        Project
            An initialized Project object.
        """
        library_path = Path(library_path).resolve()
        if not library_path.is_dir():
            raise FileExistsError(f"{library_path} cannot be found.")
        config_file = Path(config_file)
        if config_file.suffix == ".json":
            with open(library_path / "project/config" / config_file) as f:
                config_dict = dict(json.load(f))
        if config_file.suffix in (".yml", ".yaml"):
            config_dict = load_yaml(library_path / "project/config", config_file)
        else:
            raise ValueError(
                "The configuration file must be a JSON (.json) or YAML (.yaml or .yml) file."
            )
        config_dict["library_path"] = library_path
        return Project.from_dict(config_dict)

    @property
    def config_dict(self) -> dict:
        """Generates a configuration dictionary that can be saved to a new file for later
        re/use.

        Returns
        -------
        dict
            YAML-safe dictionary of a Project-loadable configuration.
        """
        wombat_config_dict = deepcopy(self.wombat_config_dict)
        config_dict = {
            "library_path": str(self.library_path),
            "orbit_config": self.orbit_config_dict,
            "wombat_config": wombat_config_dict,
            "floris_config": self.floris_config_dict,
            "weather_profile": self.weather_profile,
            "orbit_weather_cols": self.orbit_weather_cols,
            "floris_windspeed": self.floris_windspeed,
            "floris_wind_direction": self.floris_wind_direction,
            "floris_x_col": self.floris_x_col,
            "floris_y_col": self.floris_y_col,
        }
        return config_dict

    def save_config(self, config_file: str | Path) -> None:
        """Saves a copy of the Project configuration settings to recreate the results of
        the current settings.

        Parameters
        ----------
        config_file : str | Path
            The name to use for saving to a YAML configuration file.
        """
        config_dict = self.config_dict
        with open(self.library_path / "project/config" / config_file, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False)

    # **********************************************************************************************
    # Setup and setup assisting methods
    # **********************************************************************************************

    def setup_orbit(self) -> None:
        """Creates the ORBIT Project Manager object and readies it for running an analysis."""
        if self.orbit_config is None:
            print("No ORBIT configuration provided, skipping model setup.")
            return

        if isinstance(self.orbit_config, str | Path):
            orbit_config = self.library_path / "project/config" / self.orbit_config
            self.orbit_config_dict = load_config(orbit_config)
        else:
            self.orbit_config_dict = self.orbit_config

        if self.orbit_start_date is not None:
            for phase, start in self.orbit_config_dict["install_phases"].items():
                if start == 0:
                    self.orbit_config_dict["install_phases"][phase] = self.orbit_start_date

        if TYPE_CHECKING:
            assert isinstance(self.weather, pd.DataFrame)  # mypy helper
        self.orbit = ProjectManager(
            self.orbit_config_dict,
            library_path=str(self.library_path),
            weather=self.weather.loc[:, self.orbit_weather_cols],
        )

    def setup_wombat(self) -> None:
        """Creates the WOMBAT Simulation object and readies it for running an analysis."""
        if self.wombat_config is None:
            print("No WOMBAT configuration provided, skipping model setup.")
            return

        if isinstance(self.wombat_config, str | Path):
            wombat_config = (
                self.library_path / "project/config" / self.wombat_config  # type: ignore
            )
        else:
            wombat_config = self.wombat_config  # type: ignore
        self.wombat = Simulation.from_config(self.library_path, wombat_config)
        self.wombat_config_dict = attrs.asdict(self.wombat.config)
        self.operations_start = self.wombat.env.start_datetime
        self.operations_end = self.wombat.env.end_datetime

        start = self.wombat.env.start_datetime
        end = self.wombat.env.end_datetime
        diff = end - start
        self.operations_years = round((diff.days + (diff.seconds / 60 / 60) / 24) / 365.25, 1)

    def setup_floris(self) -> None:
        """Creates the FLORIS FlorisInterface object and readies it for running an
        analysis.
        """
        if self.floris_config is None:
            print("No FLORIS configuration provided, skipping model setup.")
            return

        if isinstance(self.floris_config, str | Path):
            self.floris_config_dict = load_yaml(
                self.library_path / "project/config", self.floris_config
            )
        else:
            self.floris_config_dict = self.floris_config
        self.floris = FlorisInterface(configuration=self.floris_config_dict)

    def connect_floris_to_turbines(self, x_col: str = "floris_x", y_col: str = "floris_y"):
        """Generates ``floris_turbine_order`` from the WOMBAT ``Windfarm.layout_df``.

        Parameters
        ----------
        x_col : str, optional
            The column name in the layout corresponding to the FLORIS x poszitions, by default
            "floris_x".
        y_col : str, optional
            The column name in the layout corresponding to the FLORIS y positions, by default
            "floris_y".
        """
        layout = self.wombat.windfarm.layout_df
        self.floris_turbine_order = [
            layout.loc[(layout[x_col] == x) & (layout[y_col] == y), "id"].values[0]
            for x, y in zip(self.floris.layout_x, self.floris.layout_y)
        ]

    def connect_orbit_cable_lengths(self, save_results: bool = True) -> None:
        """Runs the ORBIT design phases, so that the array system has computed the necessary cable
        length and distance measures, then attaches the cable length calculations back to the
        layout file, saves the results to the layout files, and reloads both ORBIT and WOMBAT with
        this data.

        Parameters
        ----------
        save_results : bool, optional
            Save the resulting, updated layout table to both
            ``library_path``/project/plant/``wombat_config_dict["layout"]`` and
            ``library_path``/cables/``wombat_config_dict["layout"]`` for WOMBAT and ORBIT
            compatibility, respectively.
        """
        # Get the correct design phase
        design_phases = self.orbit_config_dict["design_phases"]
        if "ArraySystemDesign" in design_phases:
            name = "ArraySystemDesign"
        elif "CustomArraySystemDesign" in design_phases:
            name = "CustomArraySystemDesign"
        else:
            raise RuntimeError(
                "None of `ArraySystemDesign` or `CustomArraySystemDesign` were included in the"
                "ORBIT configuration"
            )

        # Run the design phases if not already
        if name not in self.orbit._phases:
            self.orbit.run_design_phase(name)

        array = self.orbit._phases[name]
        locations = array.location_data.copy()
        cable_lengths = array.sections_cable_lengths.copy()

        # Loop through the substations, then strings to combine the calculated cable lengths with
        # the appropriate turbines, according to the turbine order on each string
        i = 0
        for oss in locations.substation_id.unique():
            oss_ix = locations.substation_id == oss
            oss_layout = locations.loc[oss_ix]
            string_id = np.sort(oss_layout.string.unique())
            for string in string_id:
                string_ix = oss_ix & (locations.string == string)
                cable_order = locations.loc[string_ix, "order"].values
                locations.loc[string_ix, "cable_length"] = cable_lengths[string + i, cable_order]
            i = string + 1

        # Add the cable length values to the layout file
        id_ix = locations.id.values
        self.wombat.windfarm.layout_df.loc[
            self.wombat.windfarm.layout_df.id.isin(id_ix), "cable_length"
        ] = locations.cable_length

        # Save the updated data to the original layout locations
        if save_results:
            layout_file_name = self.wombat_config_dict["layout"]
            self.wombat.windfarm.layout_df.to_csv(
                self.library_path / "project/plant" / layout_file_name,
                index=False,
            )

    def generate_floris_positions_from_layout(
        self,
        x_col: str = "easting",
        y_col: str = "northing",
        update_config: bool = True,
        config_fname: str | None = None,
    ) -> None:
        """Updates the FLORIS layout_x and layout_y based on the relative coordinates
        from the WOMBAT layout file.

        Parameters
        ----------
        x_col : str, optional
            The relative, distance-based x-coordinate column name. Defaults to "easting".
        y_col : str, optional
            The relative, distance-based y-coordinate column name. Defaults to "northing".
        update_config : bool, optional
            Run ``FlorisInterface.reinitialize`` with the updated ``layout_x`` and ``layout_y``
            values. Defaults to True.
        config_fname : str | None, optional
            Provide a file name if ``update_config`` and this new configuration should be saved.
            Defaults to None.
        """
        layout = self.wombat.windfarm.layout_df
        x_min = layout[x_col].min()
        y_min = layout[y_col].min()
        layout.assign(floris_x=layout[x_col] - x_min, floris_y=layout[y_col] - y_min)
        layout = layout.loc[
            layout.id.isin(self.wombat.windfarm.turbine_id), ["floris_x", "floris_y"]
        ]
        x, y = layout.values.T
        self.floris.reinitialize(layout_x=x, layout_y=y)
        if update_config:
            if TYPE_CHECKING:
                assert isinstance(self.floris_config_dict, dict)  # mypy helper
            self.floris_config_dict["farm"]["layout_x"] = x.tolist()
            self.floris_config_dict["farm"]["layout_y"] = y.tolist()
            if config_fname is not None:
                full_path = self.library_path / "project/config" / config_fname
                with open(full_path, "w") as f:
                    yaml.dump(self.floris_config_dict, f, default_flow_style=False)
                    print(f"Updated FLORIS configuration saved to: {full_path}.")

    # **********************************************************************************************
    # Run methods
    # **********************************************************************************************

    def preprocess_monthly_floris(
        self,
        reinitialize_kwargs: dict | None = None,
        run_kwargs: dict | None = None,
        cut_in_wind_speed: float | None = None,
        cut_out_wind_speed: float | None = None,
    ) -> tuple[
        list[tuple[FlorisInterface, pd.DataFrame, tuple[int, int], dict, dict]],
        np.ndarray,
    ]:
        """Creates the monthly chunked inputs to run a parallelized FLORIS time series
        analysis.

        Parameters
        ----------
        reinitialize_kwargs : dict | None, optional
            Any keyword arguments to be assed to ``FlorisInterface.reinitialize()``. Defaults to
            None.
        run_kwargs : dict | None, optional
            Any keyword arguments to be assed to ``FlorisInterface.calculate_wake()``.
            Defaults to None.
        cut_in_wind_speed : float, optional
            The wind speed, in m/s, at which a turbine will start producing power.
        cut_out_wind_speed : float, optional
            The wind speed, in m/s, at which a turbine will stop producing power.

        Returns
        -------
        tuple[list[tuple[FlorisInterface, pd.DataFrame, tuple[int, int], dict, dict]], np.ndarray]
            A list of tuples of:
                - a copy of the ``FlorisInterface`` object
                - tuple of year and month
                - a copy of ``reinitialize_kwargs``
                - c copy of ``run_kwargs``
        """
        if reinitialize_kwargs is None:
            reinitialize_kwargs = {}
        if run_kwargs is None:
            run_kwargs = {}

        month_list = range(1, 13)
        year_list = range(self.operations_start.year, self.operations_end.year + 1)

        if TYPE_CHECKING:
            assert isinstance(self.weather, pd.DataFrame)  # mypy helper
        weather = self.weather.loc[
            self.operations_start : self.operations_end,
            [self.floris_windspeed, self.floris_wind_direction],
        ].rename(
            columns={
                self.floris_windspeed: "windspeed",
                self.floris_wind_direction: "wind_direction",
            }
        )
        zero_power_filter = np.full((weather.shape[0]), True)
        if cut_out_wind_speed is not None:
            zero_power_filter = weather.windspeed < cut_out_wind_speed
        if cut_in_wind_speed is not None:
            zero_power_filter &= weather.windspeed >= cut_in_wind_speed

        args = [
            (
                deepcopy(self.floris),
                weather.loc[f"{month}/{year}"],
                (year, month),
                reinitialize_kwargs,
                run_kwargs,
            )
            for month, year in product(month_list, year_list)
        ]
        return args, zero_power_filter

    def run_wind_rose_aep(
        self,
        full_wind_rose: bool = False,
        run_kwargs: dict | None = None,
    ):
        """Runs the custom FLORIS WindRose AEP methodology that allows for gathering of
        intermediary results.

        Parameters
        ----------
        full_wind_rose : bool, optional
            If True, the full wind profile will be used, otherwise, if False, the wind profile will
            be limited to just the simulation period. Defaults to False.
        run_kwargs : dict | None, optional
            Arguments that are provided to ``FlorisInterface.get_farm_AEP_wind_rose_class()``.
            Defaults to None.

            From FLORIS:

                - cut_in_wind_speed (float, optional): Wind speed in m/s below which
                    any calculations are ignored and the wind farm is known to
                    produce 0.0 W of power. Note that to prevent problems with the
                    wake models at negative / zero wind speeds, this variable must
                    always have a positive value. Defaults to 0.001 [m/s].
                - cut_out_wind_speed (float, optional): Wind speed above which the
                    wind farm is known to produce 0.0 W of power. If None is
                    specified, will assume that the wind farm does not cut out
                    at high wind speeds. Defaults to None.
                - yaw_angles (NDArrayFloat | list[float] | None, optional):
                    The relative turbine yaw angles in degrees. If None is
                    specified, will assume that the turbine yaw angles are all
                    zero degrees for all conditions. Defaults to None.
                - turbine_weights (NDArrayFloat | list[float] | None, optional):
                    weighing terms that allow the user to emphasize power at
                    particular turbines and/or completely ignore the power
                    from other turbines. This is useful when, for example, you are
                    modeling multiple wind farms in a single floris object. If you
                    only want to calculate the power production for one of those
                    farms and include the wake effects of the neighboring farms,
                    you can set the turbine_weights for the neighboring farms'
                    turbines to 0.0. The array of turbine powers from floris
                    is multiplied with this array in the calculation of the
                    objective function. If None, this  is an array with all values
                    1.0 and with shape equal to (n_wind_directions, n_wind_speeds,
                    n_turbines). Defaults to None.
                - no_wake: (bool, optional): When *True* updates the turbine
                    quantities without calculating the wake or adding the wake to
                    the flow field. This can be useful when quantifying the loss
                    in AEP due to wakes. Defaults to *False*.
        """
        if run_kwargs is None:
            run_kwargs = {}
        if full_wind_rose:
            if TYPE_CHECKING:
                assert isinstance(self.weather, pd.DataFrame)  # mypy helper
            weather = self.weather.loc[:, [self.floris_wind_direction, self.floris_windspeed]]
        else:
            if TYPE_CHECKING:
                assert isinstance(self.weather, pd.DataFrame)  # mypy helper
            weather = self.weather.loc[
                self.operations_start : self.operations_end,
                [self.floris_wind_direction, self.floris_windspeed],
            ]

        # recreate the FlorisInterface object for the wind rose settings
        wd, ws = weather.values.T
        self.project_wind_rose = WindRose()
        project_wind_rose_df = self.project_wind_rose.make_wind_rose_from_user_data(
            wd, ws
        )  # noqa: F841  pylint: disable=W0612
        self.monthly_wind_rose = create_monthly_wind_rose(
            weather.rename(columns={self.floris_wind_direction: "wd", self.floris_windspeed: "ws"})
        )
        self.monthly_wind_rose = check_monthly_wind_rose(
            self.project_wind_rose, self.monthly_wind_rose
        )
        self.project_wind_rose.df.set_index(["wd", "ws"]).unstack().values
        freq_monthly = {
            k: wr.df.set_index(["wd", "ws"]).unstack().values
            for k, wr in self.monthly_wind_rose.items()
        }

        # Recreating FlorisInterface.get_farm_AEP() w/o some of the quality checks
        # because the parameters are coming directly from other FLORIS objects, and
        # not user inputs
        wd = project_wind_rose_df.wd.unique()
        ws = project_wind_rose_df.ws.unique()
        n_wd = wd.size
        n_ws = ws.size
        ix_evaluate = ws >= run_kwargs["cut_in_wind_speed"]
        if run_kwargs["cut_out_wind_speed"] is not None:
            ix_evaluate &= ws < run_kwargs["cut_out_wind_speed"]

        farm_potential_power = np.zeros((n_wd, n_ws))
        farm_production_power = np.zeros((n_wd, n_ws))
        turbine_potential_power = np.zeros((n_wd, n_ws, self.floris.floris.farm.n_turbines))
        turbine_production_power = np.zeros((n_wd, n_ws, self.floris.floris.farm.n_turbines))
        if np.any(ix_evaluate):
            ws_subset = ws[ix_evaluate]
            yaw_angles = run_kwargs.get("yaw_angles", None)
            if yaw_angles is not None:
                yaw_angles = yaw_angles[:, ix_evaluate]

            self.floris.reinitialize(wind_speeds=ws_subset, wind_directions=wd)

            # Calculate the potential energy
            self.floris.calculate_no_wake(yaw_angles=yaw_angles)
            farm_potential_power[:, ix_evaluate] = self.floris.get_farm_power(
                turbine_weights=run_kwargs["turbine_weights"]
            )
            turbine_potential_power[:, ix_evaluate, :] = self.floris.get_turbine_powers()
            if (weights := run_kwargs["turbine_weights"]) is not None:
                turbine_potential_power *= weights

            # Calculate the produced power
            self.floris.calculate_wake(yaw_angles=yaw_angles)

            farm_production_power[:, ix_evaluate] = self.floris.get_farm_power(
                turbine_weights=run_kwargs["turbine_weights"]
            )
            turbine_production_power[:, ix_evaluate, :] = self.floris.get_turbine_powers()
            if (weights := run_kwargs["turbine_weights"]) is not None:
                turbine_production_power *= weights
        else:
            self.floris.reinitialize(wind_speeds=ws, wind_directions=wd)

        # Calculate the monthly contribution to AEP from the wind rose
        self.turbine_production_energy = calculate_monthly_wind_rose_results(
            turbine_production_power, freq_monthly
        )
        self.turbine_production_energy.columns = self.floris_turbine_order  # Mwh
        self.project_production_energy = self.turbine_production_energy.values.sum()

        # Calculate the monthly potential contribution to AEP from the wind rose
        self.turbine_potential_energy = calculate_monthly_wind_rose_results(
            turbine_potential_power, freq_monthly
        )
        self.turbine_potential_energy.columns = self.floris_turbine_order  # Mwh
        self.project_potential_energy = self.turbine_potential_energy.values.sum()

    def run_floris(
        self,
        which: str,
        reinitialize_kwargs: dict | None = None,
        run_kwargs: dict | None = None,
        full_wind_rose: bool = False,
        cut_in_wind_speed: float = 0.001,
        cut_out_wind_speed: float | None = None,
        nodes: int = -1,
    ) -> None:
        """Runs either a FLORIS wind rose analysis for a simulation-level AEP value
        (``which="wind_rose"``) or a turbine-level time series for the WOMBAT simulation
        period (``which="time_series"``).

        Parameters
        ----------
        which : str
            One of "wind_rose" or "time_series" to run either a simulation-level wind rose analysis
            or hourly time-series analysis for the base AEP model.
        reinitialize_kwargs : dict | None, optional
            Any keyword arguments to be assed to ``FlorisInterface.reinitialize()``. Defaults to
            None.
        run_kwargs : dict | None, optional
            Any keyword arguments to be assed to ``FlorisInterface.calculate_wake()``.
            Defaults to None.
        full_wind_rose : bool, optional
            Indicates, for "wind_rose" analyses ONLY, if the full weather profile from ``weather``
            (True) or the limited, WOMBAT simulation period (False) should be used for analyis.
            Defaults to False.
        cut_in_wind_speed : float, optional
            The wind speed, in m/s, at which a turbine will start producing power. Should only be a
            value if running a time series analysis. Defaults to 0.001.
        cut_out_wind_speed : float, optional
            The wind speed, in m/s, at which a turbine will stop producing power. Should only be a
            value if running a time series analysis. Defaults to None.
        nodes : int, optional
            The number of nodes to parallelize over. If -1, then it will use the floor of 80% of the
            available CPUs on the computer. Defaults to -1.

        Raises
        ------
        ValueError: Raised if :py:attr:`which` is not one of "wind_rose" or "time_series".
        """
        if reinitialize_kwargs is None:
            reinitialize_kwargs = {}
        if run_kwargs is None:
            run_kwargs = {}

        if which == "wind_rose":
            # TODO: Change this to be modify the standard behavior, and get the turbine
            # powers to properly account for availability later

            # Set the FLORIS defaults
            run_kwargs.setdefault("cut_in_wind_speed", cut_in_wind_speed)
            run_kwargs.setdefault("cut_out_wind_speed", cut_out_wind_speed)
            run_kwargs.setdefault("turbine_weights", None)
            run_kwargs.setdefault("yaw_angles", None)
            run_kwargs.setdefault("no_wake", False)

            self.run_wind_rose_aep(full_wind_rose=full_wind_rose, run_kwargs=run_kwargs)
            self.floris_results_type = "wind_rose"

        elif which == "time_series":
            parallel_args, zero_power_filter = self.preprocess_monthly_floris(
                reinitialize_kwargs, run_kwargs, cut_in_wind_speed, cut_out_wind_speed
            )
            fi_dict, turbine_powers = run_parallel_time_series_floris(parallel_args, nodes)

            self._fi_dict = fi_dict
            self.turbine_aep_mwh = turbine_powers
            self.connect_floris_to_turbines(x_col=self.floris_x_col, y_col=self.floris_y_col)
            self.turbine_potential_energy.columns = self.floris_turbine_order
            self.turbine_potential_energy = (
                self.turbine_potential_energy.where(
                    np.repeat(
                        zero_power_filter.reshape(-1, 1),
                        self.turbine_aep_mwh.shape[1],
                        axis=1,
                    ),
                    0.0,
                )
                / 1e6
            )

            n_years = self.turbine_potential_energy.index.year.unique().size
            self.project_potential_energy = self.turbine_potential_energy.values.sum() / n_years
            self.floris_results_type = "time_series"
        else:
            raise ValueError(f"`which` must be one of: 'wind_rose' or 'time_series', not: {which}")

    def run(
        self,
        which_floris: str | None = None,
        floris_reinitialize_kwargs: dict | None = None,
        floris_run_kwargs: dict | None = None,
        full_wind_rose: bool = False,
        skip: list[str] | None = None,
        cut_in_wind_speed: float = 0.001,
        cut_out_wind_speed: float | None = None,
        nodes: int = -1,
    ) -> None:
        """Run all three models in serial, or a subset if ``skip`` is used.

        Parameters
        ----------
        which_floris : str | None, optional
            One of "wind_rose" or "time_series" if computing the farm's AEP based on a wind rose,
            or based on time series corresponding to the WOMBAT simulation period, respectively.
            Defaults to None.
        floris_reinitialize_kwargs : dict | None
            Any additional ``FlorisInterface.reinitialize`` keyword arguments. Defaults to None.
        floris_run_kwargs : dict | None
            Any additional ``FlorisInterface.get_farm_AEP`` or ``FlorisInterface.calculate_wake()``
            keyword arguments, depending on ``which_floris`` is used. Defaults to None.
        full_wind_rose : bool, optional
            Indicates, for "wind_rose" analyses ONLY, if the full weather profile from ``weather``
            (True) or the limited, WOMBAT simulation period (False) should be used for analyis.
            Defaults to False.
        skip : list[str] | None, optional
            A list of models to be skipped. This is intended to be used after a model is
            reinitialized with a new or modified configuration. Defaults to None.
        cut_in_wind_speed : float, optional
            The wind speed, in m/s, at which a turbine will start producing power. Can also be
            provided in ``floris_reinitialize_kwargs`` for a wind rose analysis, but must be
            provided here for a time series analysis. Defaults to 0.001.
        cut_out_wind_speed : float, optional
            The wind speed, in m/s, at which a turbine will stop producing power. Can also be
            provided in ``floris_reinitialize_kwargs`` for a wind rose analysis, but must be
            provided here for a time series analysis. Defaults to None.
        nodes : int, optional
            The number of nodes to parallelize over. If -1, then it will use the floor of 80% of the
            available CPUs on the computer. Defaults to -1.

        Raises
        ------
        ValueError
            Raised if ``which_floris`` is not one of "wind_rose" or "time_series".
        """
        if floris_reinitialize_kwargs is None:
            floris_reinitialize_kwargs = {}
        if floris_run_kwargs is None:
            floris_run_kwargs = {}
        if skip is None:
            skip = []
        if "floris" not in skip:
            if which_floris not in ("wind_rose", "time_series"):
                raise ValueError(
                    "`which_floris` must be one of: 'wind_rose' or 'time_series' when running"
                    f" FLORIS, not: {which_floris}"
                )

        if which_floris == "wind_rose":
            floris_reinitialize_kwargs.update(
                {"cut_in_wind_speed": cut_in_wind_speed, "cut_out_wind_speed": cut_out_wind_speed}
            )

        if "orbit" not in skip:
            self.orbit.run()
        if "wombat" not in skip:
            self.wombat.run()
        if "floris" not in skip:
            if TYPE_CHECKING:
                assert isinstance(which_floris, str)
            self.run_floris(
                which=which_floris,
                reinitialize_kwargs=floris_reinitialize_kwargs,
                run_kwargs=floris_run_kwargs,
                full_wind_rose=full_wind_rose,
                cut_in_wind_speed=cut_in_wind_speed,
                cut_out_wind_speed=cut_out_wind_speed,
                nodes=nodes,
            )

    def reinitialize(
        self,
        orbit_config: str | Path | dict | None = None,
        wombat_config: str | Path | dict | None = None,
        floris_config: str | Path | dict | None = None,
    ) -> None:
        """Enables a user to reinitialize one or multiple of the CapEx, OpEx, and AEP
        models.

        Parameters
        ----------
        orbit_config : str | Path | dict | None, optional
            ORBIT configuration file or dictionary. Defaults to None.
        wombat_config : str | Path | dict | None, optional
            WOMBAT configuation file or dictionary. Defaults to None.
        floris_config : (str | Path | dict | None, optional
            FLORIS configuration file or dictionary. Defaults to None.
        """
        if orbit_config is not None:
            self.orbit_config = orbit_config
            self.setup_orbit()

        if wombat_config is not None:
            self.wombat_config = wombat_config
            self.setup_wombat()

        if floris_config is not None:
            self.floris_config = floris_config
            self.setup_floris()

        return

    # **********************************************************************************************
    # Results methods
    # **********************************************************************************************

    # TODO: Figure out the actual workflows requried to have more complete/easier reporting

    def plot_farm(
        self,
        figure_kwargs: dict | None = None,
        draw_kwargs: dict | None = None,
        return_fig: bool = False,
    ) -> None | tuple[plt.figure, plt.axes]:
        """Plot the graph representation of the windfarm as represented through WOMBAT.

        Parameters
        ----------
        figure_kwargs : dict | None, optional
            Customized keyword arguments for matplotlib figure instantiation that will passed as
            ``plt.figure(**figure_kwargs)``. Defaults to None.
        draw_kwargs : dict | None, optional
            Customized keyword arguments for ``networkx.draw()`` that can will passed as
            ``nx.draw(**figure_kwargs)``. Defaults to None.
        return_fig : bool, optional
            Whether or not to return the figure and axes objects for further editing and/or saving.
            Defaults to False.

        Returns
        -------
        None | tuple[plt.figure, plt.axes]
            If :py:attr:`return_fig` is False, then None is returned, otherwise (True) the
            ``Figure`` and ``Axes`` objects are returned.
        """
        if figure_kwargs is None:
            figure_kwargs = {}
        if draw_kwargs is None:
            draw_kwargs = {}

        figure_kwargs.setdefault("figsize", (14, 12))
        figure_kwargs.setdefault("dpi", 200)

        fig = plt.figure(**figure_kwargs)
        ax = fig.add_subplot(111)

        windfarm = self.wombat.windfarm
        positions = {
            name: np.array([node["longitude"], node["latitude"]])
            for name, node in windfarm.graph.nodes(data=True)
        }

        draw_kwargs.setdefault("with_labels", True)
        draw_kwargs.setdefault("font_weight", "bold")
        draw_kwargs.setdefault("node_color", "#E37225")
        nx.draw(windfarm.graph, pos=positions, ax=ax, **draw_kwargs)

        fig.tight_layout()
        plt.show()

        if return_fig:
            return fig, ax
        return None

    # Design and installation related metrics

    def n_turbines(self) -> int:
        """Returns the number of turbines from either ORBIT, WOMBAT, or FLORIS depending on which
        model is available internally.

        Returns
        -------
        int
            The number of turbines in the project.

        Raises
        ------
        RuntimeError
            Raised if no model configurations were provided on initialization.
        """
        if self.orbit_config is not None:
            return self.orbit.num_turbines
        if self.wombat_config is not None:
            return len(self.wombat.windfarm.turbine_id)
        if self.floris_config is not None:
            return self.floris.farm.n_turbines
        raise RuntimeError("No models wer provided, cannot calculate value.")

    def turbine_rating(self) -> float:
        """Calculates the average turbine rating, in MW, of all the turbines in the project.

        Returns
        -------
        float
            The average rating of the turbines, in MW.

        Raises
        ------
        RuntimeError
            Raised if no model configurations were provided on initialization.
        """
        if self.orbit_config is not None:
            return self.orbit.turbine_rating
        if self.wombat_config is not None:
            return self.wombat.windfarm.capacity / 1000 / self.n_turbines
        raise RuntimeError("No models wer provided, cannot calculate value.")

    def n_substations(self) -> int:
        """Calculates the number of subsations in the project.

        Returns
        -------
        int
            The number of substations in the project.
        """
        if self.orbit_config is not None or "OffshoreSubstationDesign" not in self.orbit._phases:
            return self.orbit_config_dict["oss_design"]["num_substations"]
        if self.wombat_config is not None:
            return len(self.wombat.windfarm.substation_id)
        raise RuntimeError("No models wer provided, cannot calculate value.")

    def capacity(self, units: str = "mw") -> float:
        """Calculates the project's capacity in the desired units of kW, MW, or GW.

        Parameters
        ----------
        units : str, optional
            One of "kw", "mw", or "gw". Defaults to "mw".

        Returns
        -------
        float
            The project capacity, returned in the desired units

        Raises
        ------
        RuntimeError
            Raised if no model configurations were provided on initialization.
        ValueError
            Raised if an invalid units input was provided.
        """
        if self.orbit_config is not None:
            capacity = self.orbit.capacity
        elif self.wombat_config is not None:
            capacity = self.wombat.windfarm.capacity / 1000
        else:
            raise RuntimeError("No models wer provided, cannot calculate value.")

        units = units.lower()
        if units == "kw":
            return capacity * 1000
        if units == "mw":
            return capacity
        if units == "gw":
            return capacity / 1000
        raise ValueError("`units` must be one of: 'kw', 'mw', or 'gw'.")

    def capex(
        self, breakdown: bool = False, per_capacity: str | None = None
    ) -> pd.DataFrame | float:
        """Provides a thin wrapper to ORBIT's ``ProjectManager`` CapEx calculations that
        can provide a breakdown of total or normalize it by the project's capacity, in MW.

        Parameters
        ----------
        breakdown : bool, optional
            Provide a detailed view of the CapEx breakdown, and a total, which is
            the sum of the BOS, turbine, project, and soft CapEx categories. Defaults to False.
        per_capacity : str, optional
            Provide the CapEx normalized by the project's capacity, in the desired units. If
            None, then the unnormalized CapEx is returned, otherwise it must be one of "kw",
            "mw", or "gw". Defaults to None.

        Returns
        -------
        pd.DataFrame | float
            Project CapEx, normalized by :py:attr:`per_capacity`, if using, as either a
            pandas DataFrame if :py:attr:`breakdown` is True, otherwise, a float total.
        """
        if breakdown:
            capex = pd.DataFrame.from_dict(
                self.orbit.capex_breakdown, orient="index", columns=["CapEx"]
            )
            capex.loc["Total"] = self.orbit.total_capex
        else:
            capex = pd.DataFrame(
                [self.orbit.total_capex], columns=["CapEx"], index=pd.Index(["Total"])
            )

        if per_capacity is None:
            if breakdown:
                return capex
            return capex.values[0, 0]

        per_capacity = per_capacity.lower()
        capacity = self.capacity(per_capacity)
        unit_map = {"kw": "kW", "mw": "MW", "gw": "GW"}
        capex[f"CapEx per {unit_map[per_capacity]}"] = capex / capacity

        if breakdown:
            return capex
        return capex.values[0, 1]

    def array_system_total_cable_length(self):
        """Calculates the total length of the cables in the array system, in km.

        Returns
        -------
        float
            Total length, in km, of the array system cables.

        Raises
        ------
        ValueError
            Raised if neither ``ArraySystemDesign`` nor ``CustomArraySystem`` design
            were created in ORBIT.
        """
        if "ArraySystemDesign" in self.orbit._phases:
            array = self.orbit._phases["ArraySystemDesign"]
        elif "CustomArraySystemDesign" in self.orbit._phases:
            array = self.orbit._phases["CustomArraySystemDesign"]
        else:
            raise ValueError("No array system design was included in the ORBIT configuration.")

        # TODO: Fix ORBIT bug for nansum
        return np.nansum(array.sections_cable_lengths)

    def export_system_total_cable_length(self):
        """Calculates the total length of the cables in the export system, in km.

        Returns
        -------
        float
            Total length, in km, of the export system cables.

        Raises
        ------
        ValueError
            Raised if ``ExportSystemDesign`` was not created in ORBIT.
        """
        try:
            return self.orbit._phases["ExportSystemDesign"].total_length
        except KeyError:
            return self.orbit._phases["ElectricalDesign"].total_length
        except KeyError:
            raise ValueError(
                "Neither an `ElectricalDesign` nor an `ExportSystemDesign` phase were defined to be"
                " able to calculate this metric."
            )

    # Operational metrics

    def energy_potential(
        self,
        frequency: str = "project",
        by: str = "windfarm",
        units: str = "gw",
        per_capacity: str | None = None,
        aep: bool = False,
    ) -> pd.DataFrame | float:
        """Computes the potential energy production, or annual potential energy production, in GWh,
        for the simulation by extrapolating the monthly contributions to AEP if FLORIS (wtihout
        wakes) results were computed by a wind rose, or using the time series results.

        Parameters
        ----------
        frequency : str, optional
            One of "project" (project total), "annual" (annual total), or "month-year"
            (monthly totals for each year).
        by : str, optional
            One of "windfarm" (project level) or "turbine" (turbine level) to
            indicate what level to calculate the energy production.
        per_capacity : str, optional
            Provide the energy production normalized by the project's capacity, in the desired
            units. If None, then the unnormalized energy production is returned, otherwise it must
            be one of "kw", "mw", or "gw". Defaults to None.
        aep : bool, optional
            Flag to return the energy production normalized by the number of years the plan is in
            operation. Note that :py:attr:`frequency` must be "project" for this to be computed.

        Raises
        ------
        ValueError
            Raised if ``frequency`` is not one of: "project", "annual", "month-year".

        Returns
        -------
        pd.DataFrame | float
            The wind farm-level energy prodcution, in GWh, for the desired ``frequency``.
        """
        availability = self.wombat.metrics.production_based_availability(
            frequency="month-year", by="turbine"
        ).loc[:, self.floris_turbine_order]
        if self.floris_results_type == "wind_rose":
            power = pd.DataFrame(0.0, dtype=float, index=availability.index, columns=["drop"])
            power = power.merge(
                self.turbine_potential_energy,
                how="left",
                left_on="month",
                right_index=True,
            ).drop(labels=["drop"], axis=1)
            energy_gwh = power / 1000

        if self.floris_results_type == "time_series":
            energy_gwh = self.turbine_aep_mwh / 1000
            energy_gwh *= (
                self.turbine_potential_energy.assign(
                    year=energy_gwh.index.year, month=energy_gwh.index.month
                )
                .groupby(["year", "month"])
                .sum()
                .loc[energy_gwh.index]
            )

        unit_map = {"kw": "kWh", "mw": "MWh", "gw": "GWh"}
        if by == "windfarm":
            energy_gwh = energy_gwh.sum(axis=1).to_frame(f"Energy Losses ({unit_map[units]})")

        # Aggregate to the desired frequency level (nothing required for month-year)
        if frequency == "annual":
            energy_gwh = (
                energy_gwh.reset_index(drop=False).groupby("year").sum().drop(columns=["month"])
            )
        elif frequency == "project":
            if by == "turbine":
                energy_gwh = (
                    energy_gwh.sum(axis=0).to_frame(name=f"Energy Losses ({unit_map[units]})").T
                )
            else:
                energy_gwh = energy_gwh.values.sum()

        if aep:
            if frequency != "project":
                raise ValueError("`aep` can only be set to True, if `frequency`='project'.")
            energy_gwh /= self.operations_years

        if units == "kw":
            energy = energy_gwh * 1e6
        if units == "mw":
            energy = energy_gwh * 1e3
        if units == "gw":
            energy = energy_gwh

        if per_capacity is None:
            return energy
        return energy / self.capacity(per_capacity)

    def energy_production(
        self,
        frequency: str = "project",
        by: str = "windfarm",
        units: str = "gw",
        per_capacity: str | None = None,
        with_losses: bool = False,
        loss_ratio: float | None = None,
        aep: bool = False,
    ) -> pd.DataFrame | float:
        """Computes the energy production, or annual energy production, in GWh, for the simulation
        by extrapolating the monthly contributions to AEP if FLORIS (with wakes) results were
        computed by a wind rose, or using the time series results, and multiplying it by the WOMBAT
        monthly availability (``Metrics.production_based_availability``).

        Parameters
        ----------
        frequency : str, optional
            One of "project" (project total), "annual" (annual total), or "month-year"
            (monthly totals for each year).
        by : str, optional
            One of "windfarm" (project level) or "turbine" (turbine level) to
            indicate what level to calculate the energy production.
        per_capacity : str, optional
            Provide the energy production normalized by the project's capacity, in the desired
            units. If None, then the unnormalized energy production is returned, otherwise it must
            be one of "kw", "mw", or "gw". Defaults to None.
        with_losses : bool, optional
            Use the :py:attr:`loss_ratio` or :py:attr:`Project.loss_ratio` to post-hoc
            consider non-wake and non-availability losses in the energy production aggregation.
            Defaults to False.
        loss_ratio : float, optional
            The decimal non-wake and non-availability losses ratio to apply to the energy
            production. If None, then it will attempt to use the :py:attr:`loss_ratio` provided
            in the Project configuration. Defaults to None.
        aep : bool, optional
            Flag to return the energy production normalized by the number of years the plan is in
            operation. Note that :py:attr:`frequency` must be "project" for this to be computed.

        Raises
        ------
        ValueError
            Raised if ``frequency`` is not one of: "project", "annual", "month-year".

        Returns
        -------
        pd.DataFrame | float
            The wind farm-level energy prodcution, in GWh, for the desired ``frequency``.
        """
        # Check the frequency input
        opts = ("project", "annual", "month-year")
        if frequency not in opts:
            raise ValueError(f"`frequency` must be one of {opts}.")  # type: ignore

        # For the wind rose outputs, only consider project-level availability because
        # wind rose AEP is a long-term estimation of energy production
        availability = self.wombat.metrics.production_based_availability(
            frequency="month-year", by="turbine"
        ).loc[:, self.floris_turbine_order]
        if self.floris_results_type == "wind_rose":
            power = pd.DataFrame(0.0, dtype=float, index=availability.index, columns=["drop"])
            power = power.merge(
                self.turbine_production_energy,
                how="left",
                left_on="month",
                right_index=True,
            ).drop(labels=["drop"], axis=1)
            energy_gwh = availability * power / 1000

        if self.floris_results_type == "time_series":
            energy_gwh = self.turbine_aep_mwh / 1000
            energy_gwh *= (
                self.turbine_potential_energy.assign(
                    year=energy_gwh.index.year, month=energy_gwh.index.month
                )
                .groupby(["year", "month"])
                .sum()
                .loc[energy_gwh.index]
            )

        if by == "windfarm":
            energy_gwh = energy_gwh.sum(axis=1).to_frame("Energy Production (GWh)")

        # Aggregate to the desired frequency level (nothing required for month-year)
        if frequency == "annual":
            energy_gwh = (
                energy_gwh.reset_index(drop=False).groupby("year").sum().drop(columns=["month"])
            )
        elif frequency == "project":
            if by == "turbine":
                energy_gwh = energy_gwh.sum(axis=0).to_frame(name="Energy Production (GWh)").T
            else:
                energy_gwh = energy_gwh.values.sum()

        if with_losses:
            # Check that a loss_ratio exists
            if loss_ratio is None:
                if (loss_ratio := self.loss_ratio) is None:
                    raise ValueError(
                        "`loss_ratio` wasn't defined in the Project settings or in the method"
                        " keyword arguments."
                    )
            # Get the base production numbers from WOMBAT
            base_production = self.energy_potential(frequency=frequency, by=by, units="gw")
            energy_gwh -= base_production * loss_ratio

        if aep:
            if frequency != "project":
                raise ValueError("`aep` can only be set to True, if `frequency`='project'.")
            energy_gwh /= self.operations_years

        if units == "kw":
            energy = energy_gwh * 1e6
        if units == "mw":
            energy = energy_gwh * 1e3
        if units == "gw":
            energy = energy_gwh

        if per_capacity is None:
            return energy

        return energy / self.capacity(per_capacity)

    def energy_losses(
        self,
        frequency: str = "project",
        by: str = "windfarm",
        units: str = "gw",
        per_capacity: str | None = None,
        with_losses: bool = False,
        loss_ratio: float | None = None,
        aep: bool = False,
    ) -> pd.DataFrame:
        """Computes the energy losses for the simulation by subtracting the energy production from
        the potential energy production.

        Parameters
        ----------
        frequency : str, optional
            One of "project" (project total), "annual" (annual total), or "month-year"
            (monthly totals for each year).
        by : str, optional
            One of "windfarm" (project level) or "turbine" (turbine level) to
            indicate what level to calculate the energy production.
        per_capacity : str, optional
            Provide the energy production normalized by the project's capacity, in the desired
            units. If None, then the unnormalized energy production is returned, otherwise it must
            be one of "kw", "mw", or "gw". Defaults to None.
        with_losses : bool, optional
            Use the :py:attr:`loss_ratio` or :py:attr:`Project.loss_ratio` to post-hoc
            consider non-wake and non-availability losses in the energy production aggregation.
            Defaults to False.
        loss_ratio : float, optional
            The decimal non-wake and non-availability losses ratio to apply to the energy
            production. If None, then it will attempt to use the :py:attr:`loss_ratio` provided
            in the Project configuration. Defaults to None.
        aep : bool, optional
            AEP for the annualized losses. Only used for :py:attr:`frequency` = "project".

        Raises
        ------
        ValueError
            Raised if ``frequency`` is not one of: "project", "annual", "month-year".

        Returns
        -------
        pd.DataFrame | float
            The wind farm-level energy prodcution, in GWh, for the desired ``frequency``.
        """
        potential = self.energy_potential(
            "month-year",
            by="turbine",
            units="kw",
        )
        production = self.energy_production(  # type: ignore
            "month-year",
            by="turbine",
            units="kw",
            with_losses=with_losses,
            loss_ratio=loss_ratio,
        )[self.wombat.metrics.turbine_id]
        losses = potential - production

        unit_map = {"kw": "kWh", "mw": "MWh", "gw": "GWh"}
        if by == "windfarm":
            losses = losses.sum(axis=1).to_frame(f"Energy Losses ({unit_map[units]})")

        if frequency == "project":
            losses = losses.sum(axis=0)
            if by == "windfarm":
                losses = losses.values[0]
            if aep:
                losses /= self.operations_years
        elif frequency == "annual":
            losses = losses.groupby("year").sum()

        if units == "mw":
            losses /= 1e3
        elif units == "gw":
            losses /= 1e6

        if per_capacity is None:
            return losses

        return losses / self.capacity(per_capacity)

    def availability(
        self, which: str, frequency: str = "project", by: str = "windfarm"
    ) -> pd.DataFrame | float:
        """Calculates the availability based on either a time or energy basis. This is a thin
        wrapper around `self.wombat.metrics.time_based_availability()` or
        `self.wombat.metrics.production_based_availability()`.

        Parameters
        ----------
        which : str
            One of "energy" or "time" to indicate which basis to use for the availability
            calculation. For "energy", this indicates the operating capacity of the project, and
            for "time", this is the ratio of any level operational to all time.
        frequency : str, optional
            One of "project" (project total), "annual" (annual total), or "month-year"
            (monthly totals for each year).
        by : str, optional
            One of "windfarm" (project level) or "turbine" (turbine level) to
            indicate what level to calculate the availability.

        Returns
        -------
        pd.DataFrame | float
            The appropriate availability metric, as a DataFrame unless it's calculated at the
            project/windfarm level.

        Raises
        ------
        ValueError
            Raised if :py:attr:`which` is not one of "energy" or "time".
        """
        which = which.lower()
        if which == "energy":
            availability = self.wombat.metrics.production_based_availability(
                frequency=frequency, by=by
            )
        elif which == "time":
            availability = self.wombat.metrics.time_based_availability(frequency=frequency, by=by)
        else:
            raise ValueError("`which` must be one of 'energy' or 'time'.")

        if frequency == "project" and by == "windfarm":
            return availability.values[0, 0]
        return availability

    def capacity_factor(
        self,
        which: str,
        frequency: str = "project",
        by: str = "windfarm",
        with_losses: bool = False,
        loss_ratio: float | None = None,
    ) -> pd.DataFrame | float:
        """Calculates the capacity factor over a project's lifetime as a single value, annual
        average, or monthly average for the whole windfarm or by turbine.

        Parameters
        ----------
        which : str
            One of "net" (realized energy / capacity) or "gross" (potential energy production /
            capacity).
        frequency : str
            One of "project", "annual", "monthly", or "month-year". Defaults to "project".
        by : str
            One of "windfarm" or "turbine". Defaults to "windfarm".
        with_losses : bool, optional
            Use the :py:attr:`loss_ratio` or :py:attr:`Project.loss_ratio` to post-hoc
            consider non-wake and non-availability losses in the energy production aggregation.
            Defaults to False.

            .. note:: This will only be checked for :py:attr:`which` = "net".

        loss_ratio : float, optional
            The decimal non-wake and non-availability losses ratio to apply to the energy
            production. If None, then it will attempt to use the :py:attr:`loss_ratio` provided
            in the Project configuration. Defaults to None.

            .. note:: This will only be used when for :py:attr:`which` = "net".

        Returns
        -------
        pd.DataFrame | float
            The capacity factor at the desired aggregation level.
        """
        which = which.lower().strip()
        if which not in ("net", "gross"):
            raise ValueError('``which`` must be one of "net" or "gross".')

        opts = ("project", "annual", "month-year")
        if frequency not in opts:
            raise ValueError(f"`frequency` must be one of {opts}.")  # type: ignore

        by = by.lower().strip()
        if by not in ("windfarm", "turbine"):
            raise ValueError('``by`` must be one of "windfarm" or "turbine".')
        by_turbine = by == "turbine"

        if which == "net":
            numerator = self.energy_production(
                frequency="month-year",
                by="turbine",
                units="kw",
                with_losses=with_losses,
                loss_ratio=loss_ratio,
            )
        else:
            numerator = self.energy_potential(
                frequency="month-year",
                by="turbine",
                units="kw",
            )
        _potential = self.wombat.metrics.potential.loc[
            :, ["year", "month"] + self.wombat.metrics.turbine_id
        ]
        _capacity = np.ones((_potential.shape[0], self.n_turbines())) * np.array(
            self.wombat.metrics.turbine_capacities
        )
        capacity = (
            pd.DataFrame(
                np.hstack(
                    (
                        _potential.year.values.reshape(-1, 1),
                        _potential.month.values.reshape(-1, 1),
                        _capacity,
                    )
                ),
                columns=_potential.columns,
            )
            .groupby(["year", "month"])
            .sum()
        )

        if TYPE_CHECKING:
            assert isinstance(numerator, pd.DataFrame)

        if frequency == "project":
            if not by_turbine:
                return numerator.values.sum() / capacity.values.sum()
            return (
                (numerator.sum(axis=0) / capacity.sum(axis=0))
                .to_frame(f"{which.title()} Capacity Factor")
                .T
            )

        if frequency == "annual":
            group_cols = ["year"]
        elif frequency == "monthly":
            group_cols = ["month"]
        elif frequency == "month-year":
            group_cols = ["year", "month"]
        capacity = (
            capacity.reset_index(drop=False)[group_cols + self.wombat.metrics.turbine_id]
            .groupby(group_cols)
            .sum()
        )
        numerator = (
            numerator.reset_index(drop=False)[group_cols + self.wombat.metrics.turbine_id]
            .groupby(group_cols)
            .sum()
        )

        if not by_turbine:
            numerator = numerator.sum(axis=1).to_frame(name=f"{which.title()} Capacity Factor")
            capacity = capacity.sum(axis=1).to_frame(name=f"{which.title()} Capacity Factor")
        return numerator / capacity

    def opex(
        self, frequency: str = "project", per_capacity: str | None = None
    ) -> pd.DataFrame | float:
        """Calculates the operational expenditures of the project.

        Parameters
        ----------
        frequency (str, optional): One of "project", "annual", "monthly", "month-year".
            Defaults to "project".
        per_capacity : str, optional
            Provide the OpEx normalized by the project's capacity, in the desired units. If None,
            then the unnormalized OpEx is returned, otherwise it must be one of "kw", "mw", or "gw".
            Defaults to None.

        Returns
        -------
        pd.DataFrame | float
            The resulting OpEx DataFrame at the desired frequency, if more granular than the project
            frequency, otherwise a float. This will be normalized by the capacity, if
            :py:attr:`per_capacity` is not None.
        """
        opex = self.wombat.metrics.opex(frequency=frequency)
        if frequency == "project":
            opex = opex.values[0, 0]
        if per_capacity is None:
            return opex

        per_capacity = per_capacity.lower()
        return opex / self.capacity(per_capacity)

    def revenue(
        self,
        frequency: str = "project",
        offtake_price: float | None = None,
        per_capacity: str | None = None,
    ) -> pd.DataFrame | float:
        """Calculates the revenue stream using the WOMBAT availabibility, FLORIS energy
        production, and WAVES energy pricing.

        Parameters
        ----------
        frequency : str, optional
            One of "project", "annual", "monthly", or "month-year". Defaults to "project".
        offtake_price : float, optional
            Price paid per MWh of energy produced. Defaults to None.
        per_capacity : str, optional
            Provide the revenue normalized by the project's capacity, in the desired units.
            If None, then the unnormalized revenue is returned, otherwise it must
            be one of "kw", "mw", or "gw". Defaults to None.

        Returns
        -------
        pd.DataFrame | float
            The revenue stream of the wind farm at the provided frequency.
        """
        # Check the frequency input
        opts = ("project", "annual", "month-year")
        if frequency not in opts:
            raise ValueError(f"`frequency` must be one of {opts}.")  # type: ignore

        # Check that an offtake_price exists
        if offtake_price is None:
            if (offtake_price := self.offtake_price) is None:
                raise ValueError(
                    "`offtake_price` wasn't defined in the Project settings or in the method"
                    " keyword arguments."
                )

        if self.floris_results_type == "wind_rose":
            revenue = self.energy_production(frequency=frequency) * 1000 * offtake_price  # MWh
        else:
            revenue = self.energy_production(frequency=frequency) * 1000 * offtake_price  # MWh

        if frequency != "project":
            if TYPE_CHECKING:
                assert isinstance(revenue, pd.DataFrame)
            revenue.columns = ["Revenue"]

        if per_capacity is None:
            return revenue

        per_capacity = per_capacity.lower()
        return revenue / self.capacity(per_capacity)

    def capex_breakdown(
        self,
        frequency: str = "month-year",
        installation_start_date: str | None = None,
        soft_capex_date: tuple[int, int] | list[tuple[int, int]] | None = None,
        project_capex_date: tuple[int, int] | list[tuple[int, int]] | None = None,
        system_capex_date: tuple[int, int] | list[tuple[int, int]] | None = None,
        turbine_capex_date: tuple[int, int] | list[tuple[int, int]] | None = None,
        breakdown: bool = False,
    ) -> pd.DataFrame:
        """Calculates the monthly CapEx breakdwon into a DataFrame, that is returned at the desired
        frequency, allowing for custom starting dates for the varying CapEx costs.

        Parameters
        ----------
        frequency : str, optional
            The desired frequency of the outputs, where "month-year" is the monthly total over the
            course of a project's life. Must be one of: "project", "annual", "month-year". Defaults
            to "month-year"
        installation_start_date : str | None, optional
            If not provided in the ``Project`` configuration as ``orbit_start_date``, an
            installation starting date that is parseable from a string by Pandas may be provided
            here. Defaults to None
        soft_capex_date : tuple[int, int] | list[tuple[int, int]] | None, optional
            The date(s) where the ORBIT soft CapEx costs should be applied as a tuple of year and
            month, for instance: ``(2020, 1)`` for January 2020. Alternatively multiple dates can be
            set, which evenly divides the cost over all the dates, by providing a list of year and
            month combinations, for instance, a semi-annual 2 year cost starting in 2020 would look
            like: ``[(2020, 1), (2020, 7), (2021, 1), (2021, 7)]``. If None is provided, then the
            CapEx date will be the same as the start of the installation. Defaults to None
        project_capex_date : tuple[int, int] | list[tuple[int, int]] | None, optional
            The date(s) where the ORBIT project CapEx costs should be applied as a tuple of year and
            month, for instance: ``(2020, 1)`` for January 2020. Alternatively multiple dates can be
            set, which evenly divides the cost over all the dates, by providing a list of year and
            month combinations, for instance, a semi-annual 2 year cost starting in 2020 would look
            like: ``[(2020, 1), (2020, 7), (2021, 1), (2021, 7)]``. If None is provided, then the
            CapEx date will be the same as the start of the installation. Defaults to None
        system_capex_date : tuple[int, int] | list[tuple[int, int]] | None, optional
            The date(s) where the ORBIT system CapEx costs should be applied as a tuple of year and
            month, for instance: ``(2020, 1)`` for January 2020. Alternatively multiple dates can be
            set, which evenly divides the cost over all the dates, by providing a list of year and
            month combinations, for instance, a semi-annual 2 year cost starting in 2020 would look
            like: ``[(2020, 1), (2020, 7), (2021, 1), (2021, 7)]``. If None is provided, then the
            CapEx date will be the same as the start of the installation. Defaults to None.
        turbine_capex_date : tuple[int, int] | list[tuple[int, int]] | None, optional
            The date(s) where the ORBIT turbine CapEx costs should be applied as a tuple of year and
            month, for instance: ``(2020, 1)`` for January 2020. Alternatively multiple dates can be
            set, which evenly divides the cost over all the dates, by providing a list of year and
            month combinations, for instance, a semi-annual 2 year cost starting in 2020 would look
            like: ``[(2020, 1), (2020, 7), (2021, 1), (2021, 7)]``. If None is provided, then the
            CapEx date will be the same as the start of the installation. Defaults to None.
        offtake_price : int | float | None, optional
            The price paid for the energy produced, by default None.
        breakdown : bool, optional
            If True, all the CapEx categories will be provided as a column, in addition to the OpEx
            and Revenue columns, and the total cost in "cash_flow", othwerwise, only the "cash_flow"
            column will be provided. Defaults to False.

        Returns
        -------
        pd.DataFrame
            Returns the pandas DataFrame of the cashflow with a fixed monthly or annual interval,
            or a project total for the desired categories.

        Raises
        ------
        ValueError
            Raised if ``frequency`` is not one of: "project", "annual", "month-year".
        TypeError
            Raised if a valid starting date can't be found for the installation.
        """
        # Check the frequency input
        opts = ("project", "annual", "month-year")
        if frequency not in opts:
            raise ValueError(f"`frequency` must be one of {opts}.")  # type: ignore

        # Find a valid starting date for the installation processes
        if (start_date := installation_start_date) is None:
            if (start_date := self.orbit_start_date) is None:
                start_date = min(el for el in self.orbit_config_dict["install_phases"].values())
        try:
            start_date = pd.to_datetime(start_date)
        except pd.errors.ParserError:
            raise TypeError(
                "Please provide a valid `instatllation_start_date` if no configuration-based"
                " starting dates were provided."
            )

        # Create the cost dataframes that have a MultiIndex with "year" and "month" columns
        # Get the installation costs and add in each installation phase's port costs
        capex_installation = pd.DataFrame(self.orbit.logs)
        starts = capex_installation.loc[
            capex_installation.message == "SIMULATION START", ["phase", "time"]
        ]
        phase_df_list = []
        for name, time in starts.values:
            phase = self.orbit._phases[name]
            total_days = phase.total_phase_time / 24
            n_days, remainder = divmod(total_days, 1)
            day_cost = phase.port_costs / total_days
            phase_daily_port = np.repeat(day_cost, n_days).tolist() + [day_cost * remainder]
            timing = (np.arange(n_days + 1) * 24 + time).tolist()
            phase_df = pd.DataFrame(zip(timing, phase_daily_port), columns=["time", "cost"])
            phase_df["phase"] = name
            phase_df_list.append(phase_df)

        capex_installation = pd.concat([capex_installation, *phase_df_list], axis=0).sort_values(
            "time"
        )

        # Put the installation CapEx in a format that matches the OpEx output
        capex_installation["datetime"] = start_date + pd.to_timedelta(
            capex_installation.time, unit="hours"
        )
        capex_installation = (
            capex_installation.assign(
                year=capex_installation.datetime.dt.year,
                month=capex_installation.datetime.dt.month,
            )[["year", "month", "cost", "phase"]]
            .groupby(["year", "month", "phase"])
            .sum()
            .unstack(level=2, fill_value=0)
        )
        capex_installation.columns = [f"CapEx_{c}" for c in capex_installation.columns.droplevel(0)]
        capex_installation.columns.name = None  # type: ignore

        # Check the date values, ensuring a the installation start is the default if none provided
        default_start = [capex_installation.index[0]]
        if soft_capex_date is None:
            if (soft_capex_date := self.soft_capex_date) is None:
                soft_capex_date = default_start

        if project_capex_date is None:
            if (project_capex_date := self.project_capex_date) is None:
                project_capex_date = default_start

        if system_capex_date is None:
            if (system_capex_date := self.system_capex_date) is None:
                system_capex_date = default_start

        if turbine_capex_date is None:
            if (turbine_capex_date := self.turbine_capex_date) is None:
                turbine_capex_date = default_start

        # Convert the dates to a pandas MultiIndex to be compatible with concatenating later
        soft_capex_date_ix = convert_to_multi_index(soft_capex_date, "soft_capex_date")
        project_capex_date_ix = convert_to_multi_index(project_capex_date, "project_capex_date")
        system_capex_date_ix = convert_to_multi_index(system_capex_date, "system_capex_date")
        turbine_capex_date_ix = convert_to_multi_index(turbine_capex_date, "turbine_capex_date")

        # Create the remaining CapEx dataframes in the OpEx format
        capex_soft = pd.DataFrame(
            self.orbit.soft_capex / len(soft_capex_date_ix),
            index=soft_capex_date_ix,
            columns=["CapEx_Soft"],
        )
        capex_project = pd.DataFrame(
            self.orbit.project_capex / len(project_capex_date_ix),
            index=project_capex_date_ix,
            columns=["CapEx_Project"],
        )
        capex_turbine = pd.DataFrame(
            self.orbit.turbine_capex / len(turbine_capex_date_ix),
            index=turbine_capex_date_ix,
            columns=["CapEx_Turbine"],
        )
        if self.orbit.system_costs == {}:
            capex_system = pd.DataFrame([0], columns=["Installation"])
        else:
            capex_system = pd.DataFrame.from_dict(self.orbit.system_costs, orient="index").T
        capex_system = capex_system.loc[
            capex_system.index.repeat(len(system_capex_date_ix))
        ].set_index(system_capex_date_ix) / len(system_capex_date_ix)
        capex_system.columns = [
            f"CapEx_{col.replace('Installation', 'System')}" for col in capex_system
        ]

        # Combine the CapEx categories and sum for a total CapEx
        capex_df = reduce(
            lambda x, y: x.join(y, how="outer"),
            [
                capex_soft,
                capex_project,
                capex_turbine,
                capex_system,
                capex_installation,
            ],
        ).fillna(0)

        # Fill in the missing time periods to ensure a fixed-interval cash flow
        years = capex_df.index.get_level_values("year")
        years = list(range(min(years), max(years)))
        missing_ix = set(product(years, range(1, 13))).difference(capex_df.index.values)
        if missing_ix:
            capex_df = pd.concat(
                [
                    capex_df,
                    pd.DataFrame(
                        0,
                        index=convert_to_multi_index(list(missing_ix), "missing"),
                        columns=capex_df.columns,
                    ),
                ]
            ).sort_index()
        if frequency == "annual":
            capex_df = (
                capex_df.reset_index(drop=False).groupby("year").sum().drop(labels="month", axis=1)
            )
        elif frequency == "project":
            capex_df = capex_df.sum(axis=0).to_frame(name="Cash Flow").T

        capex_df["Capex"] = capex_df.sum(axis=1).sort_index()
        if breakdown:
            return capex_df
        return capex_df.CapEx.to_frame()

    def cash_flow(
        self,
        frequency: str = "month-year",
        installation_start_date: str | None = None,
        soft_capex_date: tuple[int, int] | list[tuple[int, int]] | None = None,
        project_capex_date: tuple[int, int] | list[tuple[int, int]] | None = None,
        system_capex_date: tuple[int, int] | list[tuple[int, int]] | None = None,
        turbine_capex_date: tuple[int, int] | list[tuple[int, int]] | None = None,
        offtake_price: int | float | None = None,
        breakdown: bool = False,
    ) -> pd.DataFrame:
        """Calculates the monthly cashflows into a DataFrame, that is returned at the desired
        frequency, and with or without a high level breakdown, allowing for custom starting dates
        for the varying CapEx costs.

        Parameters
        ----------
        frequency : str, optional
            The desired frequency of the outputs, where "month-year" is the monthly total over the
            course of a project's life. Must be one of: "project", "annual", "month-year". Defaults
            to "month-year"
        installation_start_date : str | None, optional
            If not provided in the ``Project`` configuration as ``orbit_start_date``, an
            installation starting date that is parseable from a string by Pandas may be provided
            here. Defaults to None
        soft_capex_date : tuple[int, int] | list[tuple[int, int]] | None, optional
            The date(s) where the ORBIT soft CapEx costs should be applied as a tuple of year and
            month, for instance: ``(2020, 1)`` for January 2020. Alternatively multiple dates can be
            set, which evenly divides the cost over all the dates, by providing a list of year and
            month combinations, for instance, a semi-annual 2 year cost starting in 2020 would look
            like: ``[(2020, 1), (2020, 7), (2021, 1), (2021, 7)]``. If None is provided, then the
            CapEx date will be the same as the start of the installation. Defaults to None
        project_capex_date : tuple[int, int] | list[tuple[int, int]] | None, optional
            The date(s) where the ORBIT project CapEx costs should be applied as a tuple of year and
            month, for instance: ``(2020, 1)`` for January 2020. Alternatively multiple dates can be
            set, which evenly divides the cost over all the dates, by providing a list of year and
            month combinations, for instance, a semi-annual 2 year cost starting in 2020 would look
            like: ``[(2020, 1), (2020, 7), (2021, 1), (2021, 7)]``. If None is provided, then the
            CapEx date will be the same as the start of the installation. Defaults to None
        system_capex_date : tuple[int, int] | list[tuple[int, int]] | None, optional
            The date(s) where the ORBIT system CapEx costs should be applied as a tuple of year and
            month, for instance: ``(2020, 1)`` for January 2020. Alternatively multiple dates can be
            set, which evenly divides the cost over all the dates, by providing a list of year and
            month combinations, for instance, a semi-annual 2 year cost starting in 2020 would look
            like: ``[(2020, 1), (2020, 7), (2021, 1), (2021, 7)]``. If None is provided, then the
            CapEx date will be the same as the start of the installation. Defaults to None.
        turbine_capex_date : tuple[int, int] | list[tuple[int, int]] | None, optional
            The date(s) where the ORBIT turbine CapEx costs should be applied as a tuple of year and
            month, for instance: ``(2020, 1)`` for January 2020. Alternatively multiple dates can be
            set, which evenly divides the cost over all the dates, by providing a list of year and
            month combinations, for instance, a semi-annual 2 year cost starting in 2020 would look
            like: ``[(2020, 1), (2020, 7), (2021, 1), (2021, 7)]``. If None is provided, then the
            CapEx date will be the same as the start of the installation. Defaults to None.
        offtake_price : int | float | None, optional
            The price paid for the energy produced, by default None.
        breakdown : bool, optional
            If True, all the CapEx categories will be provided as a column, in addition to the OpEx
            and Revenue columns, and the total cost in "cash_flow", othwerwise, only the "cash_flow"
            column will be provided. Defaults to False.

        Returns
        -------
        pd.DataFrame
            Returns the pandas DataFrame of the cashflow with a fixed monthly or annual interval,
            or a project total for the desired categories.

        Raises
        ------
        ValueError
            Raised if ``frequency`` is not one of: "project", "annual", "month-year".
        TypeError
            Raised if a valid starting date can't be found for the installation.
        """
        # Check the frequency input
        opts = ("project", "annual", "month-year")
        if frequency not in opts:
            raise ValueError(f"`frequency` must be one of {opts}.")  # type: ignore

        # Find a valid starting date for the installation processes
        if (start_date := installation_start_date) is None:
            if (start_date := self.orbit_start_date) is None:
                start_date = min(el for el in self.orbit_config_dict["install_phases"].values())
        try:
            start_date = pd.to_datetime(start_date)
        except pd.errors.ParserError:
            raise TypeError(
                "Please provide a valid `instatllation_start_date` if no configuration-based"
                " starting dates were provided."
            )

        # Create the cost dataframes that have a MultiIndex with "year" and "month" columns

        # Get the installation costs and add in each installation phase's port costs
        capex_installation = pd.DataFrame(self.orbit.logs)
        starts = capex_installation.loc[
            capex_installation.message == "SIMULATION START", ["phase", "time"]
        ]
        phase_df_list = []
        for name, time in starts.values:
            phase = self.orbit._phases[name]
            total_days = phase.total_phase_time / 24
            n_days, remainder = divmod(total_days, 1)
            day_cost = phase.port_costs / total_days
            phase_daily_port = np.repeat(day_cost, n_days).tolist() + [day_cost * remainder]
            timing = (np.arange(n_days + 1) * 24 + time).tolist()
            phase_df = pd.DataFrame(zip(timing, phase_daily_port), columns=["time", "cost"])
            phase_df["phase"] = name
            phase_df_list.append(phase_df)

        capex_installation = pd.concat([capex_installation, *phase_df_list], axis=0).sort_values(
            "time"
        )

        # Put the installation CapEx in a format that matches the OpEx output
        capex_installation["datetime"] = start_date + pd.to_timedelta(
            capex_installation.time, unit="hours"
        )
        capex_installation = (
            capex_installation.assign(
                year=capex_installation.datetime.dt.year,
                month=capex_installation.datetime.dt.month,
            )[["year", "month", "cost", "phase"]]
            .groupby(["year", "month", "phase"])
            .sum()
            .unstack(level=2, fill_value=0)
        )
        capex_installation.columns = [f"CapEx_{c}" for c in capex_installation.columns.droplevel(0)]
        capex_installation.columns.name = None  # type: ignore

        # Check the date values, ensuring a the installation start is the default if none provided
        default_start = [capex_installation.index[0]]
        if soft_capex_date is None:
            if (soft_capex_date := self.soft_capex_date) is None:
                soft_capex_date = default_start

        if project_capex_date is None:
            if (project_capex_date := self.project_capex_date) is None:
                project_capex_date = default_start

        if system_capex_date is None:
            if (system_capex_date := self.system_capex_date) is None:
                system_capex_date = default_start

        if turbine_capex_date is None:
            if (turbine_capex_date := self.turbine_capex_date) is None:
                turbine_capex_date = default_start

        # Convert the dates to a pandas MultiIndex to be compatible with concatenating later
        soft_capex_date_ix = convert_to_multi_index(soft_capex_date, "soft_capex_date")
        project_capex_date_ix = convert_to_multi_index(project_capex_date, "project_capex_date")
        system_capex_date_ix = convert_to_multi_index(system_capex_date, "system_capex_date")
        turbine_capex_date_ix = convert_to_multi_index(turbine_capex_date, "turbine_capex_date")

        # Create the remaining CapEx dataframes in the OpEx format
        capex_soft = pd.DataFrame(
            self.orbit.soft_capex / len(soft_capex_date_ix),
            index=soft_capex_date_ix,
            columns=["CapEx_Soft"],
        )
        capex_project = pd.DataFrame(
            self.orbit.project_capex / len(project_capex_date_ix),
            index=project_capex_date_ix,
            columns=["CapEx_Project"],
        )
        capex_turbine = pd.DataFrame(
            self.orbit.turbine_capex / len(turbine_capex_date_ix),
            index=turbine_capex_date_ix,
            columns=["CapEx_Turbine"],
        )
        capex_system = pd.DataFrame.from_dict(self.orbit.system_costs, orient="index").T
        capex_system = capex_system.loc[
            capex_system.index.repeat(len(system_capex_date_ix))
        ].set_index(system_capex_date_ix) / len(system_capex_date_ix)
        capex_system.columns = [
            f"CapEx_{col.replace('Installation', 'System')}" for col in capex_system
        ]

        # Combine the CapEx, Opex, and Revenue times and ensure their signs are correct
        cost_df = reduce(
            lambda x, y: x.join(y, how="outer"),
            [
                capex_soft,
                capex_project,
                capex_turbine,
                capex_system,
                capex_installation,
                self.opex(frequency="month-year"),
            ],
        ).fillna(0)
        cost_df *= -1.0
        cost_df = cost_df.join(
            self.revenue(frequency="month-year", offtake_price=offtake_price)
        ).fillna(0)

        # Fill in the missing time periods to ensure a fixed-interval cash flow
        years = cost_df.index.get_level_values("year")
        years = list(range(min(years), max(years)))
        missing_ix = set(product(years, range(1, 13))).difference(cost_df.index.values)
        if missing_ix:
            cost_df = pd.concat(
                [
                    cost_df,
                    pd.DataFrame(
                        0,
                        index=convert_to_multi_index(list(missing_ix), "missing"),
                        columns=cost_df.columns,
                    ),
                ]
            ).sort_index()
        if frequency == "annual":
            cost_df = (
                cost_df.reset_index(drop=False).groupby("year").sum().drop(labels="month", axis=1)
            )
        elif frequency == "project":
            cost_df = cost_df.sum(axis=0).to_frame(name="Cash Flow").T

        cost_df["cash_flow"] = cost_df.sum(axis=1).sort_index()
        if breakdown:
            return cost_df
        return cost_df.cash_flow.to_frame()

    def npv(
        self,
        frequency: str = "project",
        discount_rate: float | None = None,
        offtake_price: float | None = None,
        cash_flow: pd.DataFrame | None = None,
        **kwargs: dict,
    ) -> pd.DataFrame:
        """Calculates the net present value of the windfarm at a project, annual, or
        monthly resolution given a base discount rate and offtake price.

        .. note:: NPV is implemented via
            https://numpy.org/numpy-financial/latest/npv.html#numpy_financial.npv.

        Parameters
        ----------
        frequency : str
            One of "project", "annual", "monthly", or "month-year".
        discount_rate : float, optional
            The rate of return that could be earned on alternative investments. Defaults to None.
        offtake_price : float, optional
            Price of energy, per MWh. Defaults to None.
        cash_flow : pd.DataFrame, optional
            A modified cash flow DataFrame for custom workflows. Must have the "cash_flow" column
            with consistent time steps (monthly, annually, etc.). Defaults to None.
        kwargs : dict, optional
            See :py:meth:`cash_flow` for details on starting date options.

        Returns
        -------
        pd.DataFrame
            The project net prsent value at the desired time resolution.
        """
        # Check the frequency input
        opts = ("project", "annual", "month-year")
        if frequency not in opts:
            raise ValueError(f"`frequency` must be one of {opts}.")  # type: ignore

        # Check that the discout rate exists
        if discount_rate is None:
            if (discount_rate := self.discount_rate) is None:
                raise ValueError(
                    "`discount_rate` wasn't defined in the Project settings or in the method"
                    " keyword arguments."
                )

        # Check that the offtake price exists
        if offtake_price is None:
            if (offtake_price := self.offtake_price) is None:
                raise ValueError(
                    "`offtake_price` wasn't defined in the Project settings or in the method"
                    " keyword arguments."
                )

        if cash_flow is None:
            cash_flow = self.cash_flow(
                installation_start_date=kwargs.get("installation_start_date", None),  # type: ignore
                project_capex_date=kwargs.get("project_capex_date", None),  # type: ignore
                soft_capex_date=kwargs.get("soft_capex_date", None),  # type: ignore
                system_capex_date=kwargs.get("system_capex_date", None),  # type: ignore
                turbine_capex_date=kwargs.get("turbine_capex_date", None),  # type: ignore
                offtake_price=offtake_price,
            )
        return npf.npv(discount_rate, cash_flow.cash_flow.values)

    def irr(
        self,
        offtake_price: float | None = None,
        finance_rate: float | None = None,
        reinvestment_rate: float | None = None,
        cash_flow: pd.DataFrame | None = None,
        **kwargs,
    ) -> float:
        """Calculates the Internal Rate of Return using the ORBIT CapEx as the initial
        investment in conjunction with the WAVES monthly cash flows.

        .. note:: This method allows for the caluclation of the modified internal rate of return
            through https://numpy.org/numpy-financial/latest/mirr.html#numpy_financial.mirr
            if both the :py:attr:`finance_rate` and the :py:attr:`reinvestment_rate` are provided.

        Parameters
        ----------
        offtake_price : float, optional
            Price of energy, per MWh. Defaults to None.
        finance_rate : float, optional
            Interest rate paid on the cash flows. Only used if :py:attr:`reinvestment_rate` is also
            provided. Defaults to None.
        reinvestment_rate : float, optional
            Interest rate received on the cash flows upon reinvestment.  Only used if
            :py:attr:`finance_rate` is also provided.
        cash_flow : pd.DataFrame, optional
            A modified cash flow DataFrame for custom workflows. Must have the "cash_flow" column
            with consistent time steps (monthly, annually, etc.). Defaults to None.
        kwargs : dict, optional
            See :py:meth:`cash_flow` for details on starting date options.

        Returns
        -------
        float
            The IRR.
        """
        # Check that the offtake price exists
        if offtake_price is None:
            if (offtake_price := self.offtake_price) is None:
                raise ValueError(
                    "`offtake_price` wasn't defined in the Project settings or in the method"
                    " keyword arguments."
                )

        # Check to see if the Modified IRR should be used
        if finance_rate is None:
            finance_rate = self.finance_rate
        if reinvestment_rate is None:
            reinvestment_rate = self.reinvestment_rate

        if cash_flow is None:
            cash_flow = self.cash_flow(
                installation_start_date=kwargs.get("installation_start_date", None),
                project_capex_date=kwargs.get("project_capex_date", None),
                soft_capex_date=kwargs.get("soft_capex_date", None),
                system_capex_date=kwargs.get("system_capex_date", None),
                turbine_capex_date=kwargs.get("turbine_capex_date", None),
                offtake_price=offtake_price,
            )
        if finance_rate is None or reinvestment_rate is None:
            return npf.irr(cash_flow.cash_flow.values)
        return npf.mirr(cash_flow.cash_flow.values, finance_rate, reinvestment_rate)

    def lcoe(
        self,
        fixed_charge_rate: float | None = None,
        capex: float | None = None,
        opex: float | None = None,
        aep: float | None = None,
    ) -> float:
        """Calculates the levelized cost of energy (LCOE) as the following:
        LCOE = (CapEx * FCR + OpEx) / AEP, in $/MWh.

        Parameters
        ----------
        fixed_charge_rate : float, optional
            Revenue per amount of investment required to cover the investment cost. Required if no
            value was provided in the ``Project`` configuration. Defaults to None
        capex : float, optional
            Custom CapEx value, in $/kW. Defaults to None.
        opex : float, optional
            Custom OpEx value, in $/kW/year. Defaults to None.
        aep : float, optional
            Custom AEP value, in MWh/MW/year. Defaults to None.

        Returns
        -------
        float
            The levelized cost of energy.

        Raises
        ------
        ValueError
            Raised if the input to :py:attr:`units` is not one of "kw", "mw", or "gw".
        """
        # Check that the offtake price exists
        if fixed_charge_rate is None:
            if (fixed_charge_rate := self.fixed_charge_rate) is None:
                raise ValueError(
                    "`fixed_charge_rate` wasn't defined in the Project settings or in the method"
                    " keyword arguments."
                )

        # Check for custom inputs, otherwise compute the necessary metrics
        # CapEx: $/kW; OpEx: $/kW; AEP: MWh/MW -> LCOE: ($/kW/yr + $/kW/yr) / (MWh/MW): $/MW
        capex = self.capex(per_capacity="kw") if capex is None else capex
        opex = self.opex(per_capacity="kw") if opex is None else opex
        if aep is None:
            aep = self.energy_production(units="mw", per_capacity="mw", aep=True, with_losses=True)
        if TYPE_CHECKING:
            assert isinstance(capex, float) and isinstance(opex, float) and isinstance(aep, float)
        return (capex * self.fixed_charge_rate + opex / self.operations_years) / (aep / 1000)

    def generate_report(
        self,
        metrics_configuration: dict[str, dict] | None = None,
        simulation_name: str | None = None,
    ) -> pd.DataFrame:
        """Generates a single row dataframe of all the desired resulting metrics from the project.

        .. note:: This assumes all results will be a single number, and not a Pandas ``DataFrame``

        Parameters
        ----------
        metrics_dict : dict[str, dict], optional
            The dictionary of dictionaries containing the following key, value pair pattern::

                {
                    "Descriptive Name (units)": {
                        "metric": "metric_method_name",
                        "kwargs": {"kwarg1": "kwarg_value_1"}  # Exclude if not needed
                    }
                }

            For metrics that have no keyword arguments, or where the default parameter values are
            desired, either an empty dictionary or no dictionary input for "kwargs" is allowed. If
            no input is provided, then :py:attr:`report_config` will be used to populate
        simulation_name : str
            The name that should be given to the resulting index.

        Returns
        -------
        pd.DataFrame
            A pandas.DataFrame containing all of the provided outputs defined in
            :py:attr:`metrics_dict`.

        Raises
        ------
        ValueError
            Raised if any of the keys of :py:attr:`metrics_dict` aren't implemented methods.
        """
        if metrics_configuration is None:
            if self.report_config is None:
                raise ValueError(
                    "Either a `report_config` must be provided to the class, or"
                    " `metrics_configuration` as a method argument."
                )
            metrics_configuration = {k: v for k, v in self.report_config.items() if k != "name"}
        if simulation_name is None:
            if TYPE_CHECKING:
                assert isinstance(self.report_config, dict)
            if "name" not in self.report_config:
                raise ValueError(
                    "Either a `name` key, value pair must be provided in"
                    " `report_config`, or `name` as a method argument."
                )
            simulation_name = self.report_config["name"]  # type: ignore

        invalid_metrics = [
            el["metric"] for el in metrics_configuration.values() if not hasattr(self, el["metric"])
        ]
        if invalid_metrics:
            names = "', '".join(invalid_metrics)
            raise ValueError(f"None of the following are valid metrics: '{names}'.")

        results = {
            name: getattr(self, val["metric"])(**val.get("kwargs", {}))
            for name, val in metrics_configuration.items()
        }
        results_df = pd.DataFrame.from_dict(results, orient="index").T
        results_df.index = pd.Index([simulation_name])
        results_df.index.name = "Project"
        return results_df
