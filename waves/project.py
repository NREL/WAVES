"""Provides the ``Project`` class that ties to together ORBIT (CapEx), WOMBAT (OpEx), and
FLORIS (AEP) simulation libraries for a simplified modeling workflow.
"""

from __future__ import annotations

import re
import json
import math
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
from floris import WindRose, TimeSeries, FlorisModel
from landbosse import landbosse_runner  # noqa: E402
from wombat.core import Simulation
from wombat.core.data_classes import FromDictMixin

from waves.utilities import (
    load_yaml,
    resolve_path,
    check_monthly_wind_rose,
    create_monthly_wind_rose,
    calculate_monthly_wind_rose_results,
)
from waves.utilities.met_data import compute_shear, extrapolate_windspeed, fit_weibull_distribution
from waves.utilities.validators import check_dict_consistency, validate_common_inputs


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
        .resample("h")
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
    turbine_type : str
        The type of wind turbine used. Must be one of "land", "fixed", or "floating".
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
    floris_turbulence_intensity : str | None
        The turbulence intensity column in ``weather`` that will be used for the FLORIS
        wake analysis. If None, then the input from ``floris_config`` will be used. Defaults to
        "None".
    floris_x_col : str
        The column of x-coordinates in the WOMBAT layout file that corresponds to
        the ``Floris.farm.layout_x`` Defaults to "floris_x".
    floris_y_col : str
        The column of x-coordinates in the WOMBAT layout file that corresponds to
        the ``Floris.farm.layout_y`` Defaults to "floris_y".
    orbit_config : str | pathlib.Path | dict | None
        The ORBIT configuration file name or dictionary. If None, will not set up
        the ORBIT simulation component.
    landbosse_config : str | pathlib.Path | dict | None
        The LandBOSSE configuration file name or dictionary. If None, will not set up
        the LandBOSSE simulation component.
    wombat_config : str | pathlib.Path | dict | None
        The WOMBAT configuration file name or dictionary. If None, will not set up
        the WOMBAT simulation component.
    floris_config : str | pathlib.Path | dict | None
        The FLORIS configuration file name or dictionary. If None, will not set up
        the FLORIS simulation component.

        .. note:: The farm:trubine_library_path is automatically set to use the
            :py:attr:`library_path` / turbines folder to maintain consistency.

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
    environmental_loss_ratio : float, optional
        Percentage of environmental losses to deduct from the total energy production. Should be
        represented as a decimal in the range of [0, 1]. Defaults to 0.0159.
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
    turbine_type: str = field(converter=str.lower, validator=attrs.validators.instance_of(str))
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
    landbosse_config: str | Path | dict | None = field(
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
    floris_turbulence_intensity: str | None = field(default=None)
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
    environmental_loss_ratio: float = field(
        default=0.0159, validator=attrs.validators.instance_of((float, type(None)))
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
    landbosse_config_dict: dict = field(factory=dict, init=False)
    wombat_config_dict: dict = field(factory=dict, init=False)
    floris_config_dict: dict = field(factory=dict, init=False)
    wombat: Simulation = field(init=False)
    orbit: ProjectManager = field(init=False)
    landbosse: landbosse_runner.LandBOSSERunner = field(init=False)
    floris: FlorisModel = field(init=False)
    project_wind_rose: WindRose = field(init=False)
    monthly_wind_rose: WindRose = field(init=False)
    floris_turbine_order: list[str] = field(init=False, factory=list)
    turbine_potential_energy: pd.DataFrame = field(init=False)
    turbine_production_energy: pd.DataFrame = field(init=False)
    project_potential_energy: pd.DataFrame = field(init=False)
    project_production_energy: pd.DataFrame = field(init=False)
    _fi_dict: dict[tuple[int, int], FlorisModel] = field(init=False, factory=dict)
    operations_start: pd.Timestamp = field(init=False)
    operations_end: pd.Timestamp = field(init=False)
    operations_years: int = field(init=False)

    def __attrs_post_init__(self) -> None:
        """Post-initialization hook to complete the setup."""
        if isinstance(self.weather_profile, str | Path):
            weather_path = self.library_path / "weather" / self.weather_profile
            self.weather = load_weather(weather_path)

        self.setup_orbit()
        self.setup_landbosse()
        self.setup_wombat()
        self.setup_floris()

        self.check_consistent_config()

        if self.connect_floris_to_layout:
            self.connect_floris_to_turbines()
        if self.connect_orbit_array_design and self.orbit_config is not None:
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

    @turbine_type.validator  # type: ignore
    def validate_turbine_type(self, attribute: attrs.Attribute, value: str) -> None:
        """Validates that :py:attr`turbine_type` is one of "land", "fixed", or "floating".

        Parameters
        ----------
        attribute : attrs.Attribute
            The attrs Attribute information/metadata/configuration.
        value : str
            The user input.

        Raises
        ------
        ValueError
            Raised if not one of "land", "fixed", or "floating".
        """
        if value not in ("land", "fixed", "floating"):
            raise ValueError(f"{attribute.name} must be one 'land', 'fixed', or floating'.")

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
            raise KeyError("A key, value pair for `name` must be provided for the simulation name.")

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
            "landbosse_config": self.landbosse_config_dict,
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

    def check_consistent_config(self) -> None:
        """Check the configurations of the models after they have been set up
        but before running the model to ensure the basicinputs to each model are
        consistent.
        """
        num_turbines = {}
        project_capacity = {}
        num_substations = {}

        if bool(self.orbit_config_dict):
            num_turbines["orbit"] = self.orbit_config_dict["plant"]["num_turbines"]
            orbit_num_substation = self.orbit_config_dict.get("oss_design", {}).get(
                "num_substations"
            )
            if orbit_num_substation is not None:
                num_substations["orbit"] = orbit_num_substation

        if bool(self.landbosse_config_dict):
            num_turbines["landbosse"] = self.landbosse_config_dict["num_turbines"]
            project_capacity["landbosse"] = (
                self.landbosse_config_dict["turbine_rating_MW"] * num_turbines["landbosse"]
            )
            num_substations["landbosse"] = 1

        if bool(self.wombat_config_dict):
            num_turbines["wombat"] = len(self.wombat.windfarm.turbine_id)
            project_capacity["wombat"] = self.wombat.windfarm.capacity / 1000.0
            num_substations["wombat"] = len(self.wombat.windfarm.substation_id)

        if bool(self.floris_config_dict):
            num_turbines["floris"] = self.floris.n_turbines

        check_dict_consistency(num_turbines, "num_turbines")
        check_dict_consistency(project_capacity, "project_capacity")
        check_dict_consistency(num_substations, "num_substations")

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

    def setup_landbosse(self) -> None:
        """Creates the LandBOSSE runner object and readies it for running an analysis."""
        if self.landbosse_config is None:
            print("No LandBOSSE configuration provided, skipping model setup.")
            return

        if isinstance(self.landbosse_config, str | Path):
            landbosse_config = self.library_path / "project/config" / self.landbosse_config
            self.landbosse_config_dict = load_config(landbosse_config)
        else:
            self.landbosse_config_dict = self.landbosse_config

        if self.report_config is not None:
            self.landbosse_config_dict["id"] = self.report_config["name"]
        else:
            self.landbosse_config_dict["id"] = ""

        landbosse_table_path = (
            self.library_path / "project/config" / self.landbosse_config_dict["data_tables"]
        )
        landbosse_table_dict = pd.read_excel(landbosse_table_path, sheet_name=None)
        self.landbosse_config_dict["data_table"] = landbosse_table_dict

        if TYPE_CHECKING:
            assert isinstance(self.weather, pd.DataFrame)  # mypy helper

        self.landbosse = landbosse_runner.LandBOSSERunner(
            input_config=self.landbosse_config_dict,
            weather=self.weather,
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

        # Ensure the project turbine library is used for the floris turbine library
        turbine_library = self.library_path / "turbines"
        self.floris_config_dict["farm"]["turbine_library_path"] = turbine_library

        # Check that an array of turbulence intensities is provided in the weather profile to ensure
        # the wind rose and time series methods don't break during runtime.
        if self.floris_turbulence_intensity is None:
            ti = self.floris_config_dict["flow_field"]["turbulence_intensities"]
            if len(ti) > 1:
                msg = (
                    "Include floris:flow_field:turbulence_intensities in the weather profile for"
                    " more than a single value."
                )
                raise ValueError(msg)

        self.floris = FlorisModel(configuration=self.floris_config_dict)

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

        # Unset the ORBIT settings to ensure the design result isn't double counted
        self.setup_orbit()

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
        self.floris.set(layout_x=x, layout_y=y)
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

    def run_wind_rose_aep(
        self,
        full_wind_rose: bool = False,
        set_kwargs: dict | None = None,
    ):
        """Runs the custom FLORIS WindRose AEP methodology that allows for gathering of
        intermediary results.

        Parameters
        ----------
        full_wind_rose : bool, optional
            If True, the full wind profile will be used, otherwise, if False, the wind profile will
            be limited to just the simulation period. Defaults to False.
        set_kwargs : dict | None, optional
            Arguments that are provided to ``FlorisInterface.get_farm_AEP_wind_rose_class()``.
            Defaults to None.

            From FLORIS:

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
        """
        if set_kwargs is None:
            set_kwargs = {}

        if TYPE_CHECKING:
            assert isinstance(self.weather, pd.DataFrame)  # mypy helper

        weather = self.weather.copy()

        if not full_wind_rose:
            weather = weather.loc[self.operations_start : self.operations_end]

        if self.floris_turbulence_intensity is None:
            ti, *_ = self.floris_config_dict["flow_field"]["turbulence_intensities"]
            ti_col = "floris_turbulence_intensities"
            weather.loc[:, ti_col] = ti
        else:
            ti_col = self.floris_turbulence_intensity
        weather = weather.rename(
            columns={self.floris_wind_direction: "wd", self.floris_windspeed: "ws", ti_col: "ti"}
        )[["wd", "ws", "ti"]]

        # recreate the FlorisModel object for the wind rose settings
        wd, ws, ti = weather.values.T
        project_time_series = TimeSeries(
            wind_speeds=ws,
            wind_directions=wd,
            turbulence_intensities=ti,
        )
        self.project_wind_rose = project_time_series.to_WindRose()
        self.monthly_wind_rose = create_monthly_wind_rose(weather)
        self.monthly_wind_rose = check_monthly_wind_rose(
            self.project_wind_rose, self.monthly_wind_rose
        )

        self.floris.set(wind_data=self.project_wind_rose)

        # Table of frequencies (wind directions x wind speeds)
        freq_monthly = {k: wr.freq_table for k, wr in self.monthly_wind_rose.items()}

        yaw_angles = set_kwargs.get("yaw_angles", None)
        self.floris.set(yaw_angles=yaw_angles)

        # Calculate the potential energy
        self.floris.run_no_wake()
        turbine_potential_power = self.floris.get_turbine_powers()
        if (weights := set_kwargs["turbine_weights"]) is not None:
            turbine_potential_power *= weights

        # Calculate the monthly potential contribution to AEP from the wind rose
        self.turbine_potential_energy = calculate_monthly_wind_rose_results(
            turbine_potential_power, freq_monthly
        )
        self.turbine_potential_energy.columns = self.floris_turbine_order  # Mwh
        self.project_potential_energy = self.turbine_potential_energy.values.sum()

        # Calculate the produced power
        self.floris.run()

        turbine_production_power = self.floris.get_turbine_powers()
        if (weights := set_kwargs["turbine_weights"]) is not None:
            turbine_production_power *= weights

        # Calculate the monthly contribution to AEP from the wind rose
        self.turbine_production_energy = calculate_monthly_wind_rose_results(
            turbine_production_power, freq_monthly
        )
        self.turbine_production_energy.columns = self.floris_turbine_order  # Mwh
        self.project_production_energy = self.turbine_production_energy.values.sum()

    def run_floris(self, set_kwargs: dict | None = None, full_wind_rose: bool = False) -> None:
        """Runs either a FLORIS wind rose analysis for a simulation-level AEP value
        (``which="wind_rose"``) or a turbine-level time series for the WOMBAT simulation
        period (``which="time_series"``).

        Parameters
        ----------
        set_kwargs : dict | None, optional
            Any keyword arguments to be assed to ``FlorisInterface.reinitialize()``. Defaults to
            None.
        full_wind_rose : bool, optional
            Indicates, for "wind_rose" analyses ONLY, if the full weather profile from ``weather``
            (True) or the limited, WOMBAT simulation period (False) should be used for analyis.
            Defaults to False.

        Raises
        ------
        ValueError: Raised if :py:attr:`which` is not one of "wind_rose" or "time_series".
        """
        if set_kwargs is None:
            set_kwargs = {}

        # Set the FLORIS defaults
        set_kwargs.setdefault("turbine_weights", None)
        set_kwargs.setdefault("yaw_angles", None)
        set_kwargs.setdefault("no_wake", False)

        self.run_wind_rose_aep(full_wind_rose=full_wind_rose, set_kwargs=set_kwargs)

    def run(
        self,
        floris_kwargs: dict | None = None,
        full_wind_rose: bool = False,
        skip: list[str] | None = None,
    ) -> None:
        """Run all three models in serial, or a subset if ``skip`` is used.

        Parameters
        ----------
        floris_kwargs : dict | None
            Any additional ``FlorisModel.set`` keyword arguments. Defaults to None.
        full_wind_rose : bool, optional
            Indicates, for "wind_rose" analyses ONLY, if the full weather profile from ``weather``
            (True) or the limited, WOMBAT simulation period (False) should be used for analyis.
            Defaults to False.
        skip : list[str] | None, optional
            A list of models to be skipped. This is intended to be used after a model is
            reinitialized with a new or modified configuration. Defaults to None.

        """
        if floris_kwargs is None:
            floris_kwargs = {}
        if skip is None:
            skip = []

        if self.orbit_config is not None and "orbit" not in skip:
            self.orbit.run()
        elif self.landbosse_config is not None and "landbosse" not in skip:
            self.landbosse.run()
        if "wombat" not in skip:
            self.wombat.run()
        if "floris" not in skip:
            self.run_floris(set_kwargs=floris_kwargs, full_wind_rose=full_wind_rose)

    def reinitialize(
        self,
        orbit_config: str | Path | dict | None = None,
        landbosse_config: str | Path | dict | None = None,
        wombat_config: str | Path | dict | None = None,
        floris_config: str | Path | dict | None = None,
    ) -> None:
        """Enables a user to reinitialize one or multiple of the CapEx, OpEx, and AEP
        models.

        Parameters
        ----------
        orbit_config : str | Path | dict | None, optional
            ORBIT configuration file or dictionary. Defaults to None.
        landbosse_config : (str | Path | dict | None, optional
            LandBOSSE configuration file or dictionary. Defaults to None.
        wombat_config : str | Path | dict | None, optional
            WOMBAT configuation file or dictionary. Defaults to None.
        floris_config : (str | Path | dict | None, optional
            FLORIS configuration file or dictionary. Defaults to None.

        Raises
        ------
        RuntimeError
            Raised if neither ORBIT nor LandBOSSE are configured to run.
        """
        if orbit_config is not None:
            self.orbit_config = orbit_config
            self.setup_orbit()
        elif landbosse_config is not None:
            self.landbosse_config = landbosse_config
            self.setup_landbosse()
        if wombat_config is not None:
            self.wombat_config = wombat_config
            self.setup_wombat()

        if floris_config is not None:
            self.floris_config = floris_config
            self.setup_floris()

        self.check_consistent_config()

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
        elif self.landbosse_config is not None:
            return self.landbosse.result.project_parameters["Number of turbines"]
        if self.wombat_config is not None:
            return len(self.wombat.windfarm.turbine_id)
        if self.floris_config is not None:
            return self.floris.n_turbines
        raise RuntimeError("No models were provided, cannot calculate value.")

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
        elif self.landbosse_config is not None:
            return self.landbosse.result.project_parameters["Turbine rating MW"]
        if self.wombat_config is not None:
            return self.wombat.windfarm.capacity / 1000 / self.n_turbines()
        raise RuntimeError("No models were provided, cannot calculate value.")

    def n_substations(self) -> int:
        """Calculates the number of substations in the project.

        Returns
        -------
        int
            The number of substations in the project.
        """
        if self.orbit_config is not None and "OffshoreSubstationDesign" not in self.orbit._phases:
            return self.orbit_config_dict["oss_design"]["num_substations"]
        elif self.landbosse_config is not None:
            return 1
        if self.wombat_config is not None:
            return len(self.wombat.windfarm.substation_id)
        raise RuntimeError("No models were provided, cannot calculate value.")

    @validate_common_inputs(which=["units"])
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
        """
        if self.orbit_config is not None:
            capacity = self.orbit.capacity
        elif self.landbosse_config is not None:
            capacity = (
                self.landbosse.result.project_parameters["Turbine rating MW"]
                * self.landbosse.result.project_parameters["Number of turbines"]
            )
        elif self.wombat_config is not None:
            capacity = self.wombat.windfarm.capacity / 1000
        else:
            raise RuntimeError("No models were provided, cannot calculate value.")

        units = units.lower()
        if units == "kw":
            return capacity * 1000
        if units == "gw":
            return capacity / 1000
        return capacity

    @validate_common_inputs(which=["per_capacity"])
    def capex(
        self, breakdown: bool = False, per_capacity: str | None = None
    ) -> pd.DataFrame | float:
        """Calculate capital expenditure (CapEx) using ORBIT or LandBOSSE outputs.

        Parameters
        ----------
        breakdown : bool, optional
            Provide a detailed view of the CapEx breakdown, and a total, which is
            the sum of the BOS, turbine, project, and soft CapEx categories. Defaults to False.
        per_capacity : str, optional
            Provide the CapEx normalized by the project's capacity, in the desired units. If
            None, then the unnormalized CapEx is returned, otherwise it must be one of "kw",
            "mw", or "gw". Defaults to None.

        Raises
        ------
        RuntimeError
            Raised if neither ORBIT nor LandBOSSE have been run.

        Returns
        -------
        pd.DataFrame | float
            Project CapEx, normalized by :py:attr:`per_capacity`, if using, as either a
            pandas DataFrame if :py:attr:`breakdown` is True, otherwise, a float total.
        """
        if self.orbit_config is not None:
            capex = self.capex_orbit(breakdown=breakdown)
        elif self.landbosse_config is not None:
            capex = self.capex_landbosse(breakdown=breakdown)
        else:
            raise RuntimeError("Must run ORBIT or LandBOSSE to calculate capex.")

        if per_capacity is None:
            if breakdown:
                return capex
            return capex.values[0, 0]

        capacity = self.capacity(per_capacity)
        unit_map = {"kw": "kW", "mw": "MW", "gw": "GW"}
        capex[f"CapEx per {unit_map[per_capacity]}"] = capex / capacity

        if breakdown:
            return capex
        return capex.values[0, 1]

    def capex_orbit(self, *, breakdown: bool = False) -> pd.DataFrame:
        """Provides a thin wrapper to ORBIT's ``ProjectManager`` CapEx calculations that
        can provide a breakdown of total or normalize it by the project's capacity, in MW.

        Parameters
        ----------
        breakdown : bool, optional
            Provide a detailed view of the CapEx breakdown, and a total, which is
            the sum of the BOS, turbine, project, and soft CapEx categories. Defaults to False.

        Returns
        -------
        pd.DataFrame
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

        return capex

    def capex_landbosse(self, *, breakdown: bool = False) -> pd.DataFrame | float:
        """Calculates project CapEx from LandBOSSE result, in MW.

        Parameters
        ----------
        breakdown : bool, optional
            Provide a detailed view of the CapEx breakdown, and a total, which is
            the sum of the BOS, turbine, project, and soft CapEx categories. Defaults to False.

        Returns
        -------
        pd.DataFrame | float
            Project CapEx, normalized by :py:attr:`per_capacity`, if using, as either a
            pandas DataFrame if :py:attr:`breakdown` is True, otherwise, a float total.
        """
        capex_breakdown = (
            self.landbosse.result.operation_cost.groupby("operation_id")["cost_per_project"]
            .sum()
            .rename("CapEx")
        )
        capex_total = capex_breakdown.sum()
        if breakdown:
            capex = capex_breakdown
            capex.loc["Total"] = capex_total
            capex = capex.to_frame()
        else:
            capex = pd.DataFrame(data=[capex_total], columns=["CapEx"], index=pd.Index(["Total"]))

        return capex

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
        RuntimeError
            Raised if neither ORBIT nor LandBOSSE have been run.
        """
        if self.orbit_config is not None:
            if "ArraySystemDesign" in self.orbit._phases:
                array = self.orbit._phases["ArraySystemDesign"]
            elif "CustomArraySystemDesign" in self.orbit._phases:
                array = self.orbit._phases["CustomArraySystemDesign"]
            else:
                raise ValueError("No array system design was included in the ORBIT configuration.")

            # TODO: Fix ORBIT bug for nansum
            return np.nansum(array.sections_cable_lengths)

        elif self.landbosse_config is not None:
            landbosse_vars = self.landbosse.result.model_variables
            return landbosse_vars.loc[
                landbosse_vars["variable_df_key_col_name"] == "Total cable length",
                "value",
            ].squeeze()

        else:
            raise RuntimeError("Must use either ORBIT or LandBOSSE to estimate array cable length")

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
        RuntimeError
            Raised if neither ORBIT nor LandBOSSE have been run.
        """
        if self.orbit_config is not None:
            try:
                return self.orbit._phases["ExportSystemDesign"].total_length
            except KeyError:
                return self.orbit._phases["ElectricalDesign"].total_length
            except KeyError:
                raise ValueError(
                    "Neither an `ElectricalDesign` nor an `ExportSystemDesign` phase were defined"
                    " to be able to calculate this metric."
                )

        elif self.landbosse_config is not None:
            landbosse_vars = self.landbosse.result.model_variables
            return landbosse_vars.loc[
                landbosse_vars["variable_df_key_col_name"] == "Cable Length to Substation (km)",
                "value",
            ].squeeze()

        else:
            raise RuntimeError("Must use either ORBIT or LandBOSSE to estimate export cable length")

    # Operational metrics

    @validate_common_inputs(which=["frequency", "by", "units"])
    def _get_floris_energy(
        self, which: str, frequency: str = "project", by: str = "windfarm", units: str = "gw"
    ) -> pd.DataFrame | float:
        unit_map = {"kw": "kWh", "mw": "MWh", "gw": "GWh"}

        # Check the floris basis
        if which not in ("potential", "production"):
            raise ValueError("`which` should be one of 'potential' or 'production'")

        if which == "potential":
            energy = self.turbine_potential_energy
        else:
            energy = self.turbine_production_energy
        if by == "windfarm":
            energy = energy.sum(axis=1).to_frame(f"Energy {which.title()} ({unit_map[units]})")

        # Use WOMBAT availability for the base index
        base_ix = self.wombat.metrics.production_based_availability(
            frequency="month-year", by="turbine"
        ).index

        energy_gwh = pd.DataFrame(0.0, dtype=float, index=base_ix, columns=["drop"])
        energy_gwh = energy_gwh.merge(
            energy,
            how="left",
            left_on="month",
            right_index=True,
        ).drop(labels=["drop"], axis=1)
        energy_gwh /= 1000

        # Aggregate to the desired frequency level (nothing required for month-year)
        if frequency == "annual":
            energy_gwh = (
                energy_gwh.reset_index(drop=False).groupby("year").sum().drop(columns=["month"])
            )
        elif frequency == "project":
            if by == "turbine":
                energy_gwh = (
                    energy_gwh.sum(axis=0)
                    .to_frame(name=f"Energy {which.title()} ({unit_map[units]})")
                    .T
                )
            else:
                energy_gwh = energy_gwh.values.sum()

        # Convert the  units
        if units == "kw":
            return energy_gwh * 1e6
        if units == "mw":
            return energy_gwh * 1e3
        return energy_gwh

    @validate_common_inputs(which=["frequency", "by", "units", "per_capacity"])
    def energy_potential(
        self,
        frequency: str = "project",
        by: str = "windfarm",
        units: str = "gw",
        per_capacity: str | None = None,
        aep: bool = False,
    ) -> pd.DataFrame | float:
        """Computes the potential energy production, or annual potential energy production, in GWh,
        for the simulation by extrapolating the monthly contributions to AEP if FLORIS (without
        wakes) results were computed by a wind rose, or using the time series results.

        Parameters
        ----------
        frequency : str, optional
            One of "project" (project total), "annual" (annual total), or "month-year"
            (monthly totals for each year).
        by : str, optional
            One of "windfarm" (project level) or "turbine" (turbine level) to
            indicate what level to calculate the energy production.
        units : str, optional
            One of "gw", "mw", or "kw" to determine the units for energy production.
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
        if aep and frequency != "project":
            raise ValueError("`aep` can only be set to True, if `frequency`='project'.")

        energy = self._get_floris_energy(
            which="potential",
            frequency=frequency,
            by=by,
            units=units,
        )

        if aep:
            energy /= self.operations_years

        if per_capacity is None:
            return energy
        return energy / self.capacity(per_capacity)

    @validate_common_inputs(which=["frequency", "by", "units", "per_capacity"])
    def energy_production(
        self,
        frequency: str = "project",
        by: str = "windfarm",
        units: str = "gw",
        per_capacity: str | None = None,
        environmental_loss_ratio: float | None = None,
        aep: bool = False,
    ) -> pd.DataFrame | float:
        """Computes the energy production, or annual energy production.

        To compute the losses, the total loss ratio  from :py:method:`loss_ratio` is applied
        to each time step (e.g., month for month-year) rather than using each time  step's losses.
        This lower resolution apporach to turbine and time step energy stems from the nature of the
        loss method itself, which is intended for farm-level results.

        Parameters
        ----------
        frequency : str, optional
            One of "project" (project total), "annual" (annual total), or "month-year"
            (monthly totals for each year).
        by : str, optional
            One of "windfarm" (project level) or "turbine" (turbine level) to
            indicate what level to calculate the energy production.
        units : str, optional
            One of "gw", "mw", or "kw" to determine the units for energy production.
        per_capacity : str, optional
            Provide the energy production normalized by the project's capacity, in the desired
            units. If None, then the unnormalized energy production is returned, otherwise it must
            be one of "kw", "mw", or "gw". Defaults to None.
        environmental_loss_ratio : float, optional
            The decimal environmental loss ratio to apply to the energy production. If None, then
            it will attempt to use the :py:attr:`environmental_loss_ratio` provided in the Project
            configuration. Defaults to 0.0159.
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
        if aep and frequency != "project":
            raise ValueError("`aep` can only be set to True, if `frequency`='project'.")

        energy = self.energy_potential(
            frequency=frequency,
            by=by,
            units=units,
        )

        if TYPE_CHECKING:
            assert isinstance(energy, pd.DataFrame)

        # Convert naming from energy_potential()
        if by == "windfarm":
            if frequency != "project":
                energy.columns = energy.columns.str.replace("Potential", "Production")
        if frequency == "project":
            if by == "turbine":
                energy.index = energy.index.str.replace("Potential", "Production")

        energy *= 1 - self.loss_ratio(environmental_loss_ratio)

        if aep:
            energy /= self.operations_years

        if per_capacity is None:
            return energy

        return energy / self.capacity(per_capacity)

    @validate_common_inputs(which=["frequency", "by", "units", "per_capacity"])
    def energy_losses(
        self,
        frequency: str = "project",
        by: str = "windfarm",
        units: str = "gw",
        per_capacity: str | None = None,
        environmental_loss_ratio: float | None = None,
        aep: bool = False,
    ) -> pd.DataFrame:
        """Computes the energy losses for the simulation by subtracting the energy production from
        the potential energy production.

        To compute the losses, the total loss ratio  from :py:method:`loss_ratio` is applied
        to each time step (e.g., month for month-year) rather than using each time  step's losses.
        This lower resolution apporach to turbine and time step energy stems from the nature of the
        loss method itself, which is intended for farm-level results.

        Parameters
        ----------
        frequency : str, optional
            One of "project" (project total), "annual" (annual total), or "month-year"
            (monthly totals for each year).
        by : str, optional
            One of "windfarm" (project level) or "turbine" (turbine level) to
            indicate what level to calculate the energy production.
        units : str, optional
            One of "gw", "mw", or "kw" to determine the units for energy production.
        per_capacity : str, optional
            Provide the energy production normalized by the project's capacity, in the desired
            units. If None, then the unnormalized energy production is returned, otherwise it must
            be one of "kw", "mw", or "gw". Defaults to None.
        environmental_loss_ratio : float, optional
            The decimal environmental loss ratio to apply to the energy production. If None, then
            it will attempt to use the :py:attr:`environmental_loss_ratio` provided in the Project
            configuration. Defaults to 0.0159.
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
        if aep and frequency != "project":
            raise ValueError("When `aep=True`, `frequency` must be 'project'.")
        potential = self.energy_potential(
            frequency=frequency,
            by=by,
            units=units,
        )

        # Compute total loss ratio
        total_loss_ratio = self.loss_ratio(environmental_loss_ratio=environmental_loss_ratio)
        losses = potential * total_loss_ratio

        if TYPE_CHECKING:
            assert isinstance(losses, pd.DataFrame)

        # Convert naming from energy_potential()
        if by == "windfarm":
            if frequency != "project":
                losses.columns = losses.columns.str.replace("Potential", "Losses")
        if frequency == "project":
            if by == "turbine":
                losses.index = losses.index.str.replace("Potential", "Losses")

        if aep:
            losses /= self.operations_years

        if per_capacity is None:
            return losses

        return losses / self.capacity(per_capacity)

    @validate_common_inputs(which=["frequency", "by", "units", "per_capacity"])
    def wake_losses(
        self,
        frequency: str = "project",
        by: str = "windfarm",
        units: str = "kw",
        per_capacity: str | None = None,
        aep: bool = False,
        ratio: bool = False,
    ) -> pd.DataFrame | float:
        """Computes the wake losses, in GWh, for the simulation by extrapolating the
        monthly wake loss contributions to AEP if FLORIS (with wakes) results were
        computed by a wind rose, or using the time series results.

        Parameters
        ----------
        frequency : str, optional
            One of "project" (project total), "annual" (annual total), or "month-year"
            (monthly totals for each year).
        by : str, optional
            One of "windfarm" (project level) or "turbine" (turbine level) to
            indicate what level to calculate the wake losses.
        per_capacity : str, optional
            Provide the wake losses normalized by the project's capacity, in the desired
            units. If None, then the unnormalized energy production is returned, otherwise it must
            be one of "kw", "mw", or "gw". Defaults to None.
        aep : bool, optional
            Flag to return the wake losses normalized by the number of years the plan is in
            operation. Note that :py:attr:`frequency` must be "project" for this to be computed.
        ratio : bool, optional
            Flag to return the wake loss ratio or the ratio of wake losses to the potential energy
            generation.

        Raises
        ------
        ValueError
            Raised if ``frequency`` is not one of: "project", "annual", "month-year".

        Returns
        -------
        pd.DataFrame | float
            The wind farm-level losses, in GWh, for the desired ``frequency``.
        """
        unit_map = {"kw": "kWh", "mw": "MWh", "gw": "GWh"}

        # Use WOMBAT availability for the base index
        potential_energy = self._get_floris_energy(
            which="potential",
            frequency=frequency,
            by=by,
            units=units,
        )
        waked_energy = self._get_floris_energy(
            which="production",
            frequency=frequency,
            by=by,
            units=units,
        )
        if TYPE_CHECKING:
            assert isinstance(potential_energy, pd.DataFrame)
            assert isinstance(waked_energy, pd.DataFrame)

        is_float = by == "windfarm" and frequency == "project"
        if is_float:
            losses = potential_energy - waked_energy
        else:
            losses = potential_energy - waked_energy.values
            losses.columns = losses.columns.str.replace("Potential", "Losses")
            try:
                losses.index = losses.index.str.replace("Potential", "Losses")
            except AttributeError:
                pass  # no need for multi index compatibility

        unit_map = {"kw": "kWh", "mw": "MWh", "gw": "GWh"}
        name = f"Energy Losses ({unit_map[units]})"

        if ratio:
            if is_float:
                return losses / potential_energy
            losses /= potential_energy.values
            losses.columns = losses.columns.str.replace(name, "Loss Ratio")
            try:
                losses.index = losses.index.str.replace(name, "Loss Ratio")
            except AttributeError:
                pass  # no need for multi index compatibility

        if aep:
            if frequency != "project":
                raise ValueError("`aep` can only be set to True, if `frequency`='project'.")
            losses /= self.operations_years

        if per_capacity is None:
            return losses
        return losses / self.capacity(per_capacity)

    def technical_loss_ratio(self) -> float:
        """Calculate technical losses based on the project type.

        This method is adopted from ORCA where a 1% hysterisis loss is applied for fixed-bottom
        turbines. For floating turbines, this is 1% hysterisis, 0.1% for onboard equipment, and 0.1%
        for rotor misalignment (0.01197901 total).

        Returns
        -------
        float
            The technical loss ratio.
        """
        if self.turbine_type == "floating":
            return 1 - (1 - 0.01) * (1 - 0.001) * (1 - 0.001)
        elif self.turbine_type == "fixed":
            return 0.01
        else:
            return 0.0

    def electrical_loss_ratio(self) -> float:
        """Calculate electrical losses based on ORBIT parameters.

        Returns
        -------
        float
            The electrical loss ratio.

        """
        if self.orbit_config is not None:
            depth = self.orbit_config_dict["site"]["depth"]
            distance_to_landfall = self.orbit_config_dict["site"]["distance_to_landfall"]
            # ORCA formula
            electrical_loss_ratio = (
                2.20224112
                + 0.000604121 * depth
                + 0.0407303367321603 * distance_to_landfall
                + -0.0003712532582 * distance_to_landfall**2
                + 0.0000016525338 * distance_to_landfall**3
                + -0.000000003547544 * distance_to_landfall**4
                + 0.0000000000029271 * distance_to_landfall**5
            ) / 100
        else:
            # https://www.nrel.gov/docs/fy22osti/78715.pdf Fig30
            electrical_loss_ratio = 0.028

        return electrical_loss_ratio

    def loss_ratio(
        self,
        environmental_loss_ratio: float | None = None,
        breakdown: bool = False,
    ) -> pd.DataFrame | float:
        """Calculate total losses based on environmental, availability, wake, technical, and
        electrical losses.

        .. note:: This method treats different types of losses as efficiencies and is applied
        as in Equation 1 from Beiter et al. 2020 (https://www.nrel.gov/docs/fy21osti/77384.pdf).

        Parameters
        ----------
        environmental_loss_ratio : float, optional
            The decimal environmental loss ratio to apply to the energy production. If None, then
            it will attempt to use the :py:attr:`environmental_loss_ratio` provided in the Project
            configuration. Defaults to 0.0159.
        breakdown : bool, optional
            Flag to return the losses breakdown including environmental, availability, wake,
            technical, electrical losses, and total losses.

        Returns
        -------
        float | pd.DataFrame
            The total loss ratio, or the DataFrame of all loss ratios.

        """
        # Check that the environmental loos ratio exists
        if environmental_loss_ratio is None:
            if (environmental_loss_ratio := self.environmental_loss_ratio) is None:
                raise ValueError(
                    "`environmental_loss_ratio` was neither defined in the Project settings nor"
                    " provided in the keyword arguments."
                )

        availability = 1 - self.availability(which="energy", frequency="project", by="windfarm")
        wake_loss_ratio = self.wake_losses(ratio=True)
        technical_loss_ratio = self.technical_loss_ratio()
        electrical_loss_ratio = self.electrical_loss_ratio()
        total_loss_ratio = 1 - (
            (1 - environmental_loss_ratio)
            * (1 - (availability))
            * (1 - wake_loss_ratio)
            * (1 - technical_loss_ratio)
            * (1 - electrical_loss_ratio)
        )
        if breakdown:
            loss_types = [
                "Environmental Losses",
                "Availability Losses",
                "Wake Losses",
                "Technical Losses",
                "Electrical Losses",
                "Total Losses",
            ]
            loss_breakdown = pd.DataFrame(
                [
                    100 * environmental_loss_ratio,
                    100 * availability,
                    100 * wake_loss_ratio,
                    100 * technical_loss_ratio,
                    100 * electrical_loss_ratio,
                    100 * total_loss_ratio,
                ],
                index=loss_types,
                columns=["Loss Ratio (%)"],
            )
            return loss_breakdown

        return total_loss_ratio

    @validate_common_inputs(which=["frequency", "by"])
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

    @validate_common_inputs(which=["frequency", "by"])
    def capacity_factor(
        self,
        which: str,
        frequency: str = "project",
        by: str = "windfarm",
        environmental_loss_ratio: float | None = None,
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

            .. note:: This will only be checked for :py:attr:`which` = "net".
        environmental_loss_ratio : float, optional
            The decimal environmental loss ratio to apply to the energy production. If None, then
            it will attempt to use the :py:attr:`environmental_loss_ratio` provided in the Project
            configuration. Defaults to 0.0159.

        Returns
        -------
        pd.DataFrame | float
            The capacity factor at the desired aggregation level.
        """
        which = which.lower().strip()
        if which not in ("net", "gross"):
            raise ValueError('``which`` must be one of "net" or "gross".')

        by_turbine = by == "turbine"

        numerator: pd.DataFrame
        if which == "net":
            numerator = (
                1 - self.loss_ratio(environmental_loss_ratio=environmental_loss_ratio)
            ) * self.energy_potential(
                frequency="month-year",
                by="turbine",
                units="kw",
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

    @validate_common_inputs(which=["frequency", "per_capacity"])
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

    @validate_common_inputs(which=["frequency"])
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
        # Check that an offtake_price exists
        if offtake_price is None:
            if (offtake_price := self.offtake_price) is None:
                raise ValueError(
                    "`offtake_price` wasn't defined in the Project settings or in the method"
                    " keyword arguments."
                )

        revenue = self.energy_production(frequency=frequency) * 1000 * offtake_price  # MWh
        if frequency != "project":
            if TYPE_CHECKING:
                assert isinstance(revenue, pd.DataFrame)
            revenue.columns = ["Revenue"]

        if per_capacity is None:
            return revenue

        per_capacity = per_capacity.lower()
        return revenue / self.capacity(per_capacity)

    @validate_common_inputs(which=["frequency"])
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

        capex_df["CapEx"] = capex_df.sum(axis=1).sort_index()
        if breakdown:
            return capex_df
        return capex_df.CapEx.to_frame()

    @validate_common_inputs(which=["frequency"])
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

    @validate_common_inputs(which=["frequency"])
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
            aep = self.energy_production(units="mw", per_capacity="mw", aep=True)
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

    def generate_report_lcoe_breakdown(self) -> pd.DataFrame:
        """Generates a dataframe containing the detailed breakdown of LCOE (Levelized Cost of
        Energy) metrics for the project, which is used to produce LCOE waterfall charts and
        CapEx donut charts in the Cost of Wind Energy Review. The breakdown includes the
        contributions of each CapEx and OpEx component (from ORBIT and WOMBAT) to the LCOE in
        $/MWh.

        This function calculates the LCOE by considering both CapEx (from ORBIT) and OpEx (from
        WOMBAT),and incorporates the fixed charge rate (FCR) and net annual energy production
        (net AEP) into the computation for each component.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the detailed LCOE breakdown with the following columns:
                - "Component": The name of the project component (e.g., "Turbine", "Balance
                  of System CapEx", "OpEx").
                - "Category": The category of the component (e.g., "Turbine", "Balance of System
                  CapEx", "Financial CapEx", "OpEx").
                - "Value ($/kW)": The value of the component in $/kW.
                - "Fixed charge rate (FCR) (real)": The real fixed charge rate (FCR) applied to the
                  component.
                - "Value ($/kW-yr)": The value of the component in $/kW-yr, after applying the FCR.
                - "Net AEP (MWh/kW/yr)": The net annual energy production (AEP) in MWh/kW/yr.
                - "Value ($/MWh)": The value of the component in $/MWh, calculated by dividing the
                  $/kW-yr value by the net AEP.

        Notes
        -----
        - CapEx components are categorized into "Turbine", "Balance of System CapEx", and "Financial
            CapEx".
        - OpEx components are derived from WOMBAT's OpEx metrics, categorized as "OpEx".
        - The LCOE is calculated by considering both CapEx and OpEx components, and adjusting for
          net AEP and FCR.
        - Rows with a value of 0 in the "Value ($/MWh)" column are removed to avoid clutter in
          annual reporting charts.
        """
        # Static values
        fcr = self.fixed_charge_rate
        net_aep = self.energy_production(units="mw", per_capacity="kw", aep=True)

        # Handle CapEx outputs from ORBIT
        try:
            capex_data = self.orbit.capex_detailed_soft_capex_breakdown_per_kw
        except AttributeError:
            capex_data = self.orbit.capex_breakdown_per_kw

        turbine_components = ("Turbine", "Nacelle", "Blades", "Tower", "RNA")
        financial_components = (
            "Construction",
            "Decommissioning",
            "Financing",
            "Contingency",
            "Soft",
        )
        columns = [
            "Component",
            "Category",
            "Value ($/kW)",
            "Fixed charge rate (FCR) (real)",
            "Value ($/kW-yr)",
            "Net AEP (MWh/kW/yr)",
            "Value ($/MWh)",
        ]

        df = pd.DataFrame.from_dict(capex_data, orient="index", columns=["Value ($/kW)"])
        df["Category"] = "BOS"
        df.loc[df.index.isin(turbine_components), "Category"] = "Turbine"
        df.loc[df.index.isin(financial_components), "Category"] = "Financial CapEx"
        df.Category = df.Category.str.replace("BOS", "Balance of System CapEx")

        df["Fixed charge rate (FCR) (real)"] = fcr
        df["Value ($/kW-yr)"] = df["Value ($/kW)"] * fcr
        df["Value ($/MWh)"] = df["Value ($/kW-yr)"] / net_aep
        df["Net AEP (MWh/kW/yr)"] = net_aep
        df = df.reset_index(drop=False)
        df = df.rename(columns={"index": "Component"})

        # Handle OpEx outputs from WOMBAT
        opex = (
            self.wombat.metrics.opex(frequency="annual", by_category=True)
            .mean(axis=0)
            .to_frame("Value ($/kW-yr)")
            .join(
                self.wombat.metrics.opex(frequency="annual", by_category=True)
                .sum(axis=0)
                .to_frame("Value ($/kW)")
            )
            .drop("OpEx")
        )
        opex /= self.capacity("kw")
        opex.index = opex.index.str.replace("_", " ").str.title()
        opex.index.name = "Component"
        opex["Category"] = "OpEx"
        opex["Fixed charge rate (FCR) (real)"] = fcr
        opex["Net AEP (MWh/kW/yr)"] = net_aep
        opex["Value ($/MWh)"] = opex["Value ($/kW-yr)"] / net_aep
        opex = opex.reset_index(drop=False)[columns]

        # Concatenate CapEx and OpEx rows
        df = pd.concat((df, opex)).reset_index(drop=True).reset_index(names=["Original Order"])

        # Define the desired order of categories for sorting
        order_of_categories = ["Turbine", "Balance of System CapEx", "Financial CapEx", "OpEx"]

        # Sort the dataframe based on the custom category order
        df["Category"] = pd.Categorical(
            df["Category"], categories=order_of_categories, ordered=True
        )
        df = (
            df.sort_values(by=["Category", "Original Order"])
            .drop(columns=["Original Order"])
            .reset_index(drop=True)
        )

        # Remove rows where 'Value ($/MWh)' is zero to avoid 0 values on annual reporting charts
        df = df[df["Value ($/MWh)"] != 0]

        # Re-order the columns so that it is more intuitive for the analyst
        df = df[
            [
                "Component",
                "Category",
                "Value ($/kW)",
                "Fixed charge rate (FCR) (real)",
                "Value ($/kW-yr)",
                "Net AEP (MWh/kW/yr)",
                "Value ($/MWh)",
            ]
        ]

        # Reset index and return the dataframe
        df = df.reset_index(drop=True)
        return df

    def generate_report_project_details(self) -> pd.DataFrame:
        """Generates a DataFrame containing detailed project information, following the format
        from the table at slide 64 in the Cost of Wind Energy Review: 2024 Edition
        (https://www.nrel.gov/docs/fy25osti/91775.pdf).

        This function collects various project parameters such as turbine specifications, wind speed
        data,energy capture, and efficiency metrics, and formats them into a comprehensive report.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the project details, where each row corresponds to a specific
            assumption and its associated unit and value. The columns include:
                - "Assumption": Describes the specific project assumption (e.g.,
                  "Wind plant capacity","Turbine rating").
                - "Units": The unit of measurement for each assumption (e.g., "MW", "Number",
                  "m/s").
                - "Value": The actual value for each assumption, computed based on the project's
                    configurations and available data.
        """
        # Define the project details template
        project_details = {
            "Assumption": [
                "Wind plant capacity",
                "Number of turbines",
                "Turbine rating",
                "Rotor diameter",
                "Hub height",
                "Specific power",
                "Water depth",
                "Substructure type",
                "Distance to port",
                "Distance to landfall",
                "Cut-in wind speed",
                "Cut-out wind speed",
                "Average annual wind speed at 50 m",
                "Average annual wind speed at hub height",
                "Shear exponent",
                "Weibull k",
                "Total system losses",
                "Availability",
                "Gross energy capture",
                "Net energy capture",
                "Gross capacity factor",
                "Net capacity factor",
            ],
            "Units": [
                "MW",
                "Number",
                "MW",
                "m",
                "m",
                "W/m2",
                "m",
                "-",
                "km",
                "km",
                "m/s",
                "m/s",
                "m/s",
                "m/s",
                "-",
                "-",
                "%",
                "%",
                "MWh/MW/year",
                "MWh/MW/year",
                "%",
                "%",
            ],
            "Value": [
                self.capacity("mw"),
                self.orbit.num_turbines,
                self.orbit.turbine_rating,
                self.orbit.config["turbine"]["rotor_diameter"],
                self.orbit.config["turbine"]["hub_height"],
                self.orbit.config["turbine"]["turbine_rating"]
                * 1000000
                / (
                    math.pi * (self.orbit.config["turbine"]["rotor_diameter"] / 2) ** 2
                ),  # Specific power
                self.orbit.config["site"]["depth"],
                self.determine_substructure_type(),
                self.orbit.config["site"]["distance"],  # Distance to port
                self.orbit.config["site"]["distance_to_landfall"],
                self.cut_in_windspeed(),
                self.cut_out_windspeed(),
                self.average_wind_speed(50),
                self.average_wind_speed(self.orbit.config["turbine"]["hub_height"]),
                np.mean(
                    compute_shear(
                        self.wombat.env.weather.to_pandas(),
                        self.identify_windspeed_columns_and_heights(
                            self.wombat.env.weather.to_pandas()
                        ),
                        False,
                    )
                ),
                self.compute_weibull(self.orbit.config["turbine"]["hub_height"]),
                100 * self.loss_ratio(),
                100 * self.availability(which="energy", frequency="project", by="windfarm"),
                self.energy_potential(units="mw", per_capacity="mw", aep=True),
                self.energy_production(units="mw", per_capacity="mw", aep=True),
                100 * self.capacity_factor(which="gross", frequency="project", by="windfarm"),
                100 * self.capacity_factor(which="net", frequency="project", by="windfarm"),
            ],
        }

        # Create a DataFrame from the template
        df = pd.DataFrame(project_details)

        return df

    def determine_substructure_type(self):
        """Determine the substructure type based on the ORBIT configuration file.

        This function scans the "design_phases" section of the ORBIT configuration file to identify
        the substructure type used in the project. The substructure types considered are
        "Monopile", "SemiSubmersible", "Jacket", and "Spar". The function returns the substructure
        type as a string if found in the design phases, or "Unknown" if no substructure type is
        identified.

        Returns
        -------
        str
            The substructure type as one of the following: "Monopile", "SemiSubmersible",
            "Jacket", "Spar", or "Unknown" if no match is found.

        Notes
        -----
        - The search is case-insensitive and looks for the substructure types as substrings within
          the design phase names.
        - If no substructure type is found after checking all phases, the function will return
          "Unknown".
        """
        if self.turbine_type == "land":
            raise NotImplementedError("Land-based analyses are not yet supported.")

        project_data = self.orbit.config["design_phases"]

        # Iterate over the design_phases to check for the substructure type
        if "MonopileDesign" in project_data:
            return "Monopile"
        if "SemiSubmersibleDesign" in project_data:
            return "Semi-Submersible"
        if "JacketDesign" in project_data:
            return "Jacket"
        if "SparDesign" in project_data:
            return "Spar"
        return None

    def cut_in_windspeed(self):
        """Determine the cut-in wind speed for the turbine based on the power-thrust table.

        This function extracts the power and wind speed data from the turbine definitions in the
        FLORIS model and identifies the cut-in wind speed. The cut-in wind speed is the wind speed
        at which the turbine begins to produce power after zero power is achieved. The function
        returns the cut-in wind speed or `None` if no valid value can be determined.

        Returns
        -------
        float | None
            The wind speed (in m/s) at which the turbine starts producing power, or `None` if no
            valid cut-in wind speed can be found.

        Notes
        -----
        - The cut-in wind speed is identified as the wind speed immediately following the last wind
          speed with zero power that is lower than the wind speed at which the turbine produces
          maximum power.
        - If no valid zero power wind speeds are found below the maximum power wind speed, `None`
          is returned.

        """
        turbine_data = self.floris.core.farm.turbine_definitions[0]["power_thrust_table"]

        # Extract power and wind_speed lists from the floris turbine dictionary
        power = turbine_data["power"]
        wind_speed = np.array(turbine_data["wind_speed"])

        # Find the index of the maximum power
        max_power_index = np.argmax(power)

        # Identify the wind speed at the max power
        max_power_wind_speed = wind_speed[max_power_index]

        # Identify the wind speeds with 0 power
        zero_power_indices = [i for i, p in enumerate(power) if p == 0]

        # Find the first wind speed before the start of the power curve - latest zero before max ws
        latest_zero_power_ix = min(
            set(zero_power_indices).intersection(np.where(wind_speed < max_power_wind_speed)[0])
        )
        if latest_zero_power_ix == set():
            return None

        return wind_speed[latest_zero_power_ix + 1]

    def cut_out_windspeed(self):
        """Determine the cut-out wind speed for the turbine based on the power-thrust table.

        This function extracts the power and wind speed data from the turbine definitions in the
        FLORIS model and identifies the cut-out wind speed. The cut-out wind speed is the wind speed
        at which the turbine stops producing power, which occurs when the power drops to zero.
        The function returns the cut-out wind speed or `None` if no valid value can be determined.

        Returns
        -------
        float | None
            The wind speed (in m/s) at which the turbine stops producing power, or `None` if no
            valid cut-out wind speed can be found.

        Notes
        -----
        - The cut-out wind speed is identified as the wind speed immediately preceding the first
          wind speed where the turbine generates zero power and the wind speed is greater than
          the wind speed at which maximum power is produced.
        - If no valid zero power wind speeds are found above the maximum power wind speed, `None` is
          returned.
        """
        turbine_data = self.floris.core.farm.turbine_definitions[0]["power_thrust_table"]

        # Extract power and wind_speed lists from the floris turbine dictionary
        power = turbine_data["power"]
        wind_speed = np.array(turbine_data["wind_speed"])

        # Find the index of the maximum power
        max_power_index = np.argmax(power)

        # Identify the wind speed at the max power
        max_power_wind_speed = wind_speed[max_power_index]

        # Identify the wind speeds with 0 power
        zero_power_indices = [i for i, p in enumerate(power) if p == 0]

        # Find the first wind speed past the end of the power curve - earliest zero after max ws
        earliest_zero_power_ix = min(
            set(zero_power_indices).intersection(np.where(wind_speed > max_power_wind_speed)[0])
        )
        if earliest_zero_power_ix == set():
            return None

        return wind_speed[earliest_zero_power_ix - 1]

    def calculate_wind_speed(self, height: int | float) -> pd.Series:
        """Calculates a new array, series, or value of wind speed at a given height.

        Parameters
        ----------
        height : int | float
            The new height to calculate the wind speed.

        Returns
        -------
        pd.Series
            The wind speed data at :py:arg:`height`.
        """
        weather = self.wombat.env.weather.to_pandas()
        ws_heights = self.identify_windspeed_columns_and_heights(weather)
        shear = compute_shear(weather, ws_heights)
        h1, *_ = ws_heights
        shear = compute_shear(weather, ws_heights)
        ws = extrapolate_windspeed(weather[h1], ws_heights[h1], height, shear)
        return ws

    def average_wind_speed(self, height: int | float):
        """Calculates the average wind speed at a specified height.

        This method computes the mean wind speed from the weather data at the given height.
        If the wind speed data for the specified height is not found, and there are not
        at least two columns of wind speed data, an error is raised.

        Parameters
        ----------
        height : int
            The height (in meters) at which the average wind speed is to be calculated.
            The method expects the column in the weather data to be named in the format
            'windspeed_{height}m'.

        Returns
        -------
        float | str
            If the wind speed data for the specified height exists, the method returns the mean
            wind speed as a float (in meters per second). If the data is not found, it returns
            a string indicating that the wind speed data is missing.

        Raises
        ------
        KeyError
            If the column corresponding to the specified height is not present in the weather
            data.
        """
        column_name = f"windspeed_{height}m"

        weather = self.wombat.env.weather.to_pandas()
        ws_heights = self.identify_windspeed_columns_and_heights(weather)

        # Check if the column exists or if there is enough data to compute it
        if height_exists := column_name in weather.columns:
            return weather[column_name].mean()
        if (not height_exists) and len(ws_heights) < 2:
            msg = (
                f"Wind speed at {height} m not provided at {str(self.library_path)}/weather/"
                f"{str(self.weather_profile)}, nor are there at least two heights to extrapolate"
                f"the wind speed.\nPlease, add a column to the weather .csv file with the name"
                f"'windspeed_{height}m', and the respective wind speed data"
            )
            raise KeyError(msg)

        return self.calculate_wind_speed(height).mean()

    def identify_windspeed_columns_and_heights(self, df):
        """Identifies columns containing wind speed measurements and their respective sensor
        heights.

        Scans the DataFrame to find columns that match the pattern "windspeed_{number}m", where
        `{number}`corresponds to the sensor height in meters. The function returns a dictionary
        with the first two matches, where the keys are the column names and the values are the
        heights in meters.

        Parameters
        ----------
        df : pandas.DataFrame
            A DataFrame containing wind speed columns. The columns should follow the naming
            convention"windspeed_{number}m", where `{number}` represents the sensor height in
            meters.

        Returns
        -------
        dict[str, int]
            A dictionary containing the first two columns that match the "windspeed_{number}m"
            pattern. The keys are the column names (e.g., "windspeed_10m"), and the values are the
            corresponding sensor heights (e.g., 10).
        """
        windspeed_columns = {}

        # Regex pattern to match "windspeed_{number}m" where the name ends with 'm'
        pattern = r"windspeed_(\d+)m$"

        for col in df.columns:
            match = re.match(pattern, col)
            if match:
                number = match.group(1)
                windspeed_columns[col] = int(number)

        return dict(list(windspeed_columns.items())[:2])

    def compute_weibull(self, height, random_seed=1):
        """Fits a Weibull distribution to wind speed data at a specified height.

        This function fits a Weibull distribution to the wind speed data at the specified height
        from the weather data. It assumes that wind speeds are non-negative, so it fixes the
        location parameter of the Weibull distribution to 0. If the required column for the given
        height is not present in the weather data, an error message is printed and the function
        returns a string indicating the missing data.

        Parameters
        ----------
        height : int
            The height (in meters) at which the wind speed data is collected. The function looks
            for a column named "windspeed_{height}m" in the weather data.
        random_seed : int, optional
            A random seed for reproducibility. Defaults to 1. This seed is used to initialize
            the random number generator for the Weibull fitting procedure.

        Returns
        -------
        float | str
            If the wind speed data is available for the specified height, the function returns
            the shape parameter of the fitted Weibull distribution. If the necessary column is
            missing from the weather data, it returns a string indicating the error
            (e.g., "wind speed at {height}m not provided").
        """
        # Set random seed for reproducibility
        np.random.seed(random_seed)

        column_name = "windspeed_" + str(height) + "m"
        weather = self.wombat.env.weather.to_pandas()
        ws_heights = self.identify_windspeed_columns_and_heights(weather)

        # Check if the column exists or if there is enough data to compute it
        if height_exists := column_name in weather.columns:
            wind_speed_data = weather[column_name]
            return fit_weibull_distribution(wind_speed_data, random_seed)
        if (not height_exists) and len(ws_heights) < 2:
            msg = (
                f"Wind speed at {height} m not provided at {str(self.library_path)}/weather/"
                f"{str(self.weather_profile)}, nor are there at least two heights to extrapolate"
                f"the wind speed.\nPlease, add a column to the weather .csv file with the name"
                f"'windspeed_{height}m', and the respective wind speed data"
            )
            raise KeyError(msg)

        # Extract windspeed data
        wind_speed_data = self.calculate_wind_speed(height)
        return fit_weibull_distribution(wind_speed_data, random_seed)
