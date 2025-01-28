# How to Use WAVES

```{contents}
:local:
:depth: 3
```

This section will provide a guided overview of all the components of the `Project` class that are
relevant to users, and demonstrate inputs used for the 2022 Cost of Wind Energy Review (COWER)
analysis. For a complete API reference, please refer to the [API documentation](./api.md).

## Configuring

### The Project Class
```{eval-rst}
.. currentmodule:: waves.project

.. autoclass:: Project
   :undoc-members:
   :noindex:
   :exclude-members: library_path, weather_profile, orbit_weather_cols, floris_windspeed,
      floris_wind_direction, floris_x_col, floris_y_col, orbit_config, wombat_config,
      floris_config, connect_floris_to_layout, connect_orbit_array_design, offtake_price,
      fixed_charge_rate, discount_rate, finance_rate, reinvestment_rate, loss_ratio,
      orbit_start_date, soft_capex_date, project_capex_date, system_capex_date, turbine_capex_date,
      weather, orbit_config_dict, wombat_config_dict, floris_config_dict, project_wind_rose,
      monthly_wind_rose, floris_turbine_order, turbine_potential_energy, turbine_production_energy,
      project_potential_energy, project_production_energy, _fi_dict, floris_results_type,
      operations_start, operations_end, operations_years, setup_orbit, setup_wombat, setup_floris,
      wombat, orbit, floris,
      library_exists, from_file, config_dict, save_config,
      connect_floris_to_turbines, connect_orbit_cable_lengths,
      generate_floris_positions_from_layout,
      preprocess_monthly_floris, run_wind_rose_aep,
      run_floris, run_orbit, run_wombat, run, reinitialize,
      plot_farm, n_turbines, turbine_rating, n_substations, capacity,
      capex, capex_breakdown, array_system_total_cable_length, export_system_total_cable_length,
      energy_potential, energy_production, energy_losses, availability, capacity_factor, opex,
      revenue, cash_flow, npv, irr, lcoe,
      generate_report,
```

### Working With Configurations

```{eval-rst}
.. currentmodule:: waves.project

.. automethod:: Project.from_file
   :noindex:

.. autoproperty:: Project.config_dict
   :noindex:

.. automethod:: Project.save_config
   :noindex:
```

### COWER 2022 Configuration

Aligning with COWER, we have the following inputs. It should be noted that each model's
configuration is a pointer to another file to keep each configuration as tidy as possible. However,
each of `orbit_config`, `wombat_config`, and `floris_config` allow for a direct dictionary
configuration input.

```{literalinclude} ../library/base_2022/project/config/base_fixed_bottom_2022.yaml
:language: yaml
:lines: 1-36
:linenos:
```

...

```{literalinclude} ../library/base_2022/project/config/base_fixed_bottom_2022.yaml
:language: yaml
:linenos:
:lineno-start: 43
:lines: 43
```

...

```{literalinclude} ../library/base_2022/project/config/base_fixed_bottom_2022.yaml
:language: yaml
:linenos:
:lineno-start: 46
:lines: 46
```

...

```{literalinclude} ../library/base_2022/project/config/base_fixed_bottom_2022.yaml
:language: yaml
:linenos:
:lineno-start: 95
:lines: 95
```

...

```{literalinclude} ../library/base_2022/project/config/base_fixed_bottom_2022.yaml
:language: yaml
:linenos:
:lineno-start: 141
:lines: 141-143
```

### Connecting Configurations

Also, see the configurations {py:attr}`Project.connect_floris_to_layout` and
{py:attr}`Project.connect_orbit_array_design` to automatically run the below during project
initialization.

```{eval-rst}
.. currentmodule:: waves.project

The following method is run on intialization when ``Project.connect_floris_to_layout`` is set to
``True``, which is the case in the COWER example.

.. automethod:: Project.connect_floris_to_turbines
   :noindex:
```

Visually, this looks like the following workflow:

```{image} diagrams/input_flow.svg
:align: center
```

### Updating Configurations

Sometimes, additional configurations may need to be connected prior to running an analysis. For
instance, it may be cumbersome to manually compute the FLORIS layout from a traditional coordinate
system, such as WGS-84, or a localized distance-based coordinate reference system. In the latter,
WAVES can help with the ``generate_floris_positions_from_layout`` method. However, as can be seen
below, the following generation method has to be run prior to connecting the FLORIS and ORBIT/WOMBAT
layouts. As such, your workflow leading up to an analysis might look like the following.

```python
from pathlib import Path
from waves import Project
from waves.utilities import load_yaml

library_path = Path("../library/base_2022/")
config = load_yaml(library_path / "project/config", "base_fixed_bottom_2022.yaml")
config["library_path"] = library_path  # add the library path

# Ensure FLORIS is not automatically connected
config["connect_floris_to_turbines"] = False

project = Project.from_dict(config)

# Generate the layout and connect FLORIS
project.generate_floris_positions_from_layout()  # note the defaults in the section below
project.connect_floris_to_turbines()
```

```{eval-rst}
.. currentmodule:: waves.project

.. automethod:: Project.generate_floris_positions_from_layout
   :noindex:

The following method allows users to connect the ORBIT-calculated cable lengths and insert them
back into the layout configuration file. This is helpful if the base distance is to be computed,
then reused later, without re-calculating, or after modifying, if desired.

.. automethod:: Project.connect_orbit_cable_lengths
   :noindex:

.. automethod:: Project.reinitialize
   :noindex:
```

## Viewing the Wind Farm Properties

For the following set of methods, users only need to create a ``Project`` object in order to use.

```{eval-rst}
.. currentmodule:: waves.project

.. automethod:: Project.plot_farm
   :noindex:

.. automethod:: Project.determine_substructure_type
   :noindex:

.. automethod:: Project.n_turbines
   :noindex:

.. automethod:: Project.n_substations
   :noindex:

.. automethod:: Project.capacity
   :noindex:

.. automethod:: Project.turbine_rating
   :noindex:

.. automethod:: Project.cut_in_windspeed
   :noindex:

.. automethod:: Project.cut_out_windspeed
   :noindex:

.. automethod:: Project.identify_windspeed_columns_and_heights
   :noindex:

.. automethod:: Project.calculate_wind_speed
   :noindex:

.. automethod:: Project.average_wind_speed
   :noindex:

.. automethod:: Project.compute_weibull
   :noindex:
```

## Running the Models

```{eval-rst}
.. currentmodule:: waves.project

.. automethod:: Project.run
   :noindex:
```

## Results

Visually, the following is a general flow of operations for combining each model's outputs:

```{image} diagrams/results_flow.svg
:align: center
```

To quickly produce any of the high-level outputs to a single `DataFrame`, the below method can be
used in place of individually calculating each metric and combining into a report. Additionally,
users can refer to the [COWER 2022 example](example_cower_2022:results) for the reported results,
which relies on the `generate_report` method and accessing the ORBIT `ProjectManager` directly for
further CapEx breakdowns.

```{eval-rst}
.. currentmodule:: waves.project

.. automethod:: Project.generate_report
   :noindex:
```

All the models can be individually accessed to calculate results that are not integrated into
WAVES officially, but it should be noted that they should be used with caution in case there are
any interdependencies between model outputs.

```{eval-rst}
.. currentmodule:: waves.project

.. autoattribute:: waves.project.Project.orbit
   :noindex:

.. autoattribute:: waves.project.Project.wombat
   :noindex:

.. autoattribute:: waves.project.Project.floris
   :noindex:
```

### Balance of Systems Costs and Properties

```{eval-rst}
.. currentmodule:: waves.project

.. automethod:: Project.capex
   :noindex:

.. automethod:: Project.capex_breakdown
   :noindex:

.. automethod:: Project.array_system_total_cable_length
   :noindex:

.. automethod:: Project.export_system_total_cable_length
   :noindex:
```

### Operations and Maintenance Costs

```{eval-rst}
.. currentmodule:: waves.project

.. automethod:: Project.opex
   :noindex:

.. automethod:: Project.availability
   :noindex:

.. automethod:: Project.capacity_factor
   :noindex:
```

### Energy Production

```{eval-rst}
.. currentmodule:: waves.project

.. automethod:: Project.energy_potential
   :noindex:

.. automethod:: Project.energy_production
   :noindex:

.. automethod:: Project.energy_losses
   :noindex:

.. automethod:: Project.loss_ratio
   :noindex:

.. automethod:: Project.wake_losses
   :noindex:

.. automethod:: Project.technical_loss_ratio
   :noindex:
```

### Project Financials

```{eval-rst}
.. currentmodule:: waves.project

.. automethod:: Project.capex
   :noindex:

.. automethod:: Project.capex_breakdown
   :noindex:

.. automethod:: Project.opex
   :noindex:

.. automethod:: Project.revenue
   :noindex:

.. automethod:: Project.cash_flow
   :noindex:

.. automethod:: Project.npv
   :noindex:

.. automethod:: Project.irr
   :noindex:

.. automethod:: Project.lcoe
   :noindex:
```

### Report Generation

```{eval-rst}
.. currentmodule:: waves.project

.. automethod:: Project.generate_report_project_details
   :noindex:

.. automethod:: Project.generate_report
   :noindex:

.. automethod:: Project.generate_report_lcoe_breakdown
   :noindex:
```

## Command Line Interface (CLI)

Run one or multiple WAVES analyses given a configuration dictionary, and optionally output
and save the results.

**Usage**:

```console
waves [OPTIONS] LIBRARY_PATH CONFIGURATION...
```

**Arguments**:

* `LIBRARY_PATH`: The relative or absolute path to the simulation data library.
* `CONFIGURATION...`: The configuration file name(s) to run. These should be located in
  `LIBRARY_PATH/project/config/`.

**Options**:

* `--report / --no-report`: [default: report] Generate a table of metrics; `report_config` must be
  configured in the ``configuration``. See the API for `Project.generate_report()` for details.
  Use `--no-report` to just run the simulation.
* `--save-report / --no-save-report`: [default: save-report] Save the output report metrics to a
  CSV file. Use `no-save-report` to only display the results.
* `--install-completion`: Install completion for the current shell.
* `--show-completion`: Show completion for the current shell, to copy it or customize the installation.
* `--help`: Show this message and exit.


### Additional Configurations

Running one or many analyses from the command line can be used with a few additional parameters
defined in the configuration file. The [provided example](./waves_example.md) is configured to also
be run through the CLI. Below is an example of the additional configurations

```{eval-rst}
.. literalinclude:: ../library/base_2022/project/config/base_fixed_bottom_2022.yaml
   :language: yaml
   :linenos:
   :lineno-start: 146
   :lines: 146-
```
