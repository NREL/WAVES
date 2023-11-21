# How to Use WAVES

```{contents}
:local:
:depth: 3
```

## Installation

Requires Python 3.9 or 3.10.

For basic usage, users can install WAVES directly from PyPI, or from source for more advanced usage.

### Pip

`pip install waves`

### From Source

```bash
git clone https://github.com/NREL/WAVES.git
cd WAVES
pip install .
```

#### Tinkering

Use the `-e` for an editable installation, in case you plan on editing any underlying code.

```bash
pip install -e .
```

#### Developing

If you plan on contributing to the code base at any point, be sure to install the developer tools.

For more details on developing, please see the [contributor's guide](contributing.md).

```bash
pip install -e ".[dev,docs]"
pre-commit install
```

## Working With the `Project` Class

This section will provide a guided overview of all the components of the `Project` class that are
relevant to users, and demonstrate inputs used for the 2022 Cost of Wind Energy Review (COWER)
analysis. For a complete API reference, please refer to the [API documentation](./api.md).

### Configuring

```{eval-rst}
.. currentmodule:: waves.project

.. autoclass:: Project
   :undoc-members:
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

.. automethod:: Project.from_file
   :noindex:

.. autoproperty:: Project.config_dict
   :noindex:

.. automethod:: Project.save_config
   :noindex:
```

#### COWER 2022 Configuration

Aligning with COWER, we have the following inputs. It should be noted that each model's
configuration is a pointer to another file to keep each configuration as tidy as possible. However,
each of `orbit_config`, `wombat_config`, and `floris_config` allow for a direct dictionary
configuration input.

```{eval-rst}
.. literalinclude:: ../library/cower_2022/project/config/COE_2022_fixed_bottom_base.yaml
   :language: yaml
   :linenos:
```

#### Connecting Configurations

Also, see the configurations {py:attr}`Project.connect_floris_to_layout` and
{py:attr}`Project.connect_orbit_array_design` to automatically run the below during project
initialization.

```{eval-rst}
.. automethod:: waves.project.Project.connect_floris_to_turbines

.. automethod:: waves.project.Project.generate_floris_positions_from_layout

.. automethod:: waves.project.Project.connect_orbit_cable_lengths

```

#### Updating Configurations

```{eval-rst}
.. automethod:: waves.project.Project.generate_floris_positions_from_layout

.. automethod:: waves.project.Project.reinitialize
```

### Viewing the Wind Farm Properties

```{eval-rst}
.. automethod:: waves.project.Project.plot_farm

.. automethod:: waves.project.Project.n_turbines

.. automethod:: waves.project.Project.n_substations

.. automethod:: waves.project.Project.capacity

.. automethod:: waves.project.Project.turbine_rating

.. automethod:: waves.project.Project.plot_farm
```

### Running the Models

```{eval-rst}
.. automethod:: waves.project.Project.run
```

### Results

To quickly produce any of the high-level outputs to a single `DataFrame`, the below method can be
used in place of individually calculating each metric and combining into a report. Additionally,
users can refer to the [COWER 2022 example](cower_2022:results) for the reported results,
which relies on the `generate_report` method and accessing the ORBIT `ProjectManager` directly for
further CapEx breakdowns.

```{eval-rst}
.. automethod:: waves.project.Project.generate_report
   :noindex:
```

All the models can be individually accessed to calculate results that are not integrated into
WAVES officially, but it should be noted that they should be used with caution in case there are
any interdependencies between model outputs.

```{eval-rst}
.. autoattribute:: waves.project.Project.orbit

.. autoattribute:: waves.project.Project.wombat

.. autoattribute:: waves.project.Project.floris
```

#### Balance of Systems Costs and Properties

```{eval-rst}
.. automethod:: waves.project.Project.capex
   :noindex:

.. automethod:: waves.project.Project.capex_breakdown
   :noindex:

.. automethod:: waves.project.Project.array_system_total_cable_length
   :noindex:

.. automethod:: waves.project.Project.export_system_total_cable_length
   :noindex:
```

#### Operations and Maintenance Costs

```{eval-rst}
.. automethod:: waves.project.Project.opex
   :noindex:

.. automethod:: waves.project.Project.availability
   :noindex:

.. automethod:: waves.project.Project.capacity_factor
   :noindex:
```

#### Energy Production

```{eval-rst}
.. automethod:: waves.project.Project.energy_potential
   :noindex:

.. automethod:: waves.project.Project.energy_production
   :noindex:

.. automethod:: waves.project.Project.energy_losses
   :noindex:
```

#### Project Financials

```{eval-rst}
.. automethod:: waves.project.Project.capex
   :noindex:

.. automethod:: waves.project.Project.capex_breakdown
   :noindex:

.. automethod:: waves.project.Project.opex
   :noindex:

.. automethod:: waves.project.Project.revenue
   :noindex:

.. automethod:: waves.project.Project.cash_flow
   :noindex:

.. automethod:: waves.project.Project.npv
   :noindex:

.. automethod:: waves.project.Project.irr
   :noindex:

.. automethod:: waves.project.Project.lcoe
   :noindex:
```
