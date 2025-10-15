# CHANGELOG

## v0.6 - 30 September 2025

- Land-based wind analyses are now supported through the inclusion of LandBOSSE.
- Increments the required versions of ORBIT and WOMBAT to ensure the latest fixes and functionality
  are made available by default.
- Site and project detail reporting functionality has been added:
  - `Project.generate_report_lcoe_breakdown()`
  - `Project.generate_report_project_details()`
  - `Project.determine_substructure_type()`
  - `Project.identify_windspeed_columns_and_heights()`
  - `Project.cut_in_windspeed()`
  - `Project.cut_out_windspeed()`
  - `Project.calculate_wind_speed()`
  - `Project.average_wind_speed()`
  - `Project.compute_weibull()`
  - `utilities.met_data.fit_weibull_distribution()`
  - `utilities.met_data.compute_shear()`
  - `utilities.met_data.extrapolate_windspeed()`
- WAVES will now set the FLORIS `turbine_library_path` setting in the FLORIS `farm` configuration
  to use "`Project.library_path`/turbines", ensuring that all analysis data will be co-located.
- Updated dependency stack:
  - FLORIS v4.2.2 or greater is now required
    - Monthly wind rose analysis is now the only supported FLORIS analysis type due to a change in
      the underlying implementation.
    - Custom cut-in and cut-out wind speed are no longer able to be modified, and changes must be made
      to the power curve itself due to the more complicated nature of post-hoc processing.
    - `run_kwargs` is now `set_kwargs` throughout `Project` to align with the updated FLORIS API.
    - FLORIS configuration files are updated to the v4 API.
  - ORBIT v1.1 or greater is now required, and all configurations have been updated accordingly.
  - Please see model documentation for implementation specifics.
- `loss_ratio` has been replaced with `environmental_loss_ratio` to account for the only loss
  category that cannot be modeled in WAVES.
- `turbine_type` input has been added to indicate if a project is land-based (coming soon), or
  fixed or floating offshore wind
- Energy losses are now available through:
  - `Project._get_floris_energy()` to aggregate the FLORIS energy potential and waked energy
    production to the correct level in place using the same methodology in multiple methods.
  - `Project.energy_losses()` for calculating total energy losses across varying granularities
  - `Project.loss_ratio()` with an ability to provide categorical breakdowns
  - `Project.technical_loss_ratio()` to calculate the ORCA technical losses.
  - `Project.electrical_loss_ratio()` to calculate the ORCA technical losses.
- `Project.energy_potential()` now forms the basis of all energy production and loss methods to
  ensure consistent computation.
- A new series of validators have been added for commonly used parameterizations alongside a
  decorator `@validate_common_inputs(which=...)` to apply the validations automatically before the
  main methodology is run.

## 0.5.3 (7 May 2024)

- A bug was fixed where the array system installation costs were nearly doubled when compared
  to a direct ORBIT run. This was the result of a duplicated design result when calling both
  `Project.connect_orbit_cable_lengths()` and `Project.orbit.run()`.

## 0.5.2 (9 April 2024)

- Pins FLORIS to v3.6 to avoid workarounds for previous versions, and to avoid issues with
  adopting v4.
- Updates WOMBAT to 0.9.3 to account for the latest bug fixes.
- Fixes minor typos.
- Adds in mermaid markdown workflow diagrams for the documentation.
