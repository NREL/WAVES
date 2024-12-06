# CHANGELOG

## Unreleased (TBD)

- A series of bugs in the FLORIS time series method have been fixed to ensure energy potential
  and production calculations add up as expected.
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
