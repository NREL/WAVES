# CHANGELOG

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
