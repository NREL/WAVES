# CLI Reference

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
