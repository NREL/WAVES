"""Creates the main CLI for WAVES."""
from typing import Annotated
from pathlib import Path

import typer

from waves import Project
from waves.utilities import load_yaml


app = typer.Typer()


def run_single(libary_path: str, config: str, report: bool, save_report: bool) -> None:
    """Run a single WAVES analysis, optionally output the results.

    Parameters
    ----------
    libary_path : str
        The location of the simulation data.
    config : str
        The file name of the configuration dictionary.
    report : bool
        Indicates if a report should be generated; "report_config" must be in the configuration
        dictionary, if using.
    save_report : bool
        Indicates if the report should be saved as a CSV to the results library folder.
    """
    print(f"Running WAVES for: {config}")
    config_dict = load_yaml(Path(libary_path).resolve() / "project/config", config)
    if (run_kwargs := config_dict.get("run")) is None:
        raise KeyError("The key `run` could not be found. Please see `Project.run()` for details.")

    project = Project.from_file(library_path=libary_path, config_file=config)
    project.run(**run_kwargs)
    if report:
        report_df = project.generate_report()
        print(report_df.T.to_markdown(tablefmt="grid"))
        if save_report:
            name = config_dict["report_config"]["name"]
            file_name = project.library_path / f"results/{name}.csv"
            print(f"Results saved to {file_name}")
            report_df.to_csv(file_name)

    project.wombat.env.cleanup_log_files()


@app.command()
def run(
    library_path: Annotated[str, typer.Argument(None, help="The path to the data library.")],
    configuration: Annotated[
        list[str],
        typer.Argument(
            None,
            help=(
                "The filename(s) of (with extension) of the configuration dictionary. These must"
                " include the a key, value pair for the run arguments under the key `run`"
            ),
        ),
    ],
    report: Annotated[
        bool,
        typer.Option(
            True,
            help=(
                "Generate a table of metrics. ``report_config`` must be configured in the"
                "``config_dict``. See the API for``Project.generate_report()`` for details."
            ),
        ),
    ] = True,
    save_report: Annotated[
        bool,
        typer.Option(
            True,
            help="True to save the output report metrics to a CSV file, and False to print only.",
        ),
    ] = True,
) -> None:
    """Run one or multiple WAVES analyses given a configuration dictionary, and optionally output
    and save the results.
    """
    for config in configuration:
        run_single(library_path, config, report, save_report)


if __name__ == "__main__":
    app()
