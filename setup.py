"""Installation settings."""

import codecs
from pathlib import Path

from setuptools import setup, find_packages


def read(relative_path):
    """Reads the `relative_path` file."""
    here = Path(__file__).resolve().parent
    with codecs.open(here / relative_path, "r") as fp:
        return fp.read()


def get_version(relative_path):
    """Retreives the version number."""
    for line in read(relative_path).splitlines():
        if line.startswith("__version__"):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")


with open("README.md") as fh:
    long_description = fh.read()


NAME = "WAVES"
DESCRIPTION = "Wind Asset Value Estimation System"


# Installation reuqirements
REQUIRED = [
    "tqdm",
    "attrs",
    "pyyaml",
    "matplotlib>=3.6",
    "numpy-financial>=1.0.0",
    "floris>=3.3",
    "wombat>=0.7.1",
    "orbit-nrel>=1.0.8",
]
TEST = ["pytest", "pytest-cov"]
DEV = ["pre-commit", "black", "mypy", "flake8", "flake8-docstrings", "isort", "ruff"] + TEST
DOCS = ["jupyter-book", "myst-nb", "myst-parser"]

extra_package_requirements = {"dev": DEV, "docs": DOCS, "all": DEV + DOCS}

setup(
    name=NAME,
    author="Rob Hammond",
    author_email="rob.hammond@nrel.gov",
    version=get_version(Path("waves") / "__init__.py"),
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    project_urls={
        "Source": "https://github.com/NREL/WAVES",
    },
    classifiers=[
        # TODO: https://pypi.org/pypi?%3Aaction=list_classifiers
        "Development Status :: 1 - Planning",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
    ],
    packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    include_package_data=True,
    package_data={"": ["*.yaml", "*.yml", "*.csv"]},
    install_requires=REQUIRED,
    python_requires=">=3.10",
    extras_require=extra_package_requirements,
    test_suite="pytest",
    tests_require=TEST,
)
