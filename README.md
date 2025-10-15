# WAVES: Wind Asset Value Estimation System

[![PyPI version](https://badge.fury.io/py/waves.svg)](https://badge.fury.io/py/waves)
[![PyPI downloads](https://img.shields.io/pypi/dm/waves?link=https%3A%2F%2Fpypi.org%2Fproject%2FWAVES%2F)](https://pypi.org/project/WAVES/)
[![Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![image](https://img.shields.io/pypi/pyversions/waves.svg)](https://pypi.python.org/pypi/waves)


[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/NREL/WAVES/main?filepath=examples)
[![Jupyter Book](https://jupyterbook.org/badge.svg)](https://nrel.github.io/WAVES)

[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

## Overview

Runs analyses for offshore wind projects by utilizing ORBIT (CapEx), WOMBAT (OpEx), and FLORIS (AEP)
to estimate the lifecycle costs using NREL's flagship technoeconomic models.

Please visit our [documentation site](https://nrel.github.io/WAVES/) for API documentation, a
reference guide, and examples.

## Requirements

Python 3.10+, preferably 3.12

## Environment Setup

Download the latest version of [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
for the appropriate OS. Follow the remaining [steps](https://conda.io/projects/conda/en/latest/user-guide/install/index.html#regular-installation)
for the appropriate OS version.

Using conda, create a new virtual environment:

```console
conda create -n <environment_name> python=3.12
conda activate <environment_name>
conda install -c anaconda pip
conda config --set pip_interop_enabled true

# to deactivate
conda deactivate
```

## Installation

Requires Python 3.10+.

For basic usage, users can install WAVES directly from PyPI, or from source for more advanced usage.

### Pip

`pip install waves`

### From Source

A source installation is great for users that want to work with the provided example, and
potentially modify the code at a later point in time.

```bash
git clone https://github.com/NREL/WAVES.git
cd WAVES
pip install .
```

If working with the example, or running with Jupyter Notebooks, be sure to install the examples
dependencies like the following:

```bash
pip install ".[examples]"
```

#### Tinkering

Use the `-e` for an editable installation, in case you plan on editing any underlying code.

```bash
pip install -e .
```

## Usage

After installation, the package can imported:

```console
python
import waves
waves.__version__
```

### CLI

```console
waves library-path configuration1.yaml configuration2.yaml
```
