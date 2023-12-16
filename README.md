# WAVES: Wind Asset Value Estimation System

## Overview

Runs analyses for offshore wind projects by utilizing ORBIT (CapEx), WOMBAT (OpEx), and FLORIS (AEP)
to estimate the lifecycle costs using NREL's flagship technoeconomic models.

## Requirements

Python 3.9 or 3.10

## Environment Setup

Download the latest version of `Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_
for the appropriate OS. Follow the remaining `steps <https://conda.io/projects/conda/en/latest/user-guide/install/index.html#regular-installation>`_
for the appropriate OS version.

Using conda, create a new virtual environment:

```console
conda create -n <environment_name>
conda activate <environment_name>
conda install -c anaconda pip
conda config --set pip_interop_enabled true

# to deactivate
conda deactivate
```

## Installation

Requires Python 3.10.

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
