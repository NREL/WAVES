# WAVES Offshore Wind Techno-economic Analysis

Wind Asset Value Estimation System (WAVES) is a tool for estimating the lifecycle costs of
offshore wind projects using NREL's flagship techno economic models. WAVES is in active development
due to wrapping other in-development software packages for its underlying operations: ORBIT for
CapEx, WOMBAT for OpEx, and FLORIS for AEP.

## Quick Start Guide

WAVES is a Python package, run via scripts or Jupyter notebooks, but is a low- to no-code software
that relies on configurations and post-processing method calls for results. At its core, WAVES does
the work of combining each software's configurations and results, so that users don't have to
put in the painstaking work, and can focus on what scenarios should be run--not how to run them.

At its core, WAVES is a wrapper around [ORBIT](https://wisdem.github.io/ORBIT/),
[WOMBAT](https://wisdem.github.io/WOMBAT/), and [FLORIS](https://nrel.github.io/floris/), so users
are expected to know how to work with each of these models to work with WAVES.

| Model | Purpose | GitHub | Documentation |
| :---- | :------ | :----- | :------------ |
| ORBIT | CapEx | <https://github.com/WISDEM/ORBIT/> | <https://wisdem.github.io/ORBIT/> |
| WOMBAT | OpEx, Availability | <https://github.com/WISDEM/WOMBAT/> | <https://wisdem.github.io/WOMBAT/> |
| FLORIS | Energy Production | <https://github.com/NREL/FLORIS/> | <https://nrel.github.io/FLORIS/> |

### Installation

Requires Python 3.10+.

For basic usage, users can install WAVES directly from PyPI, or from source for more advanced usage.

#### Pip

`pip install waves`

#### From Source

```bash
git clone https://github.com/NREL/WAVES.git
cd WAVES
pip install .
```

##### Tinkering

Use the `-e` for an editable installation, in case you plan on editing any underlying code.

```bash
pip install -e .
```

##### Developing

If you plan on contributing to the code base at any point, be sure to install the developer tools.

```bash
pip install -e ".[dev,docs]"
pre-commit install
```

## Engaging on GitHub

WAVES utilizes the following for coordinating development efforts

- [Issues](https://github.com/NREL/WAVES/issues): report bugs and submit feature requests
- [Pull Requests](https://github.com/NREL/WAVES/pulls): submit bug fixes, feature request,
  documentation improvements, and any other improvements

## Documentation Table of Contents

```{tableofcontents}
```
