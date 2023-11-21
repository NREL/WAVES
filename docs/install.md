# Installation Guide

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
