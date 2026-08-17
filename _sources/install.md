# Installation Guide

```{contents}
:local:
:depth: 3
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

#### Developing

If you plan on contributing to the code base at any point, be sure to install the developer tools.

For more details on developing, please see the [contributor's guide](contributing.md).

```bash
pip install -e ".[dev,docs,examples]"
pre-commit install
```
