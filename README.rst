WAVES: Wind Asset Value Estimation System
==============================================

Overview
~~~~~~~~
Runs analyses for offshore wind projects by utilizing ORBIT (CapEx), WOMBAT (OpEx), and FLORIS (AEP)
to estimate the lifecycle costs using NREL's flagship technoeconomic models.


Requirements
~~~~~~~~~~~~
- Python 3.10


Environment Setup
~~~~~~~~~~~~~~~~~

Download the latest version of `Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_
for the appropriate OS. Follow the remaining `steps <https://conda.io/projects/conda/en/latest/user-guide/install/index.html#regular-installation>`_
for the appropriate OS version.

Using conda, create a new virtual environment:

.. code-block:: console

   conda create -n <environment_name> python=3.10 --no-default-packages
   conda activate <environment_name>
   conda install -c anaconda pip

   # to deactivate
   conda deactivate


Installation From Source
~~~~~~~~~~~~~~~~~~~~~~~~

Install it directly into an activated virtual environment:

.. code-block:: console

   git clone https://github.com/NREL/WAVES.git
   cd WHaLE
   pip install -e .


or if you will be contributing and needing to run tests and/or build documentation (separately, these are ".[dev]" and ".[docs]")

.. code-block:: console

   git clone https://github.com/NREL/WAVES.git
   cd wombat
   pip install -e '.[all]'


Usage
~~~~~

After installation, the package can imported:

.. code-block:: console

   python
   import waves
   waves.__version__
