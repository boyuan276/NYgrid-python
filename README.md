# NYgrid-python

<table>
<tr>
    <td>Latest Release</td>
    <td>
        <a href="https://pypi.org/project/nygrid/">
        <img src="https://img.shields.io/pypi/v/nygrid" alt="latest release" />
        </a>
        <a href="https://github.com/boyuan276/NYgrid-python/actions/workflows/python-publish.yml">
        <img src="https://github.com/boyuan276/NYgrid-python/actions/workflows/python-publish.yml/badge.svg" alt="latest release" />
        </a>
    </td>
</tr>
<tr>
    <td>Build Status</td>
    <td>
        <a href="https://nygrid-python.readthedocs.io/en/stable/">
        <img src="https://readthedocs.org/projects/nygrid-python/badge/?version=stable" alt="documentation build status" />
        </a>
        <a href="https://github.com/boyuan276/NYgrid-python/actions/workflows/python-package.yml">
        <img src="https://github.com/boyuan276/NYgrid-python/actions/workflows/python-package.yml/badge.svg" alt="python-package" />
        </a>
        <a href="https://github.com/boyuan276/NYgrid-python/actions/workflows/github-code-scanning/codeql">
        <img src="https://github.com/boyuan276/NYgrid-python/actions/workflows/github-code-scanning/codeql/badge.svg" alt="codeql" />
        </a>
    </td>
</tr>
<tr>
    <td>License</td>
    <td>
        <a href="https://github.com/boyuan276/NYgrid-python/blob/main/LICENSE">
        <img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="license" />
        </a>
    </td>
</tr>
</table>

A python version of the NYgrid model.

It contains the following components:
* Optimal power flow (OPF) model
* Renewable integration
* Building electrification
* Battery storage
* Electric vehicle (EV) charging

## Documentation

Full documentation can be found at [readthedocs](https://nygrid-python.readthedocs.io/en/latest/).

## Installation

### Install using pip

The ``nygrid`` package can be installed using PIP.

```bash
pip install nygrid
```

### Install from the source

1. Install [Anaconda](https://www.anaconda.com/download/).

2. Install [Git](https://git-scm.com/downloads).

3. Clone the repository: 
```bash
git clone https://github.com/boyuan276/NYgrid-python.git
```

4. Create a conda environment:
```bash
conda env create -f NYgrid-python.yml
```

5. Activate the environment:
```bash
conda activate NYgrid-python
```

6. Install the package:
```bash
pip install -e .
```

## Usage

1. Activate the environment:

```bash
conda activate NYgrid-python
```

2. Run the model:

    - Go to the `examples` folder.

    - Run the base year 2018 case:

        ```bash
        python 01_opf_2018NewParams_daily.py
        ```

    - Run the future year 2030 with policy scenarios:

        ```bash
        # 1) 2030BaselineCase
        python 02_opf_2030BaselineCase_daily.py

        # 2) 2030ContractCase
        python 03_opf_2030ContractCase_daily.py

        # 3) 2030StateScenario
        python 04_opf_2030StateScenario_daily.py
        ```

    - Note: Generation and load properties and profiles need to be prepared before running these cases.

## Data

1. Generation data: See `examples/write_gen_prop_profiles_{case_name}.ipynb`.

2. Load data: See `examples/write_load_profiles_{case_name}.ipynb`.

## License

MIT license.


## See also
* [MATLAB version of the NYgrid model](https://github.com/AndersonEnergyLab-Cornell/NYgrid.git)
