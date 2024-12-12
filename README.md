# NYgrid-python

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/boyuan276/NYgrid-python/blob/main/LICENSE)
![Python](https://img.shields.io/badge/python-3.8-blue.svg)
![Python](https://img.shields.io/badge/python-3.9-blue.svg)
[![Python package](https://github.com/boyuan276/NYgrid-python/actions/workflows/python-package.yml/badge.svg)](https://github.com/boyuan276/NYgrid-python/actions/workflows/python-package.yml)
[![CodeQL](https://github.com/boyuan276/NYgrid-python/actions/workflows/github-code-scanning/codeql/badge.svg)](https://github.com/boyuan276/NYgrid-python/actions/workflows/github-code-scanning/codeql)

A python version of the NYgrid model.

It contains the following components:
* Optimal power flow (OPF) model
* Renewable integration
* Building electrification
* Battery storage
* Electric vehicle (EV) charging

## Installation

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


## See also
* [MATLAB version of the NYgrid model](https://github.com/AndersonEnergyLab-Cornell/NYgrid.git)
