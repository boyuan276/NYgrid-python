# NYgrid-python

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/boyuan276/NYgrid-python/blob/main/LICENSE)
![Python](https://img.shields.io/badge/python-3.8-blue.svg)
![Python](https://img.shields.io/badge/python-3.9-blue.svg)
[![Python package](https://github.com/boyuan276/solar-farm-design/actions/workflows/python-package.yml/badge.svg)](https://github.com/boyuan276/NYgrid-python/actions/workflows/python-package.yml)
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

    - Run the model with default parameters:

        ```bash
        python ex_opf_wo_renew.py
        ```

    - Run the model with renewable integration:

        ```bash
        # With future distributed solar integration
        python ex_opf_w_future_solar.py

        # With offshore wind integration
        python ex_opf_w_offshore_wind.py

        # With solar, offshore wind, and building electrification integration
        python ex_opf_w_renew.py
        ```

Note: Renewable timeseries data need to be prepared before running the model.


## See also
* [MATLAB version of the NYgrid model](https://github.com/AndersonEnergyLab-Cornell/NYgrid.git)
