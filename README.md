# NYgrid-python
A python version of the NYgrid model.

## Installation
1. Install [Anaconda](https://www.anaconda.com/download/).
2. Install [Git](https://git-scm.com/downloads).
3. Clone the repository: 
```
git clone https://github.com/boyuan276/NYgrid-python.git
```
4. Create a conda environment:
```
conda env create -f NYgrid-python.yml
```
5. Activate the environment:
```
conda activate NYgrid-python
```
6. Install the package:
```
pip install -e .
```

## Usage
1. Activate the environment:
```
conda activate NYgrid-python
```
2. Run the model:

Go to the `examples` folder.

1. Run the model with default parameters:
```
python ex_opf_wo_renew.py
```

2. Run the model with renewable integration:
```
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