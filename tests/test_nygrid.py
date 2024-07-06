import os
from datetime import datetime
import nygrid.nygrid as ng_grid
import nygrid.run_nygrid as ng_run
import pytest

# Set up directories
cwd = os.getcwd()
if 'tests' in cwd:
    parent_dir = os.path.dirname(cwd)
    data_dir = os.path.join(parent_dir, 'data')
else:
    data_dir = os.path.join(cwd, 'data')

grid_data_dir = os.path.join(data_dir, 'grid')
if not os.path.exists(grid_data_dir):
    raise FileNotFoundError('Grid data directory not found.')

start_datetime = datetime(2018, 1, 1, 0, 0, 0)
end_datetime = datetime(2018, 1, 2, 0, 0, 0)

# Read grid property file
grid_prop = ng_run.read_grid_prop(grid_data_dir)


def test_nygrid_obj():
    # Create NYGrid object
    nygrid_sim = ng_grid.NYGrid(grid_prop=grid_prop,
                                start_datetime=start_datetime.strftime('%m-%d-%Y %H'),
                                end_datetime=end_datetime.strftime('%m-%d-%Y %H'),
                                verbose=True)

    assert nygrid_sim is not None
    assert isinstance(nygrid_sim, ng_grid.NYGrid)
    assert nygrid_sim.start_datetime == start_datetime
    assert nygrid_sim.end_datetime == end_datetime
