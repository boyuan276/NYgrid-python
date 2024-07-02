import os
from datetime import datetime
from nygrid.nygrid import NYGrid
import pytest

# Set up directories
cwd = os.getcwd()
if 'tests' in cwd:
    parent_dir = os.path.dirname(cwd)
    data_dir = os.path.join(parent_dir, 'data')
else:
    data_dir = os.path.join(cwd, 'data')

grid_data_dir = os.path.join(data_dir, 'grid', '2018Baseline')
if not os.path.exists(grid_data_dir):
    raise FileNotFoundError('Grid data directory not found.')

start_datetime = datetime(2018, 1, 1, 0, 0, 0)
end_datetime = datetime(2018, 1, 2, 0, 0, 0)


def test_nygrid_obj():
    nygrid_sim = NYGrid(grid_data_dir,
                        start_datetime=start_datetime.strftime('%m-%d-%Y %H'),
                        end_datetime=end_datetime.strftime('%m-%d-%Y %H'),
                        dcline_prop=None,
                        esr_prop=None,
                        vre_prop=None,
                        verbose=True)

    assert nygrid_sim is not None
    assert isinstance(nygrid_sim, NYGrid)
    assert nygrid_sim.start_datetime == start_datetime
    assert nygrid_sim.end_datetime == end_datetime
