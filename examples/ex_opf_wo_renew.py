"""
Run multi-period OPF with 2018 data
without renewable generators

"""
# %% Packages

import logging
import os
import pickle
import time
from datetime import datetime, timedelta

import pandas as pd

from nygrid.run_nygrid import read_grid_data, run_nygrid_one_day

if __name__ == '__main__':

    # %% Simulation settings
    # NOTE: Change the following settings to run the simulation
    sim_name = 'wo_renew'

    start_date = datetime(2018, 1, 1, 0, 0, 0)
    end_date = datetime(2018, 1, 10, 0, 0, 0)
    timestamp_list = pd.date_range(start_date, end_date, freq='1D')

    # %% Set up logging
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler(f'ex_opf_{sim_name}.log'),
                                  logging.StreamHandler()])

    t = time.time()
    logging.info('Start running multi-period OPF without renewable generators.')

    # %% Set up directories
    cwd = os.getcwd()
    if 'examples' in cwd:
        parent_dir = os.path.dirname(cwd)
        data_dir = os.path.join(parent_dir, 'data')
    else:
        data_dir = os.path.join(cwd, 'data')

    grid_data_dir = os.path.join(data_dir, 'grid')
    if not os.path.exists(grid_data_dir):
        raise FileNotFoundError('Grid data directory not found.')

    logging.info('Grid data directory: {}'.format(grid_data_dir))

    fig_dir = os.path.join(os.path.dirname(data_dir), 'figures')
    logging.info('Figure directory: {}'.format(fig_dir))

    results_dir = os.path.join(os.path.dirname(data_dir), 'results')
    logging.info('Results directory: {}'.format(results_dir))

    sim_results_dir = os.path.join(results_dir, sim_name)
    if not os.path.exists(sim_results_dir):
        os.mkdir(sim_results_dir)

    # %% Read grid data

    # Read load and generation profiles
    grid_data = read_grid_data(grid_data_dir, start_date.year)

    # Read DC line property file
    filename = os.path.join(grid_data_dir, 'dcline_prop.csv')
    dcline_prop = pd.read_csv(filename, index_col=0)
    grid_data['dcline_prop'] = dcline_prop

    # Read ESR property file
    filename = os.path.join(grid_data_dir, 'esr_prop.csv')
    esr_prop = pd.read_csv(filename, index_col=0)
    grid_data['esr_prop'] = esr_prop

    # %% Set up OPF model

    # Set options
    options = {'UsePTDF': True,
               'solver': 'gurobi',
               'PenaltyForLoadShed': 20_000,
               'PenaltyForBranchMwViolation': 10_000,
               'PenaltyForInterfaceMWViolation': 10_000}

    # No initial condition for the first day
    last_gen = None

    # Loop through all days
    for d in range(len(timestamp_list) - 1):
        # Run OPF for two days at each iteration
        # The first day is valid, the second day is used for creating initial condition for the next iteration
        start_datetime = timestamp_list[d]
        end_datetime = start_datetime + timedelta(hours=47)

        nygrid_results = run_nygrid_one_day(start_datetime, end_datetime, grid_data, grid_data_dir, options, last_gen)

        # Set generator initial condition for the next iteration
        last_gen = nygrid_results['PG'].loc[start_datetime].to_numpy().squeeze()

        # Save simulation nygrid_results to pickle
        filename = f'nygrid_sim_{sim_name}_{start_datetime.strftime("%Y%m%d%H")}.pkl'
        with open(os.path.join(sim_results_dir, filename), 'wb') as f:
            pickle.dump(nygrid_results, f)
        logging.info(f'Saved simulation nygrid_results in {filename}')

        elapsed = time.time() - t
        logging.info(f'Finished running for {start_datetime.strftime("%Y-%m-%d")}. Elapsed time: {elapsed:.2f} seconds')
        logging.info('-' * 80)
