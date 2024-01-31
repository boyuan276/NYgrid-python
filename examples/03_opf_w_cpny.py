"""
Run multi-period OPF with 2018 data.
3. With CPNY case:
Add CPNY and CHPE HVDC lines.
No ESRs.
No future solar and offshore wind.
No residential building electrification.
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
    sim_name = 'w_cpny'
    leading_hours = 12
    w_cpny = True  # True: add CPNY and CHPE HVDC lines; False: no CPNY and CHPE HVDC lines
    w_esr = False  # True: add ESRs; False: no ESRs
    w_vre = False  # True: add future solar and offshore wind; False: no future solar and offshore wind
    w_elec = False  # True: add residential building electrification; False: no residential building electrification

    start_date = datetime(2018, 1, 1, 0, 0, 0)
    end_date = datetime(2018, 12, 31, 0, 0, 0)
    timestamp_list = pd.date_range(start_date, end_date, freq='1D')

    # %% Set up logging
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler(f'ex_opf_{sim_name}.log'),
                                  logging.StreamHandler()])

    prog_start = time.time()
    logging.info(f'Start running multi-period OPF simulation {sim_name}.')

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

    if w_cpny:
        grid_data['dcline_prop'] = dcline_prop
        logging.info('With CPNY and CHPE HVDC lines.')  # Existing and planned HVDC lines
    else:
        grid_data['dcline_prop'] = dcline_prop[:4]  # Only existing HVDC lines
        logging.info('Without CPNY and CHPE HVDC lines.')

    # Read ESR property file
    filename = os.path.join(grid_data_dir, 'esr_prop.csv')
    esr_prop = pd.read_csv(filename, index_col=0)

    if w_esr:
        logging.info('With ESRs.')
        grid_data['esr_prop'] = esr_prop  # Existing and planned ESRs
    else:
        logging.info('No ESRs.')
        grid_data['esr_prop'] = esr_prop[:8]  # Only existing ESRs

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
        t = time.time()

        # Run OPF for one day (24 hours) plus leading hours
        # The first day is valid, the leading hours are used to dispatch batteries properly
        start_datetime = timestamp_list[d]
        end_datetime = start_datetime + timedelta(hours=24 + leading_hours)

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

    tot_elapsed = time.time() - prog_start
    logging.info(f"Finished multi-period OPF simulation {sim_name}. Total elapsed time: {tot_elapsed:.2f} seconds")
