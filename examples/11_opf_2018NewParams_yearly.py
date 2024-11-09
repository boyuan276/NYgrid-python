"""
Run multi-period OPF with 2018 data.
1. Baseline case:
No CPNY and CHPE HVDC lines.
No ESRs.
No future solar and offshore wind.
No residential building electrification.
"""
# %% Packages

import logging
import os
import numpy as np
import pickle
import time
from datetime import datetime, timedelta

import pandas as pd

import nygrid.run_nygrid as ng_run

if __name__ == '__main__':

    # %% Simulation settings
    # NOTE: Change the following settings to run the simulation
    sim_name = '2018NewParams'
    leading_hours = 12
    
    start_date = datetime(2018, 1, 1, 0, 0, 0)
    end_date = datetime(2018, 12, 31, 0, 0, 0)
    timestamp_list = pd.date_range(start_date, end_date, freq='1D')
    verbose = False

    if 'examples' in os.getcwd():
        os.chdir('../')

    # %% Set up logging
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler(f'examples/logs/ex_opf_{sim_name}.log'),
                                  logging.StreamHandler()])

    prog_start = time.time()
    logging.info(f'Start running multi-period OPF simulation {sim_name}.')

    # %% Set up directories
    cwd = os.getcwd()
    data_dir = os.path.join(cwd, 'data')

    grid_data_dir = os.path.join(data_dir, 'grid', '2018NewParams')
    if not os.path.exists(grid_data_dir):
        raise FileNotFoundError('Grid data directory not found.')

    logging.info('Grid data directory: {}'.format(grid_data_dir))

    fig_dir = os.path.join(os.path.dirname(data_dir), 'figures')
    logging.info('Figure directory: {}'.format(fig_dir))

    results_dir = os.path.join(os.path.dirname(data_dir), 'results')
    logging.info('Results directory: {}'.format(results_dir))

    # NOTE: Change sim_results_dir to the directory where simulation results are saved
    # sim_results_dir = os.path.join(results_dir, sim_name)
    sim_results_dir = os.path.join(results_dir, '2018NewParams')
    if not os.path.exists(sim_results_dir):
        os.mkdir(sim_results_dir)
        logging.info(
            f'Created simulation results directory: {sim_results_dir}')

    # %% Read grid data

    # Read grid property file
    grid_prop = ng_run.read_grid_prop(grid_data_dir)

    # Read load and generation profiles
    grid_profile = ng_run.read_grid_profile(grid_data_dir, start_date.year)

    # %% Modify grid data

    # Decrease external load by 50%
    bus_idx_ext = grid_prop['bus_prop'][grid_prop['bus_prop']['BUS_ZONE'].isin(['NE','PJM','IESO'])]['BUS_I']
    load_profile_new = grid_profile['load_profile'].copy()
    load_profile_new.loc[:, bus_idx_ext] = load_profile_new.loc[:, bus_idx_ext] * 0.5
    grid_profile['load_profile'] = load_profile_new

    # Increase FO2, KER and BIT generation costs
    change_index = grid_prop["gen_fuel"]["GEN_FUEL"].isin(
        ["CT_FO2", "CT_KER", "ST_BIT"]
    ).to_numpy()

    gencost0_profile_new = grid_profile['gencost0_profile'].copy()
    gencost0_profile_new.loc[:, change_index] = gencost0_profile_new.loc[:, change_index] * 3
    grid_profile['gencost0_profile'] = gencost0_profile_new

    gencost1_profile_new = grid_profile['gencost1_profile'].copy()
    gencost1_profile_new.loc[:, change_index] = gencost1_profile_new.loc[:, change_index] * 3
    grid_profile['gencost1_profile'] = gencost1_profile_new

    gencost_startup_profile_new = grid_profile['gencost_startup_profile'].copy()
    gencost_startup_profile_new.loc[:, change_index] = gencost_startup_profile_new.loc[:, change_index] * 3
    grid_profile['gencost_startup_profile'] = gencost_startup_profile_new

    # %% Set up OPF model

    # Set options
    options = {
        'UsePTDF': True,
        'solver': 'gurobi',
        'PenaltyForLoadShed': 20_000,
        # 'PenaltyForBranchMwViolation': 10_000,
        # 'PenaltyForInterfaceMWViolation': 10_000
    }

    # No initial condition for the first day
    last_gen = None
    last_gen_cmt = None
    last_soc = None

    # Run for the entire year
    d = 0
    start_datetime = timestamp_list[d]
    end_datetime = start_datetime + timedelta(hours=24*364+23)
    print(f'Start time: {start_datetime}')
    print(f'End time: {end_datetime}')

    nygrid_results = ng_run.run_nygrid_sim(grid_prop=grid_prop,
                                           grid_profile=grid_profile,
                                           start_datetime=start_datetime,
                                           end_datetime=end_datetime,
                                           options=options,
                                           gen_init=last_gen,
                                           gen_init_cmt=last_gen_cmt,
                                           soc_init=last_soc,
                                           verbose=verbose)

    # Save simulation nygrid_results to pickle
    filename = f'nygrid_sim_{sim_name}_{start_datetime.strftime("%Y")}_yearly.pkl'
    with open(os.path.join(sim_results_dir, filename), 'wb') as f:
        pickle.dump(nygrid_results, f)
    logging.info(f'Saved simulation nygrid_results in {filename}')

    tot_elapsed = time.time() - prog_start
    logging.info(
        f"Finished multi-period OPF simulation {sim_name}. Total elapsed time: {tot_elapsed:.2f} seconds")
