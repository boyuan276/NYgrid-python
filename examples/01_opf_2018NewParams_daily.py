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
import pandas as pd
import pickle
import time
from datetime import datetime, timedelta
from tqdm import tqdm

import nygrid.run_nygrid as ng_run

if __name__ == '__main__':

    # %% Simulation settings
    # NOTE: Change the following settings to run the simulation
    ext_cost_factor = 0.2
    fo_cost_factor = 0.5
    sim_name = f'2018NewParams_ext{ext_cost_factor}_fo{fo_cost_factor}_daily'

    leading_hours = 24

    start_date = datetime(2018, 1, 1, 0, 0, 0)
    end_date = datetime(2018, 12, 31, 0, 0, 0)
    timestamp_list = pd.date_range(start_date, end_date, freq='1D')
    verbose = True

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
    sim_results_dir = os.path.join(results_dir, sim_name)
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

    # Decrease external generation cost by 50%
    change_index = grid_prop["gen_prop"]["GEN_ZONE"].isin(
        ['PJM','IESO']).to_numpy()

    gencost1_profile_new = grid_profile['gencost1_profile'].copy()
    gencost1_profile_new.loc[:, change_index] = \
        gencost1_profile_new.loc[:, change_index] * ext_cost_factor
    grid_profile['gencost1_profile'] = gencost1_profile_new

    # Decrease FO2, KER and FO6 generation costs
    change_index = grid_prop["gen_fuel"]["GEN_FUEL"].isin(
        ["CT_FO2", "CT_KER", "ST_FO6"]).to_numpy()

    gencost0_profile_new = grid_profile['gencost0_profile'].copy()
    gencost0_profile_new.loc[:, change_index] = \
        gencost0_profile_new.loc[:, change_index] * fo_cost_factor
    grid_profile['gencost0_profile'] = gencost0_profile_new

    gencost1_profile_new = grid_profile['gencost1_profile'].copy()
    gencost1_profile_new.loc[:, change_index] = \
        gencost1_profile_new.loc[:, change_index] * fo_cost_factor
    grid_profile['gencost1_profile'] = gencost1_profile_new

    gencost_startup_profile_new = grid_profile['gencost_startup_profile'].copy()
    gencost_startup_profile_new.loc[:, change_index] = \
        gencost_startup_profile_new.loc[:, change_index] * fo_cost_factor
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

    solver_options = {
        'NodefileStart': 0.5,
        'NodefileDir': sim_results_dir
    }

    # No initial condition for the first day
    last_gen = None
    last_gen_cmt = np.zeros(sum(grid_prop['gen_prop']['CMT_KEY'] == 1))
    last_soc = None
    hour_since_last_startup = None
    hour_since_last_shutdown = None

    # Loop through all days
    for d in tqdm(range(len(timestamp_list)), desc='Running OPF'):
        t = time.time()

        # Remove leading hours for the last day
        if d == len(timestamp_list) - 1:
            leading_hours = 0

        # Run OPF for one day (24 hours) plus leading hours
        # The first day is valid, the leading hours are used to dispatch batteries properly
        start_datetime = timestamp_list[d]
        end_datetime = start_datetime + timedelta(hours=23 + leading_hours)

        nygrid_results = ng_run.run_nygrid_sim(grid_prop=grid_prop,
                                               grid_profile=grid_profile,
                                               start_datetime=start_datetime,
                                               end_datetime=end_datetime,
                                               options=options,
                                               solver_options=solver_options,
                                               gen_init=last_gen,
                                               gen_init_cmt=last_gen_cmt,
                                               soc_init=last_soc,
                                               gen_last_startup_hour=hour_since_last_startup,
                                               gen_last_shutdown_hour=hour_since_last_shutdown,
                                               verbose=verbose)

        # Save simulation nygrid_results to pickle
        filename = f'nygrid_sim_{sim_name}_{start_datetime.strftime("%Y%m%d")}.pkl'
        with open(os.path.join(sim_results_dir, filename), 'wb') as f:
            pickle.dump(nygrid_results, f)
        logging.info(f'Saved simulation nygrid_results in {filename}')

        # Set initial conditions for the next iteration
        end_datetime_day1 = start_datetime + timedelta(hours=23)
        
        # Set generator initial condition
        last_gen = nygrid_results['PG'].loc[end_datetime_day1].to_numpy().squeeze()
        
        # Set generator commitment initial condition
        last_gen_cmt = nygrid_results['genCommit'].loc[end_datetime_day1].to_numpy().squeeze()
        
        # Set ESR initial condition
        last_soc = nygrid_results['esrSOC'].loc[end_datetime_day1].to_numpy().squeeze()

        # Calculate hours since last startup and shutdown
        hour_since_last_startup = ng_run.get_last_startup_hour(nygrid_results,
                                                                end_datetime_day1)
        hour_since_last_shutdown = ng_run.get_last_shutdown_hour(nygrid_results,
                                                                 end_datetime_day1)
        
        elapsed = time.time() - t
        logging.info(f'Finished running for {start_datetime.strftime("%Y-%m-%d")}.')
        logging.info(f'Elapsed time: {elapsed:.2f} seconds')
        logging.info('-' * 80)

    tot_elapsed = time.time() - prog_start
    logging.info(f"Finished multi-period OPF simulation {sim_name}.")
    logging.info("Total elapsed time: {tot_elapsed:.2f} seconds")