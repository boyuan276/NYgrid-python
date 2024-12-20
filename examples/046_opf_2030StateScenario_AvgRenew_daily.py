"""
Run multi-period OPF with 2030 data.
Case name: 2030StateScenario_AvgRenew

Authors:
- Bo Yuan (Cornell University)
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
    ext_cost_factor = 0.0
    fo_cost_factor = 0.5
    sim_name = f'2030StateScenario_NoLargeLoad_ext{ext_cost_factor}_fo{fo_cost_factor}_daily'

    # Simulation time settings
    valid_days = 14
    lookahead_days = 2

    valid_hours = 24 * valid_days
    lookahead_hours = 24 * lookahead_days

    sim_start_time = datetime(2018, 1, 1, 0, 0, 0)
    sim_end_time = datetime(2018, 12, 31, 23, 0, 0)
    timestamp_list = pd.date_range(
        sim_start_time, sim_end_time, freq=f'{valid_days}D')
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

    grid_data_dir = os.path.join(data_dir, 'grid', '2030StateScenario_NoLargeLoad')
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
    grid_prop = ng_run.read_grid_prop(grid_data_dir,
                                      if_lims_prop_file='if_lims_prop_2030StateScenario.csv',
                                      esr_prop_file='esr_prop_2030StateScenario.csv',
                                      dcline_prop_file='dcline_prop_2030StateScenario.csv')

    # Read load and generation profiles
    grid_profile = ng_run.read_grid_profile(grid_data_dir, year=2030)

    # %% Modify grid data

    # Decrease external generation cost by 50%
    change_index = grid_prop["gen_prop"]["GEN_ZONE"].isin(
        ['PJM', 'IESO', 'NE']).to_numpy()

    gencost1_profile_new = grid_profile['gencost1_profile'].copy()
    gencost1_profile_new.loc[:, change_index] = \
        gencost1_profile_new.loc[:, change_index] * ext_cost_factor
    grid_profile['gencost1_profile'] = gencost1_profile_new

    # Decrease FO2, KER and FO6 generation costs
    change_index = grid_prop["gen_fuel"]["GEN_FUEL"].isin(
        [
            "CT_FO2",
            "CT_KER",
            "ST_FO6"
        ]).to_numpy()

    gencost0_profile_new = grid_profile['gencost0_profile'].copy()
    gencost0_profile_new.loc[:, change_index] = \
        gencost0_profile_new.loc[:, change_index] * fo_cost_factor
    grid_profile['gencost0_profile'] = gencost0_profile_new

    gencost1_profile_new = grid_profile['gencost1_profile'].copy()
    gencost1_profile_new.loc[:, change_index] = \
        gencost1_profile_new.loc[:, change_index] * fo_cost_factor
    grid_profile['gencost1_profile'] = gencost1_profile_new

    gencost_startup_profile_new = grid_profile['gencost_startup_profile'].copy(
    )
    gencost_startup_profile_new.loc[:, change_index] = \
        gencost_startup_profile_new.loc[:, change_index] * fo_cost_factor
    grid_profile['gencost_startup_profile'] = gencost_startup_profile_new

    # %% Set up OPF model

    # Set options
    options = {
        'UsePTDF': True,
        'solver': 'gurobi',
        'PenaltyForLoadShed': 10_000,
        'PenaltyForBranchMwViolation': 5_000,
        'PenaltyForInterfaceMWViolation': 5_000,
    }

    solver_options = {
        'NodefileStart': 0.5,
        'NodefileDir': sim_results_dir,
        'MIPGap': 1e-3
    }

    # No initial condition for the first day
    last_gen = np.zeros(grid_prop['gen_prop'].shape[0] +
                        grid_prop['esr_prop'].shape[0] +
                        grid_prop['dcline_prop'].shape[0]*2)
    last_gen_cmt = np.zeros(sum(grid_prop['gen_prop']['CMT_KEY'] == 1))
    last_soc = None
    hour_since_last_startup = None
    hour_since_last_shutdown = None

    # Restart from the middle
    # last_cycle_idx = 0
    last_cycle_idx = 26

    # Loop through all days
    for d in tqdm(range(last_cycle_idx, len(timestamp_list)), desc='Running OPF'):
        t = time.time()

        # Set clycle start and end datetime
        cycle_start_time = timestamp_list[d]

        if d < len(timestamp_list) - 1:
            cycle_end_time = cycle_start_time + \
                timedelta(hours=valid_hours + lookahead_hours)
        else:
            cycle_end_time = sim_end_time

        if cycle_end_time > sim_end_time:
            cycle_end_time = sim_end_time

        nygrid_results = ng_run.run_nygrid_sim(grid_prop=grid_prop,
                                               grid_profile=grid_profile,
                                               start_datetime=cycle_start_time,
                                               end_datetime=cycle_end_time,
                                               options=options,
                                               solver_options=solver_options,
                                               gen_init=last_gen,
                                               gen_init_cmt=last_gen_cmt,
                                               soc_init=last_soc,
                                               gen_last_startup_hour=hour_since_last_startup,
                                               gen_last_shutdown_hour=hour_since_last_shutdown,
                                               verbose=verbose)

        # Save simulation nygrid_results to pickle
        filename = f'nygrid_sim_{sim_name}_{cycle_start_time.strftime("%Y%m%d")}_{valid_days}_{lookahead_days}.pkl'
        with open(os.path.join(sim_results_dir, filename), 'wb') as f:
            pickle.dump(nygrid_results, f)
        logging.info(f'Saved simulation nygrid_results in {filename}')

        if d < len(timestamp_list) - 1:
            # Set initial conditions for the next iteration
            time_before_next_cycle = cycle_start_time + \
                timedelta(hours=valid_hours-1)

            # Set generator initial condition
            last_gen = nygrid_results['PG'].loc[time_before_next_cycle].to_numpy(
            ).squeeze()

            # Set generator commitment initial condition
            last_gen_cmt = nygrid_results['genCommit'].loc[time_before_next_cycle].to_numpy(
            ).squeeze()

            # Set ESR initial condition
            last_soc = nygrid_results['esrSOC'].loc[time_before_next_cycle].to_numpy(
            ).squeeze()

            # Calculate hours since last startup and shutdown
            hour_since_last_startup = ng_run.get_last_startup_hour(nygrid_results,
                                                                   time_before_next_cycle)
            hour_since_last_shutdown = ng_run.get_last_shutdown_hour(nygrid_results,
                                                                     time_before_next_cycle)

        elapsed = time.time() - t
        logging.info(
            f'Finished running for {cycle_start_time.strftime("%Y-%m-%d")}.')
        logging.info(f'Elapsed time: {elapsed:.2f} seconds')
        logging.info('-' * 80)

    tot_elapsed = time.time() - prog_start
    logging.info(f"Finished multi-period OPF simulation {sim_name}.")
    logging.info(f"Total elapsed time: {tot_elapsed:.2f} seconds")
