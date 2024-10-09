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
    sim_name = '2018Baseline'
    leading_hours = 12
    w_cpny = False  # True: add CPNY and CHPE HVDC lines; False: no CPNY and CHPE HVDC lines
    w_esr = False  # True: add ESRs; False: no ESRs
    w_vre = False  # True: add future solar and offshore wind; False: no future solar and offshore wind
    w_elec = False  # True: add residential building electrification; False: no residential building electrification

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

    grid_data_dir = os.path.join(data_dir, 'grid', '2018Baseline')
    if not os.path.exists(grid_data_dir):
        raise FileNotFoundError('Grid data directory not found.')

    logging.info('Grid data directory: {}'.format(grid_data_dir))

    fig_dir = os.path.join(os.path.dirname(data_dir), 'figures')
    logging.info('Figure directory: {}'.format(fig_dir))

    results_dir = os.path.join(os.path.dirname(data_dir), 'results')
    logging.info('Results directory: {}'.format(results_dir))

    # NOTE: Change sim_results_dir to the directory where simulation results are saved
    # sim_results_dir = os.path.join(results_dir, sim_name)
    sim_results_dir = os.path.join(results_dir, 'cc_1.5x_ct_2x_st_3x')
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

    # Increase CT and ST generation costs for energy
    ct_index = grid_prop["gen_prop"]["GEN_FUEL"].isin(
        ["Combustion Turbine", "Internal Combustion", "Jet Engine"]).to_numpy()
    st_index = grid_prop["gen_prop"]["GEN_FUEL"].isin(
        ["Steam Turbine"]).to_numpy()
    cc_index = grid_prop["gen_prop"]["GEN_FUEL"].isin(
        ["Combined Cycle"]).to_numpy()

    gencost1_profile_new = grid_profile['gencost1_profile'].copy()
    gencost1_profile_new.loc[:,
                             ct_index] = gencost1_profile_new.loc[:, ct_index] * 1.5
    gencost1_profile_new.loc[:,
                             cc_index] = gencost1_profile_new.loc[:, cc_index] * 2
    gencost1_profile_new.loc[:,
                             st_index] = gencost1_profile_new.loc[:, st_index] * 3
    grid_profile['gencost1_profile'] = gencost1_profile_new

    # Increase CT and ST generation costs for startup
    gencost_startup_profile_new = grid_profile['gencost_startup_profile'].copy(
    )
    gencost_startup_profile_new.loc[:,
                                    ct_index] = gencost_startup_profile_new.loc[:, ct_index] * 1.5
    gencost_startup_profile_new.loc[:,
                                    cc_index] = gencost_startup_profile_new.loc[:, cc_index] * 2
    gencost_startup_profile_new.loc[:,
                                    st_index] = gencost_startup_profile_new.loc[:, st_index] * 3
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

    # Loop through all days
    for d in range(len(timestamp_list)):
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
                                               gen_init=last_gen,
                                               gen_init_cmt=last_gen_cmt,
                                               soc_init=last_soc,
                                               verbose=verbose)

        # Save simulation nygrid_results to pickle
        filename = f'nygrid_sim_{sim_name}_{start_datetime.strftime("%Y%m%d%H")}.pkl'
        with open(os.path.join(sim_results_dir, filename), 'wb') as f:
            pickle.dump(nygrid_results, f)
        logging.info(f'Saved simulation nygrid_results in {filename}')

        # Set initial conditions for the next iteration
        end_datetime_day1 = start_datetime + timedelta(hours=23)
        # Set generator initial condition
        last_gen = nygrid_results['PG'].loc[end_datetime_day1].to_numpy(
        ).squeeze()
        # Set generator commitment initial condition
        last_gen_cmt = nygrid_results['genCommit'].loc[end_datetime_day1].to_numpy(
        ).squeeze()
        # Set ESR initial condition
        last_soc = nygrid_results['esrSOC'].loc[end_datetime_day1].to_numpy(
        ).squeeze()

        elapsed = time.time() - t
        logging.info(f'Finished running for {start_datetime.strftime("%Y-%m-%d")}.')
        logging.info(f'Elapsed time: {elapsed:.2f} seconds')
        logging.info('-' * 80)

    tot_elapsed = time.time() - prog_start
    logging.info(
        f"Finished multi-period OPF simulation {sim_name}. Total elapsed time: {tot_elapsed:.2f} seconds")
