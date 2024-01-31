"""
Run multi-period OPF with 2018 data
without renewable generators

"""
# %% Packages

import os
import pandas as pd
from datetime import datetime, timedelta
from nygrid.run_nygrid import NYGrid
import pickle
import time
import logging


def read_grid_data(grid_data_dir, year):
    """
    Read grid data

    Parameters
    ----------
    grid_data_dir : str
        Directory of grid data
    year : int
        Year of grid data

    Returns
    -------
    grid_data : dict
        Dictionary of grid data
        Keys: 'load_profile', 'gen_profile', 'genmax_profile', 'genmin_profile', 'genramp30_profile',
                'gencost0_profile', 'gencost1_profile'
        Values: pandas.DataFrame
    """

    # Read load profile
    load_profile = pd.read_csv(os.path.join(grid_data_dir, f'load_profile_{year}.csv'),
                               parse_dates=['TimeStamp'], index_col='TimeStamp').asfreq('H')

    # Read generation profile
    gen_profile = pd.read_csv(os.path.join(grid_data_dir, f'gen_profile_{year}.csv'),
                              parse_dates=['TimeStamp'], index_col='TimeStamp').asfreq('H')

    # Read generator capacity limit profile
    genmax_profile = pd.read_csv(os.path.join(grid_data_dir, f'genmax_profile_{year}.csv'),
                                 parse_dates=['TimeStamp'], index_col='TimeStamp').asfreq('H')

    genmin_profile = pd.read_csv(os.path.join(grid_data_dir, f'genmin_profile_{year}.csv'),
                                 parse_dates=['TimeStamp'], index_col='TimeStamp').asfreq('H')

    # Read generator ramp rate profile
    genramp30_profile = pd.read_csv(os.path.join(grid_data_dir, f'genramp30_profile_{year}.csv'),
                                    parse_dates=['TimeStamp'], index_col='TimeStamp').asfreq('H')

    # Read generator cost profile (linear)
    gencost0_profile = pd.read_csv(os.path.join(grid_data_dir, f'gencost0_profile_{year}.csv'),
                                   parse_dates=['TimeStamp'], index_col='TimeStamp').asfreq('H')

    gencost1_profile = pd.read_csv(os.path.join(grid_data_dir, f'gencost1_profile_{year}.csv'),
                                   parse_dates=['TimeStamp'], index_col='TimeStamp').asfreq('H')

    grid_data = {
        'load_profile': load_profile,
        'gen_profile': gen_profile,
        'genmax_profile': genmax_profile,
        'genmin_profile': genmin_profile,
        'genramp30_profile': genramp30_profile,
        'gencost0_profile': gencost0_profile,
        'gencost1_profile': gencost1_profile,
    }

    return grid_data


def run_nygrid_one_day(start_datetime, end_datetime, grid_data, grid_data_dir, options, last_gen):
    """
    Run NYGrid simulation for one day

    Parameters
    ----------
    start_datetime : datetime.datetime
        Start datetime
    end_datetime : datetime.datetime
        End datetime
    grid_data : dict
        Dictionary of grid data
    grid_data_dir : str
        Directory of grid data
    options : dict
        Dictionary of options
    last_gen : numpy.ndarray
        Generator initial condition

    Returns
    -------
    results: dict
        Dictionary of results
    """

    # Create NYGrid object
    nygrid_sim = NYGrid(grid_data_dir,
                        start_datetime=start_datetime.strftime('%m-%d-%Y %H'),
                        end_datetime=end_datetime.strftime('%m-%d-%Y %H'),
                        dcline_prop=grid_data.get('dcline_prop', None),
                        esr_prop=grid_data.get('esr_prop', None),
                        vre_prop=grid_data.get('vre_prop', None))

    # Set load and generation time series data
    nygrid_sim.set_load_sch(grid_data['load_profile'])
    nygrid_sim.set_gen_mw_sch(grid_data['gen_profile'])
    nygrid_sim.set_gen_max_sch(grid_data['genmax_profile'])
    nygrid_sim.set_gen_min_sch(grid_data['genmin_profile'])
    nygrid_sim.set_gen_ramp_sch(grid_data['genramp30_profile'])
    nygrid_sim.set_gen_cost_sch(grid_data['gencost0_profile'], grid_data['gencost1_profile'])

    # Relax branch flow limits
    nygrid_sim.relax_external_branch_lim()

    # Set generator initial condition
    nygrid_sim.set_gen_init_data(gen_init=last_gen)

    # Set options
    nygrid_sim.set_options(options)

    # Solve DC OPF
    nygrid_sim.solve_dc_opf()

    # Get nygrid_results
    results = nygrid_sim.get_results_dc_opf()

    return results


if __name__ == '__main__':

    # %% Set up logging
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler('ex_opf_wo_renew.log'),
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

    if not os.path.exists(os.path.join(results_dir, 'wo_renew')):
        os.mkdir(os.path.join(results_dir, 'wo_renew'))

    # %% Read grid data
    start_date = datetime(2018, 1, 1, 0, 0, 0)
    end_date = datetime(2018, 1, 31, 0, 0, 0)
    timestamp_list = pd.date_range(start_date, end_date, freq='1D')

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
        filename = f'nygrid_sim_wo_renew_{start_datetime.strftime("%Y%m%d%H")}.pkl'
        with open(os.path.join(results_dir, 'wo_renew', filename), 'wb') as f:
            pickle.dump(nygrid_results, f)
        logging.info(f'Saved simulation nygrid_results in {filename}')

        elapsed = time.time() - t
        logging.info(f'Finished running for {start_datetime.strftime("%Y-%m-%d")}. Elapsed time: {elapsed:.2f} seconds')
        logging.info('-' * 80)
