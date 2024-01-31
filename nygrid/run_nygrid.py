import os

import pandas as pd

from nygrid.nygrid import NYGrid


def read_grid_data(data_dir, year):
    """
    Read grid data

    Parameters
    ----------
    data_dir : str
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
    load_profile = pd.read_csv(os.path.join(data_dir, f'load_profile_{year}.csv'),
                               parse_dates=['TimeStamp'], index_col='TimeStamp').asfreq('H')

    # Read generation profile
    gen_profile = pd.read_csv(os.path.join(data_dir, f'gen_profile_{year}.csv'),
                              parse_dates=['TimeStamp'], index_col='TimeStamp').asfreq('H')

    # Read generator capacity limit profile
    genmax_profile = pd.read_csv(os.path.join(data_dir, f'genmax_profile_{year}.csv'),
                                 parse_dates=['TimeStamp'], index_col='TimeStamp').asfreq('H')

    genmin_profile = pd.read_csv(os.path.join(data_dir, f'genmin_profile_{year}.csv'),
                                 parse_dates=['TimeStamp'], index_col='TimeStamp').asfreq('H')

    # Read generator ramp rate profile
    genramp30_profile = pd.read_csv(os.path.join(data_dir, f'genramp30_profile_{year}.csv'),
                                    parse_dates=['TimeStamp'], index_col='TimeStamp').asfreq('H')

    # Read generator cost profile (linear)
    gencost0_profile = pd.read_csv(os.path.join(data_dir, f'gencost0_profile_{year}.csv'),
                                   parse_dates=['TimeStamp'], index_col='TimeStamp').asfreq('H')

    gencost1_profile = pd.read_csv(os.path.join(data_dir, f'gencost1_profile_{year}.csv'),
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


def run_nygrid_one_day(s_time, e_time, grid_data, grid_data_dir, opts, init_gen):
    """
    Run NYGrid simulation for one day

    Parameters
    ----------
    s_time : datetime.datetime
        Start datetime
    e_time : datetime.datetime
        End datetime
    grid_data : dict
        Dictionary of grid data
    grid_data_dir : str
        Directory of grid data
    opts : dict
        Dictionary of options
    init_gen : numpy.ndarray
        Generator initial condition

    Returns
    -------
    results: dict
        Dictionary of results
    """

    # Create NYGrid object
    nygrid_sim = NYGrid(grid_data_dir,
                        start_datetime=s_time.strftime('%m-%d-%Y %H'),
                        end_datetime=e_time.strftime('%m-%d-%Y %H'),
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
    nygrid_sim.set_gen_init_data(gen_init=init_gen)

    # Set options
    nygrid_sim.set_options(opts)

    # Solve DC OPF
    nygrid_sim.solve_dc_opf()

    # Get nygrid_results
    results = nygrid_sim.get_results_dc_opf()

    return results
