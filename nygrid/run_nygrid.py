import os
import numpy as np
import pandas as pd
import pickle
from nygrid.nygrid import NYGrid
from typing import Union, Dict, Tuple, Any


def read_grid_data(data_dir: Union[str, os.PathLike],
                   year: int) -> Dict[str, pd.DataFrame]:
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


def read_vre_data(solar_data_dir: Union[str, os.PathLike],
                  onshore_wind_data_dir: Union[str, os.PathLike],
                  offshore_wind_data_dir: Union[str, os.PathLike]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """

    Parameters
    ----------
    solar_data_dir: str
        Directory of solar data
    onshore_wind_data_dir: str
        Directory of onshore wind data
    offshore_wind_data_dir: str
        Directory of offshore wind data

    Returns
    -------
    vre_prop: pandas.DataFrame
        VRE properties
    genmax_profile_vre: pandas.DataFrame
        VRE generation profiles
    """

    # Renewable generation time series
    current_solar_gen = pd.read_csv(os.path.join(solar_data_dir, f'current_solar_gen_1hr.csv'),
                                    parse_dates=['Time'], index_col='Time').asfreq('H')
    current_solar_gen.columns = current_solar_gen.columns.astype(int)

    future_solar_gen = pd.read_csv(os.path.join(solar_data_dir, f'future_solar_gen_1hr.csv'),
                                   parse_dates=['Time'], index_col='Time').asfreq('H')
    future_solar_gen.columns = future_solar_gen.columns.astype(int)

    onshore_wind_gen = pd.read_csv(os.path.join(onshore_wind_data_dir, f'current_wind_gen_1hr.csv'),
                                   parse_dates=['Time'], index_col='Time').asfreq('H')
    onshore_wind_gen.columns = onshore_wind_gen.columns.astype(int)

    offshore_wind_gen = pd.read_csv(os.path.join(offshore_wind_data_dir, f'power_load_2018.csv'),
                                    parse_dates=['timestamp'], index_col='timestamp')
    offshore_wind_gen.index = offshore_wind_gen.index.tz_localize('US/Eastern', ambiguous='infer')
    offshore_wind_gen.index.freq = 'H'

    # Wind farm capacity info
    capacity = [816, 1260, 924, 1230]
    capacity_nyc, capacity_li = np.sum(capacity[:2]), np.sum(capacity[2:])

    # Correct offshore wind generation
    offshore_wind_gen['power_nyc'] = np.where(offshore_wind_gen['power_nyc'] > capacity_nyc, capacity_nyc,
                                              offshore_wind_gen['power_nyc'])
    offshore_wind_gen['power_li'] = np.where(offshore_wind_gen['power_li'] > capacity_li, capacity_li,
                                             offshore_wind_gen['power_li'])

    # Renewable allocation table
    current_solar_2bus = pd.read_csv(os.path.join(solar_data_dir, f'solar_farms_2bus.csv'), index_col='zip_code')
    future_solar_2bus = pd.read_csv(os.path.join(solar_data_dir, f'future_solar_farms_2bus.csv'), index_col=0)
    onshore_wind_2bus = pd.read_csv(os.path.join(onshore_wind_data_dir, f'onshore_wind_2bus.csv'), index_col=0)

    # Aggregate current solar generation
    groupby_dict = current_solar_2bus['busIdx'].to_dict()
    current_solar_gen_agg = current_solar_gen.groupby(groupby_dict, axis=1).sum()
    current_solar_gen_agg = current_solar_gen_agg / 1e3  # convert from kW to MW

    # Aggregate future solar generation
    groupby_dict = future_solar_2bus['busIdx'].to_dict()
    future_solar_gen_agg = future_solar_gen.groupby(groupby_dict, axis=1).sum()
    future_solar_gen_agg = future_solar_gen_agg / 1e3  # convert from kW to MW

    # Aggregate onshore wind generation
    groupby_dict = onshore_wind_2bus['busIdx'].to_dict()
    onshore_wind_gen_agg = onshore_wind_gen.groupby(groupby_dict, axis=1).sum()
    onshore_wind_gen_agg = onshore_wind_gen_agg / 1e3  # convert from kW to MW

    # Aggregate offshore wind generation
    offshore_wind_gen_agg = offshore_wind_gen[['power_nyc', 'power_li']].rename(
        columns={'power_nyc': 81, 'power_li': 82})

    # 18.08% of current solar generation is built before 2018 (base year)
    # Scale down current solar generation by 18.08%
    pct_current_solar_built = 0.1808
    current_solar_gen_agg = current_solar_gen_agg * (1 - pct_current_solar_built)

    # 90.6% of onshore wind generation is built before 2018 (base year)
    # Scale down onshore wind generation by 90.6%
    pct_onshore_wind_built = 0.906
    onshore_wind_gen_agg = onshore_wind_gen_agg * (1 - pct_onshore_wind_built)

    vre_profiles = {'CurSol': current_solar_gen_agg,
                    'FutSol': future_solar_gen_agg,
                    'OnWind': onshore_wind_gen_agg,
                    'OffWind': offshore_wind_gen_agg}
    vre_prop_list = list()
    for key, profile in vre_profiles.items():
        vre_prop_a = pd.DataFrame(data={'VRE_BUS': profile.columns,
                                        'VRE_PMAX': profile.max(axis=0),
                                        'VRE_PMIN': 0,
                                        'VRE_TYPE': 'Solar',
                                        'VRE_NAME': [f'{key}_{col}' for col in profile.columns]})
        vre_prop_list.append(vre_prop_a)

    # Combine gen_prop tables
    vre_prop = pd.concat(vre_prop_list, ignore_index=True)

    # Combine genmax tables
    genmax_profile_vre = pd.concat([current_solar_gen_agg, future_solar_gen_agg,
                                    onshore_wind_gen_agg, offshore_wind_gen_agg], axis=1)
    genmax_profile_vre.columns = vre_prop['VRE_NAME']
    genmax_profile_vre.index = genmax_profile_vre.index.tz_convert('US/Eastern').tz_localize(None)

    return vre_prop, genmax_profile_vre


def read_electrification_data(buildings_data_dir: Union[str, os.PathLike]) -> pd.DataFrame:
    """

    Parameters
    ----------
    buildings_data_dir: str
        Directory of buildings data

    Returns
    -------
    res_load_change_bus: pandas.DataFrame
        Changes in residential load
    """
    # Read county to bus allocation table
    county_2bus = pd.read_csv(os.path.join(buildings_data_dir, 'county_centroids_2bus.csv'),
                              index_col=0)
    fips_list = ['G' + s[:2] + '0' + s[2:] + '0' for s in county_2bus['FIPS_CODE'].astype(str)]

    # County-level load data
    # Residential buildings current load (upgrade=0)
    pickle_dir = os.path.join(buildings_data_dir, 'euss_processed', 'upgrade=0')
    res_current_load = pd.DataFrame()

    for ii in range(len(fips_list)):
        pickle_file = os.path.join(pickle_dir, f'{fips_list[0]}_elec_total.pkl')
        # Read pickle file
        with open(os.path.join(pickle_dir, pickle_file), 'rb') as f:
            county_load = pickle.load(f)
            county_load = county_load.rename(columns={'total': fips_list[ii]})
            res_current_load = pd.concat([res_current_load, county_load], axis=1)

    # Residential buildings future load (upgrade=10)
    pickle_dir = os.path.join(buildings_data_dir, 'euss_processed', 'upgrade=10')
    res_elec_load = pd.DataFrame()

    for ii in range(len(fips_list)):
        pickle_file = os.path.join(pickle_dir, f'{fips_list[0]}_elec_total.pkl')
        # Read pickle file
        with open(os.path.join(pickle_dir, pickle_file), 'rb') as f:
            county_load = pickle.load(f)
            county_load = county_load.rename(columns={'total': fips_list[ii]})
            res_elec_load = pd.concat([res_elec_load, county_load], axis=1)

    # Changes in residential load
    res_load_change = res_elec_load - res_current_load
    res_load_change.index.name = 'Time'

    # Aggregate building load to buses
    groupby_dict = dict(zip(fips_list, county_2bus['busIdx']))
    res_load_change_bus = res_load_change.groupby(groupby_dict, axis=1).sum()
    res_load_change_bus = res_load_change_bus / 1e3  # convert from kW to MW

    return res_load_change_bus


def run_nygrid_one_day(s_time: pd.Timestamp,
                       e_time: pd.Timestamp,
                       grid_data: Dict[str, pd.DataFrame],
                       grid_data_dir: Union[str, os.PathLike],
                       opts: Dict[str, Any],
                       init_gen: np.ndarray) -> Dict[str, pd.DataFrame]:
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

    if grid_data.get('genmax_profile_vre', None) is not None:
        nygrid_sim.set_vre_max_sch(grid_data['genmax_profile_vre'])

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
