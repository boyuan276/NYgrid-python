import os

import numpy as np
import pandas as pd
import pickle
from typing import Union, Dict, Tuple, Any

from nygrid.nygrid import NYGrid
from nygrid.preprocessing import agg_demand_county2bus, get_building_load_change_county


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
    # Remove 'Bus' prefix in column names
    load_profile.columns = load_profile.columns.str.replace('Bus', '').astype(int)

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
                  offshore_wind_data_dir: Union[str, os.PathLike]
                  ) -> Tuple[pd.DataFrame, pd.DataFrame]:
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

    # %% Renewable generation time series
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
    offshore_wind_gen.index = offshore_wind_gen.index.tz_localize(  # type: ignore
        'US/Eastern', ambiguous='infer')
    offshore_wind_gen.index.freq = 'H'

    # Wind farm capacity info
    capacity = [816, 1260, 924, 1230]
    capacity_nyc, capacity_li = np.sum(capacity[:2]), np.sum(capacity[2:])

    # Correct offshore wind generation
    offshore_wind_gen['power_nyc'] = np.where(offshore_wind_gen['power_nyc'] > capacity_nyc, capacity_nyc,
                                              offshore_wind_gen['power_nyc'])
    offshore_wind_gen['power_li'] = np.where(offshore_wind_gen['power_li'] > capacity_li, capacity_li,
                                             offshore_wind_gen['power_li'])

    # %% Aggregate renewable generation to buses
    # Aggregate current solar generation
    current_solar_2bus = pd.read_csv(os.path.join(
        solar_data_dir, f'solar_farms_2bus.csv'), index_col='zip_code')
    groupby_dict = current_solar_2bus['busIdx'].to_dict()
    current_solar_gen_bus = current_solar_gen.T.groupby(groupby_dict).sum().T
    current_solar_gen_bus = current_solar_gen_bus / 1e3  # convert from kW to MW

    # Aggregate future solar generation
    future_solar_2bus = pd.read_csv(os.path.join(
        solar_data_dir, f'future_solar_farms_2bus.csv'), index_col=0)
    groupby_dict = future_solar_2bus['busIdx'].to_dict()
    future_solar_gen_bus = future_solar_gen.T.groupby(groupby_dict).sum().T
    future_solar_gen_bus = future_solar_gen_bus / 1e3  # convert from kW to MW

    # Aggregate onshore wind generation
    onshore_wind_2bus = pd.read_csv(os.path.join(
        onshore_wind_data_dir, f'onshore_wind_2bus.csv'), index_col=0)
    groupby_dict = onshore_wind_2bus['busIdx'].to_dict()
    onshore_wind_gen_bus = onshore_wind_gen.T.groupby(groupby_dict).sum().T
    onshore_wind_gen_bus = onshore_wind_gen_bus / 1e3  # convert from kW to MW

    # Aggregate offshore wind generation
    offshore_wind_gen_bus = offshore_wind_gen[['power_nyc', 'power_li']].rename(
        columns={'power_nyc': 81, 'power_li': 80})

    # %% Adjust renewable generation profiles
    # 18.08% of current solar generation is built before 2018 (base year)
    # Scale down current solar generation by 18.08%
    pct_current_solar_built = 0.1808
    # current_solar_gen_bus_built = current_solar_gen_bus * pct_current_solar_built
    future_solar_gen_bus_planned = current_solar_gen_bus * \
        (1 - pct_current_solar_built)
    future_solar_gen_bus = future_solar_gen_bus.add(
        future_solar_gen_bus_planned, fill_value=0)

    # 90.6% of onshore wind generation is built before 2018 (base year)
    # Scale down onshore wind generation by 90.6%
    pct_onshore_wind_built = 0.906
    onshore_wind_gen_bus = onshore_wind_gen_bus * (1 - pct_onshore_wind_built)

    vre_profiles = {
        'FutSol': future_solar_gen_bus,
        'OnWind': onshore_wind_gen_bus,
        'OffWind': offshore_wind_gen_bus
    }
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
    genmax_profile_vre = pd.concat([
        future_solar_gen_bus,
        onshore_wind_gen_bus,
        offshore_wind_gen_bus
    ], axis=1)
    genmax_profile_vre.columns = vre_prop['VRE_NAME']
    genmax_profile_vre.index = genmax_profile_vre.index.tz_convert(  # type: ignore

        'US/Eastern').tz_localize(None)
    return vre_prop, genmax_profile_vre


def read_res_building_elec_data(data_dir: Union[str, os.PathLike],
                                upgrade_id: int,
                                county_attrs: pd.DataFrame
                                ) -> pd.DataFrame:
    """
    Residential building energy changes due to electrification.

    Parameters
    ----------
    data_dir : str
        Directory of residential building data
    upgrade_id : int
        Residential building upgrade scenario ID
    county_attrs : pd.DataFrame
        County attributes

    Returns
    -------
    res_load_change_county : pd.DataFrame
        Residential building load change by county
    """

    # Directory for processed data output
    res_bldg_proc_dir = os.path.join(data_dir,
                                     'county_processed',
                                     f'upgrade={upgrade_id}')

    # Create a list of building types
    res_bldg_type_list = ['Single-Family Detached',
                          'Multi-Family with 5+ Units',
                          'Multi-Family with 2 - 4 Units',
                          'Mobile Home',
                          'Single-Family Attached']

    # Read pre-processed EUSS energy saving data
    res_county_ts_list = list()

    for i, row in county_attrs.iterrows():
        county_name = row['NAME']
        fips = row['FIPS_CODE']
        county_id = f'G{str(fips)[:2]}0{str(fips)[2:]}0'

        _, _, df_county_saving_amy2018 = get_building_load_change_county(
            county_id, upgrade_id, res_bldg_type_list, res_bldg_proc_dir)

        res_county_ts = df_county_saving_amy2018['electricity'].rename(
            county_name)
        res_county_ts_list.append(res_county_ts)

    # Convert savings to increase
    res_load_change_county = pd.concat(res_county_ts_list, axis=1) * -1
    # Convert from kW to MW
    res_load_change_county = res_load_change_county / 1e3

    return res_load_change_county


def read_com_building_elec_data(data_dir: Union[str, os.PathLike],
                                upgrade_id: int,
                                county_attrs: pd.DataFrame,
                                ) -> pd.DataFrame:

    # Directory for processed data output
    com_bldg_proc_dir = os.path.join(data_dir,
                                     'county_processed',
                                     f'upgrade={upgrade_id}')

    # Create a list of building types
    com_bldg_type_list = ['RetailStripmall',
                          'SmallOffice',
                          'RetailStandalone',
                          'LargeOffice',
                          'Warehouse',
                          'SecondarySchool',
                          'PrimarySchool',
                          'Outpatient',
                          'QuickServiceRestaurant',
                          'MediumOffice',
                          'FullServiceRestaurant',
                          'SmallHotel',
                          'LargeHotel',
                          'Hospital']

    # Read pre-processed EUSS energy saving data
    com_county_ts_list = list()

    for i, row in county_attrs.iterrows():
        county_name = row['NAME']
        fips = row['FIPS_CODE']
        county_id = f'G{str(fips)[:2]}0{str(fips)[2:]}0'

        _, _, df_county_saving_amy2018 = get_building_load_change_county(
            county_id, upgrade_id, com_bldg_type_list, com_bldg_proc_dir)

        com_county_ts = df_county_saving_amy2018['electricity'].rename(
            county_name)
        com_county_ts_list.append(com_county_ts)

    # Convert savings to increase
    com_load_change_county = pd.concat(com_county_ts_list, axis=1) * -1
    # Convert from kW to MW
    com_load_change_county = com_load_change_county / 1e3

    return com_load_change_county


def read_ev_elec_data(data_dir: Union[str, os.PathLike],
                      upgrade_id: int,
                      county_attrs: pd.DataFrame
                      ) -> pd.DataFrame:
    """
    Electric vehicle energy changes due to electrification.

    Parameters
    ----------
    data_dir : str
        Directory of EV data
    upgrade_id : int
        EV upgrade scenario ID
    county_attrs : pd.DataFrame
        County attributes

    Returns
    -------
    ev_load_change_county : pd.DataFrame
        EV load change by county
    """

    # Create a list of EV charger types
    charger_types = ['home_l1', 'home_l2', 'work_l1',
                     'work_l2', 'public_l2', 'public_l3']

    ev_county_ts_list = list()

    for i, row in county_attrs.iterrows():
        county_name = row['NAME']
        county_name = county_name.replace(' ', '_')

        monthly_list = list()

        for m in range(1, 13):
            filename = f"{county_name}_County_month{m}_scen{upgrade_id}_temp_gridLoad.csv"
            ev_load = pd.read_csv(os.path.join(data_dir, filename),
                                  parse_dates=True, index_col=0)

            # Sum by charger types and resample to hourly
            ev_load_total = ev_load[charger_types].sum(
                axis=1).resample('H').sum()

            monthly_list.append(ev_load_total)

        ev_county_ts = pd.concat(monthly_list, axis=0).rename(county_name)
        ev_county_ts_list.append(ev_county_ts)

    ev_load_change_county = pd.concat(ev_county_ts_list, axis=1)

    # Convert from kW to MW
    ev_load_change_county = ev_load_change_county / 1e3

    return ev_load_change_county


def read_electrification_data(electrification_dict: Dict[str, Any],
                              county_attrs: pd.DataFrame,
                              county_2_bus: pd.DataFrame
                              ) -> Dict[str, Any]:
    """

    Parameters
    ----------
    electrification_dict : dict
        Dictionary of electrification data
        Keys: 'res_building', 'com_building', 'electric_vehicle'
        In each dictionary, the following keys are required:
            'data_dir': str
            'upgrade_id': int
    county_attrs : pd.DataFrame
        County attributes
    county_2_bus : pd.DataFrame
        County to bus mapping

    Returns
    -------
    electrification_dict : dict
        Dictionary of electrification data
        Keys: 'res_building', 'com_building', 'electric_vehicle'
        In each dictionary, the following keys are required:
            'data_dir': str
            'upgrade_id': int
            'load_change': pd.DataFrame
    """

    print(f"Get electrification data for {len(electrification_dict)} sectors:")
    print(list(electrification_dict.keys()))

    process_functions = {
        'res_building': read_res_building_elec_data,
        'com_building': read_com_building_elec_data,
        'electric_vehicle': read_ev_elec_data
    }

    for sector, attrs in electrification_dict.items():
        if sector in process_functions:
            print(f"Processing {sector} electrification data...")
            func = process_functions[sector]
            load_change_county = func(data_dir=attrs['data_dir'],
                                      upgrade_id=attrs['upgrade_id'],
                                      county_attrs=county_attrs)
            load_change_bus = agg_demand_county2bus(load_change_county,
                                                    county_2_bus)
            electrification_dict[sector]['load_change'] = load_change_bus
        else:
            raise ValueError(f"Invalid sector: {sector}")

    return electrification_dict


def run_nygrid_one_day(s_time: pd.Timestamp,
                       e_time: pd.Timestamp,
                       grid_data: Dict[str, pd.DataFrame],
                       grid_data_dir: Union[str, os.PathLike],
                       opts: Dict[str, Any],
                       init_gen: Union[np.ndarray, None]
                       ) -> Dict[str, pd.DataFrame]:
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
    nygrid_sim.set_gen_cost_sch(grid_data['gencost0_profile'],
                                grid_data['gencost1_profile'])

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
