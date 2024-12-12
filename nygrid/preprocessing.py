"""
Preprocessing functions for NY grid data.

Created: 2023-12-26, by Bo Yuan (Cornell University)
Last modified: 2023-12-26, by Bo Yuan (Cornell University)
"""

import os
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional


ZONE_NAME2ID = {
    'WEST': 'A', 
    'GENESEE': 'B', 
    'CENTRAL': 'C', 
    'NORTH': 'D',
    'MOHAWK VALLEY': 'E', 
    'CAPITAL': 'F', 
    'HUDSON VALLEY': 'G',
    'MILLWOOD': 'H', 
    'DUNWOODIE': 'I', 
    'NYC': 'J', 
    'L ISLAND': 'K'
}


def agg_demand_county2bus(demand_inc_county: pd.DataFrame,
                          county2bus: pd.DataFrame
                          ) -> pd.DataFrame:
    """
    County-level consumption to bus-level consumption.

    Parameters
    ----------
    demand_inc_county : pd.DataFrame
        County-level consumption
    county2bus : pd.DataFrame
        County to bus mapping

    Returns
    -------
    demand_inc_bus : pd.DataFrame
        Bus-level consumption
    """

    demand_inc_county_erie = demand_inc_county['Erie']
    demand_inc_county_westchester = demand_inc_county['Westchester']
    demand_inc_county_rest = demand_inc_county.drop(
        columns=['Erie', 'Westchester'])

    county2bus_erie = county2bus[county2bus['NAME'] == 'Erie']
    county2bus_westchester = county2bus[county2bus['NAME'] == 'Westchester']
    county2bus_rest = county2bus[(county2bus['NAME'] != 'Erie') &
                                 (county2bus['NAME'] != 'Westchester')]

    demand_inc_bus_erie = demand_inc_county_erie.to_frame()
    demand_inc_bus_erie['55'] = demand_inc_bus_erie['Erie'] * 0.5
    demand_inc_bus_erie['57'] = demand_inc_bus_erie['Erie'] * 0.125
    demand_inc_bus_erie['59'] = demand_inc_bus_erie['Erie'] * 0.375
    demand_inc_bus_erie = demand_inc_bus_erie.drop(columns=['Erie'])
    demand_inc_bus_erie.columns = demand_inc_bus_erie.columns.astype(int)

    demand_inc_bus_westchester = demand_inc_county_westchester.to_frame()
    demand_inc_bus_westchester['74'] = demand_inc_bus_westchester['Westchester'] * 0.5
    demand_inc_bus_westchester['78'] = demand_inc_bus_westchester['Westchester'] * 0.5
    demand_inc_bus_westchester = demand_inc_bus_westchester.drop(columns=[
                                                                 'Westchester'])
    demand_inc_bus_westchester.columns = demand_inc_bus_westchester.columns.astype(int)

    county_bus_alloc_rest = county2bus_rest.set_index('NAME').to_dict()['busIdx']
    demand_inc_bus_rest = demand_inc_county_rest.T.groupby(
        county_bus_alloc_rest).sum().T

    demand_inc_bus = demand_inc_bus_rest.add(demand_inc_bus_erie, fill_value=0)
    demand_inc_bus = demand_inc_bus.add(demand_inc_bus_westchester, fill_value=0)
    demand_inc_bus.columns = demand_inc_bus.columns.astype(int)

    return demand_inc_bus


def get_building_load_change_county(county_id: str,
                                    upgrade_id: int,
                                    bldg_type_list: List[str],
                                    bldg_proc_dir: str
                                    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """

    Read building timeseries data aggregated by county and building type.

    Parameters
    ----------
    county_id : str
        County ID
    upgrade_id : int
        Upgrade ID
    bldg_type_list : list
        List of building types
    bldg_proc_dir : str
        Directory for processed building data

    Returns
    -------
    df_county_base: pd.DataFrame
        Dataframe with baseline energy consumption
    df_county_future: pd.DataFrame
        Dataframe with future energy consumption
    df_county_saving: pd.DataFrame
        Dataframe with energy savings
    """

    # Read building timeseries data aggregated by county and building type
    first_df = True

    for bldg_type in bldg_type_list:

        filename = os.path.join(bldg_proc_dir,
                                f"{county_id.lower()}_{bldg_type.replace(' ', '_').lower()}.parquet")

        if os.path.isfile(filename):
            # Future
            df_county_bldg_type_future = pd.read_parquet(
                filename, engine='pyarrow')
            col_total_cons = [col for col in df_county_bldg_type_future.columns if col.endswith(
                '.energy_consumption') and 'total' in col]
            df_county_bldg_type_future = df_county_bldg_type_future[col_total_cons]

            # Baseline
            df_county_bldg_type_base = pd.read_parquet(filename.replace(
                f'upgrade={upgrade_id}', 'upgrade=0'), engine='pyarrow')
            df_county_bldg_type_base = df_county_bldg_type_base[col_total_cons]

            # Savings = Baseline - Future
            df_county_bldg_type_saving = df_county_bldg_type_base - df_county_bldg_type_future

            # Add to county saving dataframe
            if first_df:
                df_county_base = df_county_bldg_type_base
                df_county_future = df_county_bldg_type_future
                df_county_saving = df_county_bldg_type_saving
                first_df = False
            else:
                df_county_base = df_county_base + df_county_bldg_type_base
                df_county_future = df_county_future + df_county_bldg_type_future
                df_county_saving = df_county_saving + df_county_bldg_type_saving

        else:
            print(
                f'Building load data is not available for county {county_id} {bldg_type}. Skipping...')
            continue

    # Rename columns
    col_rename = {col: col.split('.')[1] for col in df_county_base.columns}
    df_county_base = df_county_base.rename(columns=col_rename)
    df_county_future = df_county_future.rename(columns=col_rename)
    df_county_saving = df_county_saving.rename(columns=col_rename)

    return df_county_base, df_county_future, df_county_saving


def add_load_weighted(hourly_load_zonal: pd.DataFrame, 
                      bus_info: pd.DataFrame,
                      ) -> pd.DataFrame:
    
    """
    Distribute zonal load to individual buses based on load distribution ratio.

    Parameters
    ----------
    hourly_load_zonal : pd.DataFrame
        Zonal load timeseries
    bus_info : pd.DataFrame
        Bus information

    Returns
    -------
    bus_wload : pd.DataFrame
        Bus-level load timeseries
    """
    
    # Subset of bus in NY control area
    nys_bus = bus_info[~bus_info['zone'].isnull()]
    # Subset of bus with load
    nys_bus_wload = nys_bus[nys_bus['sumLoadP0'] > 0]

    # Load bus and ratio calculation
    zone_ids = nys_bus['zone'].unique()
    load_bus_zone = np.empty(11, dtype=object)
    load_ratio_zone = np.empty(11, dtype=object)
    num_load_bus_zone = np.zeros(11, dtype=int)

    # Calculate zonal load distribution ratio
    for i, zone_id in enumerate(zone_ids):
        load_bus_table = nys_bus_wload[nys_bus_wload['zone'] == zone_id]
        load_bus_zone[i] = load_bus_table['idx'].values
        if load_bus_table.shape[0] > 0:
            load_ratio_zone[i] = (load_bus_table['sumLoadP0'] /
                                load_bus_table['sumLoadP0'].sum()).values
        else:
            load_bus_zone[i] = nys_bus[nys_bus['zone'] == zone_id]['idx'].values
            load_ratio_zone[i] = np.ones(
                len(load_bus_zone[i])) / len(load_bus_zone[i])
        num_load_bus_zone[i] = len(load_bus_zone[i])

    num_load_bus_tot = sum(num_load_bus_zone)
    
    # Distribute zonal load to individual buses
    num_hours = hourly_load_zonal.shape[0]
    zone_load_bus = np.empty(len(zone_ids), dtype=object)
    load_bus_idx = np.zeros(num_load_bus_tot)
    load_bus_load = np.zeros((num_load_bus_tot, num_hours))
    n = 0
    for i, zone_id in enumerate(zone_ids):
        zone_load_tot = hourly_load_zonal[zone_id].values
        zone_load_bus[i] = np.outer(load_ratio_zone[i], zone_load_tot)
        load_bus_idx[n:n+num_load_bus_zone[i]] = load_bus_zone[i]
        load_bus_load[n:n+num_load_bus_zone[i], :] = zone_load_bus[i]
        n += num_load_bus_zone[i]

    bus_wload = pd.DataFrame(load_bus_load.T, columns=load_bus_idx,
                             index=hourly_load_zonal.index)
    bus_wload = bus_wload.sort_index(axis=1)
    bus_wload.columns = bus_wload.columns.astype(int)

    return bus_wload