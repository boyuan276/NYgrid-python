"""
Preprocessing functions for NY grid data.

Created: 2023-12-26, by Bo Yuan (Cornell University)
Last modified: 2023-12-26, by Bo Yuan (Cornell University)
"""

import os
import pandas as pd
from typing import List, Tuple, Optional


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
