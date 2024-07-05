"""
Post-processing functions for NY grid data.

Created: 2023-12-26, by Bo Yuan (Cornell University)
Last modified: 2023-12-26, by Bo Yuan (Cornell University)
"""

import numpy as np
import pandas as pd

from typing import List, Dict, Union

import nygrid.nygrid as ng_grid


def get_pg_by_fuel(results_list: List[Dict],
                   nygrid_sim: ng_grid.NYGrid
                   ) -> Dict[str, pd.DataFrame]:

    gen_idx_non_cvt = np.arange(0, (nygrid_sim.NG - nygrid_sim.NDCL*2
                                - nygrid_sim.NESR - nygrid_sim.NVRE))
    dcline_idx_f = np.arange((nygrid_sim.NG - nygrid_sim.NDCL * 2
                              - nygrid_sim.NESR - nygrid_sim.NVRE),
                             (nygrid_sim.NG - nygrid_sim.NDCL
                             - nygrid_sim.NESR - nygrid_sim.NVRE))
    dcline_idx_t = np.arange((nygrid_sim.NG - nygrid_sim.NDCL
                              - nygrid_sim.NESR - nygrid_sim.NVRE),
                             (nygrid_sim.NG - nygrid_sim.NESR
                              - nygrid_sim.NVRE))
    esr_idx = np.arange(nygrid_sim.NG - nygrid_sim.NESR - nygrid_sim.NVRE,
                        nygrid_sim.NG - nygrid_sim.NVRE)
    vre_idx = np.arange(nygrid_sim.NG -
                        nygrid_sim.NVRE, nygrid_sim.NG)

    pg_by_fuel_dict = dict()

    for d in range(len(results_list)):
        # Get power output by generator type
        pg_non_cvt = results_list[d]['PG'].iloc[:24, gen_idx_non_cvt]
        pg_non_cvt.columns = nygrid_sim.grid_prop['gen_prop']['GEN_NAME']

        pg_dcline_f = results_list[d]['PG'].iloc[:24, dcline_idx_f]
        pg_dcline_f.columns = nygrid_sim.grid_prop['dcline_prop']['DC_NAME']

        pg_dcline_t = results_list[d]['PG'].iloc[:24, dcline_idx_t]
        pg_dcline_t.columns = nygrid_sim.grid_prop['dcline_prop']['DC_NAME']

        pg_esr = results_list[d]['PG'].iloc[:24, esr_idx]
        pg_esr.columns = nygrid_sim.grid_prop['esr_prop']['ESR_NAME']

        if nygrid_sim.NVRE > 0:
            pg_vre = results_list[d]['PG'].iloc[:24, vre_idx]
            pg_vre.columns = nygrid_sim.grid_prop['vre_prop']['VRE_NAME']

        # Get power output by fuel type
        fuel_list_non_cvt = nygrid_sim.grid_prop['gen_prop']['GEN_FUEL']
        pg_by_fuel = get_pg_by_fuel_daily(pg_table=pg_non_cvt,
                                          genfuel_list=fuel_list_non_cvt)
        pg_by_fuel['DCLine_F'] = pg_dcline_f
        pg_by_fuel['DCLine_T'] = pg_dcline_t
        pg_by_fuel['ESR'] = pg_esr
        if nygrid_sim.NVRE > 0:
            pg_by_fuel['VRE'] = pg_vre

        for fuel_type in pg_by_fuel.keys():
            if fuel_type not in pg_by_fuel_dict:
                pg_by_fuel_dict[fuel_type] = list()
            pg_by_fuel_dict[fuel_type].append(pg_by_fuel[fuel_type])

    for fuel_type in pg_by_fuel_dict.keys():
        pg_by_fuel_dict[fuel_type] = pd.concat(pg_by_fuel_dict[fuel_type])

    return pg_by_fuel_dict


def get_pg_by_fuel_daily(pg_table: pd.DataFrame,
                         genfuel_list: pd.Series
                         ) -> Dict[str, pd.DataFrame]:
    """
    NOTE: This doesn't include Added DCline, ESR, and VRE.
    It only includes thermal generators and renewable generators
    that are defined in te gen_prop table.
    """

    fuel_types = genfuel_list.unique()
    pg_by_fuel = dict()

    for fuel_type in fuel_types:
        fuel_idx = np.where(genfuel_list == fuel_type)[0]
        fuel_pg = pg_table.iloc[:, fuel_idx]
        fuel_pg = fuel_pg.iloc[:24, :]

        pg_by_fuel[fuel_type] = fuel_pg

    return pg_by_fuel


def thermal_pg_2_heat_input(thermal_pg, gen_info):

    heat_input = thermal_pg.copy()

    # Loop through all thermal generators
    for gen_name in thermal_pg.columns:
        # Get heat rate linear model
        heat_rate_lm = gen_info[gen_info.NYISOName == gen_name][[
            'HeatRateLM_1', 'HeatRateLM_0']].to_numpy().flatten()

        # Get heat input
        heat_input[gen_name] = heat_rate_lm[0] * \
            thermal_pg[gen_name] + heat_rate_lm[1]

    return heat_input
