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


def thermal_pg_2_heat_input(thermal_pg: pd.DataFrame,
                            gen_info: pd.DataFrame
                            ) -> pd.DataFrame:

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


def get_esr_results(results_list: List[Dict],
                    nygrid_sim: ng_grid.NYGrid
                    ) -> Dict[str, pd.DataFrame]:

    esr_pcrg_list = list()
    esr_pdis_list = list()
    esr_soc_list = list()

    for d in range(len(results_list)):
        # ESR charging/discharging power and state of charge
        esr_pcrg = results_list[d]['esrPCrg'].iloc[:24, :]
        esr_pcrg.columns = nygrid_sim.grid_prop['esr_prop']['ESR_NAME']
        esr_pcrg_list.append(esr_pcrg)

        esr_pdis = results_list[d]['esrPDis'].iloc[:24, :]
        esr_pdis.columns = nygrid_sim.grid_prop['esr_prop']['ESR_NAME']
        esr_pdis_list.append(esr_pdis)

        esr_soc = results_list[d]['esrSOC'].iloc[:24, :]
        esr_soc.columns = nygrid_sim.grid_prop['esr_prop']['ESR_NAME']
        esr_soc_list.append(esr_soc)

    esr_results = {
        'esrPCrg': pd.concat(esr_pcrg_list, axis=0),
        'esrPDis': pd.concat(esr_pdis_list, axis=0),
        'esrSOC': pd.concat(esr_soc_list, axis=0)
    }

    return esr_results


def get_lmp_results(results_list: List[Dict],
                    nygrid_sim: ng_grid.NYGrid
                    ) -> Dict[str, pd.DataFrame]:

    lmp_list = list()

    for d in range(len(results_list)):
        # Locational marginal price by bus
        lmp = results_list[d]['LMP'].iloc[:24, :]
        lmp.columns = nygrid_sim.grid_prop['bus_prop']['BUS_I']
        lmp_list.append(lmp)

    lmp_by_bus = pd.concat(lmp_list, axis=0)

    # Zonal average LMP
    bus_zone_alloc = nygrid_sim.grid_prop['bus_prop'].set_index("BUS_I").to_dict()[
        "BUS_ZONE"]
    lmp_by_zone = lmp_by_bus.T.groupby(bus_zone_alloc).mean().T

    lmp_results = {
        'LMP_by_bus': lmp_by_bus,
        'LMP_by_zone': lmp_by_zone
    }

    return lmp_results


def get_flow_results(results_list: List[Dict],
                     nygrid_sim: ng_grid.NYGrid
                     ) -> Dict[str, pd.DataFrame]:

    branch_names = (nygrid_sim.grid_prop['branch_prop']['F_BUS'].astype(str)
                    + '_' + nygrid_sim.grid_prop['branch_prop']['FROM_ZONE']
                    + '-' +
                    nygrid_sim.grid_prop['branch_prop']['T_BUS'].astype(str)
                    + '_' + nygrid_sim.grid_prop['branch_prop']['TO_ZONE'])

    branch_flow_list = list()
    for d in range(len(results_list)):
        # Branch flow
        flow = results_list[d]['PF'].iloc[:24, :]
        flow.columns = branch_names
        branch_flow_list.append(flow)

    if_names = (nygrid_sim.grid_prop['if_lim_prop']['TO_ZONE'] + '-'
                + nygrid_sim.grid_prop['if_lim_prop']['FROM_ZONE'])

    if_flow_list = list()
    for d in range(len(results_list)):
        # Interface flow
        if_flow = results_list[d]['IF'].iloc[:24, :]
        if_flow.columns = if_names
        if_flow_list.append(if_flow)

    flow_results = {
        'BranchFlow': pd.concat(branch_flow_list, axis=0),
        'InterfaceFlow': pd.concat(if_flow_list, axis=0)
    }

    return flow_results


def get_slack_penalties(results_list: List[Dict],
                     nygrid_sim: ng_grid.NYGrid
                     ) -> Dict[str, np.ndarray]:

    slack_penalty_results = dict()
    slack_names = ['s_over_gen', 's_load_shed',
                   's_ramp_up', 's_ramp_down',
                   's_br_max', 's_br_min',
                   's_if_max', 's_if_min',
                   's_esr_pcrg', 's_esr_pdis',
                   's_esr_soc_min', 's_esr_soc_max',
                   's_esr_soc_overt', 's_esr_soc_undert']
    penalty_names = ['over_gen_penalty', 'load_shed_penalty',
                     'ramp_up_penalty', 'ramp_down_penalty',
                     'br_max_penalty', 'br_min_penalty',
                     'if_max_penalty', 'if_min_penalty',
                     'esr_pcrg_penalty', 'esr_pdis_penalty',
                     'esr_soc_min_penalty', 'esr_soc_max_penalty',
                     'esr_soc_overt_penalty', 'esr_soc_undert_penalty']
    slack_penalty_names = slack_names + penalty_names

    for d in range(len(results_list)):
        # Slack variables
        for var in slack_penalty_names:
            slack = results_list[d][var][:24]

            if var not in slack_penalty_results:
                slack_penalty_results[var] = list()
            slack_penalty_results[var].append(slack)

    for var in slack_penalty_results.keys():
        slack_penalty_results[var] = np.concatenate(slack_penalty_results[var],
                                                    axis=0)

    return slack_penalty_results


def get_costs(results_list: List[Dict],
              nygrid_sim: ng_grid.NYGrid
              ) -> Dict[str, np.ndarray]:

    gen_idx_non_cvt = np.arange(0, (nygrid_sim.NG - nygrid_sim.NDCL*2
                                - nygrid_sim.NESR - nygrid_sim.NVRE))

    gen_cost_list = list()
    esr_cost_list = list()

    for d in range(len(results_list)):
        # Generator cost
        gen_cost = results_list[d]['gen_cost'][:24, gen_idx_non_cvt]
        gen_cost_list.append(gen_cost)

        # ESR cost
        esr_cost = results_list[d]['esr_cost'][:24, :]
        esr_cost_list.append(esr_cost)

    costs = {
        'gen_cost': np.concatenate(gen_cost_list, axis=0),
        'esr_cost': np.concatenate(esr_cost_list, axis=0)
    }

    return costs
