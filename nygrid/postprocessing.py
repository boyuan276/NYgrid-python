"""
Post-processing functions for NY grid data.

Created: 2023-12-26, by Bo Yuan (Cornell University)
Last modified: 2023-12-26, by Bo Yuan (Cornell University)
"""

import numpy as np
import pandas as pd

from typing import List, Dict, Union

import nygrid.nygrid as ng_grid


def get_pg_by_fuel(results: dict,
                   nygrid_sim: ng_grid.NYGrid,
                   valid_hours: int = 24
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

    # Get power output by generator type
    pg_non_cvt = results['PG'].iloc[:valid_hours, gen_idx_non_cvt]
    pg_non_cvt.columns = nygrid_sim.grid_prop['gen_prop']['GEN_NAME']

    pg_dcline_f = results['PG'].iloc[:valid_hours, dcline_idx_f]
    pg_dcline_f.columns = nygrid_sim.grid_prop['dcline_prop']['DC_NAME']

    pg_dcline_t = results['PG'].iloc[:valid_hours, dcline_idx_t]
    pg_dcline_t.columns = nygrid_sim.grid_prop['dcline_prop']['DC_NAME']

    pg_esr = results['PG'].iloc[:valid_hours, esr_idx]
    pg_esr.columns = nygrid_sim.grid_prop['esr_prop']['ESR_NAME']

    if nygrid_sim.NVRE > 0:
        pg_vre = results['PG'].iloc[:valid_hours, vre_idx]
        pg_vre.columns = nygrid_sim.grid_prop['vre_prop']['VRE_NAME']

    # Get power output by fuel type
    fuel_list_non_cvt = nygrid_sim.grid_prop['gen_fuel']['GEN_FUEL']

    # For non-converted units
    fuel_types = fuel_list_non_cvt.unique()
    pg_by_fuel = dict()

    for fuel_type in fuel_types:
        fuel_idx = np.where(fuel_list_non_cvt == fuel_type)[0]
        fuel_pg = pg_non_cvt.iloc[:, fuel_idx]
        fuel_pg = fuel_pg.iloc[:valid_hours, :]

        pg_by_fuel[fuel_type] = fuel_pg

    # For converted units: DCline_F, DCline_T, ESR, VRE
    pg_by_fuel['DCLine_F'] = pg_dcline_f
    pg_by_fuel['DCLine_T'] = pg_dcline_t
    pg_by_fuel['ESR'] = pg_esr
    if nygrid_sim.NVRE > 0:
        pg_by_fuel['VRE'] = pg_vre

    return pg_by_fuel


def get_pg_by_fuel_from_list(results_list: List[Dict],
                             nygrid_sim: ng_grid.NYGrid,
                             valid_hours: int = 24
                             ) -> Dict[str, pd.DataFrame]:

    pg_by_fuel_combined = dict()

    for results in results_list:

        pg_by_fuel = get_pg_by_fuel(results, nygrid_sim, valid_hours)

        for fuel_type in pg_by_fuel.keys():
            if fuel_type not in pg_by_fuel_combined:
                pg_by_fuel_combined[fuel_type] = list()
            pg_by_fuel_combined[fuel_type].append(pg_by_fuel[fuel_type])

    for fuel_type in pg_by_fuel_combined.keys():
        pg_by_fuel_combined[fuel_type] = pd.concat(
            pg_by_fuel_combined[fuel_type])

    return pg_by_fuel_combined


def thermal_pg_2_heat_input(thermal_pg: pd.DataFrame,
                            thermal_params: pd.DataFrame
                            ) -> pd.DataFrame:

    heat_input = thermal_pg.copy()

    # Loop through all thermal generators
    for gen_name in thermal_pg.columns:
        # Get heat rate linear model
        heat_rate_lm = thermal_params[thermal_params['GEN_NAME'] == gen_name][[
            'heat_1', 'heat_0']].to_numpy().flatten()

        # Get heat input
        heat_input[gen_name] = heat_rate_lm[0] * \
            thermal_pg[gen_name] + heat_rate_lm[1]

    return heat_input


def get_esr_results(results: dict,
                    nygrid_sim: ng_grid.NYGrid,
                    valid_hours: int = 24
                    ) -> Dict[str, pd.DataFrame]:

    # ESR charging/discharging power and state of charge
    esr_pcrg = results['esrPCrg'].iloc[:valid_hours, :]
    esr_pcrg.columns = nygrid_sim.grid_prop['esr_prop']['ESR_NAME']

    esr_pdis = results['esrPDis'].iloc[:valid_hours, :]
    esr_pdis.columns = nygrid_sim.grid_prop['esr_prop']['ESR_NAME']

    esr_soc = results['esrSOC'].iloc[:valid_hours, :]
    esr_soc.columns = nygrid_sim.grid_prop['esr_prop']['ESR_NAME']

    esr_results = {
        'esrPCrg': esr_pcrg,
        'esrPDis': esr_pdis,
        'esrSOC': esr_soc
    }

    return esr_results


def get_esr_results_from_list(results_list: List[Dict],
                              nygrid_sim: ng_grid.NYGrid,
                              valid_hours: int = 24
                              ) -> Dict[str, pd.DataFrame]:

    esr_results_combined = dict()

    for results in results_list:

        esr_results = get_esr_results(results, nygrid_sim, valid_hours)

        for key in esr_results.keys():
            if key not in esr_results_combined:
                esr_results_combined[key] = list()
            esr_results_combined[key].append(esr_results[key])

    for key in esr_results_combined.keys():
        esr_results_combined[key] = pd.concat(esr_results_combined[key])

    return esr_results_combined


def get_lmp_results(results: dict,
                    nygrid_sim: ng_grid.NYGrid,
                    valid_hours: int = 24
                    ) -> Dict[str, pd.DataFrame]:

    # Locational marginal price by bus
    lmp_by_bus = results['LMP'].iloc[:valid_hours, :]
    lmp_by_bus.columns = nygrid_sim.grid_prop['bus_prop']['BUS_I']

    # Zonal average LMP
    bus_zone_alloc = nygrid_sim.grid_prop['bus_prop'].set_index("BUS_I").to_dict()[
        "BUS_ZONE"]
    lmp_by_zone = lmp_by_bus.T.groupby(bus_zone_alloc).mean().T

    lmp_results = {
        'LMP_by_bus': lmp_by_bus,
        'LMP_by_zone': lmp_by_zone
    }

    return lmp_results


def get_lmp_results_from_list(results_list: List[Dict],
                              nygrid_sim: ng_grid.NYGrid,
                              valid_hours: int = 24
                              ) -> Dict[str, pd.DataFrame]:

    lmp_results_combined = dict()

    for results in results_list:

        lmp_results = get_lmp_results(results, nygrid_sim, valid_hours)

        for key in lmp_results.keys():
            if key not in lmp_results_combined:
                lmp_results_combined[key] = list()
            lmp_results_combined[key].append(lmp_results[key])

    for key in lmp_results_combined.keys():
        lmp_results_combined[key] = pd.concat(lmp_results_combined[key])

    return lmp_results_combined


def get_flow_results(results: dict,
                     nygrid_sim: ng_grid.NYGrid,
                     valid_hours: int = 24
                     ) -> Dict[str, pd.DataFrame]:

    # Branch flow
    branch_names = (nygrid_sim.grid_prop['branch_prop']['F_BUS'].astype(str)
                    + '_' + nygrid_sim.grid_prop['branch_prop']['FROM_ZONE']
                    + '-' +
                    nygrid_sim.grid_prop['branch_prop']['T_BUS'].astype(str)
                    + '_' + nygrid_sim.grid_prop['branch_prop']['TO_ZONE'])

    branch_flow = results['PF'].iloc[:valid_hours, :]
    branch_flow.columns = branch_names

    # Interface flow
    if_names = (nygrid_sim.grid_prop['if_lim_prop']['FROM_ZONE'] + '-'
                + nygrid_sim.grid_prop['if_lim_prop']['TO_ZONE'])

    if_flow = results['IF'].iloc[:valid_hours, :]
    if_flow.columns = if_names

    flow_results = {
        'BranchFlow': branch_flow,
        'InterfaceFlow': if_flow
    }

    return flow_results


def get_flow_results_from_list(results_list: List[Dict],
                               nygrid_sim: ng_grid.NYGrid,
                               valid_hours: int = 24
                               ) -> Dict[str, pd.DataFrame]:

    flow_results_combined = dict()

    for results in results_list:

        flow_results = get_flow_results(results, nygrid_sim, valid_hours)

        for key in flow_results.keys():
            if key not in flow_results_combined:
                flow_results_combined[key] = list()
            flow_results_combined[key].append(flow_results[key])

    for key in flow_results_combined.keys():
        flow_results_combined[key] = pd.concat(flow_results_combined[key])

    return flow_results_combined


def get_slack_results(results: dict,
                      nygrid_sim: ng_grid.NYGrid,
                      valid_hours: int = 24
                      ) -> Dict[str, pd.DataFrame]:

    slack_results = dict()
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

    # Slack variables
    for var in slack_penalty_names:
        slack_results[var] = results[var][:valid_hours]

    return slack_results


def get_slack_results_from_list(results_list: List[Dict],
                                nygrid_sim: ng_grid.NYGrid,
                                valid_hours: int = 24
                                ) -> Dict[str, pd.DataFrame]:

    slack_results_combined = dict()

    for results in results_list:

        slack_results = get_slack_results(results, nygrid_sim,
                                          valid_hours)

        for key in slack_results.keys():
            if key not in slack_results_combined:
                slack_results_combined[key] = list()
            slack_results_combined[key].append(slack_results[key])

    for key in slack_results_combined.keys():
        slack_results_combined[key] = np.concatenate(slack_results_combined[key])

    return slack_results_combined


def get_cost_results(results: dict,
              nygrid_sim: ng_grid.NYGrid,
              valid_hours: int = 24
              ) -> Dict[str, np.ndarray]:

    gen_idx_non_cvt = np.arange(0, (nygrid_sim.NG - nygrid_sim.NDCL*2
                                - nygrid_sim.NESR - nygrid_sim.NVRE))

    # Generator energy cost
    gen_cost = results['gen_cost'][:valid_hours, gen_idx_non_cvt]

    # Generator non-load cost
    gen_cost = results['gen_cost_noload'][:valid_hours, :]

    # Generator start-up cost
    gen_cost = results['gen_cost_startup'][:valid_hours, :]

    # Generator shut-down cost
    gen_cost = results['gen_cost_shutdown'][:valid_hours, :]

    # ESR cost
    esr_cost = results['esr_cost'][:24, :]

    costs = {
        'gen_cost': gen_cost,
        'gen_cost_noload': gen_cost,
        'gen_cost_startup': gen_cost,
        'gen_cost_shutdown': gen_cost,
        'esr_cost': esr_cost
    }

    return costs


def get_cost_results_from_list(results_list: List[Dict],
                               nygrid_sim: ng_grid.NYGrid,
                               valid_hours: int = 24
                               ) -> Dict[str, np.ndarray]:

    cost_results_combined = dict()

    for results in results_list:

        cost_results = get_flow_results(results, nygrid_sim, valid_hours)

        for key in cost_results.keys():
            if key not in cost_results_combined:
                cost_results_combined[key] = list()
            cost_results_combined[key].append(cost_results[key])

    for key in cost_results_combined.keys():
        cost_results_combined[key] = pd.concat(cost_results_combined[key])

    return cost_results_combined
