import pandas as pd
import matplotlib.pyplot as plt
import typing
import numpy as np
from typing import List, Tuple, Optional


def plot_gen(thermal_pg: pd.Series,
             gen_hist: pd.Series,
             gen_max: pd.Series,
             gen_min: pd.Series,
             ax: plt.Axes,
             title: Optional[str] = None) -> plt.Axes:

    ax.plot(thermal_pg.index, thermal_pg,
            marker='*', label='OPF',
            lw=2, alpha=0.7)
    ax.plot(thermal_pg.index, gen_hist,
            marker='o', label='OPF MATLAB',
            lw=2, alpha=0.7)
    ax.plot(thermal_pg.index, gen_max,
            linestyle='--', label='max',
            lw=2, alpha=0.7)
    ax.plot(thermal_pg.index, gen_min,
            linestyle='--', label='min',
            lw=2, alpha=0.7)
    ax.legend()
    if title:
        ax.set_title(title)

    return ax


def calc_pg_by_fuel_sum_by_zone(pg_by_fuel_dict, 
                                grid_prop,
                                gen_fuel_rename,
                                zone_order):
    
    # Change HQ import zone from 'D' and 'J' to 'HQ'
    gen_prop = grid_prop['gen_prop'].copy()
    hq_idx = gen_prop[(gen_prop['GEN_ZONE'].isin(['D','J'])) & (gen_prop['UNIT_TYPE'] == 'Import')].index
    gen_prop.loc[hq_idx, 'GEN_ZONE'] = 'HQ'

    gen2zone_dict = gen_prop.set_index('GEN_NAME')[
        'GEN_ZONE'].to_dict()
    esr2zone_dict = grid_prop['esr_prop'].set_index('ESR_NAME')[
        'ESR_ZONE'].to_dict()
    dclinef2zone_dict = grid_prop['dcline_prop'].set_index('DC_NAME')[
        'FROM_ZONE'].to_dict()
    dclinet2zone_dict = grid_prop['dcline_prop'].set_index('DC_NAME')[
        'TO_ZONE'].to_dict()

    pg_by_fuel_sum_by_zone = dict()

    for fuel_type, pg_by_fuel in pg_by_fuel_dict.items():
        if fuel_type == 'DCLine_F':
            pg_by_fuel_by_zone = pg_by_fuel.groupby(
                [dclinef2zone_dict], axis=1).sum().sum().to_dict()
        elif fuel_type == 'DCLine_T':
            pg_by_fuel_by_zone = pg_by_fuel.groupby(
                [dclinet2zone_dict], axis=1).sum().sum().to_dict()
        elif fuel_type == 'ESR':
            pg_by_fuel_by_zone = pg_by_fuel.groupby(
                [esr2zone_dict], axis=1).sum().sum().to_dict()
        else:
            pg_by_fuel_by_zone = pg_by_fuel.groupby(
                [gen2zone_dict], axis=1).sum().sum().to_dict()
        pg_by_fuel_sum_by_zone[fuel_type] = pg_by_fuel_by_zone
        
    pg_by_fuel_sum_by_zone = pd.DataFrame(pg_by_fuel_sum_by_zone)

    # Add zone I
    if 'I' not in pg_by_fuel_sum_by_zone.index:
        pg_by_fuel_sum_by_zone.loc['I'] = np.nan

    # Rename columns
    pg_by_fuel_sum_by_zone = pg_by_fuel_sum_by_zone.rename(columns=gen_fuel_rename)

    pg_by_fuel_sum_by_zone = pg_by_fuel_sum_by_zone[gen_fuel_rename.values()]

    pg_by_fuel_sum_by_zone = pg_by_fuel_sum_by_zone/1e6 # Convert to TWh

    pg_by_fuel_sum_by_zone = pg_by_fuel_sum_by_zone.T[zone_order].T

    return pg_by_fuel_sum_by_zone


def calc_pg_by_fuel_sum_by_month(pg_by_fuel_dict,
                                 gen_fuel_rename):
       
    # Group by generator type and month
    pg_by_fuel_sum_by_month = dict()

    for fuel_type, pg_by_fuel in pg_by_fuel_dict.items():
        pg_by_fuel_by_month = pg_by_fuel.groupby(pg_by_fuel.index.month).sum().sum(axis=1).to_dict()
        pg_by_fuel_sum_by_month[fuel_type] = pg_by_fuel_by_month
        
    pg_by_fuel_sum_by_month = pd.DataFrame(pg_by_fuel_sum_by_month)

    # Rename columns
    pg_by_fuel_sum_by_month = pg_by_fuel_sum_by_month.rename(columns=gen_fuel_rename)

    pg_by_fuel_sum_by_month = pg_by_fuel_sum_by_month[gen_fuel_rename.values()]

    pg_by_fuel_sum_by_month = pg_by_fuel_sum_by_month/1e6 # Convert to TWh

    return pg_by_fuel_sum_by_month


def calc_curtail_by_zone_by_month(pg_by_fuel_dict,
                                  grid_prop,
                                  grid_profile,
                                  gen_fuel_name,
                                  zone_order):
    
    # Change HQ import zone from 'D' and 'J' to 'HQ'
    gen_prop = grid_prop['gen_prop'].copy()
    hq_idx = gen_prop[(gen_prop['GEN_ZONE'].isin(['D','J'])) & (gen_prop['UNIT_TYPE'] == 'Import')].index
    gen_prop.loc[hq_idx, 'GEN_ZONE'] = 'HQ'

    gen2zone_dict = gen_prop.set_index('GEN_NAME')[
        'GEN_ZONE'].to_dict()
    
    if gen_fuel_name == 'Load_Load':
        # Large Load curtailment (as negative generation)
        gen_index = grid_prop["gen_fuel"]["GEN_FUEL"].isin([gen_fuel_name]).to_numpy()
        gen_max_profile = grid_profile['genmin_profile'].loc[:, gen_index] * -1
        gen_profile = pg_by_fuel_dict[gen_fuel_name] * -1

    else:
        # Maximum available UPV generation
        gen_index = grid_prop["gen_fuel"]["GEN_FUEL"].isin([gen_fuel_name]).to_numpy()
        gen_max_profile = grid_profile['genmax_profile'].loc[:, gen_index]
        gen_profile = pg_by_fuel_dict[gen_fuel_name]

    # Aggregate UPV available generation by zone
    gen_max_zone = gen_max_profile.groupby(
        [gen2zone_dict], axis=1).sum()

    # Aggregate by month
    gen_max_zone_month = gen_max_zone.groupby(
        gen_max_zone.index.month).sum()

    # Calculate UPV curtailment
    gen_curtailment = gen_max_profile - gen_profile

    # Remove negative curtailment
    gen_curtailment[gen_curtailment < 0] = 0

    # Aggregate UPV curtailment by zone
    gen_curtailment_zone = gen_curtailment.groupby(
        [gen2zone_dict], axis=1).sum()

    # Aggregate by month
    gen_curtailment_zone_month = gen_curtailment_zone.groupby(
        gen_curtailment_zone.index.month).sum()

    # Calculate curtailment percentage
    gen_curtailment_pct_zone_month = gen_curtailment_zone_month / gen_max_zone_month * 100

    # Convert to GWh
    gen_curtailment_zone_month = gen_curtailment_zone_month / 1e3

    # Add missing zones
    for zone in zone_order[:11]:
        if zone not in gen_curtailment_zone_month.columns:
            gen_curtailment_zone_month[zone] = np.nan
            gen_curtailment_pct_zone_month[zone] = np.nan

    # Reorder columns
    gen_curtailment_zone_month = gen_curtailment_zone_month[zone_order[:11]]
    gen_curtailment_pct_zone_month = gen_curtailment_pct_zone_month[zone_order[:11]]

    return gen_curtailment_zone_month, gen_curtailment_pct_zone_month