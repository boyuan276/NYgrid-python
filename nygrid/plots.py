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