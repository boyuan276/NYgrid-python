"""
Run multi-period OPF with 2018 data
with renewable generators
including wind and solar

"""
# %% Packages
import os
from pyomo.environ import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from nygrid.run_nygrid import NYGrid
import pickle
import time

t = time.time()

# %% Set up directories
cwd = os.getcwd()
if 'examples' in cwd:
    parent_dir = os.path.dirname(cwd)
    data_dir = os.path.join(parent_dir, 'data')
else:
    data_dir = os.path.join(cwd, 'data')

grid_data_dir = os.path.join(data_dir, 'grid')
if not os.path.exists(grid_data_dir):
    raise FileNotFoundError('Grid data directory not found.')

print('Grid data directory: {}'.format(grid_data_dir))

fig_dir = os.path.join(os.path.dirname(data_dir), 'figures')
print('Figure directory: {}'.format(fig_dir))

results_dir = os.path.join(os.path.dirname(data_dir), 'results')
print('Results directory: {}'.format(results_dir))

if not os.path.exists(os.path.join(results_dir, 'w_offshore')):
    os.mkdir(os.path.join(results_dir, 'w_offshore'))

solar_data_dir = os.path.join(data_dir, 'solar')
print('Solar data directory: {}'.format(solar_data_dir))

onshore_wind_data_dir = os.path.join(data_dir, 'onshore_wind')
print('Onshore wind data directory: {}'.format(onshore_wind_data_dir))

offshore_wind_data_dir = os.path.join(data_dir, 'offshore_wind')
print('Offshore wind data directory: {}'.format(offshore_wind_data_dir))

# %% Read grid data
start_date = datetime(2018, 1, 1, 0, 0, 0)
end_date = datetime(2019, 1, 1, 0, 0, 0)
timestamp_list = pd.date_range(start_date, end_date, freq='1D')

# Read load profile
load_profile = pd.read_csv(os.path.join(grid_data_dir, f'load_profile_{start_date.year}.csv'), 
                           parse_dates=['TimeStamp'], index_col='TimeStamp')
load_profile.index.freq = 'H'

# Read generation profile
gen_profile = pd.read_csv(os.path.join(grid_data_dir, f'gen_profile_{start_date.year}.csv'), 
                           parse_dates=['TimeStamp'], index_col='TimeStamp')
gen_profile.index.freq = 'H'

# Read generator capacity limit profile
genmax_profile = pd.read_csv(os.path.join(grid_data_dir, f'genmax_profile_{start_date.year}.csv'), 
                           parse_dates=['TimeStamp'], index_col='TimeStamp')
genmax_profile.index.freq = 'H'

genmin_profile = pd.read_csv(os.path.join(grid_data_dir, f'genmin_profile_{start_date.year}.csv'), 
                           parse_dates=['TimeStamp'], index_col='TimeStamp')
genmin_profile.index.freq = 'H'

# Read generator ramp rate profile
genramp30_profile = pd.read_csv(os.path.join(grid_data_dir, f'genramp30_profile_{start_date.year}.csv'), 
                           parse_dates=['TimeStamp'], index_col='TimeStamp')
genramp30_profile.index.freq = 'H'

# Read generator cost profile (linear)
gencost0_profile = pd.read_csv(os.path.join(grid_data_dir, f'gencost0_profile_{start_date.year}.csv'), 
                           parse_dates=['TimeStamp'], index_col='TimeStamp')
gencost0_profile.index.freq = 'H'

gencost1_profile = pd.read_csv(os.path.join(grid_data_dir, f'gencost1_profile_{start_date.year}.csv'), 
                           parse_dates=['TimeStamp'], index_col='TimeStamp')
gencost1_profile.index.freq = 'H'

# %% Read renewable generation data
# Renewable generation time series
current_solar_gen = pd.read_csv(os.path.join(solar_data_dir, f'current_solar_gen_1hr.csv'),
                                parse_dates=['Time'], index_col='Time')
current_solar_gen.index.freq = 'H'
current_solar_gen.columns = current_solar_gen.columns.astype(int)

future_solar_gen = pd.read_csv(os.path.join(solar_data_dir, f'future_solar_gen_1hr.csv'),
                                parse_dates=['Time'], index_col='Time')
future_solar_gen.index.freq = 'H'
future_solar_gen.columns = future_solar_gen.columns.astype(int)

onshore_wind_gen = pd.read_csv(os.path.join(onshore_wind_data_dir, f'current_wind_gen_1hr.csv'),
                                parse_dates=['Time'], index_col='Time')
onshore_wind_gen.index.freq = 'H'
onshore_wind_gen.columns = onshore_wind_gen.columns.astype(int)

offshore_wind_gen = pd.read_csv(os.path.join(offshore_wind_data_dir, f'power_load_2018.csv'),
                                parse_dates=['timestamp'], index_col='timestamp')
offshore_wind_gen.index = offshore_wind_gen.index.tz_localize('US/Eastern', ambiguous='infer')
offshore_wind_gen.index.freq = 'H'

# Renewable allocation table
current_solar_2bus = pd.read_csv(os.path.join(solar_data_dir, f'solar_farms_2bus.csv'), 
                                 index_col='zip_code')
future_solar_2bus = pd.read_csv(os.path.join(solar_data_dir, f'future_solar_farms_2bus.csv'), index_col=0)
onshore_wind_2bus = pd.read_csv(os.path.join(onshore_wind_data_dir, f'onshore_wind_2bus.csv'), index_col=0)

# Aggregate current solar generation
groupby_dict = current_solar_2bus['busIdx'].to_dict()
current_solar_gen_agg = current_solar_gen.groupby(groupby_dict, axis=1).sum()
current_solar_gen_agg = current_solar_gen_agg/1e3 # convert from kW to MW

# Aggregate future solar generation
groupby_dict = future_solar_2bus['busIdx'].to_dict()
future_solar_gen_agg = future_solar_gen.groupby(groupby_dict, axis=1).sum()
future_solar_gen_agg = future_solar_gen_agg/1e3 # convert from kW to MW

# Aggregate onshore wind generation
groupby_dict = onshore_wind_2bus['busIdx'].to_dict()
onshore_wind_gen_agg = onshore_wind_gen.groupby(groupby_dict, axis=1).sum()
onshore_wind_gen_agg = onshore_wind_gen_agg/1e3 # convert from kW to MW

# Aggregate offshore wind generation
offshore_wind_gen_agg = offshore_wind_gen[['power_nyc', 'power_li']].rename(
    columns={'power_nyc': 81, 'power_li': 79})

# %% Update load profile
# Tread renewable generation as negative load
load_profile_copy = load_profile.copy()
load_profile_copy.columns = [int(col.replace('Bus', '')) for col in load_profile_copy.columns]
load_profile_copy.index = current_solar_gen_agg.index

# 18.08% of current solar generation is built before 2018 (base year)
# Scale down current solar generation by 18.08%
pct_current_solar_built = 0.1808
current_solar_gen_agg = current_solar_gen_agg * (1-pct_current_solar_built)

# 90.6% of onshore wind generation is built before 2018 (base year)
# Scale down onshore wind generation by 90.6%
pct_onshore_wind_built = 0.906
onshore_wind_gen_agg = onshore_wind_gen_agg * (1-pct_onshore_wind_built)

# Total renewable generation by bus
total_renewable = pd.DataFrame()
total_renewable = total_renewable.add(onshore_wind_gen_agg, fill_value=0)
total_renewable = total_renewable.add(offshore_wind_gen_agg, fill_value=0)
# total_renewable = total_renewable.add(current_solar_gen_agg, fill_value=0)
# total_renewable = total_renewable.add(future_solar_gen_agg, fill_value=0)

# Load profile after subtracting renewable generation
load_profile_renewable = load_profile_copy.subtract(total_renewable, fill_value=0)

# Scale down external load by the same percentage as NYS
ny_buses = np.arange(37, 83)
ext_buses = np.array(list(set(load_profile_copy.columns) - set(ny_buses)))
ny_load_profile = load_profile_copy[ny_buses]
ext_load_profile = load_profile_copy[ext_buses]

# NY load change rate in each hour
ny_change_pct = ((load_profile_renewable[ny_buses].sum(axis=1)
                  - ny_load_profile.sum(axis=1)) / ny_load_profile.sum(axis=1))
print(f'NYS annual load change: {ny_change_pct.mean()*100:.2f}%')

ext_change = ext_load_profile.multiply(ny_change_pct*0.2, axis=0)
load_profile_renewable = load_profile_renewable.add(ext_change, fill_value=0)
print(f'External load change is set to 20% of NYS load change.')

# Reset timestamp index to time zone unaware
load_profile_copy.index = load_profile.index
load_profile_renewable.index = load_profile.index

# %% Read thermal generator info table
filename = os.path.join(data_dir, 'genInfo.csv')
gen_info = pd.read_csv(filename)
num_thermal = gen_info.shape[0]
gen_rename = {gen_info.index[i]: gen_info.NYISOName[i] for i in range(num_thermal)}

# %% Set up OPF model
timestamp_list = pd.date_range(start_date, end_date, freq='1D')

# Loop through all days
for d in range(len(timestamp_list)-1):
    # Run one day at a time
    start_datetime = timestamp_list[d]
    end_datetime = start_datetime + timedelta(hours=23)

    # Read MATPOWER case file
    ppc_filename = os.path.join(data_dir, 'ny_grid.mat')

    nygrid_sim = NYGrid(ppc_filename, 
                        start_datetime=start_datetime.strftime('%m-%d-%Y %H'), 
                        end_datetime=end_datetime.strftime('%m-%d-%Y %H'),
                        verbose=True)

    # Read grid data
    nygrid_sim.set_load_sch(load_profile_renewable)
    nygrid_sim.set_gen_mw_sch(gen_profile)
    nygrid_sim.set_gen_max_sch(genmax_profile)
    nygrid_sim.set_gen_min_sch(genmin_profile)
    nygrid_sim.set_gen_ramp_sch(genramp30_profile)
    nygrid_sim.set_gen_cost_sch(gencost0_profile, gencost1_profile)

    # Process ppc
    nygrid_sim.process_ppc()

    # Set generator initial condition
    if d == 0:
        last_gen = None
        print('No initial condition.')
    else:
        last_gen = nygrid_sim.get_last_gen(model_multi_opf)
        print('Initial condition set from previous day.')
    
    nygrid_sim.set_gen_init_data(gen_init=last_gen)

    # Check input
    nygrid_sim.check_input_dim()

    # Initialize single period OPF
    model_multi_opf = nygrid_sim.create_multi_opf_soft(slack_cost_weight=1e21)

    solver = SolverFactory('gurobi')
    results_multi_opf = solver.solve(model_multi_opf, tee=True)

    if check_status(results_multi_opf):
        print(f'Objective: {model_multi_opf.obj():.4e}')

        # %% Process results
        results = nygrid_sim.get_results_multi_opf(model_multi_opf)
        print(f'Cost: {results["COST"]:.4e}')

        # Format thermal generation results
        results_pg = results['PG']
        thermal_pg = results_pg.iloc[:, :num_thermal]
        thermal_pg = thermal_pg.rename(columns=gen_rename)
        thermal_pg.index.name = 'TimeStamp'

        # Save thermal generation to CSV
        filename = f'thermal_w_offshore_{start_datetime.strftime("%Y%m%d%H")}_{end_datetime.strftime("%Y%m%d%H")}.csv'
        thermal_pg.to_csv(os.path.join(results_dir, 'w_offshore', filename))
        print(f'Saved thermal generation results in {filename}')

        # Save simulation results to pickle
        filename = f'nygrid_sim_w_offshore_{start_datetime.strftime("%Y%m%d%H")}_{end_datetime.strftime("%Y%m%d%H")}.pkl'
        with open(os.path.join(results_dir, 'w_offshore', filename), 'wb') as f:
            pickle.dump([nygrid_sim, model_multi_opf, results], f)
        print(f'Saved simulation results in {filename}')
        elapsed = time.time() - t
        print(f'Elapsed time: {elapsed:.2f} seconds')
        print('-----------------------------------------------------------------')
