"""
Run multi-period OPF with 2018 data
without renewable generators

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

# %% Read grid data
start_date = datetime(2018, 1, 1, 0, 0, 0)
end_date = datetime(2019, 1, 1, 0, 0, 0)

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
    nygrid_sim.get_load_data(load_profile)
    nygrid_sim.get_gen_data(gen_profile)
    nygrid_sim.get_genmax_data(genmax_profile)
    nygrid_sim.get_genmin_data(genmin_profile)
    nygrid_sim.get_genramp_data(genramp30_profile)
    nygrid_sim.get_gencost_data(gencost0_profile, gencost1_profile)

    # Process ppc
    nygrid_sim.process_ppc()

    # Set generator initial condition
    if d == 0:
        last_gen = None
        print('No initial condition.')
    else:
        last_gen = nygrid_sim.get_last_gen(model_multi_opf)
        print('Initial condition set from previous day.')
    
    nygrid_sim.get_gen_init_data(gen_init=last_gen)

    # Check input
    nygrid_sim.check_input_dim()

    # Initialize single period OPF
    model_multi_opf = nygrid_sim.create_multi_opf()

    solver = SolverFactory('gurobi')
    results_multi_opf = solver.solve(model_multi_opf, tee=True)

    if nygrid_sim.check_status(results_multi_opf):
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
        filename = f'thermal_wo_renew_{start_datetime.strftime("%Y%m%d%H")}_{end_datetime.strftime("%Y%m%d%H")}.csv'
        thermal_pg.to_csv(os.path.join(results_dir, 'wo_renew', filename))
        print(f'Saved thermal generation results in {filename}')

        # Save simulation results to pickle
        filename = f'nygrid_sim_wo_renew_{start_datetime.strftime("%Y%m%d%H")}_{end_datetime.strftime("%Y%m%d%H")}.pkl'
        with open(os.path.join(results_dir, 'wo_renew', filename), 'wb') as f:
            pickle.dump([nygrid_sim, model_multi_opf, results], f)
        print(f'Saved simulation results in {filename}')
        elapsed = time.time() - t
        print(f'Elapsed time: {elapsed:.2f} seconds')
        print('-----------------------------------------------------------------')
