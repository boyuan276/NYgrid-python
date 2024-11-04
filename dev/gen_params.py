import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns
from warnings import warn

HEAT_RATE_RANGE = {
    'CC': {'NG': (5, 16)},
    'CT': {'NG': (7, 20), 'FO2': (7, 17), 'KER': (3, 17)},
    'ST': {'NG': (8, 14), 'FO6': (4, 9), 'BIT': (8, 14)}
}

HEAT_RATE_DEFAULT = {
    "CC": {"NG": 7.633},
    "CT": {"NG": 11.098, "FO2": 13.315, "KER": 13.315},
    "IC": {"NG": 8.899, "FO2": 10.325, "MTE": 8.899},
    "ST": {"NG": 10.347, "FO6": 10.236, "BIT": 10.002}
}


def calc_heat_rate(data, gen_info, x_name, y_name,
                   calc_eco_min=False,
                   nonneg_intercept=True,
                   keep_in_range=True):

    # Calculate total generation and heat input
    gen_sum = data[x_name].sum()
    heat_input_sum = data[y_name].sum()

    # Calculate maximum generation
    max_gen = data[x_name].max()

    # Calculate eco-min
    if calc_eco_min:
        eco_min_ratio = calc_eco_min_ratio(slope, gen_info['Unit Type'])
    else:
        eco_min_ratio = gen_info['eco_min_ratio_obs']

    eco_min = max_gen * eco_min_ratio

    # Filter data above eco-min
    data_filtered = data[data[x_name] > eco_min]

    # Use linear regression to estimate heat rate
    X = data_filtered[x_name].values.reshape(-1, 1)
    y = data_filtered[y_name].values

    
    # Method 1: sklearn LinearRegression
    reg = LinearRegression().fit(X, y)
    slope = reg.coef_[0]
    intercept = reg.intercept_
    r2 = reg.score(X, y)
    y_pred = reg.predict(X)

    if intercept < 0 and nonneg_intercept:
        # Method 2: scipy optimize: This ensures that the intercept is non-negative
        # Define the objective function to minimize: mean squared error
        def objective(params, X, y):
            intercept, slope = params
            predictions = intercept + slope * X.squeeze()
            return np.mean((y - predictions) ** 2)

        # Set up the initial guess for intercept and slope
        initial_guess = [0, 10]

        # Define the constraint that the intercept should be non-negative
        # intercept >= 0
        constraints = [{'type': 'ineq', 'fun': lambda params: params[0]}]

        # Perform the optimization
        result = minimize(objective, initial_guess,
                          args=(X, y), constraints=constraints)

        # Get the best fit parameters with a non-negative intercept
        intercept, slope = result.x

        # Generate predictions based on the optimized model
        y_pred = intercept + slope * X.squeeze()

        r2 = r2_score(y, y_pred)

    # Bad fit if slope is non-positive or R^2 is negative
    if slope <= 0 or r2 < 0:
        warn(f'Bad fit for {gen_info["NYISO_Name"]} {x_name} {y_name}', UserWarning)
        
        # Use default heat rate
        slope = HEAT_RATE_DEFAULT[gen_info["Unit Type"]][gen_info["Fuel Type Primary"]]
        intercept = 0
        r2 = 0
        y_pred = slope * X.squeeze()

    # Bad fit if slope is outside the range
    if keep_in_range and \
        slope < HEAT_RATE_RANGE[gen_info["Unit Type"]][gen_info["Fuel Type Primary"]][0] or \
        slope > HEAT_RATE_RANGE[gen_info["Unit Type"]][gen_info["Fuel Type Primary"]][1]:
        warn(f'Heat rate {slope:.2f} outside the range for {gen_info["NYISO_Name"]}', UserWarning)

        # Use default heat rate
        slope = HEAT_RATE_DEFAULT[gen_info["Unit Type"]][gen_info["Fuel Type Primary"]]
        intercept = 0
        r2 = 0
        y_pred = slope * X.squeeze()
    
    # Plot the data
    fig, axs = plt.subplots(2, 1, figsize=(8, 6),
                            gridspec_kw={'height_ratios': [3, 1]},
                            layout='constrained')
    axs[0].scatter(data[x_name], data[y_name],
                   color='skyblue', edgecolor='black')
    axs[0].scatter(data_filtered[x_name], data_filtered[y_name],
                   color='lightcoral', edgecolor='black')
    axs[0].plot(X, y_pred, color='tab:red')
    axs[0].grid(True, linestyle='--', alpha=0.5)
    axs[0].set_xlabel(x_name)
    axs[0].set_ylabel(y_name)

    # Add line of eco-min
    axs[0].axvline(x=eco_min, color='black', linestyle='--', alpha=0.5)

    # Add line of capacity
    axs[0].axvline(x=gen_info["CAMD_Nameplate_Capacity"],
                   color='black', linestyle='--', alpha=0.5)  # Black for CAMD capacity
    # axs[0].axvline(x=gen_info["Name Plate Rating (MW)"],
    #                color='blue', linestyle=':', alpha=0.5)  # Blue for NYISO capacity
    axs[0].axvline(x=max_gen,
                   color='orange', linestyle='-.', alpha=0.5)  # Orange for max generation

    # Add text to the top left of the plot
    text = f'Heat rate = {slope:.3f} mmBtu/MWh\n'
    text += f'Intercept = {intercept:.3f} mmBtu\n'
    text += f'R$^2$ = {r2:.3f}\n'
    text += f'Capacity = {gen_info["CAMD_Nameplate_Capacity"]} MW\n'
    text += f'Maximum generation = {max_gen} MW\n'
    text += f'Eco-min = {eco_min:.3f} MW ({eco_min_ratio})\n'
    text += f'Non-zero hours = {data.shape[0]}\n'
    text += f'Total generation = {gen_sum:.3e} MWh\n'
    text += f'Total heat input = {heat_input_sum:.3e} mmBtu'
    axs[0].text(0.05, 0.95, text, transform=axs[0].transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))

    title = 'Heat Rate - '
    title += '_'.join([gen_info["ID"], gen_info["NYISO_Name"],
                       gen_info["PTID"].astype(str),
                       gen_info["Unit Type"], gen_info["Fuel Type Primary"],
                       gen_info["Fuel Type Secondary"]])
    axs[0].set_title(title, fontsize=10)

    # Histogram of generation capacity factor
    capa_factor = data[x_name] / max_gen
    axs[1].hist(capa_factor, bins=np.arange(0, 1.02, 0.02),
                color='skyblue', edgecolor='black')

    # Add line of eco-min and capacity
    axs[1].axvline(x=eco_min_ratio, color='black', linestyle='--', alpha=0.5)
    axs[1].axvline(x=1, color='black', linestyle='--', alpha=0.5)
    axs[1].set_xticks(np.arange(0, 1.1, 0.1))
    axs[1].set_xticks(np.arange(0, 1.02, 0.02), minor=True)

    return slope, intercept, r2, eco_min, gen_sum, heat_input_sum, fig, axs


def calc_emis_rate(data, gen_info, x_name, y_name, rate_name):

    # Check if all data is NaN
    if data[y_name].isnull().all():
        warn(f'{gen_info["NYISO_Name"]} {x_name} All data is NaN', UserWarning)
        return np.nan, np.nan, np.nan, np.nan, np.nan, None, None

    # Remove outliers
    # threshold = data[rate_name].quantile(0.95)
    # data = data[data[rate_name] < threshold]

    # Use linear regression to estimate heat rate
    x = data[x_name].values.reshape(-1, 1)
    y = data[y_name].values
    avg_rate = data[rate_name].mean()

    reg = LinearRegression().fit(x, y)
    slope = reg.coef_[0]
    intercept = reg.intercept_
    r2 = reg.score(x, y)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(x, y)
    ax.plot(x, reg.predict(x), color='red')
    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)

    # Calculate total generation and heat input
    gen_sum = data['GLOAD (MW)'].sum()
    heat_input_sum = data['HEAT_INPUT (mmBtu)'].sum()
    emis_sum = data[y_name].sum()

    # Add text to the top left of the plot
    text = f'Slope = {slope:.3f}\n'
    text += f'Intercept = {intercept:.3f}\n'
    text += f'R$^2$ = {r2:.2f}\n'
    text += f'Avg. rate = {avg_rate:.3f}\n'
    text += f'Non-zero hours = {data.shape[0]}\n'
    text += f'Total generation = {gen_sum:.3e} MWh\n'
    text += f'Total heat input = {heat_input_sum:.3e} mmBtu\n'
    text += f'Total emissions = {emis_sum:.3e}'
    ax.text(0.05, 0.95, text, transform=plt.gca().transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))

    title = rate_name + ' - '
    title += '_'.join([gen_info["ID"], gen_info["NYISO_Name"],
                       gen_info["PTID"].astype(str),
                       gen_info["Unit Type"], gen_info["Fuel Type Primary"],
                       gen_info["Fuel Type Secondary"]])
    ax.set_title(title, fontsize=10)

    return slope, intercept, r2, avg_rate, emis_sum, fig, ax


def calc_eco_min_ratio(heat_rate, unit_type):
    if unit_type == 'CC':
        if heat_rate <= 6:
            eco_min = 0.47
            warn(f'Heat rate {heat_rate:.2f} too low for CC', UserWarning)
        elif heat_rate <= 7:
            eco_min = 0.47
        elif heat_rate <= 8:
            eco_min = 0.51
        elif heat_rate <= 9:
            eco_min = 0.56
        elif heat_rate <= 10:
            eco_min = 0.50
        elif heat_rate <= 11:
            eco_min = 0.42
        else:
            eco_min = 0.42
            warn(f'Heat rate {heat_rate:.2f} too high for CC', UserWarning)

    elif unit_type == 'CT':
        if heat_rate <= 9:
            eco_min = 0.81
            warn(f'Heat rate {heat_rate:.2f} too low for CT', UserWarning)
        elif heat_rate <= 10:
            eco_min = 0.81
        elif heat_rate <= 11:
            eco_min = 0.78
        elif heat_rate <= 12:
            eco_min = 0.75
        elif heat_rate <= 13:
            eco_min = 0.62
        elif heat_rate <= 25:
            eco_min = 0.49
        else:
            eco_min = 0.49
            warn(f'Heat rate {heat_rate:.2f} too high for CT', UserWarning)

    elif unit_type == 'ST':
        if heat_rate <= 8:
            eco_min = 0.63
            warn(f'Heat rate {heat_rate:.2f} too low for ST', UserWarning)
        elif heat_rate <= 9:
            eco_min = 0.63
        elif heat_rate <= 10:
            eco_min = 0.55
        elif heat_rate <= 11:
            eco_min = 0.38
        elif heat_rate <= 12:
            eco_min = 0.26
        elif heat_rate <= 13:
            eco_min = 0.22
        elif heat_rate <= 25:
            eco_min = 0.29
        else:
            eco_min = 0.29
            warn(f'Heat rate {heat_rate:.2f} too high for ST', UserWarning)

    else:
        raise ValueError('Unit type not recognized')

    return eco_min