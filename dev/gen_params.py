import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
from warnings import warn


def calc_heat_rate(data, gen_info, x_name, y_name):

    # Use linear regression to estimate heat rate
    x = data[x_name].values.reshape(-1, 1)
    y = data[y_name].values

    reg = LinearRegression().fit(x, y)
    slope = reg.coef_[0]
    intercept = reg.intercept_
    r2 = reg.score(x, y)

    fig, axs = plt.subplots(2, 1, figsize=(8, 6),
                            gridspec_kw={'height_ratios': [3, 1]},
                            layout='constrained')
    axs[0].scatter(x, y)
    axs[0].plot(x, reg.predict(x), color='red')
    axs[0].set_xlabel(x_name)
    axs[0].set_ylabel(y_name)

    # Calculate total generation and heat input
    gen_sum = data[x_name].sum()
    heat_input_sum = data[y_name].sum()

    # Add line of eco-min
    eco_min_ratio = calc_eco_min_ratio(slope, gen_info['Unit Type'])
    eco_min = gen_info['CAMD_Nameplate_Capacity'] * eco_min_ratio
    axs[0].axvline(x=eco_min, color='black', linestyle='--', alpha=0.5)

    # Add line of capacity
    axs[0].axvline(x=gen_info["CAMD_Nameplate_Capacity"],
                   color='black', linestyle='--', alpha=0.5)  # Black for CAMD capacity
    axs[0].axvline(x=gen_info["Name Plate Rating (MW)"],
                   color='blue', linestyle=':', alpha=0.5)  # Blue for NYISO capacity

    # Add text to the top left of the plot
    text = f'Heat rate = {slope:.3f} mmBtu/MWh\n'
    text += f'Intercept = {intercept:.3f} mmBtu\n'
    text += f'R$^2$ = {r2:.3f}\n'
    text += f'Capacity = {gen_info["CAMD_Nameplate_Capacity"]} MW\n'
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
    capa_factor = data[x_name] / gen_info["CAMD_Nameplate_Capacity"]
    axs[1].hist(capa_factor, bins=50, color='skyblue', edgecolor='black')

    # Add line of eco-min and capacity
    axs[1].axvline(x=eco_min_ratio, color='black', linestyle='--', alpha=0.5)
    axs[1].axvline(x=1, color='black', linestyle='--', alpha=0.5)
    axs[1].set_xticks(np.arange(0, 1.1, 0.1))

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
