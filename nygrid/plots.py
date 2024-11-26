import pandas as pd
import matplotlib.pyplot as plt
import typing
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
