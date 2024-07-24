"""Generate bar plots
"""
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from matplotlib.ticker import (MaxNLocator,LinearLocator, MultipleLocator)
import matplotlib.pyplot as plt
from matplotlib import cm
from map_plotting_utils import *
from tqdm import tqdm
tqdm.pandas()

mpl.style.use('ggplot')
mpl.rcParams['font.size'] = 10.
mpl.rcParams['font.family'] = 'tahoma'
mpl.rcParams['axes.labelsize'] = 12.
mpl.rcParams['xtick.labelsize'] = 10.
mpl.rcParams['ytick.labelsize'] = 10.

def main(config):
    processed_data_path = config['paths']['data']
    output_data_path = config['paths']['results']
    figures_data_path = config['paths']['figures']

    multiply_factor = 1.0
    width = 0.5
    all_colors = ["#1f78b4","#b2df8a","#33a02c","#fb9a99","#e31a1c","#fdbf6f","#ff7f00","#cab2d6"]
    data_df = pd.read_excel(
                    os.path.join(output_data_path,
                    "country_summaries",
                    "transport_tonnage_totals_by_stage.xlsx"),
                sheet_name="country_unconstrained",index_col=[0,1])
    print (data_df)
    data_df = data_df.reset_index()
    print (data_df)
    scenarios = ["2022_baseline","2030_mid_min_threshold_metal_tons","2040_mid_min_threshold_metal_tons"]
    reference_minerals = list(set(data_df["reference_mineral"].values.tolist()))
    for rf in reference_minerals:
        df = data_df[(data_df["reference_mineral"] == rf) & (data_df["processing_stage"] > 0)]
        stages = sorted(list(set(df["processing_stage"].values.tolist())))
        stage_colors = all_colors[:len(stages)]
        countries = sorted(list(set(df["iso3"].values.tolist())))
        fig, ax = plt.subplots(1,1,figsize=(16,16),dpi=500)
        f = 0
        for sc in scenarios:
            xvals = np.arange(0,len(countries)) + f
            for idx,(st,st_c) in enumerate(zip(stages,stage_colors)):
                s_df = df[df["processing_stage"] == st]
                m_df = pd.DataFrame(countries,columns=["iso3"])
                m_df = pd.merge(m_df,s_df,how="left",on=["iso3"]).fillna(0)
                bottom  = np.zeros(len(m_df.index))
                yvals = m_df[sc].values
                ax.bar(xvals,multiply_factor*yvals,
                            width=width,
                            color=st_c, 
                            label=f"Stage {st}",
                            bottom=bottom)
                bottom += multiply_factor*yvals
                f += width

        ax.legend(prop={'size':12,'weight':'bold'})
        # ax.set_ylim([0,1.2*ymax])
        ax.set_ylabel('Processing stage tonnage (tons)',fontweight='bold',fontsize=15)
        # xticks = list(np.arange(1,len(all_hazards),1, dtype=int))
        ax.set_xticklabels(countries,
                        fontsize=15, rotation=45,
                        fontweight="bold")
        plt.tight_layout()
        save_fig(os.path.join(figures_data_path,
                    f"{rf}_country_unconstrained.png"))
        plt.close()






if __name__ == '__main__':
    CONFIG = load_config()
    main(CONFIG)
