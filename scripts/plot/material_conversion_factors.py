"""Generate bar plots
"""
import os
import sys
import pandas as pd
pd.options.mode.copy_on_write = True
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse, Circle
from matplotlib.colors import ListedColormap
from matplotlib.ticker import (MaxNLocator,LinearLocator, MultipleLocator)
from itertools import cycle, islice
import matplotlib.pyplot as plt
from matplotlib import cm
from map_plotting_utils import *
from mapping_properties import *
from tqdm import tqdm
tqdm.pandas()

# mpl.style.use('ggplot')
# mpl.rcParams['font.size'] = 10.
# mpl.rcParams['font.family'] = 'tahoma'
# mpl.rcParams['axes.labelsize'] = 12.
# mpl.rcParams['xtick.labelsize'] = 10.
# mpl.rcParams['ytick.labelsize'] = 10.

def get_mine_conversion_factors(mcf_df,pcf_df,ref_min,mine_st,exp_st,cf_column="aggregate_ratio"):
    cf_df = pcf_df[pcf_df["reference_mineral"] == ref_min]
    if mine_st == 0:
        mc = mcf_df[mcf_df["reference_mineral"] == ref_min
                    ]["metal_content_factor"].values[0]
        cf_val = mc*cf_df[cf_df["final_refined_stage"] == str(exp_st).replace(".0","")
                        ][cf_column].values[0]/cf_df[
                        cf_df["final_refined_stage"] == '1'
                        ][cf_column].values[0]
    else:
        cf_val = cf_df[cf_df["final_refined_stage"] == str(exp_st).replace(".0","")
                        ][cf_column].values[0]/cf_df[
                        cf_df["final_refined_stage"] == str(mine_st).replace(".0","")
                        ][cf_column].values[0]
    return cf_val

def get_radius(x):
    return (1/3.14)*(x**0.5)

def main(config):
    processed_data_path = config['paths']['data']
    output_data_path = config['paths']['results']
    figures_data_path = config['paths']['figures']

    vals = pd.read_csv(os.path.join(processed_data_path,
                                        "mineral_usage_factors",
                                        "material_stages_for_plotting.csv"))
    y_coords = vals["Y"].values.tolist()
    x_coords = vals["X"].values.tolist()
    wts = vals["kgs"].values.tolist()
    arrows = []
    for i in range(0,len(y_coords)-1):
        if y_coords[i] == y_coords[i+1]:
            y = y_coords[i]
            x0 = x_coords[i] + get_radius(wts[i])
            x1 = x_coords[i+1] - get_radius(wts[i+1])
            arrows.append((y,x0,x1))

    ar = pd.DataFrame(arrows,columns=["Y","X0","X1"])
    
    vals["rad"] = (1.0/3.14)*(vals["kgs"]**0.5)
    ar["dx"] = ar["X1"] - ar["X0"]
    ar["dy"] = 0
    print (vals)
    print (ar)
    plt.figure(figsize=[16, 16],dpi=500)
    ax = plt.axes([0.0,0.0,1.0,1.0], xlim=(1, 26), ylim=(1, 26))
    # ax.scatter(x="X", y="Y", s="kgs", data=vals)
    rl = []
    for row in vals.itertuples():
        circle = Circle(xy=(row.X, row.Y), radius=row.rad, 
                            edgecolor = None,fc=row.color,alpha= 0.6,zorder=5)
        ax.add_patch(circle)
        if row.kgs > 2.7:
            ax.text(row.X,row.Y - 0.1,f"{row.kgs:,.2f}",ha='center', va='center',fontsize=18,fontweight='bold',zorder=6)
        else:
            ax.text(row.X,row.Y + 0.55,f"{row.kgs:,.2f}",ha='center', va='center',fontsize=18,fontweight='bold',zorder=6)
        if row.reference_mineral not in rl:
            ax.text(row.X - 2.0,row.Y,f"{row.reference_mineral}",ha='center', va='center',fontsize=40,fontweight='bold')
            rl.append(row.reference_mineral)
        if row.stage == 0:
            ax.text(row.X,row.Y - 0.5 - row.rad,f"{row.stage_type}\n content",
                    ha='center', va='center',fontsize=18,fontweight='bold')
        else:
            ax.text(row.X,row.Y - 0.5 - row.rad,row.stage_type,ha='center', va='center',fontsize=18,fontweight='bold')
    for row in ar.itertuples():
        ax.arrow(row.X0,row.Y,row.dx,row.dy,
                head_width=0.3, head_length=0.3, 
                linewidth=1.5, color='k', length_includes_head=True,zorder=1)
    plt.tight_layout()
    plt.xticks([])
    plt.yticks([])
    plt.gca().set_aspect('equal', adjustable='box')
    save_fig(os.path.join(figures_data_path,
                "material_conversion_factors.png"))
    plt.close()

if __name__ == '__main__':
    CONFIG = load_config()
    main(CONFIG)
