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

    data_type = {"initial_refined_stage":"str","final_refined_stage":"str"}
    pr_conv_factors_df = pd.read_excel(os.path.join(processed_data_path,
                                        "mineral_usage_factors",
                                        "aggregated_stages.xlsx"),dtype=data_type)[[
                                            "reference_mineral",
                                            "initial_refined_stage",
                                            "final_refined_stage", 
                                            "aggregate_ratio"
                                            ]]
    # Read the data on how much metal concent goes into ores and concentrates
    metal_content_factors_df = pd.read_csv(os.path.join(processed_data_path,
                                            "mineral_usage_factors",
                                            "metal_content.csv"))
    metal_content_factors_df.rename(
                        columns={
                            "Reference mineral":"reference_mineral",
                            "Input metal content":"metal_content_factor"},
                        inplace=True)
    
    mp = mineral_properties()
    reference_minerals = ["cobalt","copper","graphite","lithium","manganese","nickel"]
    factor = 1.0
    figure_x_size = 10
    axis_width = 0.8
    vals = []
    arrows = []
    y = 4
    for reference_mineral in reference_minerals[::-1]:
        color = mp[reference_mineral]["mineral_color"]
        sym = mp[reference_mineral]["mineral_symbol"]
        f_stages = mp[reference_mineral]["stages"]
        i_stages = [0] + f_stages[:-1]
        t = 1.0
        st = 4.0
        if reference_mineral == "lithium":
            typ = "Carbonate"
        else:
            typ = "Metal"
        vals.append((y,st,typ,sym,0.0,factor*t,color))
        x0 = st + get_radius(factor*t)
        for idx,(in_st,m_st) in enumerate(zip(i_stages,f_stages)):
            cf = get_mine_conversion_factors(metal_content_factors_df,
                                            pr_conv_factors_df,
                                            reference_mineral,
                                            in_st,m_st)
            t = t/cf
            st += 4.0
            x1 = st - get_radius(factor*t)
            vals.append((y,st,f"Stage {str(m_st).replace('.0','')}",sym,m_st,factor*t,color))
            arrows.append((y,x0,x1))
            x0 = st + get_radius(factor*t)
        y += 4

    vals = pd.DataFrame(vals,columns=["Y","X","stage_type","reference_mineral","stage","kgs","color"])
    vals["rad"] = (1.0/3.14)*(vals["kgs"]**0.5)
    ar = pd.DataFrame(arrows,columns=["Y","X0","X1"])
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
