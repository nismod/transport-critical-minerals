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
from matplotlib.colors import ListedColormap
from matplotlib.ticker import (MaxNLocator,LinearLocator, MultipleLocator)
from itertools import cycle, islice
import matplotlib.pyplot as plt
from matplotlib import cm
from map_plotting_utils import *
from mapping_properties import *
from tqdm import tqdm
tqdm.pandas()

mpl.style.use('ggplot')
mpl.rcParams['font.size'] = 10.
mpl.rcParams['font.family'] = 'tahoma'
mpl.rcParams['axes.labelsize'] = 12.
mpl.rcParams['xtick.labelsize'] = 10.
mpl.rcParams['ytick.labelsize'] = 10.


def plot_clustered_stacked(fig,axe,
                            dfall,
                            bar_colors,
                            labels=None,
                            ylabel="Y-values",
                            title="multiple stacked bar plot",
                            H="/",**kwargs):
    """ 
        Source: https://stackoverflow.com/questions/22787209/how-to-have-clusters-of-stacked-bars
        Given a list of dataframes, with identical columns and index, create a clustered stacked bar plot. 
        labels is a list of the names of the dataframe, used for the legend
        title is a string for the title of the plot
        H is the hatch used for identification of the different dataframe
    """

    """
        # create fake dataframes
        df1 = pd.DataFrame(np.random.rand(4, 5),
                           index=["A", "B", "C", "D"],
                           columns=["I", "J", "K", "L", "M"])
        df2 = pd.DataFrame(np.random.rand(4, 5),
                           index=["A", "B", "C", "D"],
                           columns=["I", "J", "K", "L", "M"])
        df3 = pd.DataFrame(np.random.rand(4, 5),
                           index=["A", "B", "C", "D"], 
                           columns=["I", "J", "K", "L", "M"])

        # Then, just call :
        plot_clustered_stacked([df1, df2, df3],["df1", "df2", "df3"])
    """

    n_df = len(dfall)
    n_col = len(dfall[0].columns) 
    n_ind = len(dfall[0].index)
    # axe = plt.subplot(111)

    for df in dfall : # for each data frame
        bar_colors = list(islice(cycle(bar_colors), None, n_col))
        # bar_colors = list(zip(bar_colors["stages"],bar_colors["stage_colors"]))
        axe = df.plot(kind="bar",
                      linewidth=1.0,
                      stacked=True,
                      ax=axe,
                      legend=False,
                      grid=False,
                      color = bar_colors,
                      edgecolor = 'white',
                      **kwargs)  # make bar plots

    h,l = axe.get_legend_handles_labels() # get the handles we want to modify
    for i in range(0, n_df * n_col, n_col): # len(h) = n_col * n_df
        for j, pa in enumerate(h[i:i+n_col]):
            for rect in pa.patches: # for each index
                rect.set_x(rect.get_x() + 1 / float(n_df + 1) * i / float(n_col))
                # rect.set_hatch(H * int(i / n_col)) #edited part 
                rect.set_width(1 / float(n_df + 1))

    # sec.set_xticks([5, 15, 25], labels=['\nOughts', '\nTeens', '\nTwenties'])
    if labels is not None:
        locs = []
        lbs = []
        for i in range(len(labels)):
            lb = labels[i]
            locs += list((np.arange(0, 2 * n_ind, 2) + (2*i-1)*(1/ float(n_df + 1))) / 2.)
            if i == 1:
                lbs += [f"{lb[:4]}\n\n{d}" for d in df.index.values.tolist()]
            else:
                lbs += [lb[:4]]*len(df.index)
        
        axe.set_xticks(locs)
        axe.set_xticklabels(lbs, rotation = 0,fontsize=14,fontweight="bold")
    
    # axe.set_xticks((np.arange(0, 2 * n_ind, 2) + 1 / float(n_df + 1)) / 2.)
    # axe.set_xticklabels(
    #                 [f"\n{d}" for d in df.index.values.tolist()], 
    #                 rotation = 0,fontsize=15,fontweight="bold"
    #                 )
    axe.set_xlabel('')
    axe.set_ylabel(ylabel,fontweight='bold',fontsize=15)
    axe.tick_params(axis='y',labelsize=15)
    axe.set_title(title,fontweight='bold',fontsize=18)
    ymin, ymax = axe.get_ylim()
    axe.set_ylim(ymin,1.2*ymax)

    legend_handles = []
    titles = ["$\\bf{Mineral \, processing \, stages}$","$\\bf Scenarios$"]
    # legend_handles.append(axe.plot([],[],
    #                                 color="none",
    #                                 label="$\\bf{Mineral \, processing \, stages}$")[0])
    for idx,(bc,bl) in enumerate(zip(bar_colors,l[:n_col])):
        legend_handles.append(mpatches.Patch(color=bc,
                                        label=bl))

    # l1 = axe.legend(h[:n_col], l[:n_col], loc="upper right",prop={'size':15,'weight':'bold'})
    # legend_handles.append(axe.plot([],[],
    #                                 color="none",
    #                                 label="$\\bf Scenarios$")[0])
    
    # for idx in range(len(labels)):
    #     legend_handles.append(mpatches.Patch(facecolor="#000000",edgecolor='white',
    #                                     label=labels[idx],hatch=H*idx))
    leg = axe.legend(
                handles=legend_handles, 
                fontsize=15, 
                loc='upper left',
                # bbox_to_anchor=(0.7,-0.1),
                ncols=3,
                title="$\\bf{Mineral \, processing \, stages}$",
                title_fontsize=15,
                frameon=False)

    # Move titles to the left 
    for item, label in zip(leg.legend_handles, leg.texts):
        if label._text in titles:
            width = item.get_window_extent(fig.canvas.get_renderer()).width
            label.set_ha('left')
            label.set_position((-10.0*width,0))
    return axe

def main(config):
    processed_data_path = config['paths']['data']
    output_data_path = config['paths']['results']
    figures_data_path = config['paths']['figures']

    figures = os.path.join(figures_data_path,"southern_africa_figures")
    if os.path.exists(figures) is False:
        os.mkdir(figures)

    figures = os.path.join(figures_data_path,"southern_africa_figures","bar_plots")
    if os.path.exists(figures) is False:
        os.mkdir(figures)

    columns = [
                "production_tonnes", 
                "export_tonnes",
                "export_transport_cost_usd", 
                "tonsCO2eq",
                "revenue_usd",
                "production_cost_usd"
            ]
    column_titles = [
                        "production volume (million tonnes)",
                        "export volume (million tonnes)",
                        "transport costs (billion USD)",
                        "transport carbon emissions (tonsCO2eq)",
                        "revenue (billion USD)",
                        "production costs (billion USD)"
                    ]
    multiply_factors = [1.0e-6,1.0e-6,1.0e-9,1,1.0e-9,1.0e-9]
    # stage_colors = [
    #                 "#fed976","#fdae6b","#fc8d59","#8c6bb1","#fb6a4a","#810f7c","#cb181d","#cc4c02","#7f0000"
    #                 ]
    stage_colors = [
                    "#fed976","#fb6a4a","#7f0000","#fb6a4a","#810f7c","#cb181d","#cc4c02"
                    ]
    column_titles = [f"Annual {c}" for c in column_titles]
    scenarios_descriptions = [
                                {
                                    "scenario_type":"country_unconstrained",
                                    "scenario_name":"Mid National",
                                    "scenarios":[
                                                    "2022_baseline",
                                                    "2030_mid_min_threshold_metal_tons",
                                                    "2040_mid_min_threshold_metal_tons"
                                                ],
                                    "scenario_labels":[
                                                    "2022",
                                                    "2030",
                                                    "2040"
                                                ]
                                },
                                {
                                    "scenario_type":"region_unconstrained",
                                    "scenario_name":"Mid Regional",
                                    "scenarios":[
                                                    "2022_baseline",
                                                    "2030_mid_max_threshold_metal_tons",
                                                    "2040_mid_max_threshold_metal_tons"
                                                ],
                                    "scenario_labels":[
                                                    "2022",
                                                    "2030",
                                                    "2040"
                                                ]
                                },
                                {
                                    "scenario_type":"country_constrained",
                                    "scenario_name":"Mid National",
                                    "scenarios":[
                                                    "2022_baseline",
                                                    "2030_mid_min_threshold_metal_tons",
                                                    "2040_mid_min_threshold_metal_tons"
                                                ],
                                    "scenario_labels":[
                                                    "2022",
                                                    "2030",
                                                    "2040"
                                                ]
                                },
                                {
                                    "scenario_type":"region_constrained",
                                    "scenario_name":"Mid Regional",
                                    "scenarios":[
                                                    "2022_baseline",
                                                    "2030_mid_max_threshold_metal_tons",
                                                    "2040_mid_max_threshold_metal_tons"
                                                ],
                                    "scenario_labels":[
                                                    "2022",
                                                    "2030",
                                                    "2040"
                                                ]
                                },
                    ]
    make_plot = True
    if make_plot is True:
        results_file = os.path.join(output_data_path,
                                "result_summaries",
                                "transport_totals_by_stage.xlsx")
        for sd in scenarios_descriptions:
            sc_t = sd["scenario_type"]
            sc_n = sd["scenario_name"]
            scenarios = sd["scenarios"]
            sc_l = sd["scenario_labels"]
            results_folder = os.path.join(figures,f"{sc_n}_{sc_t}")
            if os.path.exists(results_folder) == False:
                os.mkdir(results_folder)
            data_df = pd.read_excel(
                            results_file,
                            sheet_name=sc_t,
                            index_col=[0,1,2,3])
            data_df = data_df.reset_index()
            if sc_t != "country_unconstrained":
                baseline_df = pd.read_excel(
                                results_file,
                                sheet_name="country_unconstrained",
                                index_col=[0,1,2,3])
                baseline_df = baseline_df.reset_index()
                data_df = pd.concat(
                                [
                                    baseline_df[baseline_df["year"] == 2022],
                                    data_df
                                ],axis=0,ignore_index=True
                                )
            # data_df = data_df[(data_df["iso3"].isin(country_codes)) & (data_df["production_tonnes"] > 0)]
            data_df["reference_mineral"] = data_df["reference_mineral"].str.title()
            reference_minerals = sorted(list(set(data_df["reference_mineral"].values.tolist())))
            for cdx,(col,col_t,m_t) in enumerate(zip(columns,column_titles,multiply_factors)):
                fig, ax = plt.subplots(1,1,figsize=(18,9),dpi=500)
                dfall = []
                df = data_df[data_df["processing_stage"] > 0]
                df = df.groupby(["scenario","reference_mineral","processing_type"])[col].sum().reset_index()
                df = df[df[col] > 0]
                stages = sorted(list(set(df["processing_type"].values.tolist())))
                stage_colors = stage_colors[:len(stages)]
                for sc in scenarios:
                    m_df = pd.DataFrame(reference_minerals,columns=["reference_mineral"])
                    for st in stages:
                        s_df = df[(df["processing_type"] == st) & (df["scenario"] == sc)]
                        if len(s_df.index) > 0:
                            s_df[col] = m_t*s_df[col]
                            s_df.rename(columns={col:f"{st}"},inplace=True)
                            m_df = pd.merge(
                                            m_df,s_df[["reference_mineral",f"{st}"]],
                                            how="left",on=["reference_mineral"]).fillna(0)
                        else:
                            m_df[f"{st}"] = 0.0
                    dfall.append(m_df.set_index(["reference_mineral"]))
                ax = plot_clustered_stacked(
                                            fig,ax,dfall,stage_colors,
                                            labels=sc_l,
                                            ylabel=col_t, 
                                            title=f"{sc_n} scenario")
                plt.grid()
                plt.tight_layout()
                save_fig(os.path.join(results_folder,
                            f"{col}_{sc_t}.png"))
                plt.close()

if __name__ == '__main__':
    CONFIG = load_config()
    main(CONFIG)
