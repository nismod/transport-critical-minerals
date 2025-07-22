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

# mpl.style.use('ggplot')
# mpl.rcParams['font.size'] = 10.
# mpl.rcParams['font.family'] = 'tahoma'
# mpl.rcParams['axes.labelsize'] = 12.
# mpl.rcParams['xtick.labelsize'] = 10.
# mpl.rcParams['ytick.labelsize'] = 10.


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

    for df in dfall: # for each data frame
        bar_colors = list(islice(cycle(bar_colors), None, len(df)))
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
                rect.set_hatch(H * int(i / n_col)) #edited part     
                rect.set_width(1 / float(n_df + 1))

    # sec.set_xticks([5, 15, 25], labels=['\nOughts', '\nTeens', '\nTwenties'])
    axe.set_xticks((np.arange(0, 2 * n_ind, 2) + 1 / float(n_df + 1)) / 2.)
    axe.set_xticklabels(df.index, rotation = 0,fontsize=15,fontweight="bold")
    axe.set_xlabel('')
    axe.set_ylabel(ylabel,fontweight='bold',fontsize=15)
    axe.tick_params(axis='y',labelsize=15)
    axe.set_title(title,fontweight='bold',fontsize=18)
    axe.set_axisbelow(True)
    axe.grid(which='major', axis='x', linestyle='-', zorder=0)

    legend_handles = []
    titles = ["$\\bf{Mineral \, processing \, stages}$","$\\bf Scenarios$"]
    legend_handles.append(axe.plot([],[],
                                    color="none",
                                    label="$\\bf{Mineral \, processing \, stages}$")[0])
    for idx,(bc,bl) in enumerate(zip(bar_colors,l[:n_col])):
        legend_handles.append(mpatches.Patch(color=bc,
                                        label=bl))

    # l1 = axe.legend(h[:n_col], l[:n_col], loc="upper right",prop={'size':15,'weight':'bold'})
    legend_handles.append(axe.plot([],[],
                                    color="none",
                                    label="$\\bf Scenarios$")[0])
    
    for idx in range(len(labels)):
        legend_handles.append(mpatches.Patch(facecolor="black",edgecolor='white',
                                        label=labels[idx],hatch=H*idx))
    leg = axe.legend(
                handles=legend_handles, 
                fontsize=15, 
                loc='upper left',
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
    figure_path = config['paths']['figures']

    figures = os.path.join(figure_path,"regional_figures")
    if os.path.exists(figures) is False:
        os.mkdir(figures)

    figures = os.path.join(figure_path,"regional_figures","bar_plots")
    if os.path.exists(figures) is False:
        os.mkdir(figures)

    multiply_factor = 1.0
    columns = [
                "production_tonnes", 
                "export_tonnes",
                "export_transport_cost_usd", 
                "tonsCO2eq",
                "revenue_usd",
                "production_cost_usd",
                "water_usage_m3"
            ]
    column_titles = [
                        "production volume ('000 tonnes)",
                        "export volume ('000 tonnes)",
                        "transport costs (million USD)",
                        "transport carbon emissions (000' tonsCO2eq)",
                        "revenue (million USD)",
                        "production costs (million USD)",
                        "water usage ('000 m3)"
                    ]
    multiply_factors = [1.0e-3,1.0e-3,1.0e-6,1e-3,1.0e-6,1.0e-6,1.0e-3]
    column_titles = [f"Annual {c}" for c in column_titles]
    scenarios_descriptions = [
                                {
                                    "scenario_type":"country_unconstrained",
                                    "scenario_name":"MN",
                                    "scenarios":[
                                                    "2022_baseline",
                                                    "2030_mid_min_threshold_metal_tons",
                                                    "2040_mid_min_threshold_metal_tons"
                                                ],
                                    "scenario_labels":[
                                                    "2022 SQ",
                                                    "2030 MN",
                                                    "2040 MN"
                                                ]
                                },
                                {
                                    "scenario_type":"region_unconstrained",
                                    "scenario_name":"MR",
                                    "scenarios":[
                                                    "2022_baseline",
                                                    "2030_mid_max_threshold_metal_tons",
                                                    "2040_mid_max_threshold_metal_tons"
                                                ],
                                    "scenario_labels":[
                                                    "2022 SQ",
                                                    "2030 MR",
                                                    "2040 MR"
                                                ]
                                },
                    ]
    make_plot = False
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
                            index_col=[0,1,2,3,4])
            data_df = data_df.reset_index()
            if sc_t != "country_unconstrained":
                baseline_df = pd.read_excel(
                                results_file,
                                sheet_name="country_unconstrained",
                                index_col=[0,1,2,3,4])
                baseline_df = baseline_df.reset_index()
                data_df = pd.concat(
                                [
                                    baseline_df[baseline_df["year"] == 2022],
                                    data_df
                                ],axis=0,ignore_index=True
                                )

            reference_minerals = list(set(data_df["reference_mineral"].values.tolist()))
            all_properties = mineral_properties()
            for cdx,(col,col_t,m_t) in enumerate(zip(columns,column_titles,multiply_factors)):
                for rf in reference_minerals:
                    df = data_df[(data_df["reference_mineral"] == rf) & (data_df["processing_stage"] > 0)]
                    stages = sorted(list(set(df["processing_stage"].values.tolist())))
                    stage_colors = all_properties[rf]["stage_colors"][:len(stages)]
                    countries = sorted(list(set(df["iso3"].values.tolist())))
                    fig, ax = plt.subplots(1,1,figsize=(18,9),dpi=500)
                    f = 0
                    dfall = []
                    for sc in scenarios:
                        m_df = pd.DataFrame(countries,columns=["iso3"])
                        # xvals = 2*(np.arange(0,len(countries)) + f)
                        for st in stages:
                            s_df = df[(df["processing_stage"] == st) & (df["scenario"] == sc)]
                            s_df[col] = m_t*s_df[col]
                            s_df.rename(columns={col:f"Stage {st}"},inplace=True)
                            m_df = pd.merge(m_df,s_df[["iso3",f"Stage {st}"]],how="left",on=["iso3"]).fillna(0)
                        dfall.append(m_df.set_index(["iso3"]))
                    ax = plot_clustered_stacked(
                                                fig,ax,dfall,stage_colors,
                                                labels=sc_l,
                                                ylabel=col_t, 
                                                title=f"{rf.title()} {sc_n} scenario")
                    plt.grid()
                    plt.tight_layout()
                    save_fig(os.path.join(results_folder,
                                f"{rf}_{col}_{sc_t}.png"))
                    plt.close()


    """CO2 emissions and Value Added
    """
    make_plot = False
    if make_plot is True:
        results_file = os.path.join(output_data_path,
                                "result_summaries",
                                "totals_by_country.xlsx")
        df = pd.read_excel(
                            results_file,
                            index_col=[0,1,2])
        df = df.reset_index() 
        countries = sorted(list(set(df["iso3"].values.tolist())))
        scenarios_descriptions = [s for s in scenarios_descriptions if s["scenario_type"] == "country_unconstrained"]
        plot_type = [
                        {
                            "type":"carbon_emissions",
                            "columns": ["transport_tonsCO2eq","energy_tonsCO2eq"],
                            "columns_labels":["Annual transport emissions","Annual energy emissions"],
                            "columns_colors":["#bdbdbd","#969696"],
                            "ylabel":"Annual carbon emissions ('000 tonsCO2eq)",
                            "factor":1.0e-3
                        },
                        {
                            "type":"value_added",
                            "columns": ["export_value_added_usd"],
                            "columns_labels":["Annual value added"],
                            "columns_colors":["#66c2a4"],
                            "ylabel":"Annual value added (USD millions)",
                            "factor":1.0e-6
                        }
                        ] 
        for sc in scenarios_descriptions:
            sc_t = sc["scenario_type"]
            sc_n = sc["scenario_name"]
            for pt in plot_type:
                cols = pt["columns"]
                ptype = pt["type"]
                fig, ax = plt.subplots(1,1,figsize=(18,9),dpi=500)
                dfall = []
                for st in sc["scenarios"]:
                    m_df = pd.DataFrame(countries,columns=["iso3"])
                    s_df = df[df["scenario"] == st]
                    m_df = pd.merge(m_df,s_df[["iso3"] + cols],how="left",on=["iso3"]).fillna(0)
                    m_df[cols] = pt["factor"]*m_df[cols]
                    m_df.rename(columns=dict(list(zip(cols,pt["columns_labels"]))),inplace=True)
                    dfall.append(m_df.set_index(["iso3"]))
                
                ax = plot_clustered_stacked(
                                            fig,ax,dfall,pt["columns_colors"],
                                            labels=sc["scenario_labels"],
                                            ylabel=pt["ylabel"], 
                                            title=f"{sc_n} scenario")
                plt.grid()
                plt.tight_layout()
                save_fig(os.path.join(figures,
                            f"{ptype}_{sc_t}.png"))
                plt.close()

    """Delta tonnage unconstrained and constrained
    """
    make_plot = True
    if make_plot is True:
        multiply_factor = 1.0e-3
        results_file = os.path.join(output_data_path,
                                "result_summaries",
                                "transport_totals_by_stage.xlsx")
        constraints = ["country_unconstrained","country_constrained"]
        reference_minerals = ["cobalt","copper","graphite","lithium","manganese","nickel"]
        reference_mineral_colors = ["#fdae61","#f46d43","#66c2a5","#c2a5cf","#fee08b","#3288bd"]
        scenarios = ["2030_mid_min_threshold_metal_tons","2040_mid_min_threshold_metal_tons"]
        tons_column = "production_tonnes"
        index_cols = ["reference_mineral","iso3"]
        dfs = []
        for scenario in scenarios:
            for cs in constraints:
                df = pd.read_excel(
                                    results_file,
                                    sheet_name=cs,
                                    index_col=[0,1,2,3])
                df = df.reset_index() 
                m_df = pd.DataFrame(sorted(list(set(df["iso3"].values.tolist()))),columns=["iso3"])
                df = df[(df["scenario"] == scenario) & (df["processing_stage"] == 0)]
                df = df.groupby(index_cols)[tons_column].sum().reset_index()
                df[tons_column] = multiply_factor*df[tons_column]
                df = (df.set_index(["iso3"]).pivot(
                                        columns="reference_mineral"
                                        )[tons_column].reset_index().rename_axis(None, axis=1)).fillna(0)
                df_cols = df.columns.values.tolist()
                df.columns = ["iso3"] + [c.title() for c in df_cols[1:]] 
                m_df = pd.merge(m_df,df,how="left",on=["iso3"]).fillna(0)
                print (m_df)
                dfs.append(m_df.set_index("iso3"))
        delta_df = []
        delta_df.append(dfs[0].subtract(dfs[1], fill_value=0))
        delta_df.append(dfs[2].subtract(dfs[3], fill_value=0))
        fig, ax = plt.subplots(1,1,figsize=(18,9),dpi=500)
        ax = plot_clustered_stacked(
                                fig,ax,dfs,reference_mineral_colors,
                                labels=["National 2030 unconstrained",
                                        "National 2030 constrained",
                                        "National 2040 unconstrained",
                                        "National 2040 constrained"],
                                ylabel="Annual metal content produced (000' tonnes)", 
                                title=f"Mid National scenarios - metal content production")
        plt.grid()
        plt.tight_layout()
        save_fig(os.path.join(figures,
                    f"MN_scenarios_unconstrained_constrained_side_by_side.png"))
        plt.close()

        fig, ax = plt.subplots(1,1,figsize=(18,9),dpi=500)
        ax = plot_clustered_stacked(
                                fig,ax,delta_df,reference_mineral_colors,
                                labels=["2030 difference", "2040 difference"],
                                ylabel="Difference in annual metal content produced (000' tonnes)", 
                                title=f"Mid National 2030 and 2040 scenarios - Unconstrained minus Constrained production of metal content")
        plt.grid()
        plt.tight_layout()
        save_fig(os.path.join(figures,
                    f"MN_scenarios_unconstrained_constrained_delta.png"))
        plt.close()

    """Country and regional comparisons
    """
    make_plot = True
    if make_plot is True:
        multiply_factor = 1.0e-3
        results_file = os.path.join(output_data_path,
                                "result_summaries",
                                "transport_totals_by_stage.xlsx")
        constraints = ["country_unconstrained","region_unconstrained"]
        reference_mineral_colors = ["#f46d43","#fdae61","#fee08b","#c2a5cf","#66c2a5","#3288bd"]
        reference_minerals = ["copper","cobalt","manganese","lithium","graphite","nickel"]
        scenarios = [
                        "2030_mid_min_threshold_metal_tons",
                        "2030_mid_max_threshold_metal_tons",
                        "2040_mid_min_threshold_metal_tons",
                        "2040_mid_max_threshold_metal_tons"
                    ]
        col = "export_tonnes"
        index_cols = ["reference_mineral","iso3"]
        all_properties = mineral_properties()
        for rf in reference_minerals:
            dfall = []
            stages = all_properties[rf]["stage_labels"]
            stage_colors = all_properties[rf]["stage_label_colors"]
            for scenario in scenarios:
                for cs in constraints:
                    df = pd.read_excel(
                                        results_file,
                                        sheet_name=cs,
                                        index_col=[0,1,2,3,4])
                    df = df.reset_index()
                    countries = sorted(list(set(df["iso3"].values.tolist())))
                    df = df[(df["scenario"] == scenario) & (df["processing_stage"] > 0) & (df["reference_mineral"] == rf)]  
                    if len(df.index) > 0:
                         # stages = sorted(list(set(df["processing_stage"].values.tolist())))
                        # stage_colors = all_properties[rf]["stage_colors"][:len(stages)]
                        m_df = pd.DataFrame(countries,columns=["iso3"])
                        # xvals = 2*(np.arange(0,len(countries)) + f)
                        for st in stages:
                            s_df = df[df["processing_type"] == st]
                            if len(s_df.index) > 0:
                                s_df[col] = multiply_factor*s_df[col]
                                s_df = s_df.groupby(["iso3","processing_type"])[col].sum().reset_index()
                                s_df.rename(columns={col:f"{st}"},inplace=True)
                                m_df = pd.merge(m_df,s_df[["iso3",f"{st}"]],how="left",on=["iso3"]).fillna(0)
                            else:
                                m_df[f"{st}"] = 0
                        dfall.append(m_df.set_index(["iso3"]))
                    
            fig, ax = plt.subplots(1,1,figsize=(18,9),dpi=500)   
            ax = plot_clustered_stacked(
                                        fig,ax,dfall,stage_colors,
                                        labels=["National 2030","Regional 2030","National 2040","Regional 2040"],
                                        ylabel="Annual export volumes (000' tonnes)", 
                                        title=f"{rf.title()} Mid National and Mid Regional scenario comparisons")
            plt.grid()
            plt.tight_layout()
            save_fig(os.path.join(figures,
                        f"{rf}_MN_MR_comparisions_side_by_side.png"))
            plt.close()



if __name__ == '__main__':
    CONFIG = load_config()
    main(CONFIG)
