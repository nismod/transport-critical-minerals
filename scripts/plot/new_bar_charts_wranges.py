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

mpl.rcParams["font.family"] = "DejaVu Sans"
mpl.rcParams["font.size"] = 12
mpl.rcParams["axes.spines.top"] = False
mpl.rcParams["axes.spines.right"] = False
mpl.rcParams["axes.facecolor"] = "#f9f9f9"
mpl.rcParams["grid.alpha"] = 0.3
mpl.rcParams["grid.linestyle"] = "--"
# Increase figure DPI globally
mpl.rcParams["figure.dpi"] = 120

# mpl.style.use('ggplot')
# mpl.rcParams['font.size'] = 10.
# mpl.rcParams['font.family'] = 'tahoma'
# mpl.rcParams['axes.labelsize'] = 12.
# mpl.rcParams['xtick.labelsize'] = 10.
# mpl.rcParams['ytick.labelsize'] = 10.

# Define the directory
output_dir = os.path.expanduser('~/critical_minerals_Africa/transport-outputs/figures/regional_figures')

# Create the folder if it doesn't exist
os.makedirs(output_dir, exist_ok=True)
# Main logic here
print(f"Output directory created or already exists: {output_dir}")


def plot_clustered_stacked(fig,axe,
                            dfall,
                            bar_colors,
                            labels=None,
                            scenario_labels = True,
                            shift_bars = True,
                            ylabel="Y-values",
                            title="multiple stacked bar plot",
                            H="/",
                            stacked=True, 
                            df_style = 'multiple',
                            orientation="vertical",
                            width=0.7, 
                            **kwargs):

    """ 
        Source: https://stackoverflow.com/questions/22787209/how-to-have-clusters-of-stacked-bars
        Given a list of dataframes, with identical columns and index, create a clustered stacked bar plot. 
        labels is a list of the names of the dataframe, used for the legend
        title is a string for the title of the plot
        H is the hatch used for identification of the different dataframe
    """
    
    n_df = len(dfall)

    if df_style == 'multiple':
        n_col = len(dfall[0].columns) 
        n_ind = len(dfall[0].index)

        for df in dfall : # for each data frame
            # bar_colors = list(islice(cycle(bar_colors), None, len(df)))
            color_cycle = list(islice(cycle(bar_colors), None, len(df.columns)))
            if orientation == "horizontal":
                print("got multiple and horizontal")
                # Plot horizontally
                axe = df.plot(
                    kind="barh",
                    # linewidth=1.0,
                    ax=axe,
                    legend=False,
                    grid=False,
                    color=color_cycle,
                    edgecolor='white',
                    stacked=stacked,
                    width=width,
                    **kwargs
                )
            else:
                print("got multiple and vertical")
                # Original vertical plotting
                axe = df.plot(
                    kind="bar",
                    # linewidth=1.0,
                    ax=axe,
                    legend=False,
                    grid=False,
                    color=color_cycle,
                    edgecolor='white',
                    stacked=stacked,
                    width=width,
                    **kwargs
                )

    # For cases when only one year is used
    if df_style == 'single_yr':
        n_col = len(dfall.columns) 
        n_ind = len(dfall.index)
        df = dfall
    # axe = plt.subplot(111)

        # bar_colors = list(islice(cycle(bar_colors), None, len(dfall)))
        color_cycle = list(islice(cycle(bar_colors), None, n_col))
        if orientation == "horizontal":
            axe = df.plot(
                kind="barh",
                # linewidth=1.0,
                stacked=stacked,
                ax=axe,
                legend=False,
                grid=False,
                color=color_cycle,
                edgecolor='white',
                width=width,
                **kwargs
            )
        else:
            axe = df.plot(
                kind="bar",
                # linewidth=1.0,
                stacked=stacked,
                ax=axe,
                legend=False,
                grid=False,
                color=color_cycle,
                edgecolor='white',
                width=width,
                **kwargs
            )
    else:
        raise ValueError(f"Unknown df_style: {df_style}")

 # -------------------------------------------
    # SHIFT THE BARS IF MULTIPLE DATAFRAMES
    # -------------------------------------------
    if shift_bars == True:
        # For vertical bars, we shift in the x-direction.
        # For horizontal bars, we shift in the y-direction.
        h, l = axe.get_legend_handles_labels()  # bar handles & labels
        # len(h) should be n_col * n_df if stacked each DF has n_col segments
        for i in range(0, n_df * n_col, n_col):
            for j, pa in enumerate(h[i:i + n_col]):
                for rect in pa.patches:
                    if orientation == "horizontal":
                        # SHIFT VERTICALLY
                        # move the bar down so each DF cluster is separate
                        rect.set_y(rect.get_y() + 1 / float(n_df + 1) * (i / float(n_col)))
                        rect.set_height(1 / float(n_df + 1))
                    else:
                        # SHIFT HORIZONTALLY (the original code)
                        rect.set_x(rect.get_x() + 1 / float(n_df + 1) * (i / float(n_col)))
                        rect.set_width(1 / float(n_df + 1))

    # -------------------------------------------
    # SET TICKS, LABELS, ETC.
    # -------------------------------------------
    if orientation == "horizontal":
        # The numeric axis is x, the categorical axis is y
        # We'll set the y-ticks to the midpoint of each cluster
        # (similar logic as before, but for y-axis).
        if df_style == "multiple":
            df_first = dfall[0]
            # number of categories
            n_ind = len(df_first.index)
        # else:
        if df_style == "single_yr":
            n_ind = len(dfall.index)
        
        # For each category, we have 2.0 steps, plus offset
        # This is the vertical version of your original set_xticks(...) code
        new_positions = (np.arange(0, 2 * n_ind, 2) + 1 / float(n_df + 1)) / 2.0

        axe.set_yticks(new_positions)
        # label with the index
        if df_style == "multiple":
            axe.set_yticklabels(df_first.index, rotation=0, fontsize=12, fontweight="bold")
        else:
            axe.set_yticklabels(dfall.index, rotation=0, fontsize=12, fontweight="bold")

        # x-axis label is now your numeric axis (production)
        axe.set_xlabel(ylabel, fontweight='bold', fontsize=15)
        # We'll call this something like 'Countries' or 'ISO' or similar
        axe.set_ylabel('')
        axe.tick_params(axis='x', labelsize=12)

        # Optionally you can flip the grid lines so it draws major lines along x
        axe.grid(which='major', axis='y', linestyle='-', zorder=0)
        axe.set_axisbelow(True)
    else:
        # the original vertical code
        # The numeric axis is y, the categorical axis is x
        if df_style == 'multiple':
            df_first = dfall[0]
            n_ind = len(df_first.index)
        else:
            n_ind = len(dfall.index)
        
        new_positions = (np.arange(0, 2 * n_ind, 2) + 1 / float(n_df + 1)) / 2.0
        axe.set_xticks(new_positions)

        if df_style == "multiple":
            axe.set_xticklabels(df_first.index, rotation=0, fontsize=12, fontweight="bold")
        else:
            axe.set_xticklabels(dfall.index, rotation=0, fontsize=12, fontweight="bold")

        axe.set_xlabel('')
        axe.set_ylabel(ylabel, fontweight='bold', fontsize=15)
        axe.tick_params(axis='y', labelsize=12)
        axe.grid(which='major', axis='x', linestyle='-', zorder=0)
        axe.set_axisbelow(True)

    # Title
    axe.set_title(title, fontweight='bold', fontsize=16)

    # -------------------------------------------
    # BUILD LEGEND
    # -------------------------------------------
    legend_handles = []
    titles = ["$\\bf{Mineral \\, processing \\, stages}$", "$\\bf Scenarios$"]
    legend_handles.append(axe.plot([],[], color="none", label="$\\bf{Mineral \\, processing \\, stages}$")[0])

    # For the columns in the first DataFrame (the "stacked" part):
    if df_style == 'multiple':
        col_labels = dfall[0].columns
    else:
        col_labels = dfall.columns

    used_colors = list(islice(cycle(bar_colors), None, len(col_labels)))
    for bc, bl in zip(used_colors, col_labels):
        legend_handles.append(mpatches.Patch(color=bc, label=bl))

    # If we have multiple scenarios, add them with option to not have scenarios
    if scenario_labels == True:
        if df_style == 'multiple' and len(labels) == n_df:
            legend_handles.append(axe.plot([], [], color="none", label="$\\bf Scenario$")[0])
            scenario_colors = list(islice(cycle(bar_colors), None, n_df))
            for idx, lbl in enumerate(labels):
                legend_handles.append(
                    mpatches.Patch(facecolor=scenario_colors[idx], edgecolor='white', label=lbl)
                )
        elif df_style == 'single_yr' and len(labels) > 0:
            legend_handles.append(axe.plot([], [], color="none", label="$\\bf Scenario$")[0])
            for idx, lbl in enumerate(labels):
                scenario_color = list(islice(cycle(bar_colors), idx, idx+1))[0]
                legend_handles.append(
                    mpatches.Patch(facecolor=scenario_color, edgecolor='white', label=lbl)
                )
    else:
        titles = ["$\\bf{Mineral \\, processing \\, stages}$"]

    leg = axe.legend(handles=legend_handles, fontsize=12, loc='upper left', frameon=False)

    # Optionally move titles in legend
    for item, label_ in zip(leg.legend_handles, leg.texts):
        if label_._text in titles:
            # width = item.get_window_extent(fig.canvas.get_renderer()).width
            label_.set_ha('left')
            # label_.set_position((-10.0 * width, 0))

    return axe



### OLD
    # h,l = axe.get_legend_handles_labels() # get the handles we want to modify
    # for i in range(0, n_df * n_col, n_col): # len(h) = n_col * n_df
    #     for j, pa in enumerate(h[i:i+n_col]):
    #         for rect in pa.patches: # for each index
    #             rect.set_x(rect.get_x() + 1 / float(n_df + 1) * i / float(n_col))
    #             rect.set_width(1 / float(n_df + 1))

    # axe.set_xticks((np.arange(0, 2 * n_ind, 2) + 1 / float(n_df + 1)) / 2.)

    # # checking if this works
    #  # Use the index of the *first* DataFrame if multiple
    # if df_style == 'multiple':
    #     df_first = dfall[0]
    #     axe.set_xticklabels(df_first.index, rotation=0, fontsize=15, fontweight="bold")
    # else:
    #     axe.set_xticklabels(dfall.index, rotation=0, fontsize=15, fontweight="bold")
    # #
    # axe.set_xticklabels(df.index, rotation = 0,fontsize=15,fontweight="bold")
    # axe.set_xlabel('')
    # axe.set_ylabel(ylabel,fontweight='bold',fontsize=15)
    # axe.tick_params(axis='y',labelsize=15)
    # axe.set_title(title,fontweight='bold',fontsize=18)
    # axe.set_axisbelow(True)
    # axe.grid(which='major', axis='x', linestyle='-', zorder=0)


    # legend_handles = []
    # titles = ["$\\bf{Mineral \, processing \, stages}$","$\\bf Scenarios$"]
    # legend_handles.append(axe.plot([],[],
    #                                 color="none",
    #                                 label="$\\bf{Mineral \, processing \, stages}$")[0])
    # if df_style == 'multiple':
    #     # The column labels come from the *first* DataFrame
    #     col_labels = dfall[0].columns
    # else:
    #     col_labels = dfall.columns
    
    # # Zip the colors with the stacked columns
    # # (If you want each "column" or "stage" to appear in the legend)
    # used_colors = list(islice(cycle(bar_colors), None, len(col_labels)))
    # for bc, bl in zip(used_colors, col_labels):
    #     legend_handles.append(mpatches.Patch(color=bc, label=bl))

    # # If we have multiple scenarios, add them with separate colors
    # # (Instead of a hatch)
    # if df_style == 'multiple' and len(labels) == n_df:
    #     legend_handles.append(
    #         axe.plot([], [], color="none", label="$\\bf Scenario$")[0]
    #     )
    #     scenario_colors = list(islice(cycle(bar_colors), None, n_df))
    #     for idx, lbl in enumerate(labels):
    #         legend_handles.append(
    #             mpatches.Patch(
    #                 facecolor=scenario_colors[idx],
    #                 edgecolor='white',
    #                 label=lbl
    #             )
    #         )
    # elif df_style == 'single_yr' and len(labels) > 0:
    #     legend_handles.append(
    #         axe.plot([], [], color="none", label="$\\bf Scenario$")[0]
    #     )
    #     for idx, lbl in enumerate(labels):
    #         # cycle to get a color for each label
    #         scenario_color = list(islice(cycle(bar_colors), idx, idx+1))[0]
    #         legend_handles.append(
    #             mpatches.Patch(
    #                 facecolor=scenario_color,
    #                 edgecolor='white',
    #                 label=lbl
    #             )
    #         )

    # ### end of checks

    # leg = axe.legend(
    #             handles=legend_handles, 
    #             fontsize=15, 
    #             loc='upper left',
    #             frameon=False)

    # # Move titles to the left 
    # for item, label in zip(leg.legend_handles, leg.texts):
    #     if label._text in titles:
    #         width = item.get_window_extent(fig.canvas.get_renderer()).width
    #         label.set_ha('left')
    #         label.set_position((-10.0*width,0))
    # return axe

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
    make_plot = False # this no longer works with the plot function definition
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
        delta_df.append(dfs[1].subtract(dfs[0], fill_value=0))
        delta_df.append(dfs[3].subtract(dfs[2], fill_value=0))
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
                                title=f"Mid National 2030 and 2040 scenarios - Constrained minus Unconstrained production of metal content")
        plt.grid()
        plt.tight_layout()
        save_fig(os.path.join(figures,
                    f"MN_scenarios_unconstrained_constrained_delta_production.png"))
        plt.close()

    """COUNTRY Delta tonnage unconstrained and constrained for 2030 and 2040 separately
    """
    make_plot = False # these aren't very useful because the differences between minerals are too big and Mn dominates
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
        delta_df.append(dfs[1].subtract(dfs[0], fill_value=0))
        delta_df.append(dfs[3].subtract(dfs[2], fill_value=0))
        # fig, ax = plt.subplots(1,1,figsize=(18,9),dpi=500)
        # ax = plot_clustered_stacked(
        #                         fig,ax,dfs,reference_mineral_colors,
        #                         labels=["National 2030 unconstrained",
        #                                 "National 2030 constrained",
        #                                 "National 2040 unconstrained",
        #                                 "National 2040 constrained"],
        #                         ylabel="Annual metal content produced (000' tonnes)", 
        #                         title=f"Mid National scenarios - metal content production")

        # plt.grid()
        # plt.tight_layout()
        # save_fig(os.path.join(figures,
        #             f"MN_scenarios_unconstrained_constrained_side_by_side.png"))
        # plt.close()

        fig, axs = plt.subplots(1, 2, figsize=(19, 9), dpi=500)  # Create a 1-row, 2-column layout
        delta_df4r = pd.DataFrame(delta_df[0]) # otherwise they are a list
        delta_df4r2 = pd.DataFrame(delta_df[1])
        

        # Plot the 2030 difference in the first subplot (axs[0])
        axs[0] = plot_clustered_stacked(
            fig, axs[0], delta_df4r, reference_mineral_colors,  
            labels=["2030 difference"],
            ylabel="Difference in annual metal content produced (000' tonnes)", 
            title="Mid National 2030 - Constrained \n minus unconstrained (metal content) (Country)",
            df_style = 'single_yr',
            stacked = False,
            scenario_labels = False,
        )

        # Plot the 2040 difference in the second subplot (axs[1])
        axs[1] = plot_clustered_stacked(
            fig, axs[1], delta_df4r2, reference_mineral_colors,  
            labels=["2040 difference"],
            ylabel="Difference in annual metal content produced (000' tonnes)", 
            title="Mid National 2040 - Constrained \n minus unconstrained (metal content) (Country)",
            df_style = 'single_yr',
            stacked = False,
            scenario_labels = False,
        )
    
        # Disable gridlines on the x-axis
        axs[0].grid(False, axis='x')
        axs[1].grid(False, axis='x')
        # Enable gridlines on the y-axis
        axs[0].grid(which='major', axis='y', linestyle='-', zorder=0)
        axs[1].grid(which='major', axis='y', linestyle='-', zorder=0)
        
        # for ax in axs:
        #     ax.axvline(x=0, color="black", linewidth=1, zorder=3)

        # plt.grid()
        plt.tight_layout()

        # Save the figure
        save_fig(os.path.join(figures, "MN_scenarios_unconstrained_constrained_delta_production_year_separated.png"))
        plt.close()

    """
    REGION Delta tonnage unconstrained and constrained for 2030 and 2040 separately
    """
    make_plot = False # these aren't very useful because the differences between minerals are too big and Mn dominates
    if make_plot is True:
        multiply_factor = 1.0e-3
        results_file = os.path.join(output_data_path,
                                "result_summaries",
                                "transport_totals_by_stage.xlsx")
        constraints = ["region_unconstrained","region_constrained"]
        reference_minerals = ["cobalt","copper","graphite","lithium","manganese","nickel"]
        reference_mineral_colors = ["#fdae61","#f46d43","#66c2a5","#c2a5cf","#fee08b","#3288bd"]
        scenarios = ["2030_mid_max_threshold_metal_tons","2040_mid_max_threshold_metal_tons"]
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
        delta_df.append(dfs[1].subtract(dfs[0], fill_value=0))
        delta_df.append(dfs[3].subtract(dfs[2], fill_value=0))
        # fig, ax = plt.subplots(1,1,figsize=(18,9),dpi=500)
        # ax = plot_clustered_stacked(
        #                         fig,ax,dfs,reference_mineral_colors,
        #                         labels=["Regional 2030 unconstrained",
        #                                 "Regional 2030 constrained",
        #                                 "Regional 2040 unconstrained",
        #                                 "Regional 2040 constrained"],
        #                         ylabel="Annual metal content produced (000' tonnes)", 
        #                         title=f"Mid Regional scenarios - metal content production",
        #                         df_style = "multiple",
        #                         orientation = "verical")

        # plt.grid()
        # plt.tight_layout()
        # save_fig(os.path.join(figures,
        #             f"MR_scenarios_unconstrained_constrained_side_by_side.png"))
        # plt.close()

        fig, axs = plt.subplots(1, 2, figsize=(19, 9), dpi=500)  # Create a 1-row, 2-column layout
        delta_df4r = pd.DataFrame(delta_df[0]) # otherwise they are a list
        delta_df4r2 = pd.DataFrame(delta_df[1])
        

        # Plot the 2030 difference in the first subplot (axs[0])
        axs[0] = plot_clustered_stacked(
            fig, axs[0], delta_df4r, reference_mineral_colors,  
            labels=["2030 difference"],
            ylabel="Difference in annual metal content produced (000' tonnes)", 
            title="Mid Regional 2030 - Constrained \n minus unconstrained (metal content) (Region)",
            width=0.7,   
            df_style = 'single_yr',
            orientation = "vertical",
            scenario_labels = False
        )

        # Plot the 2040 difference in the second subplot (axs[1])
        axs[1] = plot_clustered_stacked(
            fig, axs[1], delta_df4r2, reference_mineral_colors,  
            labels=["2040 difference"],
            ylabel="Difference in annual metal content produced (000' tonnes) (Region)", 
            title="Mid Regional 2040 - Constrained \n minus unconstrained (metal content) (Region)",
            width=0.7,   
            df_style = 'single_yr',
            orientation = "vertical",
            scenario_labels = False
        )
    
        # Disable gridlines on the x-axis
        axs[0].grid(False, axis='x')
        axs[1].grid(False, axis='x')
        # Enable gridlines on the y-axis
        axs[0].grid(which='major', axis='y', linestyle='-', zorder=0)
        axs[1].grid(which='major', axis='y', linestyle='-', zorder=0)

        # for ax in axs:
        #     ax.axvline(x=0, color="black", linewidth=1, zorder=3)

        # plt.grid()
        plt.tight_layout()

        # Save the figure
        save_fig(os.path.join(figures, "MR_scenarios_unconstrained_constrained_delta_production_year_separated.png"))
        plt.close()

    """
    REGION Delta tonnage unconstrained and constrained for 2030 and 2040 separately in subplots
    """
    make_plot = False
    if make_plot:
        multiply_factor = 1.0e-3
        results_file = os.path.join(output_data_path, "result_summaries", "transport_totals_by_stage.xlsx")
        constraints = ["region_unconstrained", "region_constrained"]
        reference_minerals = ["cobalt", "copper", "graphite", "lithium", "manganese", "nickel"]
        reference_mineral_colors = ["#fdae61", "#f46d43", "#66c2a5", "#c2a5cf", "#fee08b", "#3288bd"]
        scenarios = [
            "2030_mid_max_threshold_metal_tons",
            "2040_mid_max_threshold_metal_tons"
        ]
        tons_column = "production_tonnes"
        index_cols = ["reference_mineral", "iso3"]
        all_properties = mineral_properties()

        for rf in reference_minerals:
            dfs = []  # To store DataFrames for each scenario and constraint
            stages_all = all_properties[rf]["stage_labels"]
            stage_colors_all = all_properties[rf]["stage_label_colors"]
            
            for scenario in scenarios:
                # Keeping only the relevant stages for each year
                if '2030' in scenario:
                    stages = [item for item in stages_all if 'Precursor related product' not in item]
                    stage_colors = [item for item in stage_colors_all if '#7f0000' not in item]
                if '2040' in scenario:
                    stages = [item for item in stages_all if 'Early refining' not in item]
                    stage_colors = [item for item in stage_colors_all if '#fb6a4a' not in item]

                for cs in constraints:
                    df = pd.read_excel(results_file, sheet_name=cs, index_col=[0, 1, 2, 3, 4])
                    df = df.reset_index()

                    # Filter by scenario, processing_stage, and mineral
                    filtered_df = df[(df["scenario"] == scenario) & 
                                    (df["processing_stage"] > 0) & 
                                    (df["reference_mineral"] == rf)]

                    if filtered_df.empty:
                        print(f"No data available for scenario '{scenario}' and constraint '{cs}'. Skipping...")
                        continue

                    m_df = pd.DataFrame(sorted(filtered_df["iso3"].unique()), columns=["iso3"])

                    # Aggregate and process stage-wise data
                    for st in stages:
                        s_df = filtered_df[filtered_df["processing_type"] == st]
                        if not s_df.empty:
                            s_df[tons_column] = multiply_factor * s_df[tons_column]
                            s_df = s_df.groupby(["iso3", "processing_type"])[tons_column].sum().reset_index()
                            s_df.rename(columns={tons_column: f"{st}"}, inplace=True)
                            m_df = pd.merge(m_df, s_df[["iso3", f"{st}"]], how="left", on="iso3").fillna(0)
                        else:
                            m_df[f"{st}"] = 0

                    dfs.append(m_df.set_index("iso3"))

            # Compute the differences for 2030 and 2040
            delta_df = []
            if len(dfs) >= 4:  # Ensure we have all the required DataFrames
                delta_df.append(dfs[1].subtract(dfs[0], fill_value=0))  # 2030 constrained minus unconstrained
                delta_df.append(dfs[3].subtract(dfs[2], fill_value=0))  # 2040 constrained minus unconstrained

            # Plot results for 2030 and 2040
            fig, axs = plt.subplots(1, 2, figsize=(19, 9), dpi=500, sharey=True)

            if not delta_df:
                print(f"No data differences to plot for {rf}. Skipping...")
                continue
            delta_df_2030 = pd.DataFrame(delta_df[0]) if delta_df else pd.DataFrame()
            delta_df_2040 = pd.DataFrame(delta_df[1]) if delta_df else pd.DataFrame()

            if delta_df_2030 is None or delta_df_2040 is None:
                print(f"Missing data for 2030 or 2040 for {rf}. Skipping...")
                continue
            
            if not delta_df_2030.empty:
                axs[0] = plot_clustered_stacked(
                    fig, axs[0], delta_df_2030, stage_colors,
                    labels=["2030 difference"],
                    ylabel="Difference in annual production (000' tonnes)",
                    title=f"{rf.title()} 2030 - Constrained minus unconstrained (Regional)",
                    df_style = 'single_yr',
                    stacked=False,
                    scenario_labels = False,
                    width = 0.8
                )

            if not delta_df_2040.empty:
                axs[1] = plot_clustered_stacked(
                    fig, axs[1], delta_df_2040, stage_colors,
                    labels=["2040 difference"],
                    ylabel="Difference in annual production (000' tonnes)",
                    title=f"{rf.title()} 2040 - Constrained minus unconstrained (Regional)",
                    df_style = 'single_yr',
                    stacked=False,
                    scenario_labels = False,
                    width = 0.8
                )

            # Configure gridlines
            for ax in axs:
                ax.grid(False, axis="x")
                ax.grid(which="major", axis="y", linestyle="-", zorder=0)
                ax.axhline(y=0, color="black", linewidth=1, zorder=3)

            plt.tight_layout()
            save_fig(os.path.join(figures, f"{rf}_MR_constrained_unconstrained_production_comparisons_2030_2040.png"))
            plt.close()

            # HORIZONTAL plot

            # Plot results for 2030 and 2040
            fig, axs = plt.subplots(2, 1, figsize=(9, 19), dpi=500, sharex=True)

            if not delta_df_2030.empty:
                axs[0] = plot_clustered_stacked(
                    fig, axs[0], delta_df_2030, stage_colors,
                    labels=["2030 difference"],
                    ylabel="Difference in annual production (000' tonnes)",
                    title=f"{rf.title()} 2030 - Constrained minus unconstrained (Regional)",
                    df_style = 'single_yr',
                    orientation = "horizontal",
                    stacked=False,
                    scenario_labels = False,
                    width = 0.8
                )

            if not delta_df_2040.empty:
                axs[1] = plot_clustered_stacked(
                    fig, axs[1], delta_df_2040, stage_colors,
                    labels=["2040 difference"],
                    ylabel="Difference in annual production (000' tonnes)",
                    title=f"{rf.title()} 2040 - Constrained minus unconstrained (Regional)",
                    df_style = 'single_yr',
                    orientation = "horizontal",
                    stacked=False,
                    scenario_labels = False,
                    width = 0.8
                )

            # Configure gridlines
            for ax in axs:
                ax.grid(False, axis="y")
                ax.grid(which="major", axis="x", linestyle="-", zorder=0)
                ax.axhline(y=0, color="black", linewidth=1, zorder=3)

            plt.tight_layout()
            save_fig(os.path.join(figures, f"{rf}_MR_constrained_unconstrained_production_comparisons_2030_2040_H.png"))
            plt.close()


    """
    COUNTRY Delta tonnage unconstrained and constrained for 2030 and 2040 separately in subplots
    """
    make_plot = False
    if make_plot:
        multiply_factor = 1.0e-3
        results_file = os.path.join(output_data_path, "result_summaries", "transport_totals_by_stage.xlsx")
        constraints = ["country_unconstrained", "country_constrained"]
        reference_minerals = ["cobalt", "copper", "graphite", "lithium", "manganese", "nickel"]
        reference_mineral_colors = ["#fdae61", "#f46d43", "#66c2a5", "#c2a5cf", "#fee08b", "#3288bd"]
        scenarios = [
            "2030_mid_min_threshold_metal_tons",
            "2040_mid_min_threshold_metal_tons"
        ]
        tons_column = "production_tonnes"
        index_cols = ["reference_mineral", "iso3"]
        all_properties = mineral_properties()

        for rf in reference_minerals:
            dfs = []  # To store DataFrames for each scenario and constraint
            stages_all = all_properties[rf]["stage_labels"]
            stage_colors_all = all_properties[rf]["stage_label_colors"]

            for scenario in scenarios:
                # Keeping only the relevant stages for each year
                if '2030' in scenario:
                    stages = [item for item in stages_all if 'Precursor related product' not in item]
                    stage_colors = [item for item in stage_colors_all if '#7f0000' not in item]
                if '2040' in scenario:
                    stages = [item for item in stages_all if 'Early refining' not in item]
                    stage_colors = [item for item in stage_colors_all if '#fb6a4a' not in item]

                for cs in constraints:
                    df = pd.read_excel(results_file, sheet_name=cs, index_col=[0, 1, 2, 3, 4])
                    df = df.reset_index()

                    # Filter by scenario, processing_stage, and mineral
                    filtered_df = df[(df["scenario"] == scenario) & 
                                    (df["processing_stage"] > 0) & 
                                    (df["reference_mineral"] == rf)]

                    if filtered_df.empty:
                        print(f"No data available for scenario '{scenario}' and constraint '{cs}'. Skipping...")
                        continue

                    m_df = pd.DataFrame(sorted(filtered_df["iso3"].unique()), columns=["iso3"])

                    # Aggregate and process stage-wise data
                    for st in stages:
                        s_df = filtered_df[filtered_df["processing_type"] == st]
                        if not s_df.empty:
                            s_df[tons_column] = multiply_factor * s_df[tons_column]
                            s_df = s_df.groupby(["iso3", "processing_type"])[tons_column].sum().reset_index()
                            s_df.rename(columns={tons_column: f"{st}"}, inplace=True)
                            m_df = pd.merge(m_df, s_df[["iso3", f"{st}"]], how="left", on="iso3").fillna(0)
                        else:
                            m_df[f"{st}"] = 0

                    dfs.append(m_df.set_index("iso3"))

            # Compute the differences for 2030 and 2040
            delta_df = []
            if len(dfs) >= 4:  # Ensure we have all the required DataFrames
                delta_df.append(dfs[1].subtract(dfs[0], fill_value=0))  # 2030 constrained minus unconstrained
                delta_df.append(dfs[3].subtract(dfs[2], fill_value=0))  # 2040 constrained minus unconstrained

            # Plot results for 2030 and 2040
            fig, axs = plt.subplots(1, 2, figsize=(19, 9), dpi=500, sharey=True)

            if not delta_df:
                print(f"No data differences to plot for {rf}. Skipping...")
                continue
            delta_df_2030 = pd.DataFrame(delta_df[0]) if delta_df else pd.DataFrame()
            delta_df_2040 = pd.DataFrame(delta_df[1]) if delta_df else pd.DataFrame()

            if delta_df_2030 is None or delta_df_2040 is None:
                print(f"Missing data for 2030 or 2040 for {rf}. Skipping...")
                continue
            
            # VERTICAL PLOT
            if not delta_df_2030.empty:
                axs[0] = plot_clustered_stacked(
                    fig, axs[0], delta_df_2030, stage_colors,
                    labels=["2030 difference"],
                    ylabel="Difference in annual production (000' tonnes)",
                    title=f"{rf.title()} 2030 - Constrained minus unconstrained (Country)",
                    df_style = 'single_yr',
                    stacked=False,
                    scenario_labels = False,
                    width = 0.8
                )

            if not delta_df_2040.empty:
                axs[1] = plot_clustered_stacked(
                    fig, axs[1], delta_df_2040, stage_colors,
                    labels=["2040 difference"],
                    ylabel="Difference in annual production (000' tonnes)",
                    title=f"{rf.title()} 2040 - Constrained minus unconstrained (Country)",
                    df_style = 'single_yr',
                    stacked=False,
                    scenario_labels = False,
                    width = 0.8
                )

            # Configure gridlines
            for ax in axs:
                ax.grid(False, axis="x")
                ax.grid(which="major", axis="y", linestyle="-", zorder=0)
                ax.axhline(y=0, color="black", linewidth=1, zorder=3)

            plt.tight_layout()
            save_fig(os.path.join(figures, f"{rf}_MN_constrained_unconstrained_production_comparisons_2030_2040.png"))
            plt.close()

            # HORIZONTAL PLOT
            # Plot results for 2030 and 2040
            fig, axs = plt.subplots(2, 1, figsize=(9, 19), dpi=500, sharex=True)
            
            if not delta_df_2030.empty:
                axs[0] = plot_clustered_stacked(
                    fig, axs[0], delta_df_2030, stage_colors,
                    labels=["2030 difference"],
                    ylabel="Difference in annual production (000' tonnes)",
                    title=f"{rf.title()} 2030 - Constrained minus unconstrained (Country)",
                    df_style = 'single_yr',
                    stacked=False,
                    orientation="horizontal",
                    scenario_labels = False,
                    width = 0.8
                )

            if not delta_df_2040.empty:
                axs[1] = plot_clustered_stacked(
                    fig, axs[1], delta_df_2040, stage_colors,
                    labels=["2040 difference"],
                    ylabel="Difference in annual production (000' tonnes)",
                    title=f"{rf.title()} 2040 - Constrained minus unconstrained (Country)",
                    df_style = 'single_yr',
                    stacked=False,
                    orientation="horizontal",
                    scenario_labels = False,
                    width = 0.8
                )

            # Configure gridlines
            for ax in axs:
                ax.grid(False, axis="y")
                ax.grid(which="major", axis="x", linestyle="-", zorder=0)
                ax.axvline(x=0, color="black", linewidth=1, zorder=3)

            plt.tight_layout()
            save_fig(os.path.join(figures, f"{rf}_MN_constrained_unconstrained_production_comparisons_2030_2040_H.png"))
            plt.close()
    
    
    """
    Regional constrained unconstrained production costs' comparisons for 2030 and 2040 separately in subplots
    """
    make_plot = False
    if make_plot:
        multiply_factor = 1.0e-6
        results_file = os.path.join(output_data_path, "result_summaries", "energy_transport_totals_by_stage.xlsx")
        constraints = ["region_unconstrained", "region_constrained"]
        reference_minerals = ["cobalt", "copper", "graphite", "lithium", "manganese", "nickel"]
        reference_mineral_colors = ["#fdae61", "#f46d43", "#66c2a5", "#c2a5cf", "#fee08b", "#3288bd"]
        scenarios = [
            "2030_mid_max_threshold_metal_tons",
            "2040_mid_max_threshold_metal_tons"
        ]
        tons_column = "production_cost_usd"
        index_cols = ["reference_mineral", "iso3"]
        all_properties = mineral_properties()

        for rf in reference_minerals:
            dfs = []  # To store DataFrames for each scenario and constraint
            stages_all = all_properties[rf]["stage_labels"]
            stage_colors_all = all_properties[rf]["stage_label_colors"]

            for scenario in scenarios:
                # Keeping only the relevant stages for each year
                if '2030' in scenario:
                    stages = [item for item in stages_all if 'Precursor related product' not in item]
                    stage_colors = [item for item in stage_colors_all if '#7f0000' not in item]
                if '2040' in scenario:
                    stages = [item for item in stages_all if 'Early refining' not in item]
                    stage_colors = [item for item in stage_colors_all if '#fb6a4a' not in item]


                for cs in constraints:
                    df = pd.read_excel(results_file, sheet_name=cs, index_col=[0, 1, 2, 3, 4])
                    df = df.reset_index()

                    # Filter by scenario, processing_stage, and mineral
                    filtered_df = df[(df["scenario"] == scenario) & 
                                    (df["processing_stage"] > 0) & 
                                    (df["reference_mineral"] == rf)]

                    if filtered_df.empty:
                        print(f"No data available for scenario '{scenario}' and constraint '{cs}'. Skipping...")
                        continue

                    m_df = pd.DataFrame(sorted(filtered_df["iso3"].unique()), columns=["iso3"])

                    # Aggregate and process stage-wise data
                    for st in stages:
                        s_df = filtered_df[filtered_df["processing_type"] == st]
                        if not s_df.empty:
                            s_df[tons_column] = multiply_factor * s_df[tons_column]
                            s_df = s_df.groupby(["iso3", "processing_type"])[tons_column].sum().reset_index()
                            s_df.rename(columns={tons_column: f"{st}"}, inplace=True)
                            m_df = pd.merge(m_df, s_df[["iso3", f"{st}"]], how="left", on="iso3").fillna(0)
                        else:
                            m_df[f"{st}"] = 0

                    dfs.append(m_df.set_index("iso3"))

            # Compute the differences for 2030 and 2040
            delta_df = []
            if len(dfs) >= 4:  # Ensure we have all the required DataFrames
                delta_df.append(dfs[1].subtract(dfs[0], fill_value=0))  # 2030 constrained minus unconstrained
                delta_df.append(dfs[3].subtract(dfs[2], fill_value=0))  # 2040 constrained minus unconstrained

            # VERTICAL
            # Plot results for 2030 and 2040
            fig, axs = plt.subplots(1, 2, figsize=(19, 9), dpi=500, sharey=True)

            if not delta_df:
                print(f"No data differences to plot for {rf}. Skipping...")
                continue
            delta_df_2030 = pd.DataFrame(delta_df[0]) if delta_df else pd.DataFrame()
            delta_df_2040 = pd.DataFrame(delta_df[1]) if delta_df else pd.DataFrame()

            if delta_df_2030 is None or delta_df_2040 is None:
                print(f"Missing data for 2030 or 2040 for {rf}. Skipping...")
                continue
            
            if not delta_df_2030.empty:
                axs[0] = plot_clustered_stacked(
                    fig, axs[0], delta_df_2030, stage_colors,
                    labels=["2030 difference"],
                    ylabel="Difference in annual production costs (million USD)",
                    title=f"{rf.title()} 2030 - Constrained minus unconstrained (Region)",
                    df_style = 'single_yr',
                    scenario_labels = False,
                )

            if not delta_df_2040.empty:
                axs[1] = plot_clustered_stacked(
                    fig, axs[1], delta_df_2040, stage_colors,
                    labels=["2040 difference"],
                    ylabel="Difference in annual production costs (million USD)",
                    title=f"{rf.title()} 2040 - Constrained minus unconstrained (Region)",
                    df_style = 'single_yr',
                    scenario_labels = False,
                )

            # Configure gridlines
            for ax in axs:
                ax.grid(False, axis="x")
                ax.grid(which="major", axis="y", linestyle="-", zorder=0)
                ax.axhline(y=0, color="black", linewidth=1, zorder=3)

            plt.tight_layout()
            save_fig(os.path.join(figures, f"{rf}_MR_constrained_unconstrained_costs_comparisons_2030_2040.png"))
            plt.close()

            # HORIZONTAL
            # Plot results for 2030 and 2040
            fig, axs = plt.subplots(2, 1, figsize=(9, 19), dpi=500, sharex=True)

            if not delta_df_2030.empty:
                axs[0] = plot_clustered_stacked(
                    fig, axs[0], delta_df_2030, stage_colors,
                    labels=["2030 difference"],
                    ylabel="Difference in annual production costs (million USD)",
                    title=f"{rf.title()} 2030 - Constrained minus unconstrained (Region)",
                    df_style = 'single_yr',
                    scenario_labels = False,
                    stacked = False,
                    orientation = 'horizontal'
                )

            if not delta_df_2040.empty:
                axs[1] = plot_clustered_stacked(
                    fig, axs[1], delta_df_2040, stage_colors,
                    labels=["2040 difference"],
                    ylabel="Difference in annual production costs (million USD)",
                    title=f"{rf.title()} 2040 - Constrained minus unconstrained (Region)",
                    df_style = 'single_yr',
                    scenario_labels = False,
                    stacked = False,
                    orientation = 'horizontal'
                )

            # Configure gridlines
            for ax in axs:
                ax.grid(False, axis="y")
                ax.grid(which="major", axis="x", linestyle="-", zorder=0)
                ax.axvline(x=0, color="black", linewidth=1, zorder=3)

            plt.tight_layout()
            save_fig(os.path.join(figures, f"{rf}_MR_constrained_unconstrained_costs_comparisons_2030_2040_H.png"))
            plt.close()
    """
    Country constrained unconstrained production costs' comparisons for 2030 and 2040 separately in subplots
    """
    make_plot = False
    if make_plot:
        multiply_factor = 1.0e-6
        results_file = os.path.join(output_data_path, "result_summaries", "energy_transport_totals_by_stage.xlsx")
        constraints = ["country_unconstrained", "country_constrained"]
        reference_minerals = ["cobalt", "copper", "graphite", "lithium", "manganese", "nickel"]
        reference_mineral_colors = ["#fdae61", "#f46d43", "#66c2a5", "#c2a5cf", "#fee08b", "#3288bd"]
        scenarios = [
            "2030_mid_min_threshold_metal_tons",
            "2040_mid_min_threshold_metal_tons"
        ]
        tons_column = "production_cost_usd"
        index_cols = ["reference_mineral", "iso3"]
        all_properties = mineral_properties()

        for rf in reference_minerals:
            dfs = []  # To store DataFrames for each scenario and constraint
            stages_all = all_properties[rf]["stage_labels"]
            stage_colors_all = all_properties[rf]["stage_label_colors"]

            for scenario in scenarios:
                # Keeping only the relevant stages for each year
                if '2030' in scenario:
                    stages = [item for item in stages_all if 'Precursor related product' not in item]
                    stage_colors = [item for item in stage_colors_all if '#7f0000' not in item]
                if '2040' in scenario:
                    stages = [item for item in stages_all if 'Early refining' not in item]
                    stage_colors = [item for item in stage_colors_all if '#fb6a4a' not in item]

                for cs in constraints:
                    df = pd.read_excel(results_file, sheet_name=cs, index_col=[0, 1, 2, 3, 4])
                    df = df.reset_index()

                    # Filter by scenario, processing_stage, and mineral
                    filtered_df = df[(df["scenario"] == scenario) & 
                                    (df["processing_stage"] > 0) & 
                                    (df["reference_mineral"] == rf)]

                    if filtered_df.empty:
                        print(f"No data available for scenario '{scenario}' and constraint '{cs}'. Skipping...")
                        continue

                    m_df = pd.DataFrame(sorted(filtered_df["iso3"].unique()), columns=["iso3"])

                    # Aggregate and process stage-wise data
                    for st in stages:
                        s_df = filtered_df[filtered_df["processing_type"] == st]
                        if not s_df.empty:
                            s_df[tons_column] = multiply_factor * s_df[tons_column]
                            s_df = s_df.groupby(["iso3", "processing_type"])[tons_column].sum().reset_index()
                            s_df.rename(columns={tons_column: f"{st}"}, inplace=True)
                            m_df = pd.merge(m_df, s_df[["iso3", f"{st}"]], how="left", on="iso3").fillna(0)
                        else:
                            m_df[f"{st}"] = 0

                    dfs.append(m_df.set_index("iso3"))

            # Compute the differences for 2030 and 2040
            delta_df = []
            if len(dfs) >= 4:  # Ensure we have all the required DataFrames
                delta_df.append(dfs[1].subtract(dfs[0], fill_value=0))  # 2030 constrained minus unconstrained
                delta_df.append(dfs[3].subtract(dfs[2], fill_value=0))  # 2040 constrained minus unconstrained

            # VERTICAL
            # Plot results for 2030 and 2040
            fig, axs = plt.subplots(1, 2, figsize=(19, 9), dpi=500, sharey=True)

            if not delta_df:
                print(f"No data differences to plot for {rf}. Skipping...")
                continue
            delta_df_2030 = pd.DataFrame(delta_df[0]) if delta_df else pd.DataFrame()
            delta_df_2040 = pd.DataFrame(delta_df[1]) if delta_df else pd.DataFrame()

            if delta_df_2030 is None or delta_df_2040 is None:
                print(f"Missing data for 2030 or 2040 for {rf}. Skipping...")
                continue
            
            if not delta_df_2030.empty:
                axs[0] = plot_clustered_stacked(
                    fig, axs[0], delta_df_2030, stage_colors,
                    labels=["2030 difference"],
                    ylabel="Difference in annual production costs (million USD)",
                    title=f"{rf.title()} 2030 - Constrained minus unconstrained (Country)",
                    df_style = 'single_yr',
                    scenario_labels = False,
                )

            if not delta_df_2040.empty:
                axs[1] = plot_clustered_stacked(
                    fig, axs[1], delta_df_2040, stage_colors,
                    labels=["2040 difference"],
                    ylabel="Difference in annual production costs (million USD)",
                    title=f"{rf.title()} 2040 - Constrained minus unconstrained (Country)",
                    df_style = 'single_yr',
                    scenario_labels = False,
                )

            # Configure gridlines
            for ax in axs:
                ax.grid(False, axis="x")
                ax.grid(which="major", axis="y", linestyle="-", zorder=0)
                ax.axhline(y=0, color="black", linewidth=1, zorder=3)

            plt.tight_layout()
            save_fig(os.path.join(figures, f"{rf}_MN_constrained_unconstrained_costs_comparisons_2030_2040.png"))
            plt.close()

            # HORIZONTAL
            # Plot results for 2030 and 2040
            fig, axs = plt.subplots(2, 1, figsize=(9, 19), dpi=500, sharex=True)

            if not delta_df_2030.empty:
                axs[0] = plot_clustered_stacked(
                    fig, axs[0], delta_df_2030, stage_colors,
                    labels=["2030 difference"],
                    ylabel="Difference in annual production costs (million USD)",
                    title=f"{rf.title()} 2030 - Constrained minus unconstrained (Country)",
                    df_style = 'single_yr',
                    scenario_labels = False,
                    stacked = False,
                    orientation = 'horizontal'
                )

            if not delta_df_2040.empty:
                axs[1] = plot_clustered_stacked(
                    fig, axs[1], delta_df_2040, stage_colors,
                    labels=["2040 difference"],
                    ylabel="Difference in annual production costs (million USD)",
                    title=f"{rf.title()} 2040 - Constrained minus unconstrained (Country)",
                    df_style = 'single_yr',
                    scenario_labels = False,
                    stacked = False,
                    orientation = 'horizontal'
                )

            # Configure gridlines
            for ax in axs:
                ax.grid(False, axis="y")
                ax.grid(which="major", axis="x", linestyle="-", zorder=0)
                ax.axvline(x=0, color="black", linewidth=1, zorder=3)

            plt.tight_layout()
            save_fig(os.path.join(figures, f"{rf}_MN_constrained_unconstrained_costs_comparisons_2030_2040_H.png"))
            plt.close()



    """Unconstrained country and regional comparisons: annual production
    """
    make_plot = False
    if make_plot is True:
        multiply_factor = 1.0e-3
        results_file = os.path.join(output_data_path,
                                "result_summaries",
                                "transport_totals_by_stage.xlsx")
        constraints = ["country_unconstrained","region_unconstrained"]
        reference_mineral_colors = ["#f46d43","#fdae61","#fee08b","#c2a5cf","#66c2a5","#3288bd"]
        reference_minerals = ["copper","cobalt","manganese","lithium","graphite","nickel"]
        scenarios =  [
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
                        f"{rf}_MN_MR_production_comparisions_side_by_side.png"))
            plt.close()
    
    """Country and regional comparisons for 2030 and 2040 separately in subplots: annual production
    """
    make_plot = False
    if make_plot:
        multiply_factor = 1.0e-3
        results_file = os.path.join(output_data_path, "result_summaries", "transport_totals_by_stage.xlsx")
        constraints = ["country_unconstrained", "region_unconstrained"]
        reference_minerals = ["cobalt", "copper", "graphite", "lithium", "manganese", "nickel"]
        reference_mineral_colors = ["#fdae61", "#f46d43", "#66c2a5", "#c2a5cf", "#fee08b", "#3288bd"]
        scenarios = [
            "2030_mid_min_threshold_metal_tons",
            "2030_mid_max_threshold_metal_tons",
            "2040_mid_min_threshold_metal_tons",
            "2040_mid_max_threshold_metal_tons"
        ]
        tons_column = "production_tonnes"
        index_cols = ["reference_mineral", "iso3"]
        all_properties = mineral_properties()

        for rf in reference_minerals:
            dfs = []  # To store DataFrames for each scenario and constraint
            stages = all_properties[rf]["stage_labels"]
            stage_colors = all_properties[rf]["stage_label_colors"]
            print(stage_colors)

            for scenario in scenarios:
                for cs in constraints:
                    df = pd.read_excel(results_file, sheet_name=cs, index_col=[0, 1, 2, 3, 4])
                    df = df.reset_index()

                    # Filter by scenario, processing_stage, and mineral
                    filtered_df = df[(df["scenario"] == scenario) & 
                                    (df["processing_stage"] > 0) & 
                                    (df["reference_mineral"] == rf)]

                    if filtered_df.empty:
                        print(f"No data available for scenario '{scenario}' and constraint '{cs}'. Skipping...")
                        continue

                    m_df = pd.DataFrame(sorted(filtered_df["iso3"].unique()), columns=["iso3"])

                    # Aggregate and process stage-wise data
                    for st in stages:
                        s_df = filtered_df[filtered_df["processing_type"] == st]
                        if not s_df.empty:
                            s_df[tons_column] = multiply_factor * s_df[tons_column]
                            s_df = s_df.groupby(["iso3", "processing_type"])[tons_column].sum().reset_index()
                            s_df.rename(columns={tons_column: f"{st}"}, inplace=True)
                            m_df = pd.merge(m_df, s_df[["iso3", f"{st}"]], how="left", on="iso3").fillna(0)
                        else:
                            m_df[f"{st}"] = 0

                    dfs.append(m_df.set_index("iso3"))

            # Compute the differences for 2030 and 2040
            delta_df = []
            if len(dfs) >= 4:  # Ensure we have all the required DataFrames
                delta_df.append(dfs[1].subtract(dfs[0], fill_value=0))  # 2030 Regional - National 
                delta_df.append(dfs[3].subtract(dfs[2], fill_value=0))  # 2040 Regional - National

            # Plot results for 2030 and 2040
            fig, axs = plt.subplots(2, 1, figsize=(9, 18), dpi=500, sharex=True)

            if not delta_df:
                print(f"No data differences to plot for {rf}. Skipping...")
                continue
            delta_df_2030 = pd.DataFrame(delta_df[0]) if delta_df else pd.DataFrame()
            delta_df_2040 = pd.DataFrame(delta_df[1]) if delta_df else pd.DataFrame()

            # print('2030 data: Regional - National')
            # print(delta_df_2030)

            if delta_df_2030 is None or delta_df_2040 is None:
                print(f"Missing data for 2030 or 2040 for {rf}. Skipping...")
                continue
            if not delta_df_2030.empty:
                axs[0] = plot_clustered_stacked(
                    fig, axs[0], delta_df_2030, stage_colors,
                    labels=["2030 difference"],
                    ylabel="Difference in annual production (000' tonnes)",
                    width=0.7,   
                    title=f"{rf.title()} 2030 - Regional minus National (Unconstrained)",
                    df_style = 'single_yr',
                    orientation="horizontal",
                    scenario_labels = False,
                    shift_bars = False
                )

            if not delta_df_2040.empty:
                axs[1] = plot_clustered_stacked(
                    fig, axs[1], delta_df_2040, stage_colors,
                    labels=["2040 difference"],
                    ylabel="Difference in annual production (000' tonnes)",
                    width=0.7,   
                    title=f"{rf.title()} 2040 - Regional minus National (Unconstrained)",
                    df_style = 'single_yr',
                    orientation="horizontal",
                    scenario_labels = False,
                    shift_bars = False
                )

            # Configure gridlines
            for ax in axs:
                ax.grid(False, axis="y")
                ax.grid(which="major", axis="x", linestyle="-", zorder=0)
                ax.axvline(x=0, color="black", linewidth=1, zorder=3)

            plt.tight_layout()
            save_fig(os.path.join(figures, f"{rf}_MN_MR_production_comparisons_2030_2040.png"))
            plt.close()

    """
    Country and regional comparisons for 2030 and 2040 separately in subplots: annual production costs
    """
    make_plot = False
    if make_plot:
        multiply_factor = 1.0e-6
        results_file = os.path.join(output_data_path, "result_summaries", "energy_transport_totals_by_stage.xlsx")
        constraints = ["country_unconstrained", "region_unconstrained"]
        reference_minerals = ["cobalt", "copper", "graphite", "lithium", "manganese", "nickel"]
        reference_mineral_colors = ["#fdae61", "#f46d43", "#66c2a5", "#c2a5cf", "#fee08b", "#3288bd"]
        scenarios = [
            "2030_mid_min_threshold_metal_tons",
            "2030_mid_max_threshold_metal_tons",
            "2040_mid_min_threshold_metal_tons",
            "2040_mid_max_threshold_metal_tons"
        ]
        tons_column = "production_cost_usd"
        index_cols = ["reference_mineral", "iso3"]
        all_properties = mineral_properties()

        for rf in reference_minerals:
            dfs = []  # To store DataFrames for each scenario and constraint
            stages = all_properties[rf]["stage_labels"]
            stage_colors = all_properties[rf]["stage_label_colors"]

            for scenario in scenarios:
                for cs in constraints:
                    df = pd.read_excel(results_file, sheet_name=cs, index_col=[0, 1, 2, 3, 4])
                    df = df.reset_index()

                    # Filter by scenario, processing_stage, and mineral
                    filtered_df = df[(df["scenario"] == scenario) & 
                                    (df["processing_stage"] > 0) & 
                                    (df["reference_mineral"] == rf)]

                    if filtered_df.empty:
                        print(f"No data available for scenario '{scenario}' and constraint '{cs}'. Skipping...")
                        continue

                    m_df = pd.DataFrame(sorted(filtered_df["iso3"].unique()), columns=["iso3"])

                    # Aggregate and process stage-wise data
                    for st in stages:
                        s_df = filtered_df[filtered_df["processing_type"] == st]
                        if not s_df.empty:
                            s_df[tons_column] = multiply_factor * s_df[tons_column]
                            s_df = s_df.groupby(["iso3", "processing_type"])[tons_column].sum().reset_index()
                            s_df.rename(columns={tons_column: f"{st}"}, inplace=True)
                            m_df = pd.merge(m_df, s_df[["iso3", f"{st}"]], how="left", on="iso3").fillna(0)
                        else:
                            m_df[f"{st}"] = 0

                    dfs.append(m_df.set_index("iso3"))

            # Compute the differences for 2030 and 2040
            delta_df = []
            if len(dfs) >= 4:  # Ensure we have all the required DataFrames
                delta_df.append(dfs[1].subtract(dfs[0], fill_value=0))  # 2030 Regional - National
                delta_df.append(dfs[3].subtract(dfs[2], fill_value=0))  # 2040 Regional - National

            # Plot results for 2030 and 2040
            fig, axs = plt.subplots(1, 2, figsize=(19, 9), dpi=500, sharey=True)

            if not delta_df:
                print(f"No data differences to plot for {rf}. Skipping...")
                continue
            delta_df_2030 = pd.DataFrame(delta_df[0]) if delta_df else pd.DataFrame()
            delta_df_2040 = pd.DataFrame(delta_df[1]) if delta_df else pd.DataFrame()

            if delta_df_2030 is None or delta_df_2040 is None:
                print(f"Missing data for 2030 or 2040 for {rf}. Skipping...")
                continue
            
            if not delta_df_2030.empty:
                axs[0] = plot_clustered_stacked(
                    fig, axs[0], delta_df_2030, stage_colors,
                    labels=["2030 difference"],
                    ylabel="Difference in annual production costs (million USD)",
                    title=f"{rf.title()} 2030 - Regional minus National",
                    df_style = 'single_yr'
                )

            if not delta_df_2040.empty:
                axs[1] = plot_clustered_stacked(
                    fig, axs[1], delta_df_2040, stage_colors,
                    labels=["2040 difference"],
                    ylabel="Difference in annual production costs (million USD)",
                    title=f"{rf.title()} 2040 - Regional minus National",
                    df_style = 'single_yr'
                )

            # Configure gridlines
            for ax in axs:
                ax.grid(False, axis="x")
                ax.grid(which="major", axis="y", linestyle="-", zorder=0)

            plt.tight_layout()
            save_fig(os.path.join(figures, f"{rf}_MN_MR_costs_comparisons_2030_2040.png"))
            plt.close()

    """
    Country and regional comparisons for 2030 and 2040 separately in subplots: annual production costs
    """
    make_plot = False # these are now no longer right, need fixing
    if make_plot:
        multiply_factor = 1.0e-6 # make 1 for those w per tonne
        units = 'million USD per tonne'
        results_file = os.path.join(output_data_path, "result_summaries", "energy_transport_totals_by_stage.xlsx")
        constraints = ["country_unconstrained", "region_unconstrained"]
        reference_minerals = ["cobalt", "copper", "graphite", "lithium", "manganese", "nickel"]
        reference_mineral_colors = ["#fdae61", "#f46d43", "#66c2a5", "#c2a5cf", "#fee08b", "#3288bd"]
        scenarios = [
                    "2030_mid_min_threshold_metal_tons",
                    "2030_mid_max_threshold_metal_tons",
                    "2040_mid_min_threshold_metal_tons",
                    "2040_mid_max_threshold_metal_tons"
                ]
        cost_column = ['export_transport_cost_usd_per_tonne', 
                       'import_transport_cost_usd_per_tonne', # import aren't as useful but plotting just in case
                        # 'total_gcosts_per_tons', #  general total transport costs per tonnes along roads and rail routes within a country
                        'energy_opex'
                    ]
        index_cols = ["reference_mineral", "iso3"]
        en_tr = ['transport',
                'transport',
                # 'transport',
                'energy']
        all_properties = mineral_properties()
        
        for rf in reference_minerals:
            dfs = []  # To store DataFrames for each scenario and constraint
            stages = all_properties[rf]["stage_labels"]
            stage_colors = all_properties[rf]["stage_label_colors"]

            for scenario in scenarios:
                for cs in constraints:
                    df = pd.read_excel(results_file, sheet_name=cs, index_col=[0, 1, 2, 3, 4])
                    df = df.reset_index()

                    # Filter by scenario, processing_stage, and mineral
                    filtered_df = df[(df["scenario"] == scenario) & 
                                    (df["processing_stage"] > 0) & 
                                    (df["reference_mineral"] == rf)]

                    if filtered_df.empty:
                        print(f"No data available for scenario '{scenario}' and constraint '{cs}'. Skipping...")
                        continue

                    m_df = pd.DataFrame(sorted(filtered_df["iso3"].unique()), columns=["iso3"])

                    for c, et in zip(cost_column, en_tr):
                        if c != 'energy_opex':
                            multiply_factor = 1
                            units = 'USD per tonne'
                        # Aggregate and process stage-wise data
                        for st in stages:
                            s_df = filtered_df[filtered_df["processing_type"] == st]
                            if not s_df.empty:
                                s_df[c] = multiply_factor * s_df[c]
                                s_df = s_df.groupby(["iso3", "processing_type"])[c].sum().reset_index()
                                s_df.rename(columns={c: f"{st}"}, inplace=True)
                                m_df = pd.merge(m_df, s_df[["iso3", f"{st}"]], how="left", on="iso3").fillna(0)
                            else:
                                m_df[f"{st}"] = 0
                    # shifting one indent up from here

                        dfs.append(m_df.set_index("iso3"))

                        # Compute the differences for 2030 and 2040
                        delta_df = []
                        if len(dfs) >= 4:  # Ensure we have all the required DataFrames
                            delta_df.append(dfs[1].subtract(dfs[0], fill_value=0))  # 2030 Regional - National
                            # print('National')
                            # print(dfs[0])
                            delta_df.append(dfs[3].subtract(dfs[2], fill_value=0))  # 2040 Regional - National

                        # Plot results for 2030 and 2040
                        fig, axs = plt.subplots(1, 2, figsize=(19, 9), dpi=500, sharey=True)

                        if not delta_df:
                            print(f"No data differences to plot for {rf}. Skipping...")
                            continue
                        delta_df_2030 = pd.DataFrame(delta_df[0]) if delta_df else pd.DataFrame()
                        delta_df_2040 = pd.DataFrame(delta_df[1]) if delta_df else pd.DataFrame()

                        if delta_df_2030 is None or delta_df_2040 is None:
                            print(f"Missing data for 2030 or 2040 for {rf}. Skipping...")
                            continue
                        
                        if not delta_df_2030.empty:
                            axs[0] = plot_clustered_stacked(
                                fig, axs[0], delta_df_2030, stage_colors,
                                labels=["2030 difference"],
                                ylabel=f"Difference in annual {et} costs ({units})",
                                title=f"{rf.title()} 2030 - Regional minus National (Unconstrained)",
                                df_style = 'single_yr',
                                scenario_labels = False,
                            )

                        if not delta_df_2040.empty:
                            axs[1] = plot_clustered_stacked(
                                fig, axs[1], delta_df_2040, stage_colors,
                                labels=["2040 difference"],
                                ylabel=f"Difference in annual {et} costs ({units})",
                                title=f"{rf.title()} 2040 - Regional minus National (Unconstrained)",
                                df_style = 'single_yr',
                                scenario_labels = False
                            )

                        # Configure gridlines
                        for ax in axs:
                            ax.grid(False, axis="x")
                            ax.grid(which="major", axis="y", linestyle="-", zorder=0)
                            ax.axhline(y=0, color="black", linewidth=1, zorder=3)

                        plt.tight_layout()
                        save_fig(os.path.join(figures, f"{rf}_MN_MR_{et}_comparisons_2030_2040.png"))
                        plt.close()

    """
    Country and regional uncontrained comparisons for 2030 and 2040 separately in single plots:
    annual energy and transport costs,
    with dictionary-based stage & color selection.
    """

    make_plot = True # the amounts in some years are not correct, check calculation and vis
    if make_plot:
        results_file = os.path.join(output_data_path, "result_summaries", "energy_transport_totals_by_stage.xlsx")

        constraints = ["country_unconstrained", "region_unconstrained"]
        reference_minerals = ["copper","cobalt","manganese","lithium","graphite","nickel"]

        allowed_mineral_processing = {
                    "nickel": {
                        "processing_stage": [3, 5], # sulphate
                        "processing_type": ["Early refining", "Precursor related product"]
                    },
                    "copper": {
                        "processing_stage": [3, 5], # sulphate
                        "processing_type": ["Early refining", "Precursor related product"]
                    },
                    "cobalt": {
                        "processing_stage": [4.1, 5], # sulphate
                        "processing_type": ["Early refining", "Precursor related product"]
                    }
                    ,
                    "graphite": {
                        "processing_stage": [3, 4], # 
                        "processing_type": ["Early refining", "Precursor related product"]
                    }
                    ,
                    "manganese": {
                        "processing_stage": [3.1, 4.1], # 
                        "processing_type": ["Early refining", "Precursor related product"]
                    }
                    ,
                    "lithium": {
                        "processing_stage": [3, 4.2], # 
                        "processing_type": ["Early refining", "Precursor related product"] #
                    }
                }
        
        # Cost columns to plot
        cost_columns = [
            "export_transport_cost_usd_per_tonne",
            "import_transport_cost_usd_per_tonne",# import aren't as useful but plotting just in case
            "energy_opex"
        ]
        en_tr = ["transport", "transport", "energy"]
        short_col_name = [
                    "export transport cost",
                    "import transport cost",# import aren't as useful but plotting just in case
                    "energy OPEX"
                        ]

        scenarios = [
            "2030_mid_min_threshold_metal_tons",
            "2030_mid_max_threshold_metal_tons",
            "2040_mid_min_threshold_metal_tons",
            "2040_mid_max_threshold_metal_tons"
        ]

        extended_scenarios = [
            "2030_high_min_threshold_metal_tons", "2030_low_min_threshold_metal_tons", 
            "2040_high_min_threshold_metal_tons", "2040_low_min_threshold_metal_tons", 
            "2030_high_max_threshold_metal_tons", "2030_low_max_threshold_metal_tons", 
            "2040_high_max_threshold_metal_tons", "2040_low_max_threshold_metal_tons", 
        ]

        # 1) Define scenario→stage mapping
        scenario_stage_map = {
            "2030_mid_min_threshold_metal_tons": "Early refining",
            "2030_mid_max_threshold_metal_tons": "Early refining",
            "2040_mid_min_threshold_metal_tons": "Precursor related product",
            "2040_mid_max_threshold_metal_tons": "Precursor related product"
        }

        # If you have a function that returns stage labels/colors:
        all_properties = mineral_properties()

        for rf in reference_minerals:
            stages_full = all_properties[rf]["stage_labels"]
            stage_colors_full = all_properties[rf]["stage_label_colors"]

            # Build a dictionary: stage→color (for this mineral)
            stage_color_map = dict(zip(stages_full, stage_colors_full))

            for c, et, sh_c in zip(cost_columns, en_tr, short_col_name):
                print(c)
                if c == 'energy_opex': # all those that are not unit costs
                            multiply_factor = 1.0e-6 
                            units = 'million USD per tonne'
                if c != 'energy_opex': # all those that are unit costs
                            multiply_factor = 1
                            units = 'USD per tonne'
                
                for cs in constraints:
                    print(cs)
                    # We'll accumulate data_2030 & data_2040
                    data_2030 = []
                    data_2040 = []

                    min_2030 = []
                    max_2030 = []
                    min_2040 = []
                    max_2040 = []
                    # Also keep track of colors for each scenario DataFrame
                    bar_colors_2030 = []
                    bar_colors_2040 = []

                    for scenario in scenarios:
                        # Determine year from scenario name
                        scenario_year = 2030 if "2030" in scenario else 2040
                    # else:
                    #     print(f"Skipping scenario '{scenario}' - cannot determine year.")
                    #     continue
                        
                        cs_l = "country" if "country" in cs else "region"
                        
                        df = pd.read_excel(results_file, sheet_name=cs, index_col=[0,1,2,3,4]).reset_index()

                        filtered_df = df[
                            (df["scenario"] == scenario)
                            & (df["reference_mineral"] == rf)
                        ]

                        if rf in allowed_mineral_processing:
                            allowed_stages = allowed_mineral_processing[rf].get("processing_stage", [])
                            allowed_types = allowed_mineral_processing[rf].get("processing_type", [])
                            # Keep only rows that match both stage & type
                            filtered_df = filtered_df[
                                filtered_df["processing_stage"].isin(allowed_stages)
                                & filtered_df["processing_type"].isin(allowed_types)
                            ]
                        
                        if filtered_df.empty:
                            print(f"No data after dictionary filter for {rf}, scenario '{scenario}', constraint '{cs}'")
                            continue

                        # group by iso3 and sum up cost for whichever stage remains
                        m_df = pd.DataFrame(sorted(filtered_df["iso3"].unique()), columns=["iso3"])
                        s_df = filtered_df.copy()
                        s_df[c] = multiply_factor * s_df[c]

                        # group by iso3 and processing_type => pivot
                        grouped = s_df.groupby(["iso3", "processing_type"])[c].sum().reset_index()
                        pivoted = grouped.pivot(index="iso3", columns="processing_type", values=c).fillna(0)
                        pivoted.reset_index(inplace=True)

                        # Merge into m_df
                        m_df = pd.merge(m_df, pivoted, on="iso3", how="left").fillna(0)
                        m_df.set_index("iso3", inplace=True)
                        
                        if "2030" in scenario:
                            m_df = m_df[['Early refining']]
                            color_for_this_df = stage_color_map['Early refining']
                            data_2030.append(m_df)
                            bar_colors_2030.append(color_for_this_df)

                        if "2040" in scenario:
                            m_df = m_df[['Precursor related product']]
                            color_for_this_df = stage_color_map['Precursor related product']
                            data_2040.append(m_df)
                            bar_colors_2040.append(color_for_this_df)

                    # Calculate Error Bars
                    for scenario in extended_scenarios:
                        scenario_year = 2030 if "2030" in scenario else 2040

                        # Re-load df to ensure it is defined for this loop
                        df_ext = pd.read_excel(results_file, sheet_name=cs, index_col=[0,1,2,3,4]).reset_index()

                        df_ext = df_ext[(df_ext["scenario"] == scenario) & (df_ext["reference_mineral"] == rf)]

                        if not df_ext.empty:
                            s_df_ext = df_ext.copy()
                            s_df_ext[c] = multiply_factor * s_df_ext[c]
                            grouped_ext = s_df_ext.groupby(["iso3", "processing_type"])[c].sum().reset_index()
                            pivoted_ext = grouped_ext.pivot(index="iso3", columns="processing_type", values=c).fillna(0)

                            # Store Min & Max Values Instead of Standard Deviation

                            if scenario_year == 2030 and "low" in scenario and "Early refining" in pivoted_ext.columns:
                                min_2030.append(pivoted_ext[['Early refining']])
                            if scenario_year == 2030 and "high" in scenario and "Early refining" in pivoted_ext.columns:
                                max_2030.append(pivoted_ext[['Early refining']])

                            if scenario_year == 2040 and "low" in scenario and "Precursor related product" in pivoted_ext.columns:
                                min_2040.append(pivoted_ext[['Precursor related product']])
                            if scenario_year == 2040 and "high" in scenario and "Precursor related product" in pivoted_ext.columns:
                                max_2040.append(pivoted_ext[['Precursor related product']])
                           
                    fig, axs = plt.subplots(1, 2, figsize=(19, 9), dpi=500, sharey=True)

                    # LEFT: 2030
                    if data_2030:
                        plot_data_2030 = pd.concat(data_2030, axis=1)
                        print("2030")
                        print(plot_data_2030)

                        # Calculate min and max range
                        if min_2030:
                            min_2030 = pd.concat(min_2030, axis=1).reindex(plot_data_2030.index)
                        if max_2030:
                            max_2030 = pd.concat(max_2030, axis=1).reindex(plot_data_2030.index)

                        # Compute error bars (distance from mean to min/max)
                        # lower_error_2030 = (plot_data_2030 - min_2030).fillna(0).squeeze()
                        # upper_error_2030 = (max_2030 - plot_data_2030).fillna(0).squeeze()

                        lower_error_2030 = (min_2030).fillna(0).squeeze()
                        upper_error_2030 = (max_2030).fillna(0).squeeze()

                        # yerr_2030 = np.vstack([lower_error_2030.to_numpy().flatten(), 
                        #     upper_error_2030.to_numpy().flatten()])
                        
                        # Convert errors to 1D NumPy arrays using .ravel() (or .flatten())
                        lower_err_arr = lower_error_2030.to_numpy().ravel()
                        upper_err_arr = upper_error_2030.to_numpy().ravel()

                        # Create yerr as a tuple of 1D arrays; plt.errorbar expects yerr to be either a 1D array
                        # (if symmetric) or a tuple (lower, upper) of 1D arrays.
                        yerr_2030 = (lower_err_arr, upper_err_arr)

                        print("err 2030")
                        print(yerr_2030)

                        # Get y-values as a 1D array (squeeze to remove extra dimensions)
                        yvals_2030 = pd.Series(0, index=plot_data_2030.index).to_numpy().ravel()
                        # yvals_2030 = plot_data_2030.squeeze().to_numpy().ravel()

                        axs[0] = plot_clustered_stacked(
                                        fig, axs[0],
                                        plot_data_2030,
                                        bar_colors_2030,  
                                        labels=[f"2030_{cs_l}"],
                                        ylabel=f"Annual {sh_c} ({units})",
                                        title=f"{rf.title()} 2030 - Unconstrained ({cs_l.capitalize()})", # \n  {sh_c} 
                                        df_style='single_yr',
                                        stacked = False,
                                        shift_bars = False,
                                        scenario_labels = False
                                    )
                        if min_2030 is not None and max_2030 is not None:
                            axs[0].errorbar(plot_data_2030.index, yvals_2030,
                                            yerr=yerr_2030, fmt='none', ecolor='black', capsize=5)

                    else:
                        axs[0].set_title(f"{rf.title()} 2030 - Unconstrained No Data ({cs_l.capitalize()})")
                    
                    # RIGHT: 2040
                    if data_2040:
                        plot_data_2040 = pd.concat(data_2040, axis=1)
                        print("2040")
                        print(plot_data_2040)

                        # Calculate min and max range
                        if min_2040:
                            min_2040 = pd.concat(min_2040, axis=1).reindex(plot_data_2040.index)
                        if max_2040:
                            max_2040 = pd.concat(max_2040, axis=1).reindex(plot_data_2040.index)

                        lower_error_2040 = (min_2040).fillna(0).squeeze()
                        upper_error_2040 = (max_2040).fillna(0).squeeze()

                        # Convert errors to 1D NumPy arrays using .ravel() (or .flatten())
                        lower_err_arr = lower_error_2040.to_numpy().ravel()
                        upper_err_arr = upper_error_2040.to_numpy().ravel()

                        yerr_2040 = (lower_err_arr, upper_err_arr)

                        print("err 2040")
                        print(yerr_2040)

                        # yvals_2040 = plot_data_2040.squeeze().to_numpy().ravel()
                        yvals_2040 = pd.Series(0, index=plot_data_2040.index).to_numpy().ravel()

                        # Calculate min and max range
                        # min_2040 = pd.concat(min_2040, axis=1) if min_2040 else None
                        # max_2040 = pd.concat(max_2040, axis=1) if max_2040 else None

                        # # Compute error bars (distance from mean to min/max)
                        # lower_error_2040 = plot_data_2040 - min_2040
                        # upper_error_2040 = max_2040 - plot_data_2040
                        # error_range_2040 = [lower_error_2040, upper_error_2040]

                        axs[1] = plot_clustered_stacked(
                            fig, axs[1],
                            plot_data_2040,
                            bar_colors_2040,
                            labels=[f"2040_{cs_l}"],
                            ylabel=f"Annual {sh_c} ({units})",
                            title=f"{rf.title()} 2040 - Unconstrained ({cs_l.capitalize()})",
                            df_style='single_yr',
                            stacked = False,
                            shift_bars = False,
                            scenario_labels = False
                        )
                        if min_2040 is not None and max_2040 is not None:
                            axs[1].errorbar(plot_data_2040.index, yvals_2040,
                                            yerr=yerr_2040, fmt='none', ecolor='black', capsize=5)
                        
                    else:
                        axs[1].set_title(f"{rf.title()} 2040 - Unconstrained No Data ({cs_l.capitalize()})") #, fontsize=14

                    # Configure gridlines
                    for ax in axs:
                        ax.grid(which="major", axis="y", linestyle="-", zorder=0)
                        ax.grid(False, axis="x")
                        ax.axhline(y=0, color="black", linewidth=1, zorder=3)

                    plt.tight_layout()
                    # Save figure
                    filename = f"{rf}_{cs}_{c}_comparisons_2030_2040_errorbars.png"
                    save_fig(os.path.join(figures, filename))
                    plt.close()



if __name__ == '__main__':
    CONFIG = load_config()
    main(CONFIG)
