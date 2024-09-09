"""Road network risks and adaptation maps
"""
import os
import sys
from collections import OrderedDict
import pandas as pd
import geopandas as gpd
import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from map_plotting_utils import *
from mapping_properties import *
from tqdm import tqdm
tqdm.pandas()

def get_stages(x,mineral_columns):
    string = []
    for mc in mineral_columns:
        if x[mc] > 0:
            r = mc.split("_")[0]
            # s = mc.split("_final_stage_production_tons_")[-1]
            # s = s.split("_")[0]
            # string.append(f"{r.title()} Stage {s}")
            string.append(r.title())

    return "+".join(sorted(list(set(string)))) 



def main(config):
    processed_data_path = config['paths']['data']
    output_path = config['paths']['results']
    figure_path = config['paths']['figures']

    figures = os.path.join(figure_path,"regional_figures")
    if os.path.exists(figures) is False:
        os.mkdir(figures)

    figures = os.path.join(figure_path,"regional_figures","mine_and_processing_locations")
    if os.path.exists(figures) is False:
        os.mkdir(figures)
    reference_minerals = ["copper","cobalt","manganese","lithium","graphite","nickel"]
    # reference_minerals_columns = [f"{rf}_initial_stage_production_tons_0.0_in_country" for rf in reference_minerals]
    # reference_mineral_colors = [
    #                             "#662506","#023858","#4d004b",
    #                             "#004529","#000000","#67000d",
    #                             "#dd3497","#ae017e","#7a0177","#49006a",
    #                             "#3690c0","#02818a","#016c59","#014636"
    #                             ]
    reference_mineral_colors = [
                                "#f46d43","#fdae61","#fee08b","#c2a5cf","#66c2a5","#3288bd",
                                "#67000d","#dd3497","#225ea8","#238443","#7a0177","#bf812d",
                                "#d4b9da","#000000","#016c59","#014636"
                                ]
    all_properties = mineral_properties()
    plot_mine_sites = True
    if plot_mine_sites is True:
        scenarios_descriptions = [
                                    {
                                        "scenario":"country_unconstrained",
                                        "scenario_name":"country",
                                        "layers":[
                                                    "2030_mid_min_threshold_metal_tons",
                                                    "2040_mid_min_threshold_metal_tons"],
                                        "layers_names":["2030 Mid National - Unconstrained",
                                                        "2040 Mid National - Unconstrained"]
                                    },
                                    {
                                        "scenario":"region_unconstrained",
                                        "scenario_name":"region",
                                        "layers":[
                                                    "2030_mid_max_threshold_metal_tons",
                                                    "2040_mid_max_threshold_metal_tons"],
                                        "layers_names":["2030 Mid Regional - Unconstrained",
                                                        "2040 Mid Regional - Unconstrained"]
                                    },
                                ]
        # scenarios = ["country_unconstrained"]
        # layers = ["2022_baseline"]
        for scenario in scenarios_descriptions:
            sc = scenario["scenario"]
            sn = scenario["scenario_name"]
            lyrs = scenario["layers"]
            lyr_nm = scenario["layers_names"]
            reference_minerals_columns = []
            for rf in reference_minerals:
                stages = all_properties[rf]["stages"][1:]
                reference_minerals_columns += [f"{rf}_final_stage_production_tons_{st}_in_{sn}" for st in stages]

            dfs = []
            weights = []
            minerals_classes = []
            for lyr in lyrs:
                mine_sites_df = gpd.read_file(os.path.join(output_path,
                                            "optimised_processing_locations",
                                            f"node_locations_for_energy_conversion_{sc}.gpkg"),
                                            layer=lyr)
                mine_sites_df = mine_sites_df[~mine_sites_df["mode"].isin(["mine","city"])]
                mine_columns = [m for m in mine_sites_df.columns.values.tolist() if m in reference_minerals_columns]
                mine_sites_df["total_tons"] = mine_sites_df[mine_columns].sum(axis=1)
                mine_sites_df["mineral_types"] = mine_sites_df.progress_apply(lambda x:get_stages(x,mine_columns),axis=1)
                mine_sites_df = mine_sites_df[mine_sites_df["total_tons"] > 0]
                dfs.append(mine_sites_df)
                weights += mine_sites_df["total_tons"].values.tolist()

                minerals_classes += mine_sites_df["mineral_types"].values.tolist()
            
            minerals_classes = sorted(list(set(minerals_classes)))
            rt = [rf.title() for rf in reference_minerals]
            minerals = rt + [m for m in minerals_classes if m not in rt]

            ax_proj = get_projection(epsg=4326)
            fig, ax_plots = plt.subplots(1,len(lyrs),
                            subplot_kw={'projection': ax_proj},
                            figsize=(21,12),
                            dpi=500)
            ax_plots = ax_plots.flatten()
            for idx, (df,nm) in enumerate(zip(dfs,lyr_nm)):
                ax = plot_ccg_basemap(ax_plots[idx])
                legend_handles = []
                # titles = [
                #             "$\\bf{Minerals \, produced}$",
                #             "$\\bf{Processing \, annual \, output \,(tonnes)}$"
                #         ]

                titles = [
                            "$\\bf{Processing \, annual \, output \,(tonnes)}$",
                            "$\\bf{Minerals \, processed}$"
                        ]


                # legend_handles.append(plt.plot([],[],
                #                                 color="none",
                #                                 label="$\\bf{Minerals \, produced}$")[0])
                # for jdx,(l,nc) in enumerate(zip(minerals,reference_mineral_colors[:len(minerals)])):
                #     legend_handles.append(mpatches.Patch(color=nc,
                #                                 label=l))

                legend_handles.append(plt.plot([],[],
                                                color="none",
                                                label="$\\bf{Processing \, annual \, output \,(tonnes)}$")[0])
                if sc == "country_unconstrained":
                    leg_size = 10.5
                    pts = 15
                else:
                    leg_size = 12
                    pts = 12

                ax,legend = point_map_plotting_colors_width(ax,df,
                                    "total_tons",
                                    weights,
                                    point_classify_column="mineral_types",
                                    point_categories=minerals,
                                    point_colors=reference_mineral_colors,
                                    point_labels=minerals,
                                    point_zorder=[6+n for n in range(len(minerals))],
                                    point_steps=pts,
                                    width_step = 40.0,
                                    interpolation = 'fisher-jenks',
                                    legend_label="Annual output (tonnes)",
                                    legend_size=16,
                                    legend_weight=1.5,
                                    no_value_label="No output",
                                    no_value_color="#ffffff"
                                    )
                legend_handles += legend
                legend_handles.append(plt.plot([],[],
                                                color="none",
                                                label="$\\bf{Minerals \, processed}$")[0])
                for jdx,(l,nc) in enumerate(zip(minerals,reference_mineral_colors[:len(minerals)])):
                    legend_handles.append(mpatches.Patch(color=nc,
                                                label=l))
                leg = ax.legend(
                    handles=legend_handles, 
                    fontsize=leg_size,
                    ncol = 2, 
                    loc='lower right',
                    frameon=False)
                # Move titles to the left 
                for item, label in zip(leg.legend_handles, leg.texts):
                    if label._text in titles:
                        width = item.get_window_extent(fig.canvas.get_renderer()).width
                        label.set_ha('left')
                        label.set_position((-10.0*width,0))

                ax.set_title(
                    nm, 
                    fontsize=14,fontweight="bold")

            plt.tight_layout()
            save_fig(os.path.join(figures,f"processing_locations_maps_{sc}.png"))
            plt.close()

    plot_mine_sites = True
    if plot_mine_sites is True:
        scenarios_descriptions = [
                                    {
                                        "type":"2030_unconstrained",
                                        "scenarios":[
                                                        "country_unconstrained",
                                                        "region_unconstrained"
                                                    ],
                                        "scenario_names":["country","region"],
                                        "layers":[
                                                    "2030_mid_min_threshold_metal_tons",
                                                    "2030_mid_max_threshold_metal_tons"],
                                        "layers_names":["2030 Mid National - Unconstrained",
                                                        "2030 Mid Regional - Unconstrained"]
                                    },
                                    {
                                        "type":"2030_constrained",
                                        "scenarios":[
                                                        "country_constrained",
                                                        "region_constrained"
                                                    ],
                                        "scenario_names":["country","region"],
                                        "layers":[
                                                    "2030_mid_min_threshold_metal_tons",
                                                    "2040_mid_max_threshold_metal_tons"],
                                        "layers_names":["2030 Mid National - Constrained",
                                                        "2030 Mid Regional - Constrained"]
                                    },
                                    {
                                        "type":"2040_unconstrained",
                                        "scenarios":[
                                                        "country_unconstrained",
                                                        "region_unconstrained"
                                                    ],
                                        "scenario_names":["country","region"],
                                        "layers":[
                                                    "2040_mid_min_threshold_metal_tons",
                                                    "2040_mid_max_threshold_metal_tons"],
                                        "layers_names":["2040 Mid National - Unconstrained",
                                                        "2040 Mid Regional - Unconstrained"]
                                    },
                                    {
                                        "type":"2040_constrained",
                                        "scenarios":[
                                                        "country_constrained",
                                                        "region_constrained"
                                                    ],
                                        "scenario_names":["country","region"],
                                        "layers":[
                                                    "2040_mid_min_threshold_metal_tons",
                                                    "2040_mid_max_threshold_metal_tons"],
                                        "layers_names":["2040 Mid National - Constrained",
                                                        "2040 Mid Regional - Constrained"]
                                    },
                                ]
        # scenarios = ["country_unconstrained"]
        # layers = ["2022_baseline"]
        for scenario in scenarios_descriptions:
            styp = scenario["type"]
            scenarios = scenario["scenarios"]
            scenario_names = scenario["scenario_names"]
            lyrs = scenario["layers"]
            lyr_nm = scenario["layers_names"]
            # reference_minerals_columns = []
            # for rf in reference_minerals:
            #     stages = all_properties[rf]["stages"][1:]
            #     reference_minerals_columns += [f"{rf}_final_stage_production_tons_{st}_in_{sn}" for st in stages]

            dfs = []
            weights = []
            minerals_classes = []
            for idx,(sc,sn,lyr) in enumerate(zip(scenarios,scenario_names,lyrs)):
                reference_minerals_columns = []
                for rf in reference_minerals:
                    stages = all_properties[rf]["stages"][1:]
                    reference_minerals_columns += [f"{rf}_final_stage_production_tons_{st}_in_{sn}" for st in stages]
                mine_sites_df = gpd.read_file(os.path.join(output_path,
                                            "optimised_processing_locations",
                                            f"node_locations_for_energy_conversion_{sc}.gpkg"),
                                            layer=lyr)
                mine_sites_df = mine_sites_df[~mine_sites_df["mode"].isin(["mine","city"])]
                mine_columns = [m for m in mine_sites_df.columns.values.tolist() if m in reference_minerals_columns]
                mine_sites_df["total_tons"] = mine_sites_df[mine_columns].sum(axis=1)
                mine_sites_df["mineral_types"] = mine_sites_df.progress_apply(lambda x:get_stages(x,mine_columns),axis=1)
                mine_sites_df = mine_sites_df[mine_sites_df["total_tons"] > 0]
                dfs.append(mine_sites_df)
                weights += mine_sites_df["total_tons"].values.tolist()

                minerals_classes += mine_sites_df["mineral_types"].values.tolist()
            
            minerals_classes = sorted(list(set(minerals_classes)))
            rt = [rf.title() for rf in reference_minerals]
            minerals = rt + [m for m in minerals_classes if m not in rt]

            ax_proj = get_projection(epsg=4326)
            fig, ax_plots = plt.subplots(1,len(lyrs),
                            subplot_kw={'projection': ax_proj},
                            figsize=(21,12),
                            dpi=500)
            ax_plots = ax_plots.flatten()
            for idx, (df,nm) in enumerate(zip(dfs,lyr_nm)):
                ax = plot_ccg_basemap(ax_plots[idx])
                legend_handles = []
                # titles = [
                #             "$\\bf{Minerals \, produced}$",
                #             "$\\bf{Processing \, annual \, output \,(tonnes)}$"
                #         ]

                titles = [
                            "$\\bf{Processing \, annual \, output \,(tonnes)}$",
                            "$\\bf{Minerals \, processed}$"
                        ]


                # legend_handles.append(plt.plot([],[],
                #                                 color="none",
                #                                 label="$\\bf{Minerals \, produced}$")[0])
                # for jdx,(l,nc) in enumerate(zip(minerals,reference_mineral_colors[:len(minerals)])):
                #     legend_handles.append(mpatches.Patch(color=nc,
                #                                 label=l))

                legend_handles.append(plt.plot([],[],
                                                color="none",
                                                label="$\\bf{Processing \, annual \, output \,(tonnes)}$")[0])
                if styp == "2030_unconstrained":
                    leg_size = 12
                    pts = 12
                elif styp == "2040_unconstrained":
                    leg_size = 10
                    pts = 16
                elif styp == "2040_constrained":
                    leg_size = 11
                    pts = 13
                else:
                    leg_size = 12
                    pts = 12

                ax,legend = point_map_plotting_colors_width(ax,df,
                                    "total_tons",
                                    weights,
                                    point_classify_column="mineral_types",
                                    point_categories=minerals,
                                    point_colors=reference_mineral_colors,
                                    point_labels=minerals,
                                    point_zorder=[6+n for n in range(len(minerals))],
                                    point_steps=pts,
                                    width_step = 40.0,
                                    interpolation = 'fisher-jenks',
                                    legend_label="Annual output (tonnes)",
                                    legend_size=16,
                                    legend_weight=1.5,
                                    no_value_label="No output",
                                    no_value_color="#ffffff"
                                    )
                legend_handles += legend
                legend_handles.append(plt.plot([],[],
                                                color="none",
                                                label="$\\bf{Minerals \, processed}$")[0])
                for jdx,(l,nc) in enumerate(zip(minerals,reference_mineral_colors[:len(minerals)])):
                    legend_handles.append(mpatches.Patch(color=nc,
                                                label=l))
                leg = ax.legend(
                    handles=legend_handles, 
                    fontsize=leg_size,
                    ncol = 2, 
                    loc='lower right',
                    frameon=False)
                # Move titles to the left 
                for item, label in zip(leg.legend_handles, leg.texts):
                    if label._text in titles:
                        width = item.get_window_extent(fig.canvas.get_renderer()).width
                        label.set_ha('left')
                        label.set_position((-10.0*width,0))

                ax.set_title(
                    nm, 
                    fontsize=14,fontweight="bold")

            plt.tight_layout()
            save_fig(os.path.join(figures,f"processing_locations_maps_{styp}.png"))
            plt.close()




            

    



if __name__ == '__main__':
    CONFIG = load_config()
    main(CONFIG)
