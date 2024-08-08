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

    country_codes = ["ZMB"]
    boundary_codes = ["AGO","COD","TZA","MWI","MOZ","ZWE","NAM","BWA"]
    figures = os.path.join(figure_path,f"{'_'.join(country_codes)}_figures")
    if os.path.exists(figures) is False:
        os.mkdir(figures)

    reference_minerals = ["copper","cobalt","manganese","lithium","graphite","nickel"]
    reference_minerals_columns = [f"{rf}_initial_stage_production_tons_0.0_in_country" for rf in reference_minerals]
    reference_mineral_colors = [
                                "#f46d43","#fdae61","#fee08b","#c2a5cf","#66c2a5","#3288bd",
                                "#67000d","#dd3497","#225ea8","#238443","#7a0177","#49006a",
                                "#3690c0","#02818a","#016c59","#014636"
                                ]
    all_properties = mineral_properties()
    plot_mine_sites = True
    if plot_mine_sites is True:
        scenarios = ["country_unconstrained","country_constrained"]
        layers = ["2022_baseline","2030_mid_max_threshold_metal_tons","2040_mid_max_threshold_metal_tons"]
        layers_names = ["2022 SQ","2030 MN (or MR)","2040 MN (or MR)"]
        # scenarios = ["country_unconstrained"]
        # layers = ["2022_baseline"]
        for sc in scenarios:
            if sc == "country_constrained":
                lyrs = layers[1:]
                lyr_nm = [f"{l} - Constrained" for l in layers_names[1:]]
            else:
                lyrs = layers
                lyr_nm = [f"{l} - Unconstrained" for l in layers_names]
            dfs = []
            weights = []
            minerals_classes = []
            for lyr in lyrs:
                mine_sites_df = gpd.read_file(os.path.join(output_path,
                                            "optimised_processing_locations",
                                            f"node_locations_for_energy_conversion_{sc}.gpkg"),layer=lyr)
                mine_sites_df = mine_sites_df[mine_sites_df["iso3"].isin(country_codes)]
                mine_sites_df = mine_sites_df[mine_sites_df["mode"] == "mine"]
                for idx,(rf,rc) in enumerate(zip(reference_minerals,reference_minerals_columns)):
                    mine_sites_df[rf] = mine_sites_df.progress_apply(lambda x:rf.title() if x[rc] > 0 else np.nan,axis=1)

                mine_sites_df["total_tons"] = mine_sites_df[reference_minerals_columns].sum(axis=1)
                mine_sites_df["mineral_types"] = mine_sites_df[reference_minerals].apply(lambda x: '+'.join(x.dropna()), axis=1)
                
                dfs.append(mine_sites_df)
                weights += mine_sites_df["total_tons"].values.tolist()

                minerals_classes += mine_sites_df["mineral_types"].values.tolist()
            
            minerals_classes = list(set(minerals_classes))
            rt = [rf.title() for rf in reference_minerals]
            minerals = rt + [m for m in minerals_classes if m not in rt]

            ax_proj = get_projection(epsg=4326)
            if len(lyrs) == 3:
                figsize = (24,8)
            else:
                figsize = (24,12)
            fig, ax_plots = plt.subplots(1,len(lyrs),
                            subplot_kw={'projection': ax_proj},
                            figsize=figsize,
                            dpi=500)
            ax_plots = ax_plots.flatten()
            for idx, (df,nm) in enumerate(zip(dfs,lyr_nm)):
                ax = plot_ccg_country_basemap(ax_plots[idx],country_codes,boundary_codes)
                legend_handles = []
                titles = ["$\\bf{Mine \, annual \, output \,(tonnes)}$","$\\bf{Minerals \, produced}$"]
                legend_handles.append(plt.plot([],[],
                                                color="none",
                                                label="$\\bf{Mine \, annual \, output \,(tonnes)}$")[0])
                ax,legend = point_map_plotting_colors_width(ax,df,
                                    "total_tons",
                                    weights,
                                    point_classify_column="mineral_types",
                                    point_categories=minerals,
                                    point_colors=reference_mineral_colors,
                                    point_labels=minerals,
                                    point_zorder=[6+n for n in range(len(minerals))],
                                    point_steps=10,
                                    width_step = 40.0,
                                    interpolation = 'linear',
                                    legend_label="Annual output (tonnes)",
                                    legend_size=16,
                                    legend_weight=1.5,
                                    no_value_label="No output",
                                    no_value_color="#ffffff"
                                    )
                legend_handles += legend
                legend_handles.append(plt.plot([],[],
                                                color="none",
                                                label="$\\bf{Minerals \, produced}$")[0])
                for jdx,(l,nc) in enumerate(zip(minerals,reference_mineral_colors[:len(minerals)])):
                    legend_handles.append(mpatches.Patch(color=nc,
                                                label=l))
                if sc == "country_unconstrained":
                    leg_size = 11
                else:
                    leg_size = 13
                leg = ax.legend(
                    handles=legend_handles, 
                    fontsize=leg_size,
                    ncol = 2, 
                    loc='upper left',
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
            save_fig(os.path.join(figures,f"mine_maps_{sc}.png"))
            plt.close()

    plot_processing_sites = True
    if plot_processing_sites is True:
        scenarios_descriptions = [
                                    {
                                        "scenario":"country_unconstrained",
                                        "scenario_name":"country",
                                        "layers":[
                                                    "2030_mid_min_threshold_metal_tons",
                                                    "2040_mid_min_threshold_metal_tons"],
                                        "layers_names":["2030 MN - Unconstrained","2040 MN - Unconstrained"]
                                    },
                                    {
                                        "scenario":"region_unconstrained",
                                        "scenario_name":"region",
                                        "layers":[
                                                    "2030_mid_max_threshold_metal_tons",
                                                    "2040_mid_max_threshold_metal_tons"],
                                        "layers_names":["2030 MR - Unconstrained","2040 MR - Unconstrained"]
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
                mine_sites_df = mine_sites_df[mine_sites_df["iso3"].isin(country_codes)]
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
                            figsize=(24,12),
                            dpi=500)
            ax_plots = ax_plots.flatten()
            for idx, (df,nm) in enumerate(zip(dfs,lyr_nm)):
                ax = plot_ccg_country_basemap(ax_plots[idx],country_codes,boundary_codes)
                legend_handles = []
                # titles = [
                #             "$\\bf{Minerals \, produced}$",
                #             "$\\bf{Processing \, annual \, output \,(tonnes)}$"
                #         ]

                titles = [
                            "$\\bf{Processing \, annual \, output \,(tonnes)}$",
                            "$\\bf{Minerals \, produced}$"
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
                    leg_size = 11
                    pts = 9
                else:
                    leg_size = 12
                    pts = 9

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
                                    interpolation = 'linear',
                                    legend_label="Annual output (tonnes)",
                                    legend_size=16,
                                    legend_weight=1.5,
                                    no_value_label="No output",
                                    no_value_color="#ffffff"
                                    )
                legend_handles += legend
                legend_handles.append(plt.plot([],[],
                                                color="none",
                                                label="$\\bf{Minerals \, produced}$")[0])
                for jdx,(l,nc) in enumerate(zip(minerals,reference_mineral_colors[:len(minerals)])):
                    legend_handles.append(mpatches.Patch(color=nc,
                                                label=l))
                leg = ax.legend(
                    handles=legend_handles, 
                    fontsize=leg_size,
                    ncol = 2, 
                    loc='upper left',
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

if __name__ == '__main__':
    CONFIG = load_config()
    main(CONFIG)
