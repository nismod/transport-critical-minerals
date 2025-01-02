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
from tqdm import tqdm
tqdm.pandas()

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
    reference_mineral_colors = [
                                "#f46d43","#fdae61","#fee08b","#c2a5cf","#66c2a5","#3288bd"
                                ]
    layers = ["2022_baseline","2030_mid_min_threshold_metal_tons","2040_mid_min_threshold_metal_tons"]
    layers_names = [2022,2030,2040]
    for idx, (lyr,lyr_nm) in enumerate(zip(layers,layers_names)):
        if lyr_nm == 2022:
            scenarios = ["country_unconstrained"]
        else:
            scenarios = ["country_unconstrained","country_constrained"]
        sc_dfs = []
        tmax = []
        for sc in scenarios:
            mine_sites_df = gpd.read_file(
                                        os.path.join(
                                            output_path,
                                            "optimised_processing_locations",
                                            f"node_locations_for_energy_conversion_{sc}.gpkg"),
                                        layer=lyr)
            mine_sites_df = mine_sites_df[mine_sites_df["mode"] == "mine"]
            dfs = []
            for rf in reference_minerals:
                df = mine_sites_df[mine_sites_df[f"{rf}_initial_stage_production_tons_0.0_in_country"] > 0]
                df = mine_sites_df[[f"{rf}_initial_stage_production_tons_0.0_in_country","geometry"]]
                df["reference_mineral"] = rf
                df.rename(
                        columns={
                                f"{rf}_initial_stage_production_tons_0.0_in_country":"total_tons"
                                },
                        inplace=True)
                dfs.append(df)
                tmax += df["total_tons"].values.tolist()
            dfs = pd.concat(dfs,axis=0,inplace=True)
            sc_dfs.append(dfs)
        
        tmax = max(tmax)

        for df in sc_dfs:
            


        dfs = []
        weights = []
        minerals_classes = []
        for lyr in lyrs:
            mine_sites_df = gpd.read_file(os.path.join(output_path,
                                        "optimised_processing_locations",
                                        f"node_locations_for_energy_conversion_{sc}.gpkg"),layer=lyr)
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
        fig, ax_plots = plt.subplots(1,len(lyrs),
                        subplot_kw={'projection': ax_proj},
                        figsize=(24,12),
                        dpi=500)
        ax_plots = ax_plots.flatten()
        for idx, (df,nm) in enumerate(zip(dfs,lyr_nm)):
            ax = plot_ccg_basemap(ax_plots[idx])
            legend_handles = []
            titles = ["$\\bf{Mine \, annual \, output \,(tonnes)}$","$\\bf{Minerals \, produced}$"]
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
                                interpolation = 'fisher-jenks',
                                legend_label="Annual output (tonnes)",
                                legend_size=16,
                                legend_weight=1.5,
                                no_value_label="No output",
                                no_value_color="#ffffff"
                                )
            if idx == len(dfs) - 2:
                # legend_handles.append(plt.plot([],[],
                #                                 color="none",
                #                                 label="$\\bf{Mine \, annual \, output \,(tonnes)}$")[0])
                legend_handles += legend
                ncols = 1
                leg_size = 13
                legend_title = titles[0]
            elif idx == len(dfs) - 1:
                # legend_handles.append(plt.plot([],[],
                #                                 color="none",
                #                                 label="$\\bf{Minerals \, produced}$")[0])
                for jdx,(l,nc) in enumerate(zip(minerals,reference_mineral_colors[:len(minerals)])):
                    legend_handles.append(mpatches.Patch(color=nc,
                                                label=l))
                ncols = 1
                leg_size = 14
                legend_title = titles[1]
            # if sc == "country_unconstrained":
            #     leg_size = 15
            # else:
            #     leg_size = 15
            if len(legend_handles) > 0:
                leg = ax.legend(
                    handles=legend_handles, 
                    fontsize=leg_size,
                    title=legend_title,
                    title_fontsize=14,
                    ncol = ncols, 
                    loc='lower right',
                    frameon=False)
                # Move titles to the left 
                for item, label in zip(leg.legend_handles, leg.texts):
                    if label._text in titles:
                        width = item.get_window_extent(fig.canvas.get_renderer()).width
                        label.set_ha('left')
                        label.set_position((-10.0*width,0))

            total_tons = 1.0e-6*df["total_tons"].sum()
            ax.text(
                    0.05,
                    0.20,
                    f"Total tonnes = {total_tons:,.2f} million",
                    horizontalalignment='left',
                    transform=ax.transAxes,
                    size=18,
                    weight='bold'
                    )

            ax.set_title(
                nm, 
                fontsize=24,fontweight="bold")

        plt.tight_layout()
        save_fig(os.path.join(figures,f"mine_maps_{sc}.png"))
        plt.close()




            

    



if __name__ == '__main__':
    CONFIG = load_config()
    main(CONFIG)
