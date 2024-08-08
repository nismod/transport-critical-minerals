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

    figures = os.path.join(figure_path)
    reference_minerals = ["copper","cobalt","manganese","lithium","graphite","nickel"]
    reference_minerals_columns = [f"{rf}_initial_stage_production_tons_0.0_in_country" for rf in reference_minerals]
    'copper_initial_stage_production_tons_0.0_in_country'
    # reference_mineral_colors = [
    #                             "#662506","#023858","#4d004b",
    #                             "#004529","#000000","#67000d",
    #                             "#dd3497","#ae017e","#7a0177","#49006a",
    #                             "#3690c0","#02818a","#016c59","#014636"
    #                             ]
    reference_mineral_colors = [
                                "#f46d43","#fdae61","#fee08b","#c2a5cf","#66c2a5","#3288bd",
                                "#67000d","#dd3497","#225ea8","#238443","#7a0177","#49006a",
                                "#3690c0","#02818a","#016c59","#014636"
                                ]
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
                                    point_steps=12,
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
            save_fig(os.path.join(figures,f"mine_maps_{sc}.png"))
            plt.close()




            

    



if __name__ == '__main__':
    CONFIG = load_config()
    main(CONFIG)
