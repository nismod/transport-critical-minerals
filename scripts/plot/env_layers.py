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

    figures = os.path.join(figure_path,"biodiversity_and_water_stress")
    if os.path.exists(figures) == False:
        os.mkdir(figures)
    
    global_epsg=4326
    layer_details = [
                        {
                            "layer_type":"Key Biodiversity Areas",
                            "layer_file":"Environmental datasets/KeyBiodiversityAreas/Africa_KBA/Africa_KBA.shp",
                            "layer_name":None,
                            "layer_column":"intname",
                            "layer_colors":["#00441b"]
                        },
                        {
                            "layer_type":"Last Of Wild",
                            "layer_file":"Environmental datasets/LastOfWildSouthernAfrica/low_southern_africa.shp",
                            "layer_name":None,
                            "layer_column":"biome",
                            "layer_colors":["#238b45"]
                        },
                        {
                            "layer_type":"Protected Areas",
                            "layer_file":"Environmental datasets/ProtectedAreasSouthernAfrica/protected_areas_southern_africa.shp",
                            "layer_name":None,
                            "layer_column":"DESIG_ENG",
                            "layer_colors":["#02818a"]
                        },
                        {
                            "layer_type":"Waterstress",
                            "layer_file":"water_stress/water_stress_data.gpkg",
                            "layer_name":"future_annual",
                            "layer_column":"bau30_ws_x_l",
                            "layer_colors":["#fdbb84","#7f0000","#d7301f"]
                        },
            ]
    ax_proj = get_projection(epsg=global_epsg)
    fig, ax_plots = plt.subplots(1,1,
                    subplot_kw={'projection': ax_proj},
                    figsize=(10,14),
                    dpi=500)
    ax = plot_ccg_basemap(ax_plots)
    legend_handles = []
    titles = [
                "$\\bf{Environmental \, areas}$",
                "$\\bf{Water \, stress \, areas}$"
            ]
    for lyr in layer_details:
        layer_gdf = gpd.read_file(
                        os.path.join(
                            processed_data_path,lyr["layer_file"]),
                        layer=lyr["layer_name"])
        layer_gdf = layer_gdf.to_crs(epsg=global_epsg)
        if lyr["layer_type"] == "Waterstress":
            layer_gdf = layer_gdf[layer_gdf["bau30_ws_x_c"].isin([-1,3,4])]
            layer_gdf.rename(columns={lyr["layer_column"]:"layer_classes"},inplace=True)
            legend_handles.append(plt.plot([],[],color="none",label=titles[1])[0])
        else:
            layer_gdf["layer_classes"] = lyr["layer_type"]
            if lyr["layer_type"] == "Key Biodiversity Areas":
                legend_handles.append(plt.plot([],[],color="none",label=titles[0])[0])

        layer_classes = sorted(list(set(layer_gdf["layer_classes"].values.tolist())))
        layer_colors = lyr["layer_colors"]
        for idx,(lc,nc) in enumerate(zip(layer_classes,layer_colors)):
            df = layer_gdf[layer_gdf["layer_classes"] == lc]
            df['geometry'].plot(ax=ax, color=nc, alpha=0.5,label=lc,zorder=10+idx)
            legend_handles.append(mpatches.Patch(color=nc,
                                                label=lc,alpha=0.5))

    if len(legend_handles) > 0:
        leg = ax.legend(
            handles=legend_handles, 
            fontsize=18,
            loc='lower right',
            frameon=False)
        # Move titles to the left 
        for item, label in zip(leg.legend_handles, leg.texts):
            if label._text in titles:
                width = item.get_window_extent(fig.canvas.get_renderer()).width
                label.set_ha('left')
                label.set_position((-10.0*width,0))
        
            
    plt.tight_layout()
    save_fig(os.path.join(figures,"biodiversity_and_water_stress_layers.png"))
    plt.close()
    



if __name__ == '__main__':
    CONFIG = load_config()
    main(CONFIG)
