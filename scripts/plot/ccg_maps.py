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
    data_path = config['paths']['data']
    output_path = config['paths']['results']
    figure_path = config['paths']['figures']

    figures = os.path.join(figure_path)
    ccg_countries = pd.read_csv(os.path.join(data_path,"admin_boundaries","ccg_country_codes.csv"))
    ccg_isos = ccg_countries[ccg_countries["ccg_country"] == 1]["iso_3digit_alpha"].values.tolist()
    year_percentile_combinations = [
                                    (2022,"baseline"),
                                    (2030,"low"),
                                    (2030,"mid"),
                                    (2030,"high"),
                                    (2040,"low"),
                                    (2040,"mid"),
                                    (2040,"high")
                                    ]
    tonnage_thresholds = ["min_threshold_metal_tons","max_threshold_metal_tons"]
    location_cases = ["country","region"]
    optimisation_type = ["unconstrained","constrained"]
    reference_minerals = ["copper","cobalt","manganese","lithium","graphite","nickel"]
    reference_mineral_colors = ["#cc4c02","#3690c0","#88419d","#d7301f","#252525","#737373"]

    ax_proj = get_projection(epsg=4326)
    fig, ax_plots = plt.subplots(1,1,
                    subplot_kw={'projection': ax_proj},
                    figsize=(12,12),
                    dpi=500)
    ax = plot_ccg_basemap(ax_plots)
    plt.tight_layout()
    save_fig(os.path.join(figures,"ccg_basemap.png"))
    plt.close()

if __name__ == '__main__':
    CONFIG = load_config()
    main(CONFIG)
