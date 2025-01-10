#!/usr/bin/env python
# coding: utf-8

import sys
import os
import ast
import pandas as pd
import numpy as np
pd.options.mode.copy_on_write = True
import geopandas as gpd
from shapely.geometry import LineString
from map_plotting_utils import *
from mapping_properties import *
from tqdm import tqdm
tqdm.pandas()

config = load_config()
processed_data_path = config['paths']['data']
figure_path = config['paths']['figures']

def add_geometries_to_flows(flows_dataframe,
                        merge_column="id",
                        modes=["rail","sea","road","IWW","mine","city"],
                        layer_type="edges",merge=True):    
    flow_edges = []
    for mode in modes:
        if mode == "rail":
            edges = gpd.read_file(os.path.join(
                            processed_data_path,
                            "infrastructure",
                            "africa_railways_network.gpkg"
                                ), layer=layer_type
                    )
            if mode == "nodes":
                edges["mode"] = edges[""]
            else:
                edges["mode"] = edges[""]
        elif mode == "road":
            if layer_type == "edges":
                edges = gpd.read_parquet(os.path.join(
                                processed_data_path,
                                "infrastructure",
                                "africa_roads_edges.geoparquet"))
                edges["mode"] = edges["tag_highway"]
            else:
                edges = gpd.read_parquet(os.path.join(
                                processed_data_path,
                                "infrastructure",
                                "africa_roads_nodes.geoparquet"))
                edges.rename(columns={"road_id":merge_column,"iso_a3":"iso3"},inplace=True)
                edges["infra"] = mode
        elif mode == "sea":
            edges = gpd.read_file(
                    os.path.join(processed_data_path,
                        "infrastructure",
                        "global_maritime_network.gpkg"
                    ),layer=layer_type)

            edges["mode"] = edges["infra"]
        if layer_type == "edges":
            edges = edges[[merge_column,"from_id","to_id","mode","geometry"]]
        else:
            edges = edges[[merge_column,"iso3","infra","mode","geometry"]]
        if merge is True:
            flow_edges.append(
                edges[
                    edges[merge_column].isin(flows_dataframe[merge_column].values.tolist())
                    ]
                )
        else:
            flow_edges.append(edges)

    flow_edges = pd.concat(flow_edges,axis=0,ignore_index=True)
    if merge is True:
        return gpd.GeoDataFrame(
                    pd.merge(
                            flows_dataframe,flow_edges,
                            how="left",on=[merge_column]
                            ),
                        geometry="geometry",crs="EPSG:4326")
    else:
        return gpd.GeoDataFrame(
                        flow_edges,
                        geometry="geometry",
                        crs="EPSG:4326")
def main(config):
    processed_data_path = config['paths']['data']
    output_data_path = config['paths']['results']
    figure_path = config['paths']['figures']


    figures = os.path.join(figure_path,"transport_networks")
    os.makedirs(figures,exist_ok=True)
    

    ccg_countries = pd.read_csv(os.path.join(processed_data_path,"admin_boundaries","ccg_country_codes.csv"))
    ccg_isos = ccg_countries[ccg_countries["ccg_country"] == 1]["iso_3digit_alpha"].values.tolist()
    
    link_color = "#525252"
    
    _,_,xl,yl = map_background_and_bounds(include_countries=ccg_isos)
    dxl = abs(np.diff(xl))[0]
    dyl = abs(np.diff(yl))[0]
    w = 0.03
    dt = 0.05
    panel_span = 2
    line_width_max = 0.4
    line_steps = 6
    width_step = 0.08
    interpolation='fisher-jenks'

    modes = ["road","rail","sea"]
    networks = []
    for m in range(len(modes)):
        n_dfs = []
        for lt in ["edges","nodes"]: 
            if modes[m] == "road" and lt == "nodes":
                n_dfs.append(pd.DataFrame())
            else:
                df = add_geometries_to_flows([],
                        modes=[modes[m]],
                        layer_type=lt,
                        merge=False) 
                n_dfs.append(df) 
        networks.append(tuple([modes[m]] + n_dfs + [2*m,panel_span]))

    figwidth = 16
    figheight = figwidth/(3+2*w)/dxl*dyl/(1-dt)
    textfontsize = 12
    fig = plt.figure(figsize=(figwidth,figheight))
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1-dt,wspace=w)
    for jdx, (mode,e_df,n_df,pos,span) in enumerate(networks):
        ax = plt.subplot2grid([1,6],[0,pos],1,colspan=span)
        ax.spines[['top','right','bottom','left']].set_visible(False)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        
        ax = plot_ccg_basemap(
                    ax,
                    include_continents=["Africa"],
                    include_countries=ccg_isos,
                    include_labels=True
                    )
        ax.set_title(mode,fontsize=textfontsize,fontweight="bold")
        e_df.geometry.plot(ax=ax,facecolor=link_color,edgecolor='none',linewidth=0,alpha=0.7)
        n_df.geometry.plot(
            ax=ax, 
            color=n_df["color"], 
            edgecolor='none',
            markersize=n_df["markersize"],
            alpha=0.7)

    plt.tight_layout()
    save_fig(os.path.join(figures,"ccg_transport_networks"))
    plt.close()


if __name__ == '__main__':
    main()