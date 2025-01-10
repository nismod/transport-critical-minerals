#!/usr/bin/env python
# coding: utf-8

import sys
import os
import ast
import pandas as pd
import numpy as np
pd.options.mode.copy_on_write = True
import geopandas as gpd
from matplotlib.lines import Line2D
from map_plotting_utils import *
from tqdm import tqdm
tqdm.pandas()

config = load_config()
processed_data_path = config['paths']['data']
figure_path = config['paths']['figures']

def create_legend(
                legend_type,
                width_by_range,
                legend_colors,
                legend_labels,marker='o'):
    legend_weight = 0.05
    legend_handles = []
    for dx, (width,color,label) in enumerate(zip(width_by_range,legend_colors,legend_labels)):
        if legend_type == 'marker':
            legend_handles.append(plt.plot([],[],
                                marker=marker, 
                                ms=width/legend_weight, 
                                ls="",
                                color=color,
                                label=label.title())[0])
        else:
            legend_handles.append(Line2D([0], [0], 
                            color=color, lw=width/legend_weight, label=label.title()))

    return legend_handles
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
            if layer_type == "nodes":
                edges["mode"] = edges["infra"]
            else:
                edges["mode"] = edges["status"]
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
                        "africa_maritime_network.gpkg"
                    ),layer=layer_type)
            if layer_type == "edges":
                edges["mode"] = "maritime routes"
            else:
                edges["mode"] = edges["infra"]
        if layer_type == "edges":
            edges = edges[[merge_column,"from_id","to_id","mode","geometry"]]
        else:
            edges = edges[[merge_column,"iso3","mode","geometry"]]
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
def main():
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
    # modes = ["rail","sea"]
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
        ax = plt.subplot2grid([1,2*len(modes)],[0,pos],1,colspan=span)
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
        ax.set_title(f"{mode.title()} network",fontsize=textfontsize,fontweight="bold")
        legend_handles = []
        if mode == "road":
            title = "$\\bf{Road \, class}$"
            legend_handles.append(plt.plot([],[],
                                            color="none",
                                            label=title)[0])
            tags = ["other","primary","motorway and trunk"]
            tag_buffers = [0.04,0.08,0.12]
            tag_colors = ["#969696","#fc9272","#800026"]
            for idx,(t,tb,tc) in enumerate(zip(tags,tag_buffers,tag_colors)):
                if t == "other":
                    df = e_df[~e_df["mode"].isin(["primary","motorway","trunk"])]
                elif t == "motorway and trunk":
                    df = e_df[e_df["mode"].isin(["motorway","trunk"])]
                else:
                    df = e_df[e_df["mode"] == t]
                df["geometry"] = df.progress_apply(lambda x:x.geometry.buffer(tb),axis=1)
                df.geometry.plot(
                            ax=ax,
                            facecolor=tc,
                            edgecolor='none',
                            linewidth=0,
                            alpha=0.7,
                            label=t.title())
            legend_handles += create_legend(
                            "line",
                            tag_buffers,
                            tag_colors,
                            tags)
        else:
            title = "$\\bf{Asset \, types}$"
            legend_handles.append(plt.plot([],[],
                                            color="none",
                                            label=title)[0])
            if mode == "rail":
                e_df = e_df[e_df["mode"] == "open"]
                nids = list(set(e_df["from_id"].values.tolist() + e_df["to_id"].values.tolist()))
                t = "operational lines" 
                tb = 0.08
                tc = "#238b45"
                n_df = n_df[(n_df["mode"] == "station") & (n_df["id"].isin(nids))]
                nt = "station"
                nc = "#737373"
                ns = 6
            elif mode == "sea":
                t = "maritime routes" 
                tb = 0.08
                tc = "#3690c0"
                n_df = n_df[(n_df["mode"] == "port") & (n_df["iso3"].isin(ccg_isos))]
                nt = "port"
                nc = "#6a51a3"
                ns = 20

            e_df["geometry"] = e_df.progress_apply(lambda x:x.geometry.buffer(tb),axis=1)
            e_df.geometry.plot(
                        ax=ax,
                        facecolor=tc,
                        edgecolor='none',
                        linewidth=0,
                        alpha=0.7,
                        label=t.title())
            legend_handles += create_legend(
                            "line",
                            [tb],
                            [tc],
                            [t])
            n_df.geometry.plot(
                ax=ax, 
                color=nc, 
                edgecolor='none',
                markersize=ns,
                label=nt.title())
            legend_handles += create_legend(
                                "marker",
                                [0.5],
                                [nc],
                                [nt])

        leg = ax.legend(
                handles=legend_handles, 
                fontsize=10, 
                loc='lower right',
                frameon=False)

        # Move titles to the left 
        for item, label in zip(leg.legend_handles, leg.texts):
            if label._text in [title]:
                width = item.get_window_extent(fig.canvas.get_renderer()).width
                label.set_ha('left')
                label.set_position((-4.0*width,0))

    plt.tight_layout()
    save_fig(os.path.join(figures,"ccg_transport_networks"))
    plt.close()


if __name__ == '__main__':
    main()