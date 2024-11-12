#!/usr/bin/env python
# coding: utf-8

import sys
import os
import ast
import pandas as pd
import numpy as np
pd.options.mode.copy_on_write = True
import geopandas as gpd
from map_plotting_utils import *
from mapping_properties import *
from tqdm import tqdm
tqdm.pandas()
def assign_processing_type(x):
    if x["mode"] == "mine":
        return "Mine"
    elif x["mode"] == "city":
        return "Existing processing location"
    else:
        return "New processing location"

def main(config,y,p,e,cnt,con):
    processed_data_path = config['paths']['data']
    output_data_path = config['paths']['results']
    figure_path = config['paths']['figures']

    country_codes = ["ZMB"]
    boundary_codes = ["AGO","COD","TZA","MWI","MOZ","ZWE","NAM","BWA"]
    figures = os.path.join(figure_path,f"{'_'.join(country_codes)}_figures")
    if os.path.exists(figures) is False:
        os.mkdir(figures)

    figures = os.path.join(figure_path,f"{'_'.join(country_codes)}_figures","flow_figures")
    if os.path.exists(figures) is False:
        os.mkdir(figures)

    flow_outputs = os.path.join(output_data_path,f"{'_'.join(country_codes)}_node_edge_flows")
    if os.path.exists(flow_outputs) is False:
        os.mkdir(flow_outputs)

    flow_data_folder = os.path.join(output_data_path,"aggregated_node_edge_flows")
    node_data_folder = os.path.join(output_data_path,"optimised_processing_locations")
    make_plot = True
        
    flow_column = f"total_final_stage_production_tons"
    processing_types = ["Mine","Existing processing location","New processing location"]
    reference_minerals = ["cobalt","copper","graphite","lithium","manganese","nickel"]
    
    # nodes_dfs = []
    # edges_dfs = []
    nodes_range = []
    edges_range = []
    new_combinations = []
    if y == 2022:
        layer_name = f"{p}"
    else:
        layer_name = f"{p}_{e}"
    edge_file_path = os.path.join(flow_data_folder,
                        f"edges_flows_{layer_name}_{y}_{cnt}_{con}.geoparquet")
    if os.path.exists(edge_file_path):
        edges_flows_df = gpd.read_parquet(os.path.join(flow_data_folder,
                            f"edges_flows_{layer_name}_{y}_{cnt}_{con}.geoparquet"))
        edges_flows_df = edges_flows_df[~edges_flows_df.geometry.isna()]
        nodes_flows_df = gpd.read_parquet(os.path.join(flow_data_folder,
                            f"nodes_flows_{layer_name}_{y}_{cnt}_{con}.geoparquet"))
        nodes_flows_df = nodes_flows_df[
                        (
                            nodes_flows_df["iso3"].isin(country_codes)
                        ) & (
                            nodes_flows_df["mode"].isin(["road","rail"])
                        )
                        ]
        nodes = nodes_flows_df["id"].values.tolist()
        del nodes_flows_df
        if len(nodes) > 0:
            edges_flows_df = edges_flows_df[
                                    (
                                        edges_flows_df["from_id"].isin(nodes)
                                    ) | (
                                        edges_flows_df["to_id"].isin(nodes)
                                    )]
            del nodes
            edges_range += edges_flows_df[flow_column].values.tolist()   
            if y == 2022:
                layer_name = f"{y}_{p}"
            else:
                layer_name = f"{y}_{p}_{e}"          
            nodes_flows_df = gpd.read_file(
                                os.path.join(
                                    node_data_folder,
                                    f"node_locations_for_energy_conversion_{cnt}_{con}.gpkg"),
                                layer=layer_name)
            nodes_flows_df = nodes_flows_df[~nodes_flows_df.geometry.isna()]
            nodes_flows_df = nodes_flows_df[nodes_flows_df["iso3"].isin(country_codes)]
            if len(nodes_flows_df.index) > 0:
                nodes_flows_df["processing_type"
                    ] = nodes_flows_df.progress_apply(lambda x:assign_processing_type(x),axis=1)
                fcols = []
                for rf in reference_minerals:
                    fcols += [c for c in nodes_flows_df.columns.values.tolist() if f"{rf}_final_stage_production_tons" in c]
                nodes_flows_df[flow_column] = nodes_flows_df[fcols].sum(axis=1)
                nodes_flows_df = nodes_flows_df[nodes_flows_df[flow_column]>0]
                nodes_range += nodes_flows_df[flow_column].values.tolist()
                # nodes_dfs.append(nodes_flows_df)
                new_combinations.append((y,p,e,cnt,con,nodes_flows_df,edges_flows_df))
            else:
                new_combinations.append((y,p,e,cnt,con,pd.DataFrame(),edges_flows_df))
        else:
            make_plot = False
    else:
        make_plot = False

    if make_plot is True:
        combinations = new_combinations.copy()
        titles = [
                    "$\\bf{Links \, annual \, output \,(tonnes)}$",
                    "$\\bf{Locations \, annual \, output \, (tonnes)}$",
                    "$\\bf{Location \,types}$"
                ]
        ax_proj = get_projection(epsg=4326)
        
        fig, ax_plots = plt.subplots(1,1,
                        subplot_kw={'projection': ax_proj},
                        figsize=(12,12),
                        dpi=500)
        include_titles = []
        
        ax = plot_ccg_country_basemap(ax_plots,country_codes,boundary_codes)
        legend_handles = []
        legend_handles.append(plt.plot([],[],
                                        color="none",
                                        label=titles[0])[0])
        include_titles.append(titles[0])
            
        ax, legend = line_map_plotting_colors_width(
                                            ax,
                                            edges_flows_df,
                                            flow_column,
                                            edges_range,
                                            1.0,
                                            "None",
                                            "None",
                                            line_colors = 10*["#525252"],
                                            no_value_color = '#969696',
                                            line_steps = 10,
                                            width_step = 0.01,
                                            interpolation='fisher-jenks')
        legend_handles += legend

        legend_handles.append(plt.plot([],[],
                                        color="none",
                                        label=titles[1])[0])
        include_titles.append(titles[1])
        ax, legend = point_map_plotting_colors_width(
                                    ax,
                                    nodes_flows_df,
                                    flow_column,
                                    nodes_range,
                                    point_classify_column="processing_type",
                                    point_categories=processing_types,
                                    point_colors=["#662506","#cc4c02","#fe9929"],
                                    point_labels=processing_types,
                                    point_zorder=[20,21,22,23],
                                    point_steps=6,
                                    width_step = 40.0,
                                    interpolation = 'fisher-jenks',
                                    legend_label="Annual output (tons)",
                                    legend_size=16,
                                    legend_weight=2.0,
                                    no_value_label="No output",
                                    )

        legend_handles += legend
        legend_handles.append(plt.plot([],[],
                                        color="none",
                                        label=titles[2])[0])
        include_titles.append(titles[2])
        for idx,(l,nc) in enumerate(zip(processing_types,["#662506","#cc4c02","#fe9929"])):
            legend_handles.append(mpatches.Patch(color=nc,
                                           label=l))
        if len(include_titles) > 1:
            ncol = 2
        else:
            ncol = 1
        leg = ax.legend(
            handles=legend_handles, 
            fontsize=14, 
            ncol = ncol,
            loc='upper left',
            frameon=False)

        # Move titles to the left 
        for item, label in zip(leg.legend_handles, leg.texts):
            if label._text in include_titles:
                width = item.get_window_extent(fig.canvas.get_renderer()).width
                label.set_ha('left')
                label.set_position((-10.0*width,0))
        total_tons_df = nodes_flows_df.groupby(["processing_type"])[flow_column].sum().reset_index()
        total_tons = 1e-6*total_tons_df[flow_column].sum()
        mine_tons = 1e-6*total_tons_df[total_tons_df["processing_type"] == "Mine"][flow_column].sum()
        processing_tons = total_tons - mine_tons
        ax.text(
                0.02,
                0.65,
                f"Total mine output = {mine_tons:,.2f} million",
                horizontalalignment='left',
                transform=ax.transAxes,
                size=18,
                weight='bold'
                )
        ax.text(
                0.02,
                0.60,
                f"Total processinng output = {processing_tons:,.2f} million",
                horizontalalignment='left',
                transform=ax.transAxes,
                size=18,
                weight='bold'
                )
        ax.set_title(
            f"$\\bf Total \,flows: \, {y} \, - \, {p.title()} \, scenario \, - \, {cnt.title()} \, {con}$", 
            fontsize=18)
        plt.tight_layout()
        
        st = f"aggregated_flows_{y}_{p}_{cnt}_{con}"

        save_fig(os.path.join(figures,f"{st}_scenarios.png"))
        plt.close()


if __name__ == '__main__':
    CONFIG = load_config()
    try:
        year = int(sys.argv[1])
        percentile = str(sys.argv[2])
        efficient_scale = str(sys.argv[3])
        country_case = str(sys.argv[4])
        constraint = str(sys.argv[5])
    except IndexError:
        print("Got arguments", sys.argv)
        exit()
    main(CONFIG,year,percentile,efficient_scale,country_case,constraint)