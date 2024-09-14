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

def main(config,years,percentiles,efficient_scales,country_cases,constraints):
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


    flow_data_folder = os.path.join(output_data_path,"aggregated_node_edge_flows")
    make_plot = True
        
    flow_column = f"total_final_stage_production_tons"
    
    combinations = list(zip(years,percentiles,efficient_scales,country_cases,constraints))
    edges_dfs = []
    edges_range = []
    for idx, (y,p,e,cnt,con) in enumerate(combinations):
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
            nodes = nodes_flows_df[
                            (
                                nodes_flows_df["iso3"].isin(country_codes)
                            ) & (
                                nodes_flows_df["mode"].isin(["road","rail"])
                            )
                            ]["id"].values.tolist()
            del nodes_flows_df
            edges_flows_df = edges_flows_df[
                                    (
                                        edges_flows_df["from_id"].isin(nodes)
                                    ) | (
                                        edges_flows_df["to_id"].isin(nodes)
                                    )]
            del nodes
            edges_range += edges_flows_df[flow_column].values.tolist()

            edges_dfs.append(edges_flows_df)
        else:
            make_plot = False

    if make_plot is True:
        combinations = list(
                            zip(
                                    years,
                                    percentiles,
                                    efficient_scales,
                                    country_cases,
                                    constraints,
                                    edges_dfs
                                )
                            )
        ax_proj = get_projection(epsg=4326)
        fig, ax_plots = plt.subplots(1,len(years),
                        subplot_kw={'projection': ax_proj},
                        figsize=(20,10),
                        dpi=500)
        ax_plots = ax_plots.flatten()
        for idx, (y,p,e,cnt,con,edf) in enumerate(combinations):
            ax = plot_ccg_country_basemap(ax_plots[idx],country_codes,boundary_codes)
            legend_handles = []
            titles = ["$\\bf{Links \, annual \, output \,(tonnes)}$"]
            legend_handles.append(plt.plot([],[],
                                            color="none",
                                            label="$\\bf{Links \, annual \, output \,(tonnes)}$")[0])
            ax, legend = line_map_plotting_colors_width(
                                                ax,
                                                edf,
                                                flow_column,
                                                edges_range,
                                                1.0,
                                                "None",
                                                "None",
                                                line_colors = 8*["#525252"],
                                                no_value_color = '#969696',
                                                line_steps = 8,
                                                width_step = 0.01,
                                                interpolation='fisher-jenks')
            legend_handles += legend
            leg = ax.legend(
                handles=legend_handles, 
                fontsize=10, 
                loc='upper left',
                frameon=False)

            # Move titles to the left 
            for item, label in zip(leg.legend_handles, leg.texts):
                if label._text in titles:
                    width = item.get_window_extent(fig.canvas.get_renderer()).width
                    label.set_ha('left')
                    label.set_position((-4.0*width,0))

            ax.set_title(
                f"$\\bf {y} \, - \, {p.title()} \, scenario \, - \, {cnt.title()} \, {con}$", 
                fontsize=12)
        plt.tight_layout()
        
        scenario = [
                        years,
                        percentiles,
                        country_cases,
                        constraints
                    ]
        st = "agg"
        for sc in scenario:
            st += "_" + '_'.join(list(set(map(str,sc))))
            print (st)

        save_fig(os.path.join(figures,f"{st}_scenarios.png"))
        plt.close()


if __name__ == '__main__':
    CONFIG = load_config()
    try:
        years = ast.literal_eval(str(sys.argv[1]))
        percentiles = ast.literal_eval(str(sys.argv[2]))
        efficient_scales = ast.literal_eval(str(sys.argv[3]))
        country_cases = ast.literal_eval(str(sys.argv[4]))
        constraints = ast.literal_eval(str(sys.argv[5]))
    except IndexError:
        print("Got arguments", sys.argv)
        exit()
    main(CONFIG,years,percentiles,efficient_scales,country_cases,constraints)