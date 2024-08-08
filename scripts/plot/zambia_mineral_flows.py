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

def main(config,reference_mineral,years,percentiles,efficient_scales,country_cases,constraints):
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

    flow_data_folder = os.path.join(output_data_path,"node_edge_flows")
    node_data_folder = os.path.join(output_data_path,"optimised_processing_locations")
    make_plot = True
        
    mp = mineral_properties()[reference_mineral]
    flow_column = f"{reference_mineral}_final_stage_production_tons"
    # ccg_countries = pd.read_csv(os.path.join(processed_data_path,"admin_boundaries","ccg_country_codes.csv"))
    # ccg_isos = ccg_countries[ccg_countries["ccg_country"] == 1]["iso_3digit_alpha"].values.tolist()
    processing_types = ["Mine","Existing processing location","New processing location"]
    
    combinations = list(zip(years,percentiles,efficient_scales,country_cases,constraints))
    # nodes_dfs = []
    # edges_dfs = []
    nodes_range = []
    edges_range = []
    new_combinations = []
    for idx, (y,p,e,cnt,con) in enumerate(combinations):
        if y == 2022:
            layer_name = f"{reference_mineral}_{p}"
        else:
            layer_name = f"{reference_mineral}_{p}_{e}"
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
                    fcols = [c for c in nodes_flows_df.columns.values.tolist() if flow_column in c]
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
                    "$\\bf{Links \, annual \, output \,(tons)}$",
                    "$\\bf{Locations \, annual \, output \, (tons)}$",
                    "$\\bf{Location \,types}$"
                ]
        ax_proj = get_projection(epsg=4326)
        if len(combinations) == 3:
                figsize = (24,8)
        else:
            figsize = (24,12)
        fig, ax_plots = plt.subplots(1,len(combinations),
                        subplot_kw={'projection': ax_proj},
                        figsize=figsize,
                        dpi=500)
        ax_plots = ax_plots.flatten()
        include_titles = []
        for idx, (y,p,e,cnt,con,ndf,edf) in enumerate(combinations):
            ax = plot_ccg_country_basemap(ax_plots[idx],country_codes,boundary_codes)
            legend_handles = []
            legend_handles.append(plt.plot([],[],
                                            color="none",
                                            label=titles[0])[0])
            include_titles.append(titles[0])
            if len(ndf.index) > 0:
                ax, legend = line_map_plotting_colors_width(
                                                    ax,
                                                    edf,
                                                    flow_column,
                                                    edges_range,
                                                    1.0,
                                                    "None",
                                                    "None",
                                                    line_colors = 6*[mp["edge_color"]],
                                                    no_value_color = '#969696',
                                                    line_steps = 6,
                                                    width_step = 0.02,
                                                    interpolation='fisher-jenks')
                legend_handles += legend

                legend_handles.append(plt.plot([],[],
                                                color="none",
                                                label=titles[1])[0])
                include_titles.append(titles[1])
                ax, legend = point_map_plotting_colors_width(
                                            ax,
                                            ndf,
                                            flow_column,
                                            nodes_range,
                                            point_classify_column="processing_type",
                                            point_categories=processing_types,
                                            point_colors=mp["node_colors"],
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
                for idx,(l,nc) in enumerate(zip(processing_types,mp["node_colors"])):
                    legend_handles.append(mpatches.Patch(color=nc,
                                                   label=l))
            if len(include_titles) > 1:
                ncol = 2
            else:
                ncol = 1
            leg = ax.legend(
                handles=legend_handles, 
                fontsize=10, 
                ncol = ncol,
                loc='upper left',
                frameon=False)

            # Move titles to the left 
            for item, label in zip(leg.legend_handles, leg.texts):
                if label._text in include_titles:
                    width = item.get_window_extent(fig.canvas.get_renderer()).width
                    label.set_ha('left')
                    label.set_position((-4.0*width,0))

            ax.set_title(
                f"$\\bf {reference_mineral.title()}: \, {y} \, - \, {p.title()} \, scenario \, - \, {cnt.title()} \, {con}$", 
                fontsize=12)
        plt.tight_layout()
        
        scenario = [
                        years,
                        percentiles,
                        country_cases,
                        constraints
                    ]
        st = ""
        for sc in scenario:
            st += "_" + '_'.join(list(set(map(str,sc))))

        save_fig(os.path.join(figures,f"{reference_mineral}{st}_scenarios.png"))
        plt.close()


if __name__ == '__main__':
    CONFIG = load_config()
    try:
        reference_mineral = str(sys.argv[1])
        years = ast.literal_eval(str(sys.argv[2]))
        percentiles = ast.literal_eval(str(sys.argv[3]))
        efficient_scales = ast.literal_eval(str(sys.argv[4]))
        country_cases = ast.literal_eval(str(sys.argv[5]))
        constraints = ast.literal_eval(str(sys.argv[6]))
    except IndexError:
        print("Got arguments", sys.argv)
        exit()
    main(CONFIG,reference_mineral,years,percentiles,efficient_scales,country_cases,constraints)