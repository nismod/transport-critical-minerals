#!/usr/bin/env python
# coding: utf-8

import sys
import os
import re
import json
import pandas as pd
import numpy as np
pd.options.mode.copy_on_write = True
import igraph as ig
import geopandas as gpd
from collections import defaultdict
from utils import *
from transport_cost_assignment import *
from tqdm import tqdm
tqdm.pandas()

def main(
            config,
            year,
            percentile,
            efficient_scale,
            country_case,
            constraint,
            combination = None,
            distance_from_origin=0.0,
            environmental_buffer=0.0
        ):
    processed_data_path = config['paths']['data']
    output_data_path = config['paths']['results']
    figure_path = config['paths']['figures']

    baseline_year = 2022
    results_folder = os.path.join(output_data_path,"aggregated_node_edge_flows")
    if os.path.exists(results_folder) == False:
        os.mkdir(results_folder)


    flow_data_folder = os.path.join(output_data_path,"node_edge_flows")
    
    reference_minerals = ["copper","cobalt","manganese","lithium","graphite","nickel"]
    nodes_dfs = []
    edges_dfs = []
    sum_dict = []
    for reference_mineral in reference_minerals:
        flow_column = f"{reference_mineral}_final_stage_production_tons"
        if year == baseline_year:
            layer_name = f"{reference_mineral}_{percentile}"
        else:
            layer_name = f"{reference_mineral}_{percentile}_{efficient_scale}"
        
        if combination is None:
            input_gpq = f"flows_{layer_name}_{year}_{country_case}_{constraint}.geoparquet"
        else:
            if distance_from_origin > 0.0 or environmental_buffer > 0.0:
                ds = str(distance_from_origin).replace('.','p')
                eb = str(environmental_buffer).replace('.','p')
                input_gpq = f"{combination}_flows_{layer_name}_{year}_{country_case}_{constraint}_op_{ds}km_eb_{eb}km.geoparquet"
            else:
                input_gpq = f"{combination}_flows_{layer_name}_{year}_{country_case}_{constraint}.geoparquet"
        
        edge_file_path = os.path.join(flow_data_folder,
                            f"edges_{input_gpq}")
        if os.path.exists(edge_file_path):
            edges_flows_df = gpd.read_parquet(edge_file_path)
            nodes_flows_df = gpd.read_parquet(os.path.join(flow_data_folder,
                                f"nodes_{input_gpq}"))
            edges_dfs.append(edges_flows_df[["id","from_id","to_id","mode",flow_column,"geometry"]])
            nodes_dfs.append(nodes_flows_df[["id","iso3","infra","mode",flow_column,"geometry"]])
            sum_dict.append((flow_column,"sum"))
    
    nodes_dfs = pd.concat(nodes_dfs,axis=0,ignore_index=True)
    edges_dfs = pd.concat(edges_dfs,axis=0,ignore_index=True)

    edges = edges_dfs[["id","geometry"]].drop_duplicates(subset=["id"],keep="first")
    edges_flows_df = edges_dfs.groupby(["id","from_id","to_id","mode"]).agg(dict(sum_dict)).reset_index()
    edges_flows_df["total_final_stage_production_tons"] = edges_flows_df[[c[0] for c in sum_dict]].sum(axis=1)
    edges_flows_df = pd.merge(edges_flows_df,edges,how="left",on=["id"])

    nodes = nodes_dfs[["id","geometry"]].drop_duplicates(subset=["id"],keep="first")
    nodes_flows_df = nodes_dfs.groupby(["id","iso3","infra","mode"]).agg(dict(sum_dict)).reset_index()
    nodes_flows_df["total_final_stage_production_tons"] = nodes_flows_df[[c[0] for c in sum_dict]].sum(axis=1)
    nodes_flows_df = pd.merge(nodes_flows_df,nodes,how="left",on=["id"])

    edges_flows_df = gpd.GeoDataFrame(edges_flows_df,geometry="geometry",crs="EPSG:4326")
    nodes_flows_df = gpd.GeoDataFrame(nodes_flows_df,geometry="geometry",crs="EPSG:4326")

    if year == baseline_year:
        layer_name = f"{percentile}"
    else:
        layer_name = f"{percentile}_{efficient_scale}"
    if combination is None:
        output_gpq = f"flows_{layer_name}_{year}_{country_case}_{constraint}.geoparquet"
    else:
        if distance_from_origin > 0.0 or environmental_buffer > 0.0:
            ds = str(distance_from_origin).replace('.','p')
            eb = str(environmental_buffer).replace('.','p')
            output_gpq = f"{combination}_flows_{layer_name}_{year}_{country_case}_{constraint}_op_{ds}km_eb_{eb}km.geoparquet"
        else:
            output_gpq = f"{combination}_flows_{layer_name}_{year}_{country_case}_{constraint}.geoparquet"
    
    edges_flows_df.to_parquet(os.path.join(results_folder,
                        f"edges_{output_gpq}"))
    nodes_flows_df.to_parquet(os.path.join(results_folder,
                        f"nodes_{output_gpq}"))

if __name__ == '__main__':
    CONFIG = load_config()
    try:
        if len(sys.argv) > 6:
            year = int(sys.argv[1])
            percentile = str(sys.argv[2])
            efficient_scale = str(sys.argv[3])
            country_case = str(sys.argv[4])
            constraint = str(sys.argv[5])
            combination = str(sys.argv[6])
            distance_from_origin = float(sys.argv[7])
            environmental_buffer = float(sys.argv[8])
        else:
            year = int(sys.argv[1])
            percentile = str(sys.argv[2])
            efficient_scale = str(sys.argv[3])
            country_case = str(sys.argv[4])
            constraint = str(sys.argv[5])
            combination = None
            distance_from_origin = 0.0
            environmental_buffer = 0.0
    except IndexError:
        print("Got arguments", sys.argv)
        exit()
    main(
            CONFIG,
            year,
            percentile,
            efficient_scale,
            country_case,
            constraint,
            combination = combination,
            distance_from_origin=distance_from_origin,
            environmental_buffer=environmental_buffer)