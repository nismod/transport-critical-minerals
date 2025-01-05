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

def get_export_import_columns(
                    flows_dataframe,
                    sum_dictionary,
                    location_types,
                    trade_types,
                    trade_ton_column="final_stage_production_tons"
                ):
    for idx, (ly,ty) in enumerate(zip(location_types,trade_types)):
        tag_columns = [c for c in flows_dataframe.columns.values.tolist() if ly in c and trade_ton_column in c]
        for tg in tag_columns:
            tg_c = tg.split(ly)[-1]
            if ty == "export":
                sum_dictionary[f"{trade_ton_column}_{tg_c}_export"].append(tg)
            elif ty == "import":
                sum_dictionary[f"{trade_ton_column}_{tg_c}_import"].append(tg)
            elif ty == "inter":
                o_d = tg_c.split("_")
                sum_dictionary[f"{trade_ton_column}_{o_d[0]}_export"].append(tg)
                sum_dictionary[f"{trade_ton_column}_{o_d[1]}_import"].append(tg)

    return sum_dictionary

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

    ccg_countries = pd.read_csv(os.path.join(processed_data_path,"admin_boundaries","ccg_country_codes.csv"))
    ccg_isos = ccg_countries[ccg_countries["ccg_country"] == 1]["iso_3digit_alpha"].values.tolist()
    del ccg_countries
    
    reference_minerals = ["copper","cobalt","manganese","lithium","graphite","nickel"]
    location_types = ["_origin_","_destination_","_inter_"]
    trade_types = ["export","import","inter"]
    trade_column = "final_stage_production_tons"
    nodes_dfs = []
    edges_dfs = []
    sum_dict = defaultdict(list) 
    for reference_mineral in reference_minerals:
        flow_columns = [
                        f"{reference_mineral}_{trade_column}_{tc}" for tc in trade_types
                        ] + [f"{reference_mineral}_{trade_column}"]
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
            edges_dfs.append(edges_flows_df)
            nodes_dfs.append(nodes_flows_df)
            for dx, (tt,fc) in enumerate(zip(trade_types + ["total"],flow_columns)):
                sum_dict[f"{trade_column}_{tt}"].append(fc)
             
            sum_dict = get_export_import_columns(
                                    edges_flows_df,
                                    sum_dict,
                                    location_types,
                                    trade_types)
            
    
    nodes_dfs = pd.concat(nodes_dfs,axis=0,ignore_index=True)
    edges_dfs = pd.concat(edges_dfs,axis=0,ignore_index=True)

    all_sums = [(c,"sum") for c in list(set(sum_dict.values()))]
    edges = edges_dfs[["id","geometry"]].drop_duplicates(subset=["id"],keep="first")
    edges_flows_df = edges_dfs.groupby(["id","from_id","to_id","mode"]).agg(dict(all_sums)).reset_index()
    for k,v in sum_dict.items():
        edges_flows_df[k] = edges_flows_df[list(set(v))].sum(axis=1)

    edges_flows_df.drop(list(set(sum_dict.values())),axis=1,inplace=True)
    edges_flows_df = pd.merge(edges_flows_df,edges,how="left",on=["id"])

    nodes = nodes_dfs[["id","geometry"]].drop_duplicates(subset=["id"],keep="first")
    nodes_flows_df = nodes_dfs.groupby(["id","iso3","infra","mode"]).agg(dict(all_sums)).reset_index()
    for k,v in sum_dict.items():
        nodes_flows_df[k] = nodes_flows_df[list(set(v))].sum(axis=1)
    nodes_flows_df.drop(list(set(sum_dict.values())),axis=1,inplace=True)
    
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