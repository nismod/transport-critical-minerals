#!/usr/bin/env python
# coding: utf-8
"""This code finds the route between mines (in Africa) and all ports (outside Africa) 
"""
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

def find_country_edges(edges,network_edges):
    weights = defaultdict(int)
    for i, e in enumerate(edges):
        weights[e] += i

    result = sorted(set(edges) & set(network_edges), key=lambda i: weights[i])
    return result
    # items = set(edges) & set(network_edges)
    # result = sorted(items, key=lambda element: edges.index(element))
    # return result
    # return [e for e in edges if e in network_edges]

def main(config,reference_mineral,year,percentile):
    incoming_data_path = config['paths']['incoming_data']
    processed_data_path = config['paths']['data']
    output_data_path = config['paths']['results']

    results_folder = os.path.join(output_data_path,"flow_mapping")
    if os.path.exists(results_folder) == False:
        os.mkdir(results_folder)

    """Step 1: Get the input datasets
    """
    # reference_mineral = "copper"
    # final_refined_stage = 3.0
    trade_ton_column = "mine_output_tons"
    # trade_usd_column = "mine_output_thousandUSD"
    # years = [2021,2030]
    # find_flows = True
    
    od_df = pd.read_parquet(
                        os.path.join(results_folder,
                            f"{reference_mineral}_flow_paths_{year}_{percentile}.parquet")
                        )
    origin_isos = list(set(od_df["export_country_code"].values.tolist()))
    stages = list(set(od_df["final_refined_stage"].values.tolist()))
    country_df = []
    edges_flows_df = []
    nodes_flows_df = []
    sum_dict = []    
    for o_iso in origin_isos:
        for stage in stages:
            df = od_df[(od_df["export_country_code"] == o_iso) & (od_df["final_refined_stage"] == stage)]
            # for flow_column in [trade_ton_column,trade_usd_column]:
            for flow_column in [trade_ton_column]:
                sum_dict.append((f"{reference_mineral}_{flow_column}_stage_{stage}_origin_{o_iso}","sum"
                                    ))
                for path_type in ["full_edge_path","full_node_path"]:
                    f_df = get_flow_on_edges(
                                   df,
                                    "id",path_type,
                                    flow_column)
                    f_df.rename(columns={flow_column:f"{reference_mineral}_{flow_column}_stage_{stage}_origin_{o_iso}"},
                        inplace=True)
                    if path_type == "full_edge_path":
                        edges_flows_df.append(f_df)
                    else:
                        nodes_flows_df.append(f_df)
        print ("* Done with:",o_iso)

    degree_df = pd.DataFrame()
    for path_type in ["edges","nodes"]:
        if path_type == "edges":
            flows_df = pd.concat(edges_flows_df,axis=0,ignore_index=True).fillna(0)
        else:
            flows_df = pd.concat(nodes_flows_df,axis=0,ignore_index=True).fillna(0)
        flows_df = flows_df.groupby(["id"]).agg(dict(sum_dict)).reset_index()

        # for flow_column in [trade_ton_column,trade_usd_column]:
        for flow_column in [trade_ton_column]:
            flow_sums = []
            for stage in stages:
                stage_sums = []
                for o_iso in origin_isos:
                    stage_sums.append(f"{reference_mineral}_{flow_column}_stage_{stage}_origin_{o_iso}")

                flows_df[f"{reference_mineral}_{flow_column}_stage_{stage}"] = flows_df[stage_sums].sum(axis=1)
                flow_sums.append(f"{reference_mineral}_{flow_column}_stage_{stage}")

            flows_df[f"{reference_mineral}_{flow_column}"] = flows_df[flow_sums].sum(axis=1) 

        flows_df = add_geometries_to_flows(flows_df,
                                merge_column="id",
                                modes=["rail","sea","road"],
                                layer_type=path_type)
        if path_type == "edges":
            degree_df = flows_df[["from_id","to_id"]].stack().value_counts().rename_axis('id').reset_index(name='degree')
        elif path_type == "nodes" and len(degree_df.index) > 0:
            flows_df = pd.merge(flows_df,degree_df,how="left",on=["id"])

        flows_df = gpd.GeoDataFrame(flows_df,
                                geometry="geometry",
                                crs="EPSG:4326")
        if year == 2022:
            layer_name = f"{reference_mineral}"
        else:
            layer_name = f"{reference_mineral}_{percentile}"
        flows_df.to_file(os.path.join(results_folder,
                            f"{path_type}_flows_{year}.gpkg"),
                            layer=layer_name,driver="GPKG")


if __name__ == '__main__':
    CONFIG = load_config()
    try:
        reference_mineral = str(sys.argv[1])
        year = int(sys.argv[2])
        percentile = int(sys.argv[3])
    except IndexError:
        print("Got arguments", sys.argv)
        exit()
    main(CONFIG,reference_mineral,year,percentile)