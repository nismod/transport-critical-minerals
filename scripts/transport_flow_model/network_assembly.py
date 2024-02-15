#!/usr/bin/env python
# coding: utf-8
"""This code finds the route between mines (in Africa) and all ports (outside Africa) 
"""
import sys
import os
import json
import pandas as pd
import geopandas as gpd
from utils import *
from transport_cost_assignment import *
from tqdm import tqdm
tqdm.pandas()

config = load_config()
incoming_data_path = config['paths']['incoming_data']
processed_data_path = config['paths']['data']
output_data_path = config['paths']['results']

def multimodal_network_assembly(modes=["IWW","rail","road","sea","intermodal"],
                            rail_status=["open"],intermodal_ports="all",
                            cargo_type="general_cargo",
                            default_capacity=1e10,
                            port_to_land_capacity=None):    
    multi_modal_df = []
    for mode in modes:
        if mode == "IWW":
            edges = gpd.read_file(os.path.join(
                                    processed_data_path,
                                    "infrastructure",
                                    "africa_iww_network.gpkg"
                                        ), layer="edges"
                            )
            edges['capacity'] = default_capacity
        elif mode == "rail":
            edges = gpd.read_file(os.path.join(
                            processed_data_path,
                            "infrastructure",
                            "africa_railways_network.gpkg"
                                ), layer="edges"
                    )
            edges = edges[edges["status"].isin(rail_status)]
            edges['capacity'] = default_capacity
        elif mode == "road":
            edges = gpd.read_parquet(os.path.join(
                            processed_data_path,
                            "infrastructure",
                            "africa_roads_edges.geoparquet"))
            edges['capacity'] = default_capacity
        elif mode == "sea":
            if cargo_type != "general_cargo":
                c_df = pd.read_parquet(os.path.join(
                        processed_data_path,
                        "shipping_network",
                        f"maritime_base_network_{cargo_type}.parquet"))
                g_df = pd.read_parquet(os.path.join(
                    processed_data_path,
                    "shipping_network",
                    "maritime_base_network_general_cargo.parquet")) 
                edges = pd.concat([c_df,g_df],axis=0,ignore_index=True)
                edges = edges.drop_duplicates(subset=["from_id","to_id"],keep="first")
            else:
                edges = pd.read_parquet(os.path.join(
                        processed_data_path,
                        "shipping_network",
                        "maritime_base_network_general_cargo.parquet"))
            edges.rename(columns={"from_iso3":"from_iso_a3","to_iso3":"to_iso_a3"},inplace=True)
            edges["id"] = edges.index.values.tolist()
            edges["id"] = edges.apply(lambda x:f"maritimeroute{x.id}",axis=1)
            edges['capacity'] = default_capacity
        elif mode == "intermodal":
            edges = gpd.read_file(os.path.join(
                            processed_data_path,
                            "infrastructure",
                            "africa_multimodal.gpkg"
                                ), layer="edges")
            link_types = [f"{fm}-{tm}" for idx,(fm,tm) in enumerate(itertools.permutations(modes,2))]
            edges = edges[edges["link_type"].isin(link_types)]
            if intermodal_ports != "all":
                port_edges = edges[(edges["from_infra"] == "sea") | (edges["to_infra"] == "sea")]
                port_edges = port_edges[~((port_edges["from_id"].isin(intermodal_ports)) | (port_edges["to_id"].isin(intermodal_ports)))]
                edges = edges[~edges["id"].isin(port_edges.id.values.tolist())]

            edges["from_id"] = edges.progress_apply(
                                lambda x: f"{x.from_id}_land" if "port" in str(x.from_id) else x.from_id,
                                axis=1)
            edges["to_id"] = edges.progress_apply(
                                lambda x: f"{x.to_id}_land" if "port" in str(x.to_id) else x.to_id,
                                axis=1)
            edges['capacity'] = default_capacity
            if port_to_land_capacity is not None:
                for idx,(port,capacity) in enumerate(port_to_land_capacity):
                    cap_factor = 0.5*len(edges[((edges["from_id"] == f"{port}_land") | (edges["to_id"] == f"{port}_land"))].index)
                    edges.loc[((edges["from_id"] == f"{port}_land") | (edges["to_id"] == f"{port}_land")),"capacity"] = 1.0*capacity/cap_factor
        edges["mode"] = mode
        edges = transport_cost_assignment_function(edges,mode)
        multi_modal_df.append(edges[["from_id","to_id","id","mode","capacity","gcost_usd_tons"]])
        if mode in ("road","rail"):
            add_edges = edges[["from_id","to_id","id","mode","capacity","gcost_usd_tons"]].copy()
            add_edges.columns = ["to_id","from_id","id","mode","capacity","gcost_usd_tons"]
            multi_modal_df.append(add_edges)


    return multi_modal_df

def main():
    network_graph = multimodal_network_assembly(
                            modes=["rail","road","sea","intermodal"],
                            rail_status=["open"],
                            intermodal_ports="all",
                            cargo_type="general_cargo",
                            default_capacity=1e10,
                            port_to_land_capacity=None)
    network_graph = pd.concat(network_graph,axis=0,ignore_index=True)
    print (network_graph)


if __name__ == '__main__':
    main()