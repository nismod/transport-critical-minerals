#!/usr/bin/env python
# coding: utf-8
"""This code finds the route between mines (in Africa) and all ports (outside Africa) 
"""
import sys
import os
import re
import json
import pandas as pd
import igraph as ig
import geopandas as gpd
from utils import *
from transport_cost_assignment import *
from trade_functions import *
from tqdm import tqdm
tqdm.pandas()



def main(config):
    incoming_data_path = config['paths']['incoming_data']
    processed_data_path = config['paths']['data']
    output_data_path = config['paths']['results']

    input_folder = os.path.join(output_data_path,"flow_node_ods")
    results_folder = os.path.join(output_data_path,"flow_od_paths")
    os.makedirs(results_folder,exist_ok=True)

    # cargo_type = "Dry bulk"
    od_columns = [
                    "reference_mineral",
                    "export_country_code",
                    "import_country_code",
                    "export_continent",
                    "import_continent",
                    "trade_type",
                    "initial_processing_stage",
                    "final_processing_stage",
                    "initial_processing_location",
                    "final_processing_location",
                    "initial_stage_production_tons",    
                    "final_stage_production_tons"
                ]
    od_merge_columns = [
                        "origin_id",
                        "destination_id",
                        "reference_mineral",
                        "export_country_code",
                        "import_country_code",
                        "export_continent",
                        "import_continent",
                        "trade_type",
                        "initial_processing_stage",
                        "final_processing_stage",
                        "initial_processing_location",
                        "final_processing_location",
                        ]
    data_type = {"initial_refined_stage":"str","final_refined_stage":"str"}
    """Step 1: Get the OD matrix
    """
    # scenario = scenario.replace(" ","_")
    # if year == 2022:
    #     od_file_name = f"mining_city_node_level_ods_{year}_{percentile}.csv"
    # else:
    #     od_file_name = f"mining_city_node_level_ods_{scenario}_{year}_{percentile}_{efficient_scale}.csv"

    
    od_file_name = "mining_city_node_level_ods_2022_baseline.csv"
    # od_file_name = "mining_city_node_level_ods_precursor_2040_mid_max_threshold_metal_tons.csv"
    c_t_df = pd.read_csv(os.path.join(
                                input_folder,
                                od_file_name),dtype=data_type)
    
    origin_id = "origin_id"
    destination_id = "destination_id"
    final_ton_column = "final_stage_production_tons"
    initial_ton_column = "initial_stage_production_tons"
    edges_flows_df = []
    nodes_flows_df = []
    flow_column_types = []
    network_graph = pd.read_parquet(os.path.join(processed_data_path,"shipping_network","network_graph.parquet"))
    network_graph["capacity"] = np.where(network_graph["mode"]=="rail",1e6,network_graph["capacity"])
    network_graph[final_ton_column] = 0
    mine_routes, unassinged_routes, network_graph = od_flow_allocation_capacity_constrained(
                                        c_t_df,network_graph,
                                        final_ton_column,"gcost_usd_tons",
                                        "distance_km","time_hr","land_border_cost_usd_tons",
                                        "id",origin_id,
                                        destination_id)
    network_graph = network_graph[network_graph[final_ton_column] > 0]
    network_graph.to_csv(os.path.join(results_folder,
                f"total_flows_2022_baseline_rail_capacity.csv"),
                index=False)
    # network_graph = pd.read_csv(os.path.join(results_folder,
    #             f"total_flows_2022_baseline_rail_capacity.csv"))
    network_graph = add_geometries_to_flows(network_graph,
                        merge_column=["id","from_id","to_id"],
                        modes=["rail","road"])
    network_graph.to_file(os.path.join(results_folder,
                f"total_flows_2022_baseline_rail_capacity.gpkg"),layer="flows",driver="GPKG")
    # network_graph.to_csv(os.path.join(results_folder,
    #             f"total_flows_2040_precursor.csv"),
    #             index=False)
    del network_graph
    if len(unassinged_routes) > 0:
        unassinged_routes = pd.concat(unassinged_routes,axis=0,ignore_index=True)
        if "geometry" in unassinged_routes.columns.values.tolist():
            unassinged_routes.drop("geometry",axis=1,inplace=True)
        unassinged_routes.to_csv(
                os.path.join(results_folder,
                f"unassigned_flow_paths_2022_baseline_rail_capacity.csv"),
                index=False)
        # unassinged_routes.to_csv(
        #         os.path.join(results_folder,
        #         f"unassigned_flow_paths_2040_precursor.csv"),
        #         index=False)
    del unassinged_routes

    if len(mine_routes) > 0:
        mine_routes = pd.concat(mine_routes,axis=0,ignore_index=True)
        c_t_df.rename(columns={final_ton_column:"initial_tonnage"},inplace=True)
        mine_routes = pd.merge(mine_routes,
                        c_t_df[od_merge_columns + ["initial_tonnage"]],
                        how="left",on=od_merge_columns)
        mine_routes[initial_ton_column] = mine_routes[
                                                initial_ton_column]*mine_routes[
                                                    final_ton_column]/mine_routes["initial_tonnage"]
        mine_routes.drop("initial_tonnage",axis=1,inplace=True)
        if "geometry" in mine_routes.columns.values.tolist():
            mine_routes.drop("geometry",axis=1,inplace=True)
        
        mine_routes = convert_port_routes(mine_routes,"edge_path","node_path")
        # if year > 2022:
        #     file_name = f"{reference_mineral}_flow_paths_{scenario}_{year}_{percentile}_{efficient_scale}.parquet"
        # else:
        #     file_name = f"{reference_mineral}_flow_paths_{year}_{percentile}.parquet"
        # file_name = "baseline_flows.parquet"
        file_name = "baseline_flows_rail_capacity.parquet"
        # file_name = "precursor_flows.parquet"

        mine_routes[[origin_id,destination_id] + od_columns + [
                                "edge_path",
                                "node_path",
                                "full_edge_path",
                                "full_node_path",
                                "gcost_usd_tons_path",
                                "distance_km_path",
                                "time_hr_path",
                                "land_border_cost_usd_tons_path",
                                "gcost_usd_tons"]].to_parquet(
                os.path.join(results_folder,file_name),
                index=False)


if __name__ == '__main__':
    CONFIG = load_config()
    main(CONFIG)