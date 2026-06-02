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
                        "origin_id",
                        "destination_id",
                        "final_stage_production_tons",
                        "distance_km",
                        "time_hr",
                        "vehicle_op_cost_usd_tons",
                        "vot_cost_usd_tons",
                        "border_cost_usd_tons",
                        "gcost_usd_tons"
                        ]
    # od_columns = [
    #                     "origin_id",
    #                     "destination_id",
    #                     "final_stage_production_tons",
    #                     "time_hr"
    #                     ]
    data_type = {"initial_refined_stage":"str","final_refined_stage":"str"}

    origin_id = "origin_id"
    destination_id = "destination_id"
    final_ton_column = "final_stage_production_tons"
    attribute_dictionary = {
                            "distance_km_path":["distance_km",list],
                            "distance_km":["distance_km",float],
                            "time_hr_path":["time_hr",list],
                            "time_hr":["time_hr",float],
                            "gcost_usd_tons_path":["gcost_usd_tons",list],
                            # "gcost_usd_tons":["gcost_usd_tons",float],
                            "vehicle_op_cost_usd_tons":["vehicle_op_cost_usd_tons",float],
                            "vot_cost_usd_tons":["vot_cost_usd_tons",float],
                            "border_cost_usd_tons":["border_cost_usd_tons",float]
                            }
    """Step 1: Get the OD matrix
    """
    origins = ["s_and_p_30884"]
    destinations = [
                    "port_656","port_702","port_784",
                    "port_137","port_1381","port_668",
                    "port_278","port_1133","port_311",
                    "port_948","port_315","port_2063"]
    destinations = [f"{d}_land" for d in destinations]
    destinations += ["city_1499"]

    c_t_df = pd.DataFrame(list(zip(origins*len(destinations),destinations)),columns=["origin_id","destination_id"])
    c_t_df[final_ton_column] = 1

    print (c_t_df)
    

    edges_flows_df = []
    nodes_flows_df = []
    flow_column_types = []
    network_graph = pd.read_parquet(os.path.join(processed_data_path,"shipping_network","network_graph.parquet"))
    # network_graph["vehicle_op_cost_usd_tons"] = np.where(
    #                                                     (
    #                                                         network_graph["mode"]=="road"
    #                                                     ) & (
    #                                                         network_graph["vehicle_op_cost_usd_tons"
    #                                                         ]/network_graph["distance_km"] >= 0.07
    #                                                     ),0.5*network_graph["vehicle_op_cost_usd_tons"],
    #                                                     network_graph["vehicle_op_cost_usd_tons"]
    #                                                     )
    # network_graph["gcost_usd_tons"] = network_graph["vehicle_op_cost_usd_tons"] + network_graph["vot_cost_usd_tons"] + network_graph["border_cost_usd_tons"]
    # network_graph["capacity"] = np.where(network_graph["mode"]=="rail",1e6,network_graph["capacity"])
    # network_graph = network_graph[~network_graph["mode"].isin(["rail","sea"])]
    # network_graph = network_graph[~network_graph["mode"].isin(["rail"])]

    network_graph[final_ton_column] = 0
    # mine_routes, unassinged_routes, network_graph = od_flow_allocation_capacity_constrained(
    #                                     c_t_df,network_graph,
    #                                     final_ton_column,"gcost_usd_tons",
    #                                     "distance_km","time_hr","land_border_cost_usd_tons",
    #                                     "id",origin_id,
    #                                     destination_id)
    mine_routes, unassinged_routes, network_graph = od_flow_allocation_capacity_constrained(
                                            c_t_df,network_graph,
                                            final_ton_column,"gcost_usd_tons",
                                            "id",origin_id,
                                            destination_id,
                                            attribute_dict=attribute_dictionary,
                                            get_node_path=True)
    # mine_routes, unassinged_routes, network_graph = od_flow_allocation_capacity_constrained(
    #                                     c_t_df,network_graph,
    #                                     final_ton_column,"time_hr",
    #                                     "distance_km","time_hr","land_border_cost_usd_tons",
    #                                     "id",origin_id,
    #                                     destination_id)
    network_graph = network_graph[network_graph[final_ton_column] > 0]
    network_graph["speed_kph"] = np.where(network_graph["time_hr"] > 0,
                                            network_graph["distance_km"]/network_graph["time_hr"],
                                            0)
    print (network_graph)
    mine_routes = pd.concat(mine_routes,axis=0,ignore_index=True)
    print (mine_routes)
    mine_routes = convert_port_routes(mine_routes,"edge_path","node_path")
    print (mine_routes)
    flow_df = get_flow_on_edges(mine_routes,"id","full_edge_path",final_ton_column)
    print (flow_df)
    # print (network_graph)
    network_graph = add_geometries_to_flows(flow_df,
                        merge_column=["id"],
                        modes=["rail","road","sea"])
    print (network_graph)
    network_graph.to_file(os.path.join(results_folder,
                f"test_routes_roads_border.gpkg"),layer="flows",driver="GPKG")
    del network_graph
    # if len(unassinged_routes) > 0:
    #     unassinged_routes = pd.concat(unassinged_routes,axis=0,ignore_index=True)
    #     if "geometry" in unassinged_routes.columns.values.tolist():
    #         unassinged_routes.drop("geometry",axis=1,inplace=True)
    #     unassinged_routes.to_csv(
    #             os.path.join(results_folder,
    #             f"test_unassigned_roads.csv"),
    #             index=False)
    # del unassinged_routes

    print (mine_routes.columns)
    mine_routes[od_columns].to_csv(os.path.join(results_folder,
                f"test_routes_roads_border.csv"),index=False)


if __name__ == '__main__':
    CONFIG = load_config()
    main(CONFIG)