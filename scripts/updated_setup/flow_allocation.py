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



def main(config,reference_mineral,scenario,year,percentile,efficient_scale):
    incoming_data_path = config['paths']['incoming_data']
    processed_data_path = config['paths']['data']
    output_data_path = config['paths']['results']

    input_folder = os.path.join(output_data_path,"flow_node_ods")
    results_folder = os.path.join(output_data_path,"flow_od_paths")
    os.makedirs(results_folder,exist_ok=True)

    # cargo_type = "Dry bulk"
    cargo_type = "General cargo"
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
    scenario = scenario.replace(" ","_")
    # print (year,percentile)
    if year == 2022:
        od_file_name = f"mining_city_node_level_ods_{year}_{percentile}.csv"
        # mine_layer = f"{reference_mineral}"
    else:
        od_file_name = f"mining_city_node_level_ods_{scenario}_{year}_{percentile}_{efficient_scale}.csv"
        # mine_layer = f"{reference_mineral}_{percentile}"

    # print (od_file_name)

    combined_trade_df = pd.read_csv(os.path.join(
                                input_folder,
                                od_file_name),dtype=data_type)
    od_locations = list(
                        set(
                            combined_trade_df["origin_id"].values.tolist() + combined_trade_df["destination_id"].values.tolist()
                            )
                        )

    # Mine locations in Africa with the mineral tonages
    mine_id_col = "mine_id"
    mines_df = get_mine_layer(reference_mineral,year,percentile,
                        mine_id_col=mine_id_col,return_columns=[mine_id_col,"geometry"])
    mines_df = mines_df[mines_df[mine_id_col].isin(od_locations)]

    # Population locations for urban cities
    pop_id_col = "city_id"
    un_pop_df = gpd.read_file(os.path.join(processed_data_path,
                                    "admin_boundaries",
                                    "un_urban_population",
                                    "un_pop_df.gpkg"))
    # un_pop_df = un_pop_df[un_pop_df["CONTINENT"] == "Africa"]
    un_pop_df = un_pop_df[un_pop_df[pop_id_col].isin(od_locations)]
    un_pop_df = un_pop_df[[pop_id_col,"geometry"]]

    port_commodities_df = pd.read_csv(os.path.join(processed_data_path,
                                    "port_statistics",
                                    "port_known_commodities_traded.csv"))
    """Step 2: Identify all the countries outside Africa 
    """
    # Amount of copper restricted for Biera and to disallow transhipment via Nacala ports
    set_port_capacity = [("port137",0.27*1e6),("port784",0.0)]
    origin_id = "origin_id"
    destination_id = "destination_id"
    final_ton_column = "final_stage_production_tons"
    initial_ton_column = "initial_stage_production_tons"
    edges_flows_df = []
    nodes_flows_df = []
    flow_column_types = []  
    c_t_df = combined_trade_df[combined_trade_df.reference_mineral == reference_mineral]
    export_ports_africa = port_commodities_df[port_commodities_df[f"{reference_mineral}_export_binary"] == 1]
    if len(export_ports_africa.index) == 0:
        export_ports_africa = port_commodities_df[
                            port_commodities_df[
                            "total_annual_vessel_capacity_tons"] > 0
                            ]
    set_port_capacity = None
    if set_port_capacity is not None:
        for idx,(prt,cap) in enumerate(set_port_capacity):
            export_ports_africa.loc[
                    export_ports_africa["id"] == prt,
                    f"total_annual_vessel_capacity_tons"] = cap

    export_port_ids = list(zip(export_ports_africa["id"].values.tolist(),
                    export_ports_africa[
                        "total_annual_vessel_capacity_tons"]))
    network_graph = create_mines_and_cities_to_port_network(
                            mines_df,mine_id_col,
                            un_pop_df,pop_id_col,
                            modes=["sea","intermodal","road","rail"],
                            intermodal_ports=export_ports_africa["id"].values.tolist(),
                            cargo_type=f"{cargo_type.lower().replace(' ','_')}",
                            port_to_land_capacity=export_port_ids,
                            distance_threshold=1500
                            )
    network_graph[final_ton_column] = 0
    mine_routes, unassinged_routes, network_graph = od_flow_allocation_capacity_constrained(
                                        c_t_df,network_graph,
                                        final_ton_column,"gcost_usd_tons",
                                        "distance_km","time_hr","land_border_cost_usd_tons",
                                        "id",origin_id,
                                        destination_id)
    network_graph = network_graph[network_graph[final_ton_column] > 0]
    network_graph.to_csv(os.path.join(results_folder,
                f"{reference_mineral}_total_flows_{year}_{percentile}.csv"),
                index=False)
    del network_graph
    if len(unassinged_routes) > 0:
        unassinged_routes = pd.concat(unassinged_routes,axis=0,ignore_index=True)
        if "geometry" in unassinged_routes.columns.values.tolist():
            unassinged_routes.drop("geometry",axis=1,inplace=True)
        unassinged_routes.to_csv(
                os.path.join(results_folder,
                f"{reference_mineral}_unassigned_flow_paths_{year}_{percentile}.csv"),
                index=False)
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
        if year > 2022:
            file_name = f"{reference_mineral}_flow_paths_{scenario}_{year}_{percentile}_{efficient_scale}.parquet"
        else:
            file_name = f"{reference_mineral}_flow_paths_{year}_{percentile}.parquet"

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
    try:
        reference_mineral = str(sys.argv[1])
        scenario = str(sys.argv[2])
        year = int(sys.argv[3])
        percentile = str(sys.argv[4])
        efficient_scale = str(sys.argv[5])
    except IndexError:
        print("Got arguments", sys.argv)
        exit()
    main(CONFIG,reference_mineral,scenario,year,percentile,efficient_scale)