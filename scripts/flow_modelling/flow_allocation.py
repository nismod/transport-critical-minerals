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
from tqdm import tqdm
tqdm.pandas()



def main(config):
    incoming_data_path = config['paths']['incoming_data']
    processed_data_path = config['paths']['data']
    output_data_path = config['paths']['results']

    results_folder = os.path.join(output_data_path,"flow_mapping")
    if os.path.exists(results_folder) == False:
        os.mkdir(results_folder)

    mine_id_col = "mine_cluster_mini"
    # cargo_type = "Dry bulk"
    cargo_type = "General cargo"
    trade_groupby_columns = ["export_country_code", 
                            "import_country_code", 
                            "export_continent","export_landlocked",
                            "import_continent","import_landlocked"]
    trade_value_columns = ["trade_value_thousandUSD","trade_quantity_tons"]
    od_columns = ["reference_mineral","process_binary",
                    "final_refined_stage",
                    "export_country_code",
                    "import_country_code",
                    "export_continent",
                    "import_continent",
                    "mine_output_tons",    
                    "mine_output_thousandUSD"]
    """Step 1: Get the input datasets
    """
    # Mine locations in Africa with the copper tonages
    mines_df = gpd.read_file(os.path.join(processed_data_path,
                                    "minerals",
                                    "copper_mines_tons.gpkg"))
    mines_df[mine_id_col] = mines_df.progress_apply(lambda x:f"mine_{x[mine_id_col]}",axis=1)
    # Select only tonnages > 0
    mines_df = mines_df[mines_df["mine_output_approx_copper"] > 0]

    # Get the global port network data for Dry Bulk transport
    # We assume CCG critical minerals are transported as Dry Bulk Cargo (General Cargo maybe)
    port_df = pd.read_csv(os.path.join(
                            processed_data_path,
                            "port_statistics",
                            "port_vessel_types_and_capacities.csv"
                                )
                            )
    port_df = port_df[port_df["vessel_type_main"] == cargo_type]
    # Get the max capacity port in each country
    port_df = port_df.sort_values(by=["annual_vessel_capacity_tons"],ascending=False)
    port_df = port_df.drop_duplicates(subset=["iso3"],keep="first")

    # Get the perferred port mapping of the landlocked countries outside Africa
    landlocked_port_df = pd.read_csv(
                            os.path.join(
                                processed_data_path,
                                "port_statistics",
                                "ccg_importing_landlocked_countries.csv"))
    # Create a list of all ports with unique port assumed for each country
    all_ports_df = pd.concat(
                    [port_df[["id","iso3"]],landlocked_port_df[["id","iso3"]]],
                    axis=0,ignore_index=True)
    del port_df, landlocked_port_df

    port_commodities_df = pd.read_csv(os.path.join(processed_data_path,
                                    "port_statistics",
                                    "port_known_commodities_traded.csv"))
    """Step 2: Identify all the countries outside Africa 
    """
    baci_codes_types = pd.read_csv(os.path.join(processed_data_path,
                            "baci",
                            "commodity_codes_refined_unrefined.csv"))
    # Amount of copper restricted for Biera and to disallow transhipment via Nacala ports
    # set_port_capacity = [("port137",0.27*1e6),("port784",0.0)]
    set_port_capacity = [("port137",0.27*1e6),("port784",0.0)]
    origin_id = "origin_id"
    destination_id = "destination_id"
    trade_ton_column = "mine_output_tons"
    trade_usd_column = "mine_output_thousandUSD"
    years = [2021,2030]
    for year in years:
        combined_trade_df = pd.read_csv(os.path.join(
                                                results_folder,
                                                f"mining_node_level_ods_{year}.csv"))
        mineral_classes = list(set(combined_trade_df.reference_mineral.values.tolist()))
        edges_flows_df = []
        nodes_flows_df = []
        flow_column_types = []
        for mineral_class in mineral_classes:
            c_t_df = combined_trade_df[combined_trade_df.reference_mineral == mineral_class]
            export_ports_africa = port_commodities_df[port_commodities_df[f"{mineral_class}_export_binary"] == 1]
            if len(export_ports_africa.index) == 0:
                export_ports_africa = port_commodities_df[
                                    port_commodities_df[
                                    f"{cargo_type.lower().replace(' ','_')}_annual_vessel_capacity_tons"] > 0
                                    ]
            if set_port_capacity is not None:
                for idx,(prt,cap) in enumerate(set_port_capacity):
                    export_ports_africa.loc[
                            export_ports_africa["id"] == prt,
                            f"{cargo_type.lower().replace(' ','_')}_annual_vessel_capacity_tons"] = cap

            export_port_ids = list(zip(export_ports_africa["id"].values.tolist(),
                            export_ports_africa[
                                f"{cargo_type.lower().replace(' ','_')}_annual_vessel_capacity_tons"]))
            network_graph = create_mines_to_port_network(mines_df,mine_id_col,
                        modes=["sea","intermodal","road","rail"],
                        intermodal_ports=export_ports_africa["id"].values.tolist(),
                        cargo_type=f"{cargo_type.lower().replace(' ','_')}",
                        port_to_land_capacity=export_port_ids
                        )

            network_graph[trade_ton_column] = 0
            mine_routes, unassinged_routes = od_flow_allocation_capacity_constrained(
                                                c_t_df,network_graph,
                                                trade_ton_column,"gcost_usd_tons",
                                                "id",origin_id,
                                                destination_id)
            # mine_routes = pd.read_parquet(
            #             os.path.join(results_folder,f"{mineral_class}.parquet"))
            if len(mine_routes) > 0:
                mine_routes = pd.concat(mine_routes,axis=0,ignore_index=True)
                c_t_df.rename(columns={trade_ton_column:"initial_tonnage"},inplace=True)
                mine_routes = pd.merge(mine_routes,
                                c_t_df[[origin_id,destination_id,"initial_tonnage"]],
                                how="left",on=[origin_id,destination_id])
                mine_routes[trade_usd_column] = mine_routes[
                                                            trade_usd_column]*mine_routes[
                                                            trade_ton_column]/mine_routes["initial_tonnage"]
                mine_routes.drop("initial_tonnage",axis=1,inplace=True)
                # mine_routes[c_t_df.columns.values.tolist()].to_csv("copper_ods_assigned.csv",index=False)
                del c_t_df

                # print (mine_routes[[origin_id,destination_id,trade_ton_column,"gcost_usd_tons"]])
                if "geometry" in mine_routes.columns.values.tolist():
                    mine_routes.drop("geometry",axis=1,inplace=True)
                # mine_routes = add_node_paths(mine_routes,network_graph,"id","edge_path")
                # mine_routes.to_csv("test.csv")
                mine_routes = convert_port_routes(mine_routes,"edge_path","node_path")
                mine_routes[[origin_id,destination_id] + od_columns + [
                                        "edge_path","node_path","full_edge_path",
                                        "full_node_path","gcost_usd_tons"]].to_parquet(
                        os.path.join(results_folder,f"{mineral_class}_flow_paths_{year}.parquet"),
                        index=False)

                for flow_column in [trade_ton_column,trade_usd_column]:
                    for refined_type in ["unrefined","refined"]:
                        for path_type in ["full_edge_path","full_node_path"]:
                            if refined_type == "unrefined":
                                f_df = get_flow_on_edges(
                                                mine_routes[mine_routes["process_binary"] == 0],
                                                "id",path_type,
                                                flow_column)
                            else:
                                f_df = get_flow_on_edges(
                                                mine_routes[mine_routes["process_binary"] == 1],
                                                "id",path_type,
                                                flow_column)
                            f_df.rename(columns={flow_column:f"{mineral_class}_{refined_type}_{flow_column}"},
                                    inplace=True)
                            if path_type == "full_edge_path":
                                edges_flows_df.append(f_df)
                            else:
                                nodes_flows_df.append(f_df)
                        flow_column_types.append(f"{mineral_class}_{refined_type}_{flow_column}")

        for mineral_class in mineral_classes:
            for path_type in ["edges","nodes"]:
                if path_type == "edges":
                    flows_df = pd.concat(edges_flows_df,axis=0,ignore_index=True).fillna(0)
                else:
                    flows_df = pd.concat(nodes_flows_df,axis=0,ignore_index=True).fillna(0)
                flows_df = flows_df.groupby(["id"]).agg(
                            dict(
                                    [(ft,"sum") for ft in flow_column_types]
                                )
                            ).reset_index()

                for flow_column in [trade_ton_column,trade_usd_column]:
                    flows_df[
                            f"{mineral_class}_{flow_column}"
                            ] = flows_df[
                                    f"{mineral_class}_unrefined_{flow_column}"
                                ] + flows_df[f"{mineral_class}_refined_{flow_column}"]
                    
                    # flows_df = flows_df.reset_index()
                flows_df = add_geometries_to_flows(flows_df,
                                    merge_column="id",
                                    modes=["rail","sea","road"],
                                    layer_type=path_type)
                if path_type == "nodes":
                    flows_df = add_node_degree_to_flows(flows_df,mineral_class)

                print (flows_df)
                
                flows_df.to_file(os.path.join(results_folder,
                                        f"{mineral_class}_flows_{year}.gpkg"),
                                        layer=path_type,driver="GPKG")









if __name__ == '__main__':
    CONFIG = load_config()
    main(CONFIG)