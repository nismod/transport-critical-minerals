# TZA - Dar es Salaam
# NAM - Waalis Bay
# ZAF - Durban
# AGO - Lobito
# MOZ - Biera 

#!/usr/bin/env python
# coding: utf-8
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
    
    epsg_meters = 3395 # To convert geometries to measure distances in meters
    baci_codes = [260300,740200,740311]
    mine_id_col = "mine_cluster_mini"
    mine_iso = "shapeGroup_primary_admin0"
    port_utilization = gpd.read_file(os.path.join(
                            processed_data_path,
                            "infrastructure",
                            "global port utilisation.gpkg"
                                ), layer="nodes"
                            )
    port_utilization["max_capacity"] = port_utilization["count_max"]*port_utilization["dwt_max"]

    origin_port_ids = ["port311","port1381","port656","port137","port278"]

    global_ports_nodes =  gpd.read_file(os.path.join(
                            processed_data_path,
                            "infrastructure",
                            "global_maritime_network.gpkg"
                                ), layer="nodes"
                            )
    global_ports_nodes = pd.merge(global_ports_nodes,
                            port_utilization[["id","max_capacity"]],
                            how="left",on=["id"]).fillna(0)
    origin_ports = global_ports_nodes[global_ports_nodes["id"].isin(origin_port_ids)]
    global_ports_nodes = global_ports_nodes.sort_values(by="max_capacity",ascending=False)
    global_ports_nodes = global_ports_nodes.drop_duplicates(subset=["iso3","Continent_Code"],keep="first")
    global_ports_nodes = global_ports_nodes[
                            (global_ports_nodes["max_capacity"] > 0
                                ) & (
                            global_ports_nodes["Continent_Code"] != "AF")]

    global_ports_edges =  gpd.read_file(os.path.join(
                            processed_data_path,
                            "infrastructure",
                            "global_maritime_network.gpkg"
                                ), layer="edges"
                            )
    global_ports_edges = global_ports_edges.to_crs(epsg=epsg_meters)
    global_ports_edges["length_m"] = global_ports_edges.geometry.length
    
    trade_df = pd.read_csv(os.path.join(processed_data_path,
                            "baci",
                            "ccg_country_critical_mineral_commodity_trade_all.csv"))

    mines_df = gpd.read_file(os.path.join(processed_data_path,
                                    "Minerals",
                                    "copper_mines_tons.gpkg"))
    mines_df = mines_df[mines_df["mine_output_approx_copper"] > 0]
    mine_countries = list(set(mines_df["shapeGroup_primary_admin0"].values.tolist()))
    copper_trade = trade_df[(
                    trade_df["from_iso"].isin(mine_countries)
                    ) & (trade_df["k"].isin(baci_codes))]
    # print (copper_trade)
    cols = ["t","i","j","from_country",
            "from_iso","to_country","to_iso","refined"]
    copper_trade["refined"] = np.where(copper_trade["k"] == 260300,0,1)
    copper_trade = copper_trade.groupby(cols)[["v","q_mod"]].sum().reset_index()
    # copper_trade.to_csv("test.csv")
    destination_isos = list(set(copper_trade["to_iso"].values.tolist()))
    global_ports_nodes = global_ports_nodes[global_ports_nodes["iso3"].isin(destination_isos)]
    # print (global_ports_nodes)

    """Find port-to-port route distances
    """
    destination_port_ids = global_ports_nodes["id"].values.tolist()

    graph = create_igraph_from_dataframe(
                    global_ports_edges[["from_id","to_id","id","length_m"]])
    
    od_matrix = pd.DataFrame(origin_port_ids,
                    columns=["origin_id"]).merge(
                pd.DataFrame(destination_port_ids,columns=["destination_id"]), how='cross')
    # print (od_matrix)
    port_routes = network_od_paths_assembly(od_matrix, graph,
                                "length_m","id")
    # print (port_routes)
    port_routes = pd.merge(port_routes,
                    origin_ports[["id","iso3"]],
                    how="left",left_on=["origin_id"],right_on=["id"])
    port_routes.rename(columns={"iso3":"from_iso_a3"},inplace=True)
    port_routes.drop("id",axis=1,inplace=True)
    port_routes = pd.merge(port_routes,
                    global_ports_nodes[["id","iso3"]],
                    how="left",left_on=["destination_id"],right_on=["id"])
    port_routes.rename(columns={"iso3":"to_iso_a3"},inplace=True)
    port_routes.drop("id",axis=1,inplace=True)

    port_routes["rank"] = port_routes.groupby(
                            ["destination_id"])["length_m"].rank(method="dense", ascending=True)
    port_routes.to_csv(os.path.join(
                        processed_data_path,
                        "flow_paths","port_routes.csv"),index=False)

    mines_in_port_countries = mines_df[
                                mines_df["shapeGroup_primary_admin0"
                                ].isin(origin_ports["iso3"].values.tolist())]
    mines_not_in_port_countries = mines_df[~
                                mines_df["shapeGroup_primary_admin0"
                                ].isin(origin_ports["iso3"].values.tolist())]
    
    rail_edges = gpd.read_file(os.path.join(
                            processed_data_path,
                            "infrastructure",
                            "africa_railways_network.gpkg"
                                ), layer="edges"
                    )
    rail_edges.rename(columns={"from_iso":"from_iso_a3","to_iso":"to_iso_a3"},inplace=True)
    rail_edges = transport_cost_assignment_function(rail_edges,"rail")
    rail_edges = rail_edges[rail_edges["status"] == "open"]
    # rail_edges.head(50).to_csv("test.csv")
    rail_node_ids = list(set(rail_edges["from_id"].values.tolist() + rail_edges["to_id"].values.tolist()))
    rail_nodes = gpd.read_file(os.path.join(
                            processed_data_path,
                            "infrastructure",
                            "africa_railways_network.gpkg"
                                ), layer="nodes"
                    )
    rail_nodes = rail_nodes[(rail_nodes["id"].isin(rail_node_ids)) & (rail_nodes["type"].isin(['stop','station']))]

    """Find mines attached to rail and roads
    """
    mine_rail_intersects = gpd.sjoin_nearest(mines_df[[mine_id_col,mine_iso,"geometry"]].to_crs(epsg=epsg_meters),
                            rail_nodes[["id","facility","geometry"]].to_crs(epsg=epsg_meters),
                            how="left",distance_col="distance").reset_index()
    mine_rail_intersects = mine_rail_intersects[mine_rail_intersects["distance"] < 50]
    mine_rail_intersects = mine_rail_intersects.drop_duplicates(subset=[mine_id_col],keep="first")
    print (mine_rail_intersects)

    road_edges = gpd.read_parquet(os.path.join(
                            processed_data_path,
                            "infrastructure",
                            "africa_roads_edges.geoparquet"))
    road_edges = transport_cost_assignment_function(road_edges,"road")
    # road_edges.head(50).to_csv("test2.csv")
    road_nodes = gpd.read_parquet(os.path.join(
                            processed_data_path,
                            "infrastructure",
                            "africa_roads_nodes.geoparquet"))
    mine_road_intersects = gpd.sjoin_nearest(mines_df[[mine_id_col,mine_iso,"geometry"]].to_crs(epsg=epsg_meters),
                            road_nodes[["road_id","geometry"]].to_crs(epsg=epsg_meters),
                            how="left",distance_col="distance").reset_index()
    mine_road_intersects = mine_road_intersects.drop_duplicates(subset=[mine_id_col],keep="first")
    print (mine_road_intersects)

    """Find port to rail and road connections
    """
    rail_nodes.rename(columns={"id":"rail_id"},inplace=True)
    port_rail_connections = ckdnearest(origin_ports[["id","iso3","geometry"]].to_crs(epsg=epsg_meters),
                                  rail_nodes[["rail_id","geometry"]].to_crs(epsg=epsg_meters))
    print (port_rail_connections)
    # port_rail_connections = port_rail_connections[port_rail_connections["distance"] < 100]
    # print (port_rail_connections)
    port_road_connections = ckdnearest(origin_ports[["id","iso3","geometry"]].to_crs(epsg=epsg_meters),
                                  road_nodes[["road_id","geometry"]].to_crs(epsg=epsg_meters))
    print (port_road_connections)

    rail_od_matrix = pd.DataFrame(port_rail_connections["rail_id"].values.tolist(),
                    columns=["origin_id"]).merge(
                    pd.DataFrame(mine_rail_intersects["id"].values.tolist(),
                    columns=["destination_id"]), how='cross')
    road_od_matrix = pd.DataFrame(port_road_connections["road_id"].values.tolist(),
                        columns=["origin_id"]).merge(
                    pd.DataFrame(mine_road_intersects["road_id"].values.tolist(),
                        columns=["destination_id"]), how='cross')

    graph = create_igraph_from_dataframe(
                    rail_edges[["from_id","to_id","id","gcost_tons"]])
    
    # print (od_matrix)
    rail_routes = network_od_paths_assembly(rail_od_matrix, graph,
                                "gcost_tons","id")

    graph = create_igraph_from_dataframe(
                    road_edges[["from_id","to_id","id","gcost_tons"]])
    
    # print (od_matrix)
    road_routes = network_od_paths_assembly(road_od_matrix, graph,
                                "gcost_tons","id")
    rail_routes = rail_routes[rail_routes["gcost_tons"] > 0]
    rail_routes = pd.merge(rail_routes,port_rail_connections[["id","iso3","rail_id"]],
                    how="left",left_on=["origin_id"],right_on=["rail_id"])
    rail_routes.rename(columns={"id":"port_id","iso3":"to_iso_a3"},inplace=True)
    rail_routes.drop("rail_id",axis=1,inplace=True)
    rail_routes = pd.merge(rail_routes,mine_rail_intersects[[mine_id_col,mine_iso,"id"]],
                    how="left",left_on=["destination_id"],right_on=["id"])
    rail_routes.rename(columns={mine_iso:"from_iso_a3"},inplace=True)
    print (rail_routes)
    rail_routes.drop("id",axis=1,inplace=True)
    rail_routes["rank"] = rail_routes.groupby(["destination_id"])["gcost_tons"].rank(
                            method="dense", 
                            ascending=False)
    rail_routes.to_csv(os.path.join(
                        processed_data_path,
                        "flow_paths","rail_routes.csv"),index=False)
    rail_routes.drop("edge_path",axis=1,inplace=True)
    rail_routes.to_csv(os.path.join(
                        processed_data_path,
                        "flow_paths","rail_routes_no_paths.csv"),index=False)
    road_routes = pd.merge(road_routes,port_road_connections[["id","iso3","road_id"]],
                    how="left",left_on=["origin_id"],right_on=["road_id"])
    road_routes.rename(columns={"id":"port_id","iso3":"to_iso_a3"},inplace=True)
    road_routes.drop("road_id",axis=1,inplace=True)
    road_routes = pd.merge(road_routes,mine_road_intersects[[mine_id_col,mine_iso,"road_id"]],
                    how="left",left_on=["destination_id"],right_on=["road_id"])
    road_routes.rename(columns={mine_iso:"from_iso_a3"},inplace=True)
    road_routes.drop("road_id",axis=1,inplace=True)
    road_routes = road_routes[road_routes["gcost_tons"] > 0]
    road_routes["rank"] = road_routes.groupby(["destination_id"])["gcost_tons"].rank(
                            method="dense", 
                            ascending=False)
    road_routes.to_csv(os.path.join(
                        processed_data_path,
                        "flow_paths","road_routes.csv"),index=False)
    road_routes.drop("edge_path",axis=1,inplace=True)
    road_routes.to_csv(os.path.join(
                        processed_data_path,
                        "flow_paths","road_routes_no_paths.csv"),index=False)







if __name__ == '__main__':
    CONFIG = load_config()
    main(CONFIG)