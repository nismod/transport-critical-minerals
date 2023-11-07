#!/usr/bin/env python
# coding: utf-8
# (1) Merge three datasets; (2)Add ISO3 (4) extraxt non_intersected
import sys
import os
import re
import pandas as pd
import geopandas as gpd
import igraph as ig
from shapely.geometry import LineString
from utils import *
from tqdm import tqdm
tqdm.pandas()

def main(config):
    incoming_data_path = config['paths']['incoming_data']
    processed_data_path = config['paths']['data']
    
    epsg_meters = 3395 # To convert geometries to measure distances in meters
    # We first took some data on IWW ports from different datasets and combined them
    # We produced a final version of the selected ports by manual cleaning
    df_ports = pd.read_excel(os.path.join(incoming_data_path,
                                    "IWW_ports",
                                    "africa_IWW_ports.xlsx"),
                            sheet_name="selected_ports")
    df_ports["geometry"] = gpd.points_from_xy(
                            df_ports["lon"],df_ports["lat"])
    df_ports["infra"] = "IWW port"
    # df_ports["id"] = df_ports.index.values.tolist()
    # df_ports["id"] = df_ports.progress_apply(lambda x: f"iwwn{x.id}",axis=1)
    df_ports = gpd.GeoDataFrame(df_ports,geometry="geometry",crs="EPSG:4326")
    # df_ports.to_file(os.path.join(incoming_data_path,
    #                             "IWW_ports",
    #                             "africa_IWW_ports.gpkg"),layer="nodes",driver="GPKG")

    # Add lines from known known connections between Lake ports
    df_lake_routes = pd.read_excel(os.path.join(incoming_data_path,
                                    "IWW_ports",
                                    "africa_IWW_ports.xlsx"),
                            sheet_name="known_connections")
    df_lake_routes = pd.merge(df_lake_routes,
                    df_ports[["name","geometry"]],
                    how="left",left_on=["from_port"],right_on=["name"])
    df_lake_routes.drop("name",axis=1,inplace=True)
    df_lake_routes.rename(columns={"geometry":"from_geometry"},inplace=True)
    df_lake_routes = pd.merge(df_lake_routes,
                    df_ports[["name","geometry"]],
                    how="left",left_on=["to_port"],right_on=["name"])
    df_lake_routes.drop("name",axis=1,inplace=True)
    df_lake_routes.rename(columns={"geometry":"to_geometry"},inplace=True)
    df_lake_routes["geometry"] = df_lake_routes.progress_apply(
                                    lambda x:LineString([x.from_geometry,x.to_geometry]),
                                    axis=1)
    df_lake_routes.drop(["from_geometry","to_geometry"],axis=1,inplace=True)
    
    # Add lines for Congo ports based on the routing along the rivers
    df_congo_rivers = gpd.read_file(os.path.join(incoming_data_path,
                            "IWW_ports",
                            "edges_port_IWW_af.gpkg"))
    # df_congo_ports = df_ports[df_ports["iso3"].isin(["CAF","COD","COG"])]
    # lake_ids = list(set(df_lake_routes["from_id"].values.tolist() + df_lake_routes["to_id"].values.tolist())) 
    # df_congo_ports = df_congo_ports[~df_congo_ports["id"].isin(lake_ids)]

    df_south_sudan = gpd.read_file(os.path.join(incoming_data_path,
                            "IWW_ports",
                            "hotosm_ssd_waterways.gpkg"))
    df_south_sudan = df_south_sudan.loc[df_south_sudan.geometry.geometry.type =='LineString']
    df_south_sudan = df_south_sudan[df_south_sudan["waterway"] == "river"]

    df_routes = gpd.GeoDataFrame(pd.concat([df_lake_routes[["geometry"]],
                        df_congo_rivers[["geometry"]],
                        df_south_sudan[["geometry"]]],axis=0,ignore_index=True),
                        geometry="geometry",crs="EPSG:4326")
    network = create_network_from_nodes_and_edges(df_ports.to_crs(epsg=epsg_meters),
                        df_routes.to_crs(epsg=epsg_meters),"iww",snap_distance=6000)
    edges = network.edges.set_crs(epsg=epsg_meters)
    nodes = network.nodes.set_crs(epsg=epsg_meters)
    edges, nodes = components(edges,nodes,"node_id")

    # Get the specific routes that connect IWW ports in Congo basin
    # Reject other routes
    routing_edges = edges[["from_node","to_node","edge_id","component","geometry"]]
    routing_edges["distance"] = routing_edges.geometry.length
    G = ig.Graph.TupleList(routing_edges.itertuples(index=False), edge_attrs=list(routing_edges.columns)[2:])
    # print (G)

    all_edges = []
    ports = nodes[nodes["infra"] == "IWW port"]["node_id"].values.tolist()
    for o in range(len(ports)-1):
        origin = ports[o]
        destinations = ports[o+1:]
        e,c = network_od_path_estimations(G,origin,destinations,"distance","edge_id")
        all_edges += e

    all_edges = list(set([item for sublist in all_edges for item in sublist]))
    africa_edges = edges[edges["edge_id"].isin(all_edges)]
    
    all_nodes = list(set(africa_edges["from_node"].values.tolist() + africa_edges["to_node"].values.tolist()))
    africa_nodes = nodes[nodes["node_id"].isin(all_nodes)]

    africa_edges = africa_edges.to_crs(epsg=4326)
    africa_nodes = africa_nodes.to_crs(epsg=4326)
    africa_nodes.to_file(os.path.join(
                            processed_data_path,
                            "infrastructure",
                            "africa_iww_network.gpkg"),
                        layer="nodes",driver="GPKG")
    africa_edges.to_file(os.path.join(processed_data_path,
                            "infrastructure",
                            "africa_iww_network.gpkg"),
                        layer="edges",driver="GPKG")

    

if __name__ == '__main__':
    CONFIG = load_config()
    main(CONFIG)