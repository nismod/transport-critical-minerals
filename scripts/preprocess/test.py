#!/usr/bin/env python
# coding: utf-8
import sys
import os
import re
import json
import pandas as pd
import geopandas as gpd
import snkit
from shapely.geometry import LineString
from updated_utils import *
from tqdm import tqdm
tqdm.pandas()

def get_line_status(x):
    if "abandon" in x:
        return "abandoned"
    else:
        return "open"

def add_attributes(dataframe,columns_attributes):
    for column_name,attribute_value in columns_attributes.items():
        dataframe[column_name] = attribute_value

    return dataframe


def main(config):
    incoming_data_path = config['paths']['incoming_data']
    processed_data_path = config['paths']['data']
    
    epsg_meters = 3395 # To convert geometries to measure distances in meters
    distance_threshold = 1000 # Join point within 50cm of each other
    # Read a number of rail network files and convert them to networks first
    rail_paths = os.path.join(incoming_data_path,"africa_corridor_developments")

    condition = False
    if condition is True:
        del_lines = ["SGR Phases 3 - 5","Morogoro - Makutupora SGR","Dar es Salaam - Morogoro SGR"]
        # Read the rail edges data for Africa
        rail_edges = json.load(open(os.path.join(incoming_data_path,
                                "africa_rail_network",
                                "network_data",
                                "africa_rail_network.geojson")))
        rail_edges = convert_json_geopandas(rail_edges)
        rail_edges = rail_edges.to_crs(epsg=4326)
        tza_lines = rail_edges[rail_edges["line"].isin(del_lines)]

        rail_nodes = json.load(open(os.path.join(incoming_data_path,
                                "africa_rail_network",
                                "network_data",
                                "africa_rail_stops.geojson")))
        rail_nodes = convert_json_geopandas(rail_nodes)
        rail_nodes = rail_nodes.to_crs(epsg=4326)

        tza_node_ids = list(set(tza_lines["source"].values.tolist() + tza_lines["target"].values.tolist()))
        tza_nodes = rail_nodes[rail_nodes["oid"].isin(tza_node_ids)]

        rail_edges[~rail_edges["line"].isin(del_lines)].to_file(os.path.join(incoming_data_path,
                                "africa_rail_network",
                                "network_data",
                                "africa_railways.gpkg"),layer="edges",driver="GPKG")

        rail_nodes[~rail_nodes["oid"].isin(tza_node_ids)].to_file(os.path.join(incoming_data_path,
                                "africa_rail_network",
                                "network_data",
                                "africa_railways.gpkg"),layer="nodes",driver="GPKG")

        tza_lines = tza_lines[tza_lines["line"] != "SGR Phases 3 - 5"]
        tza_node_ids = list(set(tza_lines["source"].values.tolist() + tza_lines["target"].values.tolist()))
        tza_nodes = tza_nodes[tza_nodes["oid"].isin(tza_node_ids)]

        tza_l = gpd.read_file(os.path.join(rail_paths,"tanzania_sgr_v2.gpkg"))
        tza_lines = pd.concat([tza_lines,tza_l],axis=0,ignore_index=True)
        tza_lines["gauge"] = 1435
        tza_lines["mode"] = "mixed"
        tza_lines["country"] = "Tanzania"
        tza_lines["oid"] = tza_lines.index.values.tolist()
        tza_lines.drop("objectid",axis=1,inplace=True)
        gpd.GeoDataFrame(tza_lines,
                        geometry="geometry",
                        crs="EPSG:4326").to_file(os.path.join(rail_paths,
                            "tanzania_sgr_lines.gpkg"),layer="lines",driver="GPKG")
        gpd.GeoDataFrame(tza_nodes,
                        geometry="geometry",
                        crs="EPSG:4326").to_file(os.path.join(rail_paths,
                            "tanzania_sgr_lines.gpkg"),layer="nodes",driver="GPKG")
    if condition is True:
        del_lines = ["Lomé to Lomé Port",
                        "Lomé to Blitta",
                        "Diamond Cement Alfao",
                        "Dalavé-Adétikopé",
                        "Tabligbo-Dalavé",
                        "Route into Lomé station"]
        # Read the rail edges data for Africa
        rail_edges = gpd.read_file(os.path.join(incoming_data_path,
                                "africa_rail_network",
                                "network_data",
                                "africa_railways.gpkg"),layer="edges")
        togo_lines = rail_edges[rail_edges["line"].isin(del_lines)]

        rail_nodes = gpd.read_file(os.path.join(incoming_data_path,
                                "africa_rail_network",
                                "network_data",
                                "africa_railways.gpkg"),layer="nodes")

        togo_node_ids = list(set(togo_lines["source"].values.tolist() + togo_lines["target"].values.tolist()))
        togo_nodes = rail_nodes[rail_nodes["oid"].isin(togo_node_ids)]

        rail_edges[~rail_edges["line"].isin(del_lines)].to_file(os.path.join(incoming_data_path,
                                "africa_rail_network",
                                "network_data",
                                "africa_railways_v2.gpkg"),layer="edges",driver="GPKG")

        rail_nodes[~rail_nodes["oid"].isin(togo_node_ids)].to_file(os.path.join(incoming_data_path,
                                "africa_rail_network",
                                "network_data",
                                "africa_railways_v2.gpkg"),layer="nodes",driver="GPKG")


        togo_l = gpd.read_file(os.path.join(rail_paths,"lome_cinkasse_railway.gpkg"),layer="edges")
        togo_n = gpd.read_file(os.path.join(rail_paths,"lome_cinkasse_railway.gpkg"),layer="nodes")
        
        togo_lines = togo_lines[["country","line","status","mode","gauge","comment","geometry"]]
        togo_lines = pd.concat([togo_lines,togo_l],axis=0,ignore_index=True)
        gpd.GeoDataFrame(togo_lines,
                        geometry="geometry",
                        crs="EPSG:4326").to_file(os.path.join(rail_paths,
                            "togo_lines.gpkg"),layer="lines",driver="GPKG")

        togo_nodes = togo_nodes[["country","type","name","facility","status","comment","geometry"]]
        togo_nodes = pd.concat([togo_nodes,togo_n],axis=0,ignore_index=True)
        gpd.GeoDataFrame(togo_nodes,
                        geometry="geometry",
                        crs="EPSG:4326").to_file(os.path.join(rail_paths,
                            "togo_lines.gpkg"),layer="nodes",driver="GPKG")
    condition = True
    if condition is True:
        del_country = ["Guinea"]
        # Read the rail edges data for Africa
        rail_edges = gpd.read_file(os.path.join(incoming_data_path,
                                "africa_rail_network",
                                "network_data",
                                "africa_railways_v0.gpkg"),layer="edges")
        guinea_lines = rail_edges[rail_edges["country"].isin(del_country)]

        rail_nodes = gpd.read_file(os.path.join(incoming_data_path,
                                "africa_rail_network",
                                "network_data",
                                "africa_railways_v0.gpkg"),layer="nodes")

        guinea_node_ids = list(set(guinea_lines["source"].values.tolist() + guinea_lines["target"].values.tolist()))
        guinea_nodes = rail_nodes[rail_nodes["oid"].isin(guinea_node_ids)]

        rail_edges[~rail_edges["country"].isin(del_country)].to_file(os.path.join(incoming_data_path,
                                "africa_rail_network",
                                "network_data",
                                "africa_railways_v2.gpkg"),layer="edges",driver="GPKG")

        rail_nodes[~rail_nodes["oid"].isin(guinea_node_ids)].to_file(os.path.join(incoming_data_path,
                                "africa_rail_network",
                                "network_data",
                                "africa_railways_v2.gpkg"),layer="nodes",driver="GPKG")


        guinea_l = gpd.read_file(os.path.join(rail_paths,"conakry-kankan_railway.gpkg"),layer="edges")
        guinea_n = gpd.read_file(os.path.join(rail_paths,"conakry-kankan_railway.gpkg"),layer="nodes")
        guinea_l.drop(["edge_id","from_node","to_node","component"],axis=1,inplace=True)
        guinea_n.drop(["node_id","component"],axis=1,inplace=True)

        
        guinea_lines = pd.concat([guinea_lines,guinea_l],axis=0,ignore_index=True)
        guinea_nodes = pd.concat([guinea_nodes,guinea_n],axis=0,ignore_index=True)

        network = create_network_from_nodes_and_edges(
                                            guinea_nodes,
                                            guinea_lines,"",
                                            geometry_precision=False)
        edges, nodes = components(network.edges,network.nodes,"node_id")
        gpd.GeoDataFrame(edges,
                        geometry="geometry",
                        crs="EPSG:4326").to_file(os.path.join(rail_paths,
                            "guinea_lines.gpkg"),layer="edges",driver="GPKG")

        gpd.GeoDataFrame(nodes,
                        geometry="geometry",
                        crs="EPSG:4326").to_file(os.path.join(rail_paths,
                            "guinea_lines.gpkg"),layer="nodes",driver="GPKG")




if __name__ == '__main__':
    CONFIG = load_config()
    main(CONFIG)