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
    # Read a number of rail network files and convert them to networks first
    rail_paths = os.path.join(incoming_data_path,"africa_corridor_developments")
    input_descriptions = [
                            {
                                "project_name":"Conakry-Kankan Railway",
                                "points_file": None,
                                "lines_file": "guinea_rail.shp",
                                "points_layer":None,
                                "lines_layer":None,
                                "project_attributes":{"country":"Guinea",
                                                    "status":"abandoned",
                                                    "mode":"mixed",
                                                    "gauge":1000,
                                                    "line":"Conakry-Kankan Railway"},
                                "distance_threshold":300

                            },
                            {
                                "project_name":"Simandou railway project",
                                "points_file": "17D.1.gpkg",
                                "lines_file": "17D.1.gpkg",
                                "points_layer":"17d1_points",
                                "lines_layer":"17d1_lines",
                                "project_attributes":{"country":"Guinea",
                                                    "status":"construction",
                                                    "mode":"freight",
                                                    "gauge":1435,
                                                    "line":"Simandou-Conakry Railway (or Transguinean Railway)"},
                                "distance_threshold":300

                            },
                            {
                                "project_name":"Togo Rail",
                                "points_file": "togo_lines.gpkg",
                                "lines_file": "togo_lines.gpkg",
                                "points_layer":"nodes",
                                "lines_layer":"lines",
                                "project_attributes":{"country":"Togo",
                                                    "status":"proposed",
                                                    "mode":"mixed",
                                                    "gauge":1435,
                                                    "line":"Lomé – Cinkassé Railway"},
                                "distance_threshold":300

                            },
                            {
                                "project_name":"Ghana Burkina Faso",
                                "points_file": "Team.gpkg",
                                "lines_file": "Team.gpkg",
                                "points_layer":"team_points",
                                "lines_layer":"team_lines",
                                "project_attributes":{"country":"Ghana-Burkina Faso",
                                                    "status":"construction",
                                                    "mode":"mixed",
                                                    "gauge":1435,
                                                    "line":"Tema-Ouagadougou Railway"},

                            },
                            {
                                "project_name":"Isaka Kigali Gitega Railway",
                                "points_file": "Dar_es.gpkg",
                                "lines_file": "Dar_es.gpkg",
                                "points_layer":"dar_es_points",
                                "lines_layer":"dar_es_lines",
                                "project_attributes":{"country":"Tanzania-Rwanda-Burundi",
                                                    "status":"proposed",
                                                    "mode":"mixed",
                                                    "gauge":1435,
                                                    "line":"Isaka - Kigali - Gitega Standard Gauge Railway"},
                                "distance_threshold":300

                            },
                            {
                                "project_name":"East Africa Railway",
                                "points_file": "eastafrica_rail.gpkg",
                                "lines_file": "eastafrica_rail.gpkg",
                                "points_layer":"eastafrica_points",
                                "lines_layer":"eastafrica_edges",
                                "project_attributes":{"country":"Kenya-Uganda-DRC-Rwanda-Burundi-South Sudan-Ethopia",
                                                    "status":"proposed",
                                                    "mode":"mixed",
                                                    "gauge":1435,
                                                    "line":"East Africa Standard Gauge Railway"},
                                "distance_threshold":1000

                            },
                            {
                                "project_name":"Kinshasa Ilebo Railway",
                                "points_file": None,
                                "lines_file": "kinsasha_rail.gpkg",
                                "points_layer":None,
                                "lines_layer":None,
                                "project_attributes":{"country":"DRC",
                                					"status":"proposed",
                                                    "mode":"mixed",
                                                    "gauge":1435,
                                                    "line":"Kinshasa - Ilebo Railway"},
                                "distance_threshold":300

                            },
                            {
                                "project_name":"Tanzania Standard Gauge Railway",
                                "points_file": "tanzania_sgr_lines.gpkg",
                                "lines_file": "tanzania_sgr_lines.gpkg",
                                "points_layer":"nodes",
                                "lines_layer":"lines",
                                "project_attributes":{"country":"Tanzania",
                                                    "status":"contruction",
                                                    "mode":"mixed",
                                                    "gauge":1435,
                                                    "line":"Tanzania SGR Project"},
                                "distance_threshold":200

                            },

                        ]
    for inputs in input_descriptions:
        distance_threshold = inputs["distance_threshold"]
        if inputs["points_file"] is not None:
            df_points = gpd.read_file(os.path.join(rail_paths,
                                inputs["points_file"]),layer=inputs["points_layer"])
            df_points.columns = map(str.lower, df_points.columns)
        else:
            df_points = inputs["points_file"]
        
        df_lines = gpd.read_file(os.path.join(rail_paths,
                    inputs["lines_file"]),layer=inputs["lines_layer"])

        df_lines.columns = map(str.lower, df_lines.columns)
        
        if inputs["project_name"] == "Conakry-Kankan Railway":
            df_lines = df_lines[df_lines["name"] == "Conakry-Kankan Railway"]
            df_lines.rename(columns={"name":"line"},inplace=True)
            df_lines["status"] = df_lines.progress_apply(lambda x:get_line_status(str(x["comments"]).lower()),axis=1)
            df_lines["country"] = "Guinea"
            df_lines["mode"] = np.where(df_lines["type2"] == "Freight & Passenger","mixed","freight")
            df_lines["gauge"] = 1000
            df_lines = df_lines[["country","line","status","mode","gauge","geometry"]]
        # elif inputs["project_name"] in ["Simandou railway project","Lome Cinkasse Railway","Ghana Burkina Faso"]:
        elif inputs["project_name"] in ["Tanzania Standard Gauge Railway","Togo Rail"]:
            df_lines = df_lines[["country","line","status","mode","gauge","comment","geometry"]]
        else:
            df_lines = add_attributes(df_lines,inputs["project_attributes"])
            df_lines = df_lines[["country","line","status","mode","gauge","geometry"]]

        # df_lines = df_lines[["country","line","status","mode","gauge","geometry"]]
        df_crs = int(str(df_lines.crs).split(":")[1])
        network = create_network_from_nodes_and_edges(
                                    df_points,
                                    df_lines,"",
                                    geometry_precision=False)
        edges, nodes = components(network.edges,network.nodes,"node_id")
        edges = edges.set_crs(epsg=df_crs)
        nodes = nodes.set_crs(epsg=df_crs)
        max_edge_id = max([int(re.findall(r'\d+',v)[0]) for v in edges["edge_id"].values.tolist()])
        # Join network components which are very close
        all_components = list(set(nodes["component"].values.tolist()))
        component_dfs = []
        nodes = nodes.to_crs(epsg=epsg_meters)
        for c in range(len(all_components)-1):
            for p in range(c+1, len(all_components)):
                c1 = nodes[nodes["component"] == all_components[c]][["node_id","geometry"]]
                c2 = nodes[nodes["component"] == all_components[p]][["node_id","geometry"]]
                distances = ckdnearest(c1,c2)
                distances = distances[distances["dist"] <= distance_threshold]
                if len(distances.index) > 0:
                    distances.columns = ["from_node","from_geometry","node_id","dist"]
                    distances = pd.merge(distances,c2,how="left",on=["node_id"])
                    distances.rename(columns={"node_id":"to_node","geometry":"to_geometry"},inplace=True)
                    distances["geometry"] = distances.progress_apply(lambda x:LineString([x.from_geometry,x.to_geometry]),axis=1)
                    distances["edge_id"] = list(max_edge_id + 1 + distances.index.values)
                    distances["edge_id"] = distances.progress_apply(lambda x: f"e_{x.edge_id}",axis=1)
                    distances.drop(["from_geometry","to_geometry","dist"],axis=1,inplace=True)
                    distances = add_attributes(distances,inputs["project_attributes"])
                    distances = gpd.GeoDataFrame(distances,geometry="geometry",crs=epsg_meters)
                    distances = distances.to_crs(epsg=df_crs)
                    component_dfs.append(distances)
                    max_edge_id = max(max_edge_id + 1 + distances.index.values)
    
        nodes = nodes.to_crs(epsg=df_crs)
        if len(component_dfs) > 0:
            edges = pd.concat([edges]+component_dfs,
                            axis=0,ignore_index=True)

        edges, nodes = components(edges,nodes,"node_id")
        output_file_name = inputs["project_name"].lower().replace(" ","_")
        gpd.GeoDataFrame(nodes,
                geometry="geometry",
                crs=df_lines.crs).to_file(os.path.join(
                                rail_paths,
                                f"{output_file_name}.gpkg"),
                            layer="nodes",driver="GPKG")
        gpd.GeoDataFrame(edges,
                geometry="geometry",
                crs=df_lines.crs).to_file(os.path.join(
                                rail_paths,
                                f"{output_file_name}.gpkg"),
                            layer="edges",driver="GPKG")



if __name__ == '__main__':
    CONFIG = load_config()
    main(CONFIG)