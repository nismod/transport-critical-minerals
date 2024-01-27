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
from shapely.geometry import LineString
from utils import *
from transport_cost_assignment import *
from tqdm import tqdm
tqdm.pandas()

def create_edges_from_nearest_node_joins(from_df,to_df,
                    from_id_column,to_id_column,
                    from_iso_column,to_iso_column,
                    from_mode,to_mode,
                    distance_threshold=2000):
    from_df.rename(columns={from_id_column:"from_id",from_iso_column:"from_iso_a3"},inplace=True)
    to_df.rename(columns={to_id_column:"to_id",to_iso_column:"to_iso_a3"},inplace=True)
    from_df["from_infra"] = from_mode
    to_df["to_infra"] = to_mode

    from_to_df = ckdnearest(from_df[["from_id","from_iso_a3","from_infra","geometry"]],
                            to_df[["to_id","to_iso_a3","to_infra","geometry"]])
    from_to_df["link_type"] = f"{from_mode}-{to_mode}"
    from_to_df = from_to_df[from_to_df["dist"] <= distance_threshold]

    if len(from_to_df.index) > 0:
        from_to_df.rename(columns = {"geometry":"from_geometry","dist":"length_m"},inplace=True)
        to_df.rename(columns = {"geometry":"to_geometry"},inplace=True)
        from_to_df = pd.merge(from_to_df,to_df[["to_id","to_geometry"]],how="left",on=["to_id"])
        from_to_df["geometry"] = from_to_df.progress_apply(
                                    lambda x:LineString([x.from_geometry,x.to_geometry]),
                                    axis=1)
        from_to_df.drop(["from_geometry","to_geometry"],axis=1,inplace=True)
    
    return from_to_df




def main(config):
    incoming_data_path = config['paths']['incoming_data']
    processed_data_path = config['paths']['data']
    
    epsg_meters = 3395 # To convert geometries to measure distances in meters


    maritime_nodes =  gpd.read_file(os.path.join(
                            processed_data_path,
                            "infrastructure",
                            "africa_maritime_network.gpkg"
                                ), layer="nodes"
                            )  
    maritime_nodes = maritime_nodes[maritime_nodes["infra"] == "port"]

    iww_nodes = gpd.read_file(os.path.join(
                            processed_data_path,
                            "infrastructure",
                            "africa_iww_network.gpkg"
                                ), layer="nodes"
                            ) 
    iww_nodes = iww_nodes[iww_nodes["infra"] == "IWW port"]

    rail_edges = gpd.read_file(os.path.join(
                            processed_data_path,
                            "infrastructure",
                            "africa_railways_network.gpkg"
                                ), layer="edges"
                    )
    rail_edges = rail_edges[rail_edges["status"] == "open"]
    rail_node_ids = list(set(rail_edges["from_id"].values.tolist() + rail_edges["to_id"].values.tolist()))
    rail_nodes = gpd.read_file(os.path.join(
                            processed_data_path,
                            "infrastructure",
                            "africa_railways_network.gpkg"
                                ), layer="nodes"
                    )
    rail_nodes = rail_nodes[(rail_nodes["id"].isin(rail_node_ids)) & (rail_nodes["infra"].isin(['stop','station']))]

    degree_df = rail_edges[["from_id","to_id"]].stack().value_counts().rename_axis('id').reset_index(name='degree')
    rail_nodes = pd.merge(rail_nodes,degree_df,how="left",on=["id"])
    rail_nodes.to_csv("test.csv")

    freight_facility_types = ["port","port (dry)",
                                "port (inland)",
                                "port (river)",
                                "road-rail transfer",
                                "container terminal",
                                "freight terminal",
                                "freight marshalling yard",
                                "mining",
                                "refinery"]
    rail_to_road_df = rail_nodes[
                                    (
                                        rail_nodes["facility"].isin(freight_facility_types)
                                    ) | (
                                        rail_nodes["degree"] == 1
                                    )
                                ] 
    road_nodes = gpd.read_parquet(os.path.join(
                            processed_data_path,
                            "infrastructure",
                            "africa_roads_nodes.geoparquet"))

    from_infras = [maritime_nodes,maritime_nodes,iww_nodes,iww_nodes,rail_to_road_df]
    from_modes = ["sea","sea","IWW","IWW","rail"]
    to_infras = [rail_nodes,road_nodes,rail_nodes,road_nodes,road_nodes]
    to_modes = ["rail","road","rail","road","road"]

    multi_df = []
    for idx,(f_df,t_df,f_m,t_m) in enumerate(zip(from_infras,to_infras,from_modes,to_modes)):
        f_t_df = create_edges_from_nearest_node_joins(
                            f_df.to_crs(epsg=epsg_meters),
                            t_df.to_crs(epsg=epsg_meters),
                            "id","id",
                            "iso3","iso3",
                            f_m,t_m)
        if len(f_t_df) > 0:
            multi_df.append(f_t_df)
            c_t_df = f_t_df[["from_id","to_id","from_infra",
                        "to_infra","from_iso_a3","to_iso_a3",
                        "link_type","geometry"]].copy()
            c_t_df.columns = ["to_id","from_id",
                        "to_infra","from_infra",
                        "to_iso_a3",
                        "from_iso_a3",
                        "link_type","geometry"]
            multi_df.append(c_t_df)

    multi_df = gpd.GeoDataFrame(
                    pd.concat(multi_df,axis=0,ignore_index=True),
                    geometry="geometry",crs=f"EPSG:{epsg_meters}")
    multi_df = multi_df.to_crs(epsg=4326)
    multi_df["id"] = multi_df.index.values.tolist()
    multi_df["id"] = multi_df.progress_apply(lambda x:f"intermodale_{x.id}",axis=1)
    print (multi_df)
    multi_df.to_file(os.path.join(
                            processed_data_path,
                            "infrastructure",
                            "africa_multimodal.gpkg"
                                ), 
                            layer="edges",
                            driver="GPKG"
                            )






if __name__ == '__main__':
    CONFIG = load_config()
    main(CONFIG)