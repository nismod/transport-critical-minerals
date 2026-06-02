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
from tqdm import tqdm
tqdm.pandas()



def main(config):
    incoming_data_path = config['paths']['incoming_data']
    processed_data_path = config['paths']['data']
    
    epsg_meters = 3395
    afdb_edges = gpd.read_parquet(os.path.join(
                            incoming_data_path,
                            "africa_roads",
                            "africa_db_roads_edges.geoparquet"))
    afdb_edges = afdb_edges[~afdb_edges["corridor_name"].isna()]
    df = afdb_edges[["osm_way_id","corridor_name"]].drop_duplicates(subset=["osm_way_id","corridor_name"],keep="first")
    df = df.groupby(["osm_way_id"])["corridor_name"].apply(list).reset_index(name='corridor_name')
    df["corridor_name"] = df.progress_apply(lambda x:"/".join(list(set(x["corridor_name"]))),axis=1)
    all_edges = gpd.read_parquet(os.path.join(
                            incoming_data_path,
                            "africa_roads",
                            "corridor_edges.gpq"))
    all_edges = pd.merge(all_edges,df,how="left",on=["osm_way_id"])
    all_edges.drop("segment_id",axis=1,inplace=True)
    
    edges = gpd.read_parquet(os.path.join(
                            processed_data_path,
                            "infrastructure",
                            "africa_roads_edges.geoparquet"))   

    if "corridor_name" in edges.columns.values.tolist():
        edges.drop("corridor_name",axis=1,inplace=True)

    edge_ids = edges["id"].values.tolist()
    df = all_edges[~all_edges["id"].isin(edge_ids)]
    # df = gpd.GeoDataFrame(df,geometry="geometry",crs="EPSG:4326")
    df = df.to_crs(epsg=epsg_meters)
    print (df)

    edges = pd.merge(edges,all_edges[["id","corridor_name"]],how="left",on=["id"])
    # edges = gpd.GeoDataFrame(edges,geometry="geometry",crs="EPSG:4326")
    edges = edges.to_crs(epsg=epsg_meters)
    print (edges)
    
    print(edges.crs == df.crs)

    edges = pd.concat([edges,df],axis=0,ignore_index=True)
    edges["border_road"] = np.where(edges["from_iso_a3"] == edges["to_iso_a3"],0,1)
    edges["length_m"] = edges.geometry.length

    edge_nodes = list(set(edges["from_id"].values.tolist() + edges["to_id"].values.tolist()))

    nodes = gpd.read_parquet(os.path.join(
                            incoming_data_path,
                            "africa_roads",
                            "nodes_with_topology.gpq"))
    nodes = nodes[nodes["id"].isin(edge_nodes)]
    edges, nodes = components(edges,nodes,node_id_column="id")
    edges = edges.to_crs(epsg=4326)

    edges.to_parquet(os.path.join(
                            processed_data_path,
                            "infrastructure",
                            "africa_roads_edges.geoparquet"))

    nodes.to_parquet(os.path.join(
                            processed_data_path,
                            "infrastructure",
                            "africa_roads_nodes.geoparquet"))

    edges.to_file(os.path.join(
                            processed_data_path,
                            "infrastructure",
                            "africa_roads.gpkg"),layer="edges",driver="GPKG")
    

    



if __name__ == '__main__':
    CONFIG = load_config()
    main(CONFIG)