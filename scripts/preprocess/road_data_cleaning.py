#!/usr/bin/env python
# coding: utf-8
import sys
import os
import re
import json
import pandas as pd
import geopandas as gpd
import pyogrio
from utils import *
from tqdm import tqdm
tqdm.pandas()

def main(config):
    incoming_data_path = config['paths']['incoming_data']
    processed_data_path = config['paths']['data']
    
    epsg_meters = 3395 # To convert geometries to measure distances in meters
    mine_id_column = "mine_id"
    road_id_column = "edge_id"
    # Read the mining data from the global extract
    mines = gpd.read_file(os.path.join(incoming_data_path,
                            "mines_spatial_locations",
                            "all_mines_adm.gpkg"))
    mines = mines[mines["continent"] == "Africa"]
    mines = mines.to_crs(epsg=epsg_meters)
    # Read the road edges data for Africa
    road_edges = gpd.read_parquet(os.path.join(incoming_data_path,
                            "africa_roads",
                            "edges_with_topology.geoparquet"))
    road_edges = road_edges.to_crs(epsg=epsg_meters)

    # intersect mines with roads first to find which mines have roads on them
    mine_intersects = gpd.sjoin(mines[[mine_id_column,"geometry"]],
                        roads[[road_id_column,"geometry"]],how="left", 
                        predicate='intersects').reset_index()
    print (mine_intersects)
    mine_intersects = mine_intersects[~mine_intersects[road_id_column].isna()]
    print (mine_intersects)




if __name__ == '__main__':
    CONFIG = load_config()
    main(CONFIG)