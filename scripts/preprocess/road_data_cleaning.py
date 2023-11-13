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
    mine_countries = list(set(mines["shapeGroup_primary_admin0"].values.tolist()))
    # Read the road edges data for Africa
    road_edges = gpd.read_parquet(os.path.join(incoming_data_path,
                            "africa_roads",
                            "edges_with_topology.geoparquet"))
    road_edges = road_edges.to_crs(epsg=epsg_meters)

    # We assume all the mines intersect the networks of the countries they are within
    # Seems like only a few mines are border mines, so our assumption is fine
    for m_c in mine_countries:
        country_roads = road_edges[(
                    road_edges["from_iso_a3"] == m_c
                    ) & (road_edges["to_iso_a3"] == m_c)]
        country_mines = mines[mines["shapeGroup_primary_admin0"] == m_c]
        # intersect mines with roads first to find which mines have roads on them
        mine_intersects = gpd.sjoin(country_mines[[mine_id_column,"geometry"]],
                            country_roads[[road_id_column,"geometry"]],how="left", 
                            predicate='intersects').reset_index()
        print (mine_intersects)
        mine_intersects = mine_intersects[~mine_intersects[road_id_column].isna()]
        print (mine_intersects)
        print (f"* Done with country - {mc}")




if __name__ == '__main__':
    CONFIG = load_config()
    main(CONFIG)