#!/usr/bin/env python
# coding: utf-8
# Code to extract the suez canal navigation route 
import os
import pandas as pd
import geopandas as gpd
from utils import *
from tqdm import tqdm
tqdm.pandas()

def main(config):
    incoming_data_path = config['paths']['incoming_data']
    # Read the OSM data and the suez canal IDs
    waterways = gpd.read_file(os.path.join(
                    incoming_data_path,
                    "egypt-latest-free.shp",
                    "gis_osm_waterways_free_1.shp"))
    print (waterways)
    suez_ids = pd.read_csv(os.path.join(
                            incoming_data_path,
                                "egypt-latest-free.shp",
                                "suez_canal_ids.csv"))
    suez_ids = suez_ids["osm_id"].values.tolist()
    print (suez_ids)
    suez_canal = waterways[waterways["osm_id"].isin(suez_ids)]
    print (suez_canal)
    # Write the Suez Canal routes to a GPKG
    suez_canal.to_file(os.path.join(
                    incoming_data_path,
                    "egypt-latest-free.shp",
                    "suez_canal.gpkg"),driver="GPKG")

if __name__ == '__main__':
    CONFIG = load_config()
    main(CONFIG)