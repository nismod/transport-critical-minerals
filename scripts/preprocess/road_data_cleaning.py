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
    
    # Read the road edges data for Africa
    road_edges = gpd.read_parquet(os.path.join(incoming_data_path,
                            "africa_roads",
                            "edges_with_topology.geoparquet"))

    # Find the count of different road classes
    counts = road_edges["tag_highway"].values_counts()
    print (counts)
    
    pyogrio.write_dataframe(road_edges, os.path.join(incoming_data_path,
                            	"africa_roads",
                            	"edges_with_topology.gpkg"))


if __name__ == '__main__':
    CONFIG = load_config()
    main(CONFIG)