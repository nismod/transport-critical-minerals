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
    
    iww_edges = gpd.read_file(os.path.join(
                            processed_data_path,
                            "infrastructure",
                            "africa_iww_network.gpkg"
                                ), layer="edges"
                    )
    iww_edges = transport_cost_assignment_function(iww_edges,"IWW")
    print (iww_edges)

    rail_edges = gpd.read_file(os.path.join(
                            processed_data_path,
                            "infrastructure",
                            "africa_railways_network.gpkg"
                                ), layer="edges"
                    )
    rail_edges = transport_cost_assignment_function(rail_edges,"rail")
    print (rail_edges)

    road_edges = gpd.read_parquet(os.path.join(
                            processed_data_path,
                            "infrastructure",
                            "africa_roads_edges.geoparquet"))
    road_edges = transport_cost_assignment_function(road_edges,"road")
    print (road_edges)

    port_edges = pd.read_parquet(os.path.join(
                    processed_data_path,
                    "shipping_network",
                    "maritime_base_network_dry_bulk.parquet"))
    port_edges.rename(columns={"from_iso3":"from_iso_a3","to_iso3":"to_iso_a3"},inplace=True)
    port_edges = transport_cost_assignment_function(port_edges,"sea")
    print (port_edges)

    intermodal_edges = gpd.read_file(os.path.join(
                            processed_data_path,
                            "infrastructure",
                            "africa_multimodal.gpkg"
                                ), layer="edges")
    print (intermodal_edges)
    intermodal_edges = transport_cost_assignment_function(intermodal_edges,"intermodal")
    print (intermodal_edges)

    

    
    
    


    


if __name__ == '__main__':
    CONFIG = load_config()
    main(CONFIG)