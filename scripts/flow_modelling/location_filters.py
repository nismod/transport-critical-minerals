#!/usr/bin/env python
# coding: utf-8
"""This code finds the optimal locations of processing in Africa 
"""
import sys
import os
import re
import json
import pandas as pd
import numpy as np
pd.options.mode.copy_on_write = True
import igraph as ig
import geopandas as gpd
from collections import defaultdict
from utils import *
from transport_cost_assignment import *
from trade_functions import * 
from tqdm import tqdm
tqdm.pandas()

def main(config):
    incoming_data_path = config['paths']['incoming_data']
    processed_data_path = config['paths']['data']
    output_data_path = config['paths']['results']

    results_folder = os.path.join(output_data_path,"location_filters")
    if os.path.exists(results_folder) == False:
        os.mkdir(results_folder)

    # Read the finalised version of the BACI trade data
    ccg_countries = pd.read_csv(
                        os.path.join(processed_data_path,
                                "baci","ccg_country_codes.csv"))
    ccg_countries = ccg_countries[ccg_countries["ccg_country"] == 1]["iso_3digit_alpha"].values.tolist()

    nodes = add_geometries_to_flows([],
                                modes=["rail","sea","road","mine","city"],
                                layer_type="nodes",merge=False)
    nodes = nodes[nodes["iso3"].isin(ccg_countries)]
    # nodes = get_distance_to_layer(nodes)
    # nodes.to_parquet(
    #         os.path.join(
    #             results_folder,
    #             "nodes_with_location_identifiers.geoparquet")
    #         )
    # nodes.to_file(
    #         os.path.join(
    #             results_folder,
    #             "nodes_with_location_identifiers.gpkg"),
    #         driver="GPKG"
    #         )

    nodes = get_distance_to_layer_global(nodes)
    nodes.to_parquet(
            os.path.join(
                results_folder,
                "nodes_with_location_identifiers_regional.geoparquet")
            )
    nodes.to_file(
            os.path.join(
                results_folder,
                "nodes_with_location_identifiers_regional.gpkg"),
            driver="GPKG"
            )



if __name__ == '__main__':
    CONFIG = load_config()
    main(CONFIG)