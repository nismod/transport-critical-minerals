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

    #  Get a number of input dataframes
    baseline_year = 2022
    data_type = {"initial_refined_stage":"str","final_refined_stage":"str"}
    (
        pr_conv_factors_df, 
        metal_content_factors_df, 
        ccg_countries, mine_city_stages, _
    ) = get_common_input_dataframes(data_type,year,baseline_year)
    
    nodes = add_geometries_to_flows([],
                                modes=["rail","sea","road","mine","city"],
                                layer_type="nodes",merge=False)
    nodes = nodes[nodes["iso3"].isin(ccg_countries)]
    nodes = get_distance_to_layer(nodes)
    nodes.to_parquet(
            os.path.join(
                results_folder,
                "nodes_with_location_identifiers.geoparquet")
            )



if __name__ == '__main__':
    CONFIG = load_config()
    main(CONFIG)