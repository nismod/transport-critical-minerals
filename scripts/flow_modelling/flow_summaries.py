#!/usr/bin/env python
# coding: utf-8
"""This code finds the route between mines (in Africa) and all ports (outside Africa) 
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
    
def main(config,year,percentile,efficient_scale):
    incoming_data_path = config['paths']['incoming_data']
    processed_data_path = config['paths']['data']
    output_data_path = config['paths']['results']

    results_folder = os.path.join(output_data_path,"flow_mapping")
    if os.path.exists(results_folder) == False:
        os.mkdir(results_folder)

    """Step 1: Get the input datasets
    """
    reference_minerals = ["graphite","lithium","cobalt","manganese","nickel","copper"]
    optimal_check_columns = ["final_stage_production_tons","gcosts","distance_km","time_hr"]
    location_types = ["origin_in_country","in_region"]
    location_binary = ["conversion_location_in_country","conversion_location_in_region"]
    #  Get a number of input dataframes
    baseline_year = 2022
    data_type = {"initial_refined_stage":"str","final_refined_stage":"str"}
    (
        _, 
        metal_content_factors_df, 
        _, _, _, _
    ) = get_common_input_dataframes(data_type,year,baseline_year)
    if year == 2022:
        layer_name = f"{year}"
    else:
        layer_name = f"{year}_{percentile}_{efficient_scale}"
    
    optimal_df = gpd.read_file(
            os.path.join(
                    results_folder,
                    "optimal_locations_for_processing.gpkg"),
            layer=layer_name)

    


if __name__ == '__main__':
    CONFIG = load_config()
    try:
        year = int(sys.argv[1])
        percentile = int(sys.argv[2])
        efficient_scale = str(sys.argv[3])
    except IndexError:
        print("Got arguments", sys.argv)
        exit()
    main(CONFIG,year,percentile,efficient_scale)