#!/usr/bin/env python
# coding: utf-8
"""This code finds the route between mines (in Africa) and all ports (outside Africa) 
"""
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



def main(config,reference_mineral,year,percentile):
    incoming_data_path = config['paths']['incoming_data']
    processed_data_path = config['paths']['data']
    output_data_path = config['paths']['results']

    reference_mineral = "copper"
    year = 2030
    percentile = "mid"
    efficient_scale = "min_threshold_metal_tons"
    od_columns = [
                    "origin_id",
                    "destination_id",
                    "reference_mineral",
                    "export_country_code",
                    "import_country_code",
                    "export_continent",
                    "import_continent",
                    "trade_type",
                    "initial_processing_stage",
                    "final_processing_stage",
                    "initial_processing_location",
                    "final_processing_location",
                    "initial_stage_production_tons",    
                    "final_stage_production_tons"
                ]
    file_name = f"{reference_mineral}_flow_paths_{year}_{percentile}_{efficient_scale}"
    od_df = pd.read_parquet(
                        os.path.join(
                            output_data_path,
                            "flow_od_paths",
                            f"{file_name}.parquet"
                            )
                        )
    od_df[od_columns].to_csv("test.csv",index=False)


if __name__ == '__main__':
    CONFIG = load_config()
    main(CONFIG)