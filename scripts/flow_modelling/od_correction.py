#!/usr/bin/env python
# coding: utf-8

import os 
import pandas as pd
import geopandas as gpd
from collections import defaultdict
pd.options.mode.chained_assignment = None  # default='warn'
from utils import *
import subprocess 

"""Notes of BACI updates
    - Correct the codes for Singapore manually
    - Run the script baci_trade_data.py
    - Run the script global_trade_balancing.py
"""
def main(config):

    incoming_data_path = config['paths']['incoming_data']
    processed_data_path = config['paths']['data']
    output_data_path = config['paths']['results']

    input_folder = os.path.join(output_data_path,"flow_mapping")
    results_folder = os.path.join(output_data_path,"flow_od_paths")
    if os.path.exists(results_folder) == False:
        os.mkdir(results_folder)
    year_percentile_combinations = [
                                    (2022,"baseline"),
                                    (2030,"low"),
                                    (2030,"mid"),
                                    (2030,"high"),
                                    (2040,"low"),
                                    (2040,"mid"),
                                    (2040,"high")
                                    ]
    tonnage_thresholds = ["min_threshold_metal_tons","max_threshold_metal_tons"]
    reference_minerals = ["graphite","lithium","cobalt","manganese","nickel","copper"]

    all_files = defaultdict(list)
    for idx, (year,percentile) in enumerate(year_percentile_combinations):
        for reference_mineral in reference_minerals:
            if year == 2022:
                all_files[
                    f"mining_city_node_level_ods_{year}_{percentile}.csv"
                    ].append(f"{reference_mineral}_flow_paths_{year}_{percentile}")
                
            else:
                for th in tonnage_thresholds:
                    od_file = f"{reference_mineral}_flow_paths_{year}_{percentile}_{th}"
                    all_files[
                    f"mining_city_node_level_ods_{year}_{percentile}_{th}.csv"
                    ].append(f"{reference_mineral}_flow_paths_{year}_{percentile}_{th}")

    print (all_files)

    od_merge_columns = [
                        "origin_id",
                        "destination_id",
                        "reference_mineral",
                        "export_country_code",
                        "import_country_code",
                        "initial_processing_stage",
                        "final_processing_stage"
                        ]

    final_ton_column = "final_stage_production_tons"
    initial_ton_column = "initial_stage_production_tons"
    for k,v in all_files.items():
        c_t_df = pd.read_csv(
                            os.path.join(input_folder,
                                k)
                            )
        c_t_df.rename(columns={final_ton_column:"initial_tonnage"},inplace=True)
        for i in v:
            mine_routes = pd.read_parquet(
                                os.path.join(input_folder,
                                    f"{i}.parquet")
                                )
            mine_routes = pd.merge(mine_routes,
                            c_t_df[od_merge_columns + ["initial_tonnage"]],
                            how="left",on=od_merge_columns)
            mine_routes[initial_ton_column] = mine_routes[
                                                    initial_ton_column]*mine_routes[
                                                        final_ton_column]/mine_routes["initial_tonnage"]
            mine_routes.drop("initial_tonnage",axis=1,inplace=True)
            mine_routes.to_parquet(os.path.join(results_folder,f"{i}.parquet"))
                    

    
if __name__ == '__main__':
    CONFIG = load_config()
    main(CONFIG)


