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



def main(config):
    incoming_data_path = config['paths']['incoming_data']
    processed_data_path = config['paths']['data']
    output_data_path = config['paths']['results']

    results_folder = os.path.join(output_data_path,"location_outputs")
    if os.path.exists(results_folder) == False:
        os.mkdir(results_folder)

    mine_id_col = "mine_cluster_mini"
    mine_tons_column = "mine_output_approx_copper"
    copper_conversion_stage = 3.0
    # cargo_type = "Dry bulk"
    # trade_groupby_columns = ["export_country_code", 
    #                         "import_country_code", 
    #                         "export_continent","export_landlocked",
    #                         "import_continent","import_landlocked"]
    # trade_value_columns = ["trade_value_thousandUSD","trade_quantity_tons"]
    # od_columns = ["reference_mineral","process_binary","export_country_code",
    #                 "import_country_code",
    #                 "export_continent",
    #                 "import_continent",
    #                 "mine_output_tons",    
    #                 "mine_output_thousandUSD"]
    """Step 1: Get the input datasets
    """
    # Network locations globally with the copper tonages
    flow_df = gpd.read_file(os.path.join(output_data_path,
                                    "flow_mapping",
                                    "copper_flows.gpkg"),
                                    layer="nodes")
    flow_df = flow_df[~flow_df.geometry.isna()]
    transport_modes = list(set(flow_df["mode"].values.tolist()))
    infra_df = []
    for tm in transport_modes:
        if tm == "sea":
            df = gpd.read_file(
                    os.path.join(processed_data_path,
                        "infrastructure",
                        "global_maritime_network.gpkg"
                    ),layer="nodes")
            infra_df.append(df[["id","infra"]])
        elif tm == "rail":
            df = gpd.read_file(os.path.join(
                            processed_data_path,
                            "infrastructure",
                            "africa_railways_network.gpkg"
                                ), layer="nodes"
                    )
        else:
            df = gpd.read_parquet(os.path.join(
                            processed_data_path,
                            "infrastructure",
                            "africa_roads_nodes.geoparquet"))
            df.rename(columns={"road_id":"id"},inplace=True)
            df["infra"] = "road"
        
        infra_df.append(df[["id","infra","iso3"]])

    infra_df = pd.concat(infra_df,axis=0,ignore_index=True)
    flow_df = pd.merge(flow_df,infra_df,how="left",on=["id"])

    flow_df.to_file(os.path.join(output_data_path,
                                    "flow_mapping",
                                    "copper_flows.gpkg"),
                                    layer="nodes",driver="GPKG")

    

    







if __name__ == '__main__':
    CONFIG = load_config()
    main(CONFIG)