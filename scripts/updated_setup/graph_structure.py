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
from trade_functions import *
from tqdm import tqdm
tqdm.pandas()



def main(config):
    incoming_data_path = config['paths']['incoming_data']
    processed_data_path = config['paths']['data']
    output_data_path = config['paths']['results']

    input_folder = os.path.join(output_data_path,"flow_node_ods")
    results_folder = os.path.join(output_data_path,"flow_od_paths")
    os.makedirs(results_folder,exist_ok=True)

    cargo_type = "Dry bulk"
    # cargo_type = "General cargo"
    # print (od_file_name)

    # Mine locations in Africa with the mineral tonages
    mine_id_col = "mine_id"
    mines_df = get_all_mines(mine_id_col=mine_id_col,return_columns=[mine_id_col,"iso3","geometry"])
    

    # Population locations for urban cities
    pop_id_col = "city_id"
    un_pop_df = gpd.read_file(os.path.join(processed_data_path,
                                    "admin_boundaries",
                                    "un_urban_population",
                                    "un_pop_df.gpkg"))
    un_pop_df = un_pop_df[un_pop_df["CONTINENT"] == "Africa"]
    un_pop_df = un_pop_df[[pop_id_col,"geometry"]]

    port_commodities_df = pd.read_csv(os.path.join(processed_data_path,
                                    "port_statistics",
                                    "port_known_commodities_traded.csv"))
    """Step 2: Identify all the countries outside Africa 
    """
    export_ports_africa = port_commodities_df[port_commodities_df["export_binary"] == 1]
    if len(export_ports_africa.index) == 0:
        export_ports_africa = port_commodities_df[
                            port_commodities_df[
                            "total_annual_vessel_capacity_tons"] > 0
                            ]

    export_port_ids = list(zip(export_ports_africa["id"].values.tolist(),
                    export_ports_africa[
                        "total_annual_vessel_capacity_tons"]))
    network_graph = create_mines_and_cities_to_port_network(
                            mines_df,mine_id_col,
                            un_pop_df,pop_id_col,
                            modes=["sea","intermodal","rail","road"],
                            intermodal_ports=export_ports_africa["id"].values.tolist(),
                            cargo_type=f"{cargo_type.lower().replace(' ','_')}",
                            port_to_land_capacity=export_port_ids,
                            distance_threshold=1500,join_largest_component=True
                            )
    print (network_graph)
    network_graph.to_parquet(os.path.join(processed_data_path,"shipping_network","network_graph.parquet"))
    


if __name__ == '__main__':
    CONFIG = load_config()
    main(CONFIG)