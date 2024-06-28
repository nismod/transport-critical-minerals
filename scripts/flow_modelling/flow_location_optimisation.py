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

def get_stage_1_tons_columns(df,reference_minerals,location_types,location_binary,mc_factors):
    for idx,(lt,lb) in enumerate(zip(location_types,location_binary)):
        for rm in reference_minerals:
            in_col = f"{rm}_initial_stage_production_tons_0.0_{lt}"
            f_col = f"{rm}_final_stage_production_tons_1.0_{lt}"
            mf = mc_factors[mc_factors["reference_mineral"] == rm]["metal_content_factor"].values[0]
            df[f_col] = 1.0*df[in_col]/mf
            df[f"{reference_mineral}_{lb}"] = 1

    return df


def find_optimal_locations(opt_list,flows_df,path_df,
                            reference_mineral,
                            loc_type,
                            columns,
                            mine_stage):
    if len(flows_df.index) > 0:
        path_df = path_df[path_df["final_processing_stage"] == mine_stage]
        node_path_indexes = get_flow_paths_indexes_and_edges_dataframe(path_df,"full_node_path")

        all_ids = flows_df["id"].values.tolist()
        node_path_indexes = node_path_indexes[node_path_indexes["id"].isin(all_ids)]
        c_df_flows = pd.merge(node_path_indexes,flows_df,how="left",on=["id"]).fillna(0)

        while len(c_df_flows.index) > 0:
            optimal_locations = defaultdict()
            c_df_flows = c_df_flows.sort_values(
                                        by=columns,
                                        ascending=[False,True,True,True])
            nid = c_df_flows["id"].values[0]
            optimal_locations["iso3"] = c_df_flows["iso3"].values[0]
            optimal_locations["id"] = c_df_flows["id"].values[0]
            optimal_locations[
                f"{reference_mineral}_{loc_type}"
                ] = c_df_flows[f"{reference_mineral}_{loc_type}"].values[0]
            for c in columns:
                optimal_locations[c] = c_df_flows[c].values[0]

            opt_list.append(optimal_locations)
            pth_idx = list(set(c_df_flows[c_df_flows["id"] == nid]["path_index"].values.tolist()))
            c_df_flows = c_df_flows[~c_df_flows["path_index"].isin(pth_idx)]

    return opt_list
    
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
        _, _, _
    ) = get_common_input_dataframes(data_type,year,baseline_year)
    if year == 2022:
        layer_name = f"{year}"
    else:
        layer_name = f"{year}_{percentile}_{efficient_scale}"
    
    all_flows = gpd.read_file(os.path.join(results_folder,
                        f"node_locations_for_energy_conversion.gpkg"),
                        layer=layer_name)
    if year > 2022:
        origins = list(set(all_flows["iso3"].values.tolist()))
        optimal_df = []
        for idx,(lt,lb) in enumerate(zip(location_types,location_binary)):
            for reference_mineral in reference_minerals:
                # Find year locations
                if year == 2022:
                    file_name = f"{reference_mineral}_flow_paths_{year}"
                else:
                    file_name = f"{reference_mineral}_flow_paths_{year}_{percentile}_{efficient_scale}"
                
                od_df = pd.read_parquet(
                                os.path.join(results_folder,
                                    f"{file_name}.parquet")
                                )
                mine_stage = all_flows[f"{reference_mineral}_mine_highest_stage"].max()
                columns = [f"{reference_mineral}_{c}_{mine_stage}_{lt}" for c in optimal_check_columns]
                if lt == "origin_in_country":
                    for o in origins:
                        flows_df = all_flows[(all_flows["iso3"] == o) & (all_flows[f"{reference_mineral}_{lb}"] == 1)]
                        optimal_df = find_optimal_locations(optimal_df,flows_df,
                                od_df.copy(),
                                reference_mineral,
                                lb,
                                columns,
                                mine_stage)

                else:
                    flows_df = all_flows[all_flows[f"{reference_mineral}_{lb}"] == 1]
                    optimal_df = find_optimal_locations(optimal_df,flows_df,
                                od_df.copy(),
                                reference_mineral,
                                lb,
                                columns,
                                mine_stage)

                print (f"Done with {lt} case for {reference_mineral}")

        optimal_df = pd.DataFrame(optimal_df).fillna(0)
        add_columns = [c for c in optimal_df.columns.values.tolist() if c not in ["iso3","id"]]
        optimal_df = optimal_df.groupby(["id","iso3"]).agg(dict([(c,"sum") for c in add_columns])).reset_index()
        flow_cols = [c for c in all_flows.columns.values.tolist() if c not in optimal_df.columns.values.tolist()]

        mc_cols = [c for c in optimal_df.columns.values.tolist() if "final_stage_production_tons" in c]
        mines_cities_cols = [c for c in all_flows.columns.values.tolist() if c not in mc_cols]
        mines_and_cities_df = all_flows[all_flows["mode"].isin(["mine","city"])]
        mines_and_cities_df = mines_and_cities_df[
                            ~mines_and_cities_df["id"].isin(
                                optimal_df["id"].values.tolist()
                            )]
        mines_and_cities_df = mines_and_cities_df[mines_cities_cols]

        mines_df = mines_and_cities_df[mines_and_cities_df["mode"] == "mine"]
        mines_df = get_stage_1_tons_columns(mines_df,reference_minerals,location_types,location_binary,metal_content_factors_df)
        cities_df = mines_and_cities_df[mines_and_cities_df["mode"] == "city"]
        optimal_df = pd.merge(optimal_df,all_flows[["id"] + flow_cols],how="left",on=["id"]).fillna(0)
        optimal_df = pd.concat([optimal_df,mines_df,cities_df],axis=0,ignore_index=True).fillna(0)
        optimal_df = gpd.GeoDataFrame(optimal_df,geometry="geometry",crs=all_flows.crs)
    else:
        optimal_df = all_flows.copy()

    print (optimal_df)
    optimal_df.to_file(
            os.path.join(
                    results_folder,
                    "optimal_locations_for_processing.gpkg"),
            layer=layer_name,driver="GPKG")


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