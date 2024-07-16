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

def main(config,country_case,constraint):
    incoming_data_path = config['paths']['incoming_data']
    processed_data_path = config['paths']['data']
    output_data_path = config['paths']['results']

    input_folder = os.path.join(output_data_path,"country_summaries")
    results_folder = os.path.join(output_data_path,"country_summaries")
    if os.path.exists(results_folder) == False:
        os.mkdir(results_folder)

    #  Get a number of input dataframes
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
    initial_tons_column = "initial_stage_production_tons"
    final_tons_column = "final_stage_production_tons"    

    
    """Step 1: get all the relevant nodes and find their distances 
                to grid and bio-diversity layers 
    """
    if country_case == "country" and constraint == "unconstrained":
        combos = year_percentile_combinations
    else:
        combos = year_percentile_combinations[1:]

    all_files = []
    all_layers = []
    for idx, (year,percentile) in enumerate(combos):
        if year == 2022:
            file_name = f"location_totals_{year}_baseline"
            layer_name = f"{year}_baseline"
            all_files.append(file_name)
            all_layers.append(layer_name)
        else:
            for th in tonnage_thresholds:
                file_name = f"location_totals_{year}_{percentile}_{th}"
                layer_name = f"{year}_{percentile}_{th}"
                all_files.append(file_name)
                all_layers.append(layer_name)
    all_dfs = []
    for idx, (f,l) in enumerate(zip(all_files,all_layers)):
        df = pd.read_csv(
                os.path.join(
                    results_folder,
                    f"{f}_{country_case}_{constraint}.csv"))
        df = df[df["trade_type"] != "Import"]
        metal_df = df[df["initial_processing_stage"] == 0]
        metal_df = metal_df.groupby(
                        ["reference_mineral","iso3","initial_processing_stage"]
                        )[initial_tons_column].sum().reset_index()
        metal_df.rename(
                    columns={
                            "initial_processing_stage":"processing_stage",
                            initial_tons_column:l},
                    inplace=True)

        non_metal_df = df.groupby(
                        ["reference_mineral","iso3","final_processing_stage"]
                        )[final_tons_column].sum().reset_index()
        non_metal_df.rename(
                    columns={
                            "final_processing_stage":"processing_stage",
                            final_tons_column:l},
                    inplace=True)
        all_dfs.append(metal_df)
        all_dfs.append(non_metal_df)

    all_dfs = pd.concat(all_dfs,axis=0,ignore_index=True)
    all_dfs = all_dfs.groupby(
                    ["reference_mineral","iso3","processing_stage"]
                    ).agg(dict([(c,"sum") for c in all_layers])).reset_index()

    all_dfs = all_dfs.set_index(["reference_mineral","iso3","processing_stage"])
    all_dfs.to_excel(
            os.path.join(
                results_folder,
                "transport_tonnage_totals_by_stage.xlsx"),
            sheet_name =f"{country_case}_{constraint}")


if __name__ == '__main__':
    CONFIG = load_config()
    try:
        country_case = str(sys.argv[1])
        constraint = str(sys.argv[2])
    except IndexError:
        print("Got arguments", sys.argv)
        exit()
    main(CONFIG,country_case,constraint)