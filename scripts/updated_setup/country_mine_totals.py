#!/usr/bin/env python
# coding: utf-8
"""This code finds the route between mines (in Africa) and all ports (outside Africa) 
"""
import sys
import os
import re
import json
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
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

    results_folder = os.path.join(output_data_path,"country_totals")
    if os.path.exists(results_folder) == False:
        os.mkdir(results_folder)

    reference_minerals = ["copper","cobalt","manganese","lithium","graphite","nickel"]
    ccg_countries = pd.read_csv(
                        os.path.join(processed_data_path,
                                "baci","ccg_country_codes.csv"))
    ccg_countries = ccg_countries[ccg_countries["ccg_country"] == 1]["iso_3digit_alpha"].values.tolist()

    # Read data on production scales
    production_scales_df = pd.read_excel(
                                os.path.join(
                                    processed_data_path,
                                    "production_costs",
                                    "scales.xlsx"),
                                sheet_name="efficient_scales"
                                )
    combined_mine_df = []
    for reference_mineral in reference_minerals:
        process_threshold = production_scales_df[
                                production_scales_df["reference_mineral"
                                ] == reference_mineral]["min_threshold_metal_tons"].values[0]
        for indx,(year,percentile) in enumerate(zip([2022,2040],["baseline","mid"])):
            mines_df = gpd.read_file(
                            os.path.join(
                                processed_data_path,
                                "minerals",
                                "s_and_p_mines_current_and_future_estimates.gpkg"),
                            layer=f"{reference_mineral}_{percentile}")
            mines_df.rename(columns={"ISO_A3":"iso3"},inplace=True)
            mines_df["weight"] = mines_df[f"{year}_metal_content"]
            mines_df = mines_df.groupby(["iso3"])["weight"].sum().reset_index()
            mines_df["reference_mineral"] = reference_mineral
            mines_df["year"] = year
            mines_df["metal_threshold"] = process_threshold
            combined_mine_df.append(mines_df)

    combined_mine_df = pd.concat(combined_mine_df,axis=0,ignore_index=True)
    combined_mine_df.to_csv(os.path.join(
                            results_folder,
                            "mines_total_metal_content.csv"),index=False)


if __name__ == '__main__':
    CONFIG = load_config()
    main(CONFIG)