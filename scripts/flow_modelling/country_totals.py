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

    input_folder = os.path.join(output_data_path,"result_summaries")

    results_folder = os.path.join(output_data_path,"result_summaries")
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
    tonnage_types = ["production","export"] 
    cost_column = "total_gcosts" 
    carbon_column = "tonsCO2eq"
    index_columns = ["year","scenario","iso3"]
    transport_columns = [
                            "production_tonnes",
                            "export_tonnes",
                            "total_transport_cost_usd",
                            "tonsCO2eq","revenue_usd",
                            "production_cost_usd",
                            "export_cost_usd"
                        ]
    scenarios = [
                        "2022_baseline",
                        "2030_mid_min_threshold_metal_tons",
                        "2040_mid_min_threshold_metal_tons",
                        ]
    cost_sum_dict = [(t,"sum") for t in transport_columns]

    transport_df = pd.read_excel(
                        os.path.join(
                            input_folder,
                            "transport_totals_by_stage.xlsx"),
                    sheet_name = "country_unconstrained",index_col=[0,1,2,3])
    transport_df = transport_df.reset_index()
    transport_df = transport_df[transport_df["processing_stage"] != 0]
    transport_df["export_cost_usd"] = transport_df["export_tonnes"]*(
                                            transport_df["capex_usd_per_tonne"
                                            ] + transport_df["opex_usd_per_tonne"]
                                            )
    transport_df = transport_df.groupby(index_columns).agg(dict(cost_sum_dict)).reset_index()
    transport_df = transport_df[transport_df["scenario"].isin(scenarios)]

    energy_df = pd.read_excel(
                        os.path.join(
                            input_folder,
                            "energy_totals_by_country.xlsx"),
                        index_col=[0,1,2,3])
    energy_df = energy_df.reset_index()
    energy_df = energy_df[index_columns + ["Emissions_tCO2eq","energy_cost_usd"]]
    transport_df = pd.merge(transport_df,energy_df,how="left",on=index_columns)
    transport_df = transport_df.set_index(index_columns)
    transport_df.to_excel(os.path.join(results_folder,"totals_by_country_v2.xlsx"))

    


    


if __name__ == '__main__':
    CONFIG = load_config()
    main(CONFIG)