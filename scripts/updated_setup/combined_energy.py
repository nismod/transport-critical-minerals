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

    energy_input_folder = os.path.join(output_data_path,"energy_results")
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
    cost_sum_dict = [(final_tons_column,"sum"),(cost_column,"sum")]
    energy_sum_cols = ['energy_cost_usd','Emissions_tCO2eq']
    constraints = ["Country_unconstrained","Region Unconstrained"]
    rename_constraints = ["country_unconstrained","region_unconstrained"]
    chosen_scenarios = [
                        "Base",
                        "2030_mid_min_max_average_threshold_metal_tons",
                        "2040_mid_min_threshold_metal_tons",
                        "2030_mid_min_max_average_threshold_metal_tons",
                        "2040_mid_min_max_average_threshold_metal_tons"
                        ]
    rename_scenarios = [
                        "2022_baseline",
                        "2030_mid_min_threshold_metal_tons",
                        "2040_mid_min_threshold_metal_tons",
                        "2030_mid_max_threshold_metal_tons",
                        "2040_mid_max_threshold_metal_tons"
                        ]
    years = [2022,2030,2040,2030,2040]
    # Read the finalised version of the BACI trade data
    ccg_countries = pd.read_csv(
                        os.path.join(processed_data_path,
                                "baci","ccg_country_codes.csv"))
    ccg_countries = ccg_countries[ccg_countries["ccg_country"] == 1]["iso_3digit_alpha"].values.tolist()

    energy_demand_df = pd.read_excel(
                        os.path.join(
                            energy_input_folder,
                            "Sceanrio Results.xlsx"),header=[1,2],index_col=[0,1])
    energy_demand_df = energy_demand_df.reset_index()
    existing_columns = energy_demand_df.columns.values.tolist()
    energy_demand_df.columns = ["constraints","original_scenario"] + existing_columns[2:]
    energy_demand_df.loc[energy_demand_df["original_scenario"] == "Base","constraints"] = "Country_unconstrained"
    energy_demand_df.to_csv("test0.csv")
    energy_demand_df = energy_demand_df[
                                        (
                                            energy_demand_df["constraints"].isin(constraints)
                                        ) & (
                                            energy_demand_df["original_scenario"].isin(chosen_scenarios)
                                        )
                                        ]
    energy_demand_df["constraints"] = energy_demand_df["constraints"].str.lower()
    energy_demand_df["constraints"] = energy_demand_df["constraints"].str.replace(" ","_")
    energy_demand_df["scenario"] = rename_scenarios
    energy_demand_df["year"] = years
    energy_demand_df.to_csv("test1.csv")
    dfs = []
    index_cols = ["constraints","original_scenario","scenario","year"] 
    cnts = [c[0] for c in existing_columns[2:]]
    for cnt in ccg_countries:
        cols = [c for c in existing_columns[2:] if c[0] == cnt]
        df = energy_demand_df[index_cols + cols]
        df.columns = index_cols + [c[1] for c in cols]
        df["iso3"] = cnt
        dfs.append(df)

    dfs = pd.concat(dfs,axis=0,ignore_index=True)
    # print (dfs.columns.values.tolist())

    sum_cols = []
    edfs = []
    for rn in rename_constraints:
        for idx,(sc,y) in enumerate(zip(rename_scenarios,years)):
            fpath = os.path.join(energy_input_folder,f"{sc}-{rn}.csv")
            if os.path.exists(fpath) is True:
                e_df = pd.read_csv(fpath)
                e_df["energy_cost_usd"] = e_df['Est_dem_kWh_year']*e_df['Est_lcoe_$perkWh']
                e_df = e_df.groupby(["iso3"]).agg(dict([(c,"sum") for c in energy_sum_cols])).reset_index()
                e_df["scenario"] = sc
                e_df["year"] = y
                e_df["constraints"] = rn
                edfs.append(e_df)

    edfs = pd.concat(edfs,axis=0,ignore_index=True)
    dfs = pd.merge(dfs,edfs,how="left",on=["constraints","year","scenario","iso3"]).fillna(0)
    dfs.drop("original_scenario",axis=1,inplace=True)
    dfs = dfs.set_index(["constraints","year","scenario","iso3"])
    dfs.to_excel(os.path.join(results_folder,"energy_totals_by_country.xlsx"))





if __name__ == '__main__':
    CONFIG = load_config()
    main(CONFIG)