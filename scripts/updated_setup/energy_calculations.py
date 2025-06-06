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

def get_stage_one_conversion_factors(x,pcf_df,cf_column="aggregate_ratio"):
    ref_min = x["reference_mineral"]
    exp_st = x["final_processing_stage"]
    cf_df = pcf_df[pcf_df["reference_mineral"] == ref_min]
    cf_val = cf_df[cf_df["final_refined_stage"] == str(exp_st).replace(".0","")
                    ][cf_column].values[0]/cf_df[
                    cf_df["final_refined_stage"] == '1'
                    ][cf_column].values[0]
    return cf_val

def main(config,country_case,constraint):
    incoming_data_path = config['paths']['incoming_data']
    processed_data_path = config['paths']['data']
    output_data_path = config['paths']['results']

    input_folder = os.path.join(output_data_path,"result_summaries")
    
    results_folder = os.path.join(output_data_path,"result_summaries")
    if os.path.exists(results_folder) == False:
        os.mkdir(results_folder)

    output_file = os.path.join(
                        results_folder,
                        "transport_totals_by_stage_with_energy.xlsx")
    if os.path.isfile(output_file) is True:
        writer = pd.ExcelWriter(output_file,mode='a',engine="openpyxl",if_sheet_exists='replace')
    else:
        writer = pd.ExcelWriter(output_file)

    output_file = os.path.join(
                        results_folder,
                        "energy_totals.xlsx")
    if os.path.isfile(output_file) is True:
        writer_t = pd.ExcelWriter(output_file,mode='a',engine="openpyxl",if_sheet_exists='replace')
    else:
        writer_t = pd.ExcelWriter(output_file)

    energy_factors = pd.read_excel(
                        os.path.join(
                            processed_data_path,
                            "mineral_usage_factors",
                            "min_processing_values_dict.xlsx"))
    tonnages_df = pd.read_excel(
                    os.path.join(
                            results_folder,
                            "transport_totals_by_stage.xlsx"),
                    sheet_name=f"{country_case}_{constraint}",index_col=[0,1,2,3])
    print (tonnages_df)
    tonnages_df = tonnages_df.reset_index()
    print (tonnages_df)
    tonnages_df = pd.merge(
                        tonnages_df,
                        energy_factors,
                        how="left",
                        on=["reference_mineral","processing_stage"]).fillna(0)
    tonnages_df["energy_production_demand_kWh"] = 1.0e3*tonnages_df["production_tonnes"]*tonnages_df["processing_energy_intensity"]
    tonnages_df["energy_export_demand_kWh"] = 1.0e3*tonnages_df["export_tonnes"]*tonnages_df["processing_energy_intensity"]

    tonnages_df = tonnages_df.set_index(["year","scenario","reference_mineral","iso3","processing_stage"])
    tonnages_df.to_excel(writer,sheet_name=f"{country_case}_{constraint}")
    writer.close()

    sum_dict = [("energy_production_demand_kWh","sum"),("energy_export_demand_kWh","sum")]
    tonnages_df = tonnages_df.groupby(["year","scenario","iso3"]).agg(dict(sum_dict)).reset_index()
    tonnages_df = tonnages_df.set_index(["year","scenario","iso3"])
    tonnages_df.to_excel(writer_t,sheet_name=f"{country_case}_{constraint}")
    writer_t.close()


if __name__ == '__main__':
    CONFIG = load_config()
    try:
        country_case = str(sys.argv[1])
        constraint = str(sys.argv[2])
    except IndexError:
        print("Got arguments", sys.argv)
        exit()
    main(CONFIG,country_case,constraint)