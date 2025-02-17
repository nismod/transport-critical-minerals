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
    
def main(config):
    output_data_path = config['paths']['results']

    results_folder = os.path.join(output_data_path,"result_summaries")
    energy_results_folder = os.path.join(output_data_path,"20250216_Run_No_GridBuffer")

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
    location_cases = ["country","region"]
    optimisation_type = ["unconstrained","constrained"]
    baseline_year = 2022

    energy_columns = [
                        "energy_req_capacity_kW",
                        "energy_tonsCO2eq",
                        "energy_investment_usd",
                        "energy_opex"
                    ]
    dfs = []
    scenarios = []
    for idx, (y,s) in enumerate(year_percentile_combinations):
        if y == baseline_year:
            scenarios.append(
                                (
                                    y,f"{y}_{s}",
                                    "country_unconstrained",
                                    f"{y}_{s}_country_unconstrained.csv_mineral_summary.xlsx"
                                )
                            )
        else:
            for jdx, (thr,loc) in enumerate(zip(tonnage_thresholds,location_cases)):
                for opt in optimisation_type:
                    scenarios.append(
                                        (
                                            y,f"{y}_{s}_{thr}",f"{loc}_{opt}",
                                            f"{y}_{s}_{thr}_{loc}_{opt}.csv_mineral_summary.xlsx"
                                        )
                                    )

    for idx,(y,sc,lc,fname) in enumerate(scenarios):
        fpath = os.path.join(
                    energy_results_folder,
                    fname)
        if os.path.exists(fpath) is True:
            df = pd.read_excel(fpath,sheet_name="By Mineral and Stage")
            df.rename(
                columns={
                            "mineral":"reference_mineral",
                            "stage":"processing_stage",
                            "ISO3":"iso3",
                            "Req_Capacity_kW":"energy_req_capacity_kW",
                            "Emissions_tCO2eq":"energy_tonsCO2eq",
                            "Final_Investment_USD":"energy_investment_usd",
                            "annual_opex":"energy_opex"
                        },inplace=True
                    )
            df["year"] = y
            df["scenario"] = sc
            df["location_constraint"] = lc
            dfs.append(df)

    dfs = pd.concat(dfs,axis=0,ignore_index=True)
    

    output_file = os.path.join(
                        results_folder,
                        "energy_transport_totals_by_stage.xlsx")
    if os.path.isfile(output_file) is True:
        writer = pd.ExcelWriter(output_file,mode='a',engine="openpyxl",if_sheet_exists='replace')
    else:
        writer = pd.ExcelWriter(output_file)

    for loc in location_cases:
        for opt in optimisation_type:
            transport_df = pd.read_excel(
                                    os.path.join(
                                        results_folder,
                                        "combined_transport_totals_by_stage.xlsx"),
                                    sheet_name=f"{loc}_{opt}",index_col=[0,1,2,3,4,5])
            transport_df = transport_df.reset_index()
            df = dfs[dfs["location_constraint"] == f"{loc}_{opt}"]
            df.drop("location_constraint",axis=1,inplace=True)
            
            transport_df = pd.merge(
                                transport_df,
                                df,how="left",
                                on=[
                                    "year","scenario",
                                    "reference_mineral",
                                    "iso3","processing_stage"
                                    ]
                                ).fillna(0)
            for e in energy_columns:
                transport_df[f"{e}_per_tonne"
                        ] = np.where(
                                    transport_df["production_tonnes"] != 0,
                                    transport_df[e]/transport_df["production_tonnes"],
                                    0
                                    )
            transport_df = transport_df.set_index(
                                        [
                                            "year","scenario",
                                            "reference_mineral",
                                            "iso3","processing_type",
                                            "processing_stage"
                                            ]
                                        )
            transport_df.to_excel(writer,sheet_name=f"{loc}_{opt}")
    
    writer.close()



    
    


if __name__ == '__main__':
    CONFIG = load_config()
    main(CONFIG)