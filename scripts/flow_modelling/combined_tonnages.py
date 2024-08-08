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

    tons_input_folder = os.path.join(output_data_path,"tonnage_summaries")
    carbon_input_folder = os.path.join(output_data_path,"carbon_emissions_summaries")

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

    price_df = pd.read_excel(
                        os.path.join(
                            processed_data_path,
                            "production_costs",
                            "Final_Price_and_Costs_RP.xlsx"),
                    sheet_name = "Price_final",index_col=[0])
    price_df = price_df.reset_index()
    capex_df = pd.read_excel(
                        os.path.join(
                            processed_data_path,
                            "production_costs",
                            "Final_Price_and_Costs_RP.xlsx"),
                    sheet_name = "CapEx_final",index_col=[0])
    capex_df = capex_df.reset_index()
    opex_df = pd.read_excel(
                        os.path.join(
                            processed_data_path,
                            "production_costs",
                            "Final_Price_and_Costs_RP.xlsx"),
                    sheet_name = "OpEx_final",index_col=[0])
    opex_df = opex_df.reset_index()
    years = [2022,2030,2040]
    price_costs_df = []
    index_cols = ["year","reference_mineral","processing_stage"]
    for y in years:
        pdf = price_df[["reference_mineral","processing_stage",y]]
        pdf["year"] = y
        pdf.rename(columns={y:"price_usd_per_tonne"},inplace=True)

        cdf = capex_df[["reference_mineral","processing_stage",y]]
        cdf["year"] = y
        cdf.rename(columns={y:"capex_usd_per_tonne"},inplace=True)

        odf = opex_df[["reference_mineral","processing_stage",y]]
        odf["year"] = y
        odf.rename(columns={y:"opex_usd_per_tonne"},inplace=True)

        pc_df = pd.concat(
                        [
                            pdf.set_index(index_cols),
                            cdf.set_index(index_cols),
                            odf.set_index(index_cols)
                        ],
                        axis=1)
        price_costs_df.append(pc_df.reset_index())

    price_costs_df = pd.concat(price_costs_df,axis=0,ignore_index=True)
    print (price_costs_df)


    output_file = os.path.join(
                        results_folder,
                        "transport_totals_by_stage.xlsx")
    if os.path.isfile(output_file) is True:
        writer = pd.ExcelWriter(output_file,mode='a',engine="openpyxl",if_sheet_exists='replace')
    else:
        writer = pd.ExcelWriter(output_file)
    
    if country_case == "country" and constraint == "unconstrained":
        combos = year_percentile_combinations
    else:
        combos = year_percentile_combinations[1:]

    all_tons_files = []
    all_carbon_files = []
    all_layers = []
    all_years = []
    for idx, (year,percentile) in enumerate(combos):
        if year == 2022:
            tons_file_name = f"location_totals_{year}_baseline"
            carbon_file_name = f"carbon_emission_totals_{year}_baseline"
            layer_name = f"{year}_baseline"
            all_tons_files.append(tons_file_name)
            all_carbon_files.append(carbon_file_name)
            all_layers.append(layer_name)
            all_years.append(year)
        else:
            for th in tonnage_thresholds:
                tons_file_name = f"location_totals_{year}_{percentile}_{th}"
                carbon_file_name = f"carbon_emission_totals_{year}_{percentile}_{th}"
                layer_name = f"{year}_{percentile}_{th}"
                all_tons_files.append(tons_file_name)
                all_carbon_files.append(carbon_file_name)
                all_layers.append(layer_name)
                all_years.append(year)

    all_dfs = []
    all_sums = [
                "production_tonnes",
                "export_tonnes",
                "total_transport_cost_usd",
                "average_transport_cost_usd_per_tonne",
                carbon_column
                ]
    for idx, (ft,fc,l,y) in enumerate(zip(all_tons_files,all_carbon_files,all_layers,all_years)):
        t_df = pd.read_csv(
                os.path.join(
                    tons_input_folder,
                    f"{ft}_{country_case}_{constraint}.csv"))
        # Get the total production volumes 
        for pt in tonnage_types:
            if pt == "production":
                p_df = t_df[t_df["trade_type"] != "Import"]
                m_df = t_df[t_df["trade_type"].isin(["Export","Domestic"])]
                m_df = m_df.groupby(
                            ["reference_mineral","iso3","final_processing_stage"]
                            ).agg(dict(cost_sum_dict)).reset_index()
                m_df["average_transport_cost_usd_per_tonne"
                    ] = m_df[cost_column]/m_df[final_tons_column]
                m_df.rename(
                            columns={
                                    "final_processing_stage":"processing_stage",
                                    cost_column:"total_transport_cost_usd"},
                            inplace=True)
                m_df.drop(final_tons_column,axis=1,inplace=True)
                m_df["scenario"] = l
                m_df["year"] = y
                all_dfs.append(m_df)
            else:
                p_df = t_df[t_df["trade_type"] == "Export"]


            metal_df = p_df[p_df["initial_processing_stage"] == 0]
            metal_df = metal_df.groupby(
                            ["reference_mineral","iso3","initial_processing_stage"]
                            )[initial_tons_column].sum().reset_index()
            metal_df.rename(
                        columns={
                                "initial_processing_stage":"processing_stage",
                                initial_tons_column:f"{pt}_tonnes"},
                        inplace=True)

            non_metal_df = p_df.groupby(
                            ["reference_mineral","iso3","final_processing_stage"]
                            )[final_tons_column].sum().reset_index()
            non_metal_df.rename(
                        columns={
                                "final_processing_stage":"processing_stage",
                                final_tons_column:f"{pt}_tonnes"},
                        inplace=True)
            metal_df["scenario"] = l
            metal_df["year"] = y
            non_metal_df["scenario"] = l
            non_metal_df["year"] = y
            all_dfs.append(metal_df)
            all_dfs.append(non_metal_df)

        c_df = pd.read_csv(
                os.path.join(
                    carbon_input_folder,
                    f"{fc}_{country_case}_{constraint}.csv"))
        c_df = c_df.groupby(
                            ["reference_mineral","iso3","processing_stage"]
                            )[carbon_column].sum().reset_index()
        c_df["scenario"] = l
        c_df["year"] = y
        all_dfs.append(c_df)

    all_dfs = pd.concat(all_dfs,axis=0,ignore_index=True)
    all_dfs = all_dfs.groupby(
                    ["year","scenario","reference_mineral","iso3","processing_stage"]
                    ).agg(dict([(c,"sum") for c in all_sums])).reset_index()

    all_dfs = pd.merge(all_dfs,price_costs_df,how="left",on=index_cols).fillna(0)
    all_dfs["revenue_usd"] = all_dfs["export_tonnes"]*all_dfs["price_usd_per_tonne"]
    all_dfs["capex_usd"] = all_dfs["production_tonnes"]*all_dfs["capex_usd_per_tonne"]
    all_dfs["opex_usd"] = all_dfs["production_tonnes"]*all_dfs["opex_usd_per_tonne"]
    all_dfs["production_cost_usd"] = all_dfs["capex_usd"] + all_dfs["opex_usd"]
    for idx,(p,r) in enumerate(zip(["price","capex","opex"],["revenue","capex","opex"])):
        all_dfs[
            f"{p}_usd_per_tonne"
            ] = np.where(
                    all_dfs[f"{r}_usd"] > 0,
                    all_dfs[f"{p}_usd_per_tonne"],
                    0
            )
    all_dfs = all_dfs.set_index(["year","scenario","reference_mineral","iso3","processing_stage"])
    all_dfs.to_excel(writer,sheet_name=f"{country_case}_{constraint}")
    writer.close()


if __name__ == '__main__':
    CONFIG = load_config()
    try:
        country_case = str(sys.argv[1])
        constraint = str(sys.argv[2])
    except IndexError:
        print("Got arguments", sys.argv)
        exit()
    main(CONFIG,country_case,constraint)