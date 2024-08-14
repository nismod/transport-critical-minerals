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
    # tonnage_types = ["production","export","import_ccg","import_nonccg"]
    tonnage_types = ["production","export","import"] 
    cost_column = "total_gcosts" 
    carbon_column = "tonsCO2eq"
    cost_sum_dict = [(final_tons_column,"sum"),(cost_column,"sum")]

    #  Get a number of input dataframes
    baseline_year = 2022
    data_type = {"initial_refined_stage":"str","final_refined_stage":"str"}
    (
        pr_conv_factors_df, 
        metal_content_factors_df, 
        _, _, _
    ) = get_common_input_dataframes(data_type,baseline_year,baseline_year)

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

    gdp_df = pd.read_excel(
                        os.path.join(
                            processed_data_path,
                            "production_costs",
                            "GDP Projections Critical Minerals 1.xlsx"),
                        sheet_name = "IMF")
    gdp_df.columns = ["country_name","iso3",2022,2030,2040]
    years = [2022,2030,2040]
    price_costs_df = []
    regional_gdp_df = []
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

        g_df = gdp_df[["iso3",y]]
        g_df["year"] = y
        g_df[y] = 1.0e9*g_df[y]
        g_df.rename(columns={y:"gdp_usd"},inplace=True)
        regional_gdp_df.append(g_df)

    price_costs_df = pd.concat(price_costs_df,axis=0,ignore_index=True)
    regional_gdp_df = pd.concat(regional_gdp_df,axis=0,ignore_index=True)

    output_file = os.path.join(
                        results_folder,
                        "transport_totals_by_stage.xlsx")
    if os.path.isfile(output_file) is True:
        writer = pd.ExcelWriter(output_file,mode='a',engine="openpyxl",if_sheet_exists='replace')
    else:
        writer = pd.ExcelWriter(output_file)

    output_file = os.path.join(
                        results_folder,
                        "value_added_totals.xlsx")
    if os.path.isfile(output_file) is True:
        writer_t = pd.ExcelWriter(output_file,mode='a',engine="openpyxl",if_sheet_exists='replace')
    else:
        writer_t = pd.ExcelWriter(output_file)
    
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
    # all_sums = [
    #             "production_tonnes",
    #             "export_tonnes",
    #             "import_ccg_tonnes",
    #             "import_nonccg_tonnes",
    #             "export_transport_cost_usd",
    #             "export_transport_cost_usd_per_tonne",
    #             "import_ccg_transport_cost_usd",
    #             "import_ccg_transport_cost_usd_per_tonne",
    #             "import_nonccg_transport_cost_usd",
    #             "import_nonccg_transport_cost_usd_per_tonne",
    #             carbon_column
    #             ]
    all_sums = [
                "production_tonnes",
                "stage_1_production_for_export_tonnes",
                "export_tonnes",
                "import_tonnes",
                "export_transport_cost_usd",
                "export_transport_cost_usd_per_tonne",
                "import_transport_cost_usd",
                "import_transport_cost_usd_per_tonne",
                carbon_column
                ]
    for idx, (ft,fc,l,y) in enumerate(zip(all_tons_files,all_carbon_files,all_layers,all_years)):
        fname = os.path.join(
                    tons_input_folder,
                    f"{ft}_{country_case}_{constraint}.csv")
        if os.path.exists(fname):
            t_df = pd.read_csv(fname)
            # Get the total production volumes 
            tr_df = pd.DataFrame()
            for pt in tonnage_types:
                if pt == "production":
                    p_df = t_df[~t_df["trade_type"].isin(["Import_CCG","Import_NonCCG"])]
                    """Total metal and stage 1 production - including export + domestic + other uses
                    """
                    metal_df = p_df[p_df["initial_processing_stage"] == 0]
                    metal_df = metal_df.groupby(
                                    ["reference_mineral","iso3","initial_processing_stage"]
                                    )[initial_tons_column].sum().reset_index()
                    metal_df.rename(
                                columns={
                                        "initial_processing_stage":"processing_stage",
                                        initial_tons_column:f"{pt}_tonnes"},
                                inplace=True)
                    metal_df["scenario"] = l
                    metal_df["year"] = y
                    all_dfs.append(metal_df)
                    st_1_df = pd.merge(metal_df,metal_content_factors_df,how="left",on=["reference_mineral"])
                    st_1_df[f"{pt}_tonnes"
                        ] = st_1_df[f"{pt}_tonnes"]/st_1_df["metal_content_factor"]
                    st_1_df["processing_stage"] = 1.0
                    st_1_df.drop("metal_content_factor",axis=1,inplace=True)
                    all_dfs.append(st_1_df)
                    p_df = p_df[p_df["final_processing_stage"] > 1.0]
                elif pt == "export":
                    p_df = t_df[t_df["trade_type"] == "Export"]
                    tr_df = t_df[t_df["trade_type"].isin(["Export","Domestic"])]
                    
                    trade_types = ["Export","Domestic"]
                    st_1_export_df = []
                    for tt in trade_types:
                        if tt == "Export":
                            st_1_df = t_df[
                                    (
                                        t_df["trade_type"] == tt
                                    ) & (
                                        t_df["initial_processing_stage"] == 0
                                    ) & (
                                        t_df["final_processing_stage"] > 1.0
                                    )
                                    ] 
                        else:
                            st_1_df = t_df[
                                    (
                                        t_df["trade_type"] == tt
                                    ) & (
                                        t_df["initial_processing_stage"] == 0
                                    ) & (
                                        t_df["final_processing_location"] != "city_demand"
                                    )
                                    ] 

                        st_1_df = st_1_df.groupby(
                                        ["reference_mineral","iso3","initial_processing_stage"]
                                        )[initial_tons_column].sum().reset_index()
                        st_1_df.rename(
                                    columns={
                                            "initial_processing_stage":"processing_stage",
                                            initial_tons_column:"stage_1_production_for_value_added_export_tonnes"},
                                    inplace=True)
                        st_1_df = pd.merge(st_1_df,metal_content_factors_df,how="left",on=["reference_mineral"])
                        st_1_df["stage_1_production_for_value_added_export_tonnes"
                            ] = st_1_df["stage_1_production_for_value_added_export_tonnes"]/st_1_df["metal_content_factor"]
                        st_1_df["processing_stage"] = 1.0
                        st_1_df.drop("metal_content_factor",axis=1,inplace=True)
                        st_1_export_df.append(st_1_df)

                    st_1_export_df = pd.concat(st_1_export_df).groupby(
                                        ["reference_mineral","iso3","processing_stage"],
                                        as_index=False).sum()
                    st_1_export_df["scenario"] = l
                    st_1_export_df["year"] = y
                    all_dfs.append(st_1_export_df)
                # elif pt == "import_ccg":
                #     p_df = t_df[t_df["trade_type"] == "Import_CCG"]
                #     tr_df = p_df.copy()
                # else:
                #     p_df = t_df[t_df["trade_type"] == "Import_NonCCG"]
                #     tr_df = p_df.copy()
                else:
                    p_df = t_df[t_df["trade_type"].isin(["Import_NonCCG","Import_CCG"])]
                    tr_df = p_df.copy()


                non_metal_df = p_df.groupby(
                                ["reference_mineral","iso3","final_processing_stage"]
                                )[final_tons_column].sum().reset_index()
                non_metal_df.rename(
                            columns={
                                    "final_processing_stage":"processing_stage",
                                    final_tons_column:f"{pt}_tonnes"},
                            inplace=True)
                non_metal_df["scenario"] = l
                non_metal_df["year"] = y
                all_dfs.append(non_metal_df)
                
                """Transport costs
                """
                if len(tr_df.index) > 0:
                    m_df = tr_df.groupby(
                                    ["reference_mineral","iso3","final_processing_stage"]
                                    ).agg(dict(cost_sum_dict)).reset_index()
                    m_df[f"{pt}_transport_cost_usd_per_tonne"
                        ] = m_df[cost_column]/m_df[final_tons_column]
                    m_df.rename(
                                columns={
                                        "final_processing_stage":"processing_stage",
                                        cost_column:f"{pt}_transport_cost_usd"},
                                inplace=True)
                    m_df.drop(final_tons_column,axis=1,inplace=True)
                    m_df["scenario"] = l
                    m_df["year"] = y
                    all_dfs.append(m_df)
            
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
    all_dfs["expenditure_usd"] = all_dfs["import_tonnes"]*all_dfs["price_usd_per_tonne"]
    all_dfs["capex_usd"] = all_dfs["production_tonnes"]*all_dfs["capex_usd_per_tonne"]
    all_dfs["opex_usd"] = all_dfs["production_tonnes"]*all_dfs["opex_usd_per_tonne"]
    all_dfs["production_cost_usd"] = all_dfs["capex_usd"] + all_dfs["opex_usd"]
    all_dfs["stage1_production_cost_usd"
        ] = all_dfs["stage_1_production_for_value_added_export_tonnes"]*(
            all_dfs["capex_usd_per_tonne"] + all_dfs["opex_usd_per_tonne"])
    all_dfs["stage1_production_cost_usd_opex_only"
        ] = all_dfs["stage_1_production_for_value_added_export_tonnes"]*all_dfs["opex_usd_per_tonne"]
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

    all_sums = ["revenue_usd","stage1_production_cost_usd","stage1_production_cost_usd_opex_only","expenditure_usd"]
    all_dfs = all_dfs.reset_index()
    all_dfs = all_dfs.groupby(
                ["year","scenario","reference_mineral","iso3"]
                ).agg(dict([(c,"sum") for c in all_sums])).reset_index()
    all_dfs["value_added_usd"] = all_dfs["revenue_usd"] - all_dfs["stage1_production_cost_usd"] - all_dfs["expenditure_usd"]
    all_dfs["value_added_usd_opex_only"] = all_dfs["revenue_usd"] - all_dfs["stage1_production_cost_usd_opex_only"] - all_dfs["expenditure_usd"]
    all_dfs = pd.merge(all_dfs,regional_gdp_df,how="left",on=["iso3","year"])
    all_dfs = all_dfs.set_index(["year","scenario","reference_mineral","iso3"])
    all_dfs.to_excel(writer_t,sheet_name=f"{country_case}_{constraint}")
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