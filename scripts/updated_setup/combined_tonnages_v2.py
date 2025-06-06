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

config = load_config()
incoming_data_path = config['paths']['incoming_data']
processed_data_path = config['paths']['data']
output_data_path = config['paths']['results']

def get_stage_one_conversion_factors(x,pcf_df,cf_column="aggregate_ratio"):
    ref_min = x["reference_mineral"]
    exp_st = x["final_processing_stage"]
    cf_df = pcf_df[pcf_df["reference_mineral"] == ref_min]
    cf_val = cf_df[cf_df["final_refined_stage"] == str(exp_st).replace(".0","")
                    ][cf_column].values[0]/cf_df[
                    cf_df["final_refined_stage"] == '1'
                    ][cf_column].values[0]
    return cf_val

def get_prices_costs_gdp_waterintensity(years=[2022,2030,2040]):
    price_df = pd.read_excel(
                        os.path.join(
                            processed_data_path,
                            "production_costs",
                            "Final_Price_and_Costs_RP.xlsx"),
                    sheet_name = "Price_final",index_col=[0])
    price_df = price_df.reset_index()

    gdp_df = pd.read_excel(
                        os.path.join(
                            processed_data_path,
                            "production_costs",
                            "GDP Projections Critical Minerals 1.xlsx"),
                        sheet_name = "IMF")
    gdp_df.columns = ["country_name","iso3"] + years
    price_costs_df = []
    regional_gdp_df = []
    index_cols = ["year","reference_mineral","processing_stage"]
    for y in years:
        pdf = price_df[["reference_mineral","processing_stage",y]]
        pdf["year"] = y
        pdf.rename(columns={y:"price_usd_per_tonne"},inplace=True)
        price_costs_df.append(pdf)

        g_df = gdp_df[["iso3",y]]
        g_df["year"] = y
        g_df[y] = 1.0e9*g_df[y]
        g_df.rename(columns={y:"gdp_usd"},inplace=True)
        regional_gdp_df.append(g_df)

    price_costs_df = pd.concat(price_costs_df,axis=0,ignore_index=True)
    regional_gdp_df = pd.concat(regional_gdp_df,axis=0,ignore_index=True)

    water_intensity_df = pd.read_csv(
                            os.path.join(
                                processed_data_path,
                                "mineral_usage_factors",
                                "water_intensities_final.csv")
                            )[["reference_mineral","processing_stage","water_intensity_m3_per_kg"]]
    return (price_costs_df,regional_gdp_df,water_intensity_df)

def get_full_file_name(
                fname,
                country_case,
                constraint,
                combination=None,
                distance_from_origin=0.0,
                environmental_buffer=0.0
                ):
    fname = f"{fname}_{country_case}_{constraint}"
    if combination is None:
        return fname
    else:
        if distance_from_origin > 0.0 or environmental_buffer > 0.0:
            ds = str(distance_from_origin).replace('.','p')
            eb = str(environmental_buffer).replace('.','p')
            return f"{combination}_{fname}_op_{ds}km_eb_{eb}km"
        else:
            return f"{combination}_{fname}"

def main(
            config,
            country_case,
            constraint,
            combination = None,
            distance_from_origin=0.0,
            environmental_buffer=0.0
        ):
    tons_input_folder = os.path.join(output_data_path,"tonnage_summaries")
    carbon_input_folder = os.path.join(output_data_path,"carbon_emissions_summaries")
    cost_input_folder = os.path.join(output_data_path,"production_costs")

    results_folder = os.path.join(output_data_path,"result_summaries")
    os.makedirs(results_folder,exist_ok=True)
    #  Get a number of input dataframes
    baseline_year = 2022
    years = [2022,2030,2040]
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
    tonnage_types = ["production","export","import"] 
    cost_column = "total_gcosts_usd" 
    carbon_tons_column = "transport_total_tonkm"
    carbon_divisor_columns = ["transport_total_tonkm","export_tonnes","import_tonnes"]
    cost_sum_dict = [(final_tons_column,"sum"),(cost_column,"sum")]
    index_cols = ["year","reference_mineral","processing_stage"]
    anchor_columns = ["year","scenario","reference_mineral","iso3","processing_type","processing_stage"]
    column_dictionary = {
                            "anchor": [
                                        "year",
                                        "scenario",
                                        "reference_mineral",
                                        "iso3",
                                        "processing_type",
                                        "processing_stage"
                                        ],
                            "tonnages": [
                                            "production_tonnes",
                                            "stage_1_production_for_value_added_export_tonnes",
                                            "export_tonnes",
                                            "import_tonnes",
                                            "production_tonnes_for_costs",
                                        ],
                            "carbon":[
                                        "transport_total_tonkm",
                                        "transport_total_tonsCO2eq",
                                        "transport_export_tonsCO2eq",
                                        "transport_import_tonsCO2eq"
                                    ],
                            "water":["water_usage_m3"],
                            "production_costs":[
                                            "production_cost_usd"
                                            ],
                            "costs":[
                                        "revenue_usd",
                                        "expenditure_usd",
                                        "gdp_usd"
                                    ],
                            "transport_costs":[
                                                "export_transport_cost_usd",
                                                "import_transport_cost_usd"
                                            ],
                            "unit_production_costs":[
                                                    "production_cost_usd_per_tonne"
                                                ],
                            "unit_costs":[
                                            "price_usd_per_tonne"
                                        ],
                            "unit_transport_costs":[
                                                        "export_transport_cost_usd_per_tonne",
                                                        "import_transport_cost_usd_per_tonne"
                                                    ],
                            "unit_carbon":[
                                                "transport_total_tonsCO2eq_pertonkm",
                                                "transport_export_tonsCO2eq_pertonne",
                                                "transport_import_tonsCO2eq_pertonne"
                                        ],
                            "unit_water":["water_intensity_m3_per_kg"]
                        }

    all_sums = column_dictionary["tonnages"
                    ] + column_dictionary["production_costs"
                    ] + column_dictionary["transport_costs"
                    ] + column_dictionary["carbon"
                    ] + column_dictionary["unit_production_costs"
                    ] + column_dictionary["unit_transport_costs"]

    #  Get a number of input dataframes
    data_type = {"initial_refined_stage":"str","final_refined_stage":"str"}
    (
        pr_conv_factors_df, 
        metal_content_factors_df, 
        _, _, _, _
    ) = get_common_input_dataframes(data_type,baseline_year,baseline_year)

    (
        price_costs_df,
        regional_gdp_df,
        water_intensity_df
    ) = get_prices_costs_gdp_waterintensity(years=years)

    stage_names_df = pd.read_excel(
                        os.path.join(
                            processed_data_path,
                            "mineral_usage_factors",
                            "stage_mapping.xlsx"),
                        sheet_name="stage_maps")[["reference_mineral","processing_stage","processing_type"]]
    if combination is None:
        t_file_name = "transport_totals_by_stage.xlsx"
        v_file_name = "value_added_totals.xlsx"
    else:
        if distance_from_origin > 0.0 or environmental_buffer > 0.0:
            ds = str(distance_from_origin).replace('.','p')
            eb = str(environmental_buffer).replace('.','p')
            t_file_name = f"{combination}_transport_totals_by_stage_op_{ds}km_eb_{eb}km.xlsx"
            v_file_name = f"{combination}_value_added_totals_op_{ds}km_eb_{eb}km.xlsx"
        else:
            t_file_name = f"{combination}_transport_totals_by_stage.xlsx"
            v_file_name = f"{combination}_value_added_totals.xlsx"
    
    output_file = os.path.join(
                        results_folder,
                        t_file_name)
    if os.path.isfile(output_file) is True:
        writer = pd.ExcelWriter(output_file,mode='a',engine="openpyxl",if_sheet_exists='replace')
    else:
        writer = pd.ExcelWriter(output_file)

    # output_file = os.path.join(
    #                     results_folder,
    #                     v_file_name)
    # if os.path.isfile(output_file) is True:
    #     writer_t = pd.ExcelWriter(output_file,mode='a',engine="openpyxl",if_sheet_exists='replace')
    # else:
    #     writer_t = pd.ExcelWriter(output_file)
    
    if country_case == "country" and constraint == "unconstrained":
        combos = year_percentile_combinations
    else:
        combos = year_percentile_combinations[1:]

    all_tons_files = []
    all_carbon_files = []
    all_cost_files = []
    all_layers = []
    all_years = []
    for idx, (year,percentile) in enumerate(combos):
        if year == baseline_year:
            tons_file_name = get_full_file_name(
                                    f"location_totals_{year}_baseline",
                                    country_case,
                                    constraint,
                                    combination=combination,
                                    distance_from_origin=distance_from_origin,
                                    environmental_buffer=environmental_buffer
                                    )
            carbon_file_name = get_full_file_name(
                                    f"carbon_emission_totals_{year}_baseline",
                                    country_case,
                                    constraint,
                                    combination=combination,
                                    distance_from_origin=distance_from_origin,
                                    environmental_buffer=environmental_buffer
                                    )
            cost_file_name = get_full_file_name(
                                    f"location_costs_{year}_baseline",
                                    country_case,
                                    constraint,
                                    combination=combination,
                                    distance_from_origin=distance_from_origin,
                                    environmental_buffer=environmental_buffer
                                    )
            layer_name = f"{year}_baseline"
            all_tons_files.append(tons_file_name)
            all_carbon_files.append(carbon_file_name)
            all_cost_files.append(cost_file_name)
            all_layers.append(layer_name)
            all_years.append(year)
        else:
            for th in tonnage_thresholds:
                tons_file_name = get_full_file_name(
                                    f"location_totals_{year}_{percentile}_{th}",
                                    country_case,
                                    constraint,
                                    combination=combination,
                                    distance_from_origin=distance_from_origin,
                                    environmental_buffer=environmental_buffer
                                    )
                carbon_file_name = get_full_file_name(
                                    f"carbon_emission_totals_{year}_{percentile}_{th}",
                                    country_case,
                                    constraint,
                                    combination=combination,
                                    distance_from_origin=distance_from_origin,
                                    environmental_buffer=environmental_buffer
                                    )
                cost_file_name = get_full_file_name(
                                    f"location_costs_{year}_{percentile}_{th}",
                                    country_case,
                                    constraint,
                                    combination=combination,
                                    distance_from_origin=distance_from_origin,
                                    environmental_buffer=environmental_buffer
                                    )
                layer_name = f"{year}_{percentile}_{th}"
                all_tons_files.append(tons_file_name)
                all_carbon_files.append(carbon_file_name)
                all_cost_files.append(cost_file_name)
                all_layers.append(layer_name)
                all_years.append(year)

    all_dfs = []
    for idx, (ft,fc,fcs,l,y) in enumerate(zip(all_tons_files,all_carbon_files,all_cost_files,all_layers,all_years)):
        fname = os.path.join(
                    tons_input_folder,
                    f"{ft}.csv")
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
                        f"{fc}.csv"))
            c_df = c_df[c_df[carbon_tons_column] > 0]
            c_df = c_df.groupby(
                                ["reference_mineral","iso3","processing_stage"]
                                ).agg(dict([(ca,"sum") for ca in column_dictionary["carbon"]])).reset_index()
            c_df["scenario"] = l
            c_df["year"] = y
            all_dfs.append(c_df)

            cst_df = pd.read_csv(
                    os.path.join(
                        cost_input_folder,
                        f"{fcs}.csv"))
            cst_df.rename(
                    columns={
                                "final_processing_stage":"processing_stage",
                                "final_stage_production_tons":"production_tonnes_for_costs"
                            }, 
                    inplace=True
                        )
            cst_df["scenario"] = l
            cst_df["year"] = y
            all_dfs.append(cst_df)


    all_dfs = pd.concat(all_dfs,axis=0,ignore_index=True)
    all_dfs = all_dfs.groupby(
                    ["year","scenario","reference_mineral","iso3","processing_stage"]
                    ).agg(dict([(c,"sum") for c in all_sums])).reset_index()
    for idx,(c,u,d) in enumerate(
                                zip(
                                    column_dictionary["carbon"][1:],
                                    column_dictionary["unit_carbon"],
                                    carbon_divisor_columns
                                    )
                                ):
        all_dfs[u] = np.where(all_dfs[d] > 0,all_dfs[c]/all_dfs[d],0)

    all_dfs = pd.merge(all_dfs,price_costs_df,how="left",on=index_cols).fillna(0)
    all_dfs = pd.merge(all_dfs,water_intensity_df,
                        how="left",
                        on=["reference_mineral","processing_stage"]
                    ).fillna(0)
    all_dfs["water_usage_m3"] = 1.0e3*all_dfs["production_tonnes"]*all_dfs["water_intensity_m3_per_kg"]
    all_dfs["revenue_usd"] = all_dfs["export_tonnes"]*all_dfs["price_usd_per_tonne"]
    all_dfs["expenditure_usd"] = all_dfs["import_tonnes"]*all_dfs["price_usd_per_tonne"]
    all_dfs = pd.merge(all_dfs,regional_gdp_df,how="left",on=["iso3","year"])
    for idx,(p,r) in enumerate(zip(["price"],["revenue"])):
        all_dfs[
            f"{p}_usd_per_tonne"
            ] = np.where(
                    all_dfs[f"{r}_usd"] > 0,
                    all_dfs[f"{p}_usd_per_tonne"],
                    0
            )
    all_dfs["gdp_usd"] = np.where(all_dfs["revenue_usd"] > 0,all_dfs["gdp_usd"],0)
    all_dfs = pd.merge(all_dfs,stage_names_df,how="left",on=["reference_mineral","processing_stage"])
    all_dfs = all_dfs[[x for xs in [v for k,v in column_dictionary.items()] for x in xs]]
    all_dfs = all_dfs.set_index(anchor_columns)
    all_dfs.to_excel(writer,sheet_name=f"{country_case}_{constraint}")
    writer.close()

    # all_sums = ["stage1_production_cost_usd","stage1_production_cost_usd_opex_only","expenditure_usd"]
    # all_dfs = all_dfs.reset_index()
    # revenue_df = all_dfs[all_dfs["processing_stage"] > 1.0]
    # revenue_df = revenue_df.groupby(
    #                     ["year","scenario","reference_mineral","iso3"]
    #                     )["revenue_usd"].sum().reset_index()
    # all_dfs = all_dfs.groupby(
    #             ["year","scenario","reference_mineral","iso3"]
    #             ).agg(dict([(c,"sum") for c in all_sums])).reset_index()
    # all_dfs = pd.merge(
    #                     all_dfs,
    #                     revenue_df,
    #                     how="left",
    #                     on=["year","scenario","reference_mineral","iso3"]
    #                 ).fillna(0)
    # all_dfs["value_added_usd"] = all_dfs["revenue_usd"] - all_dfs["stage1_production_cost_usd"] - all_dfs["expenditure_usd"]
    # all_dfs["value_added_usd_opex_only"] = all_dfs["revenue_usd"] - all_dfs["stage1_production_cost_usd_opex_only"] - all_dfs["expenditure_usd"]
    # all_dfs = pd.merge(all_dfs,regional_gdp_df,how="left",on=["iso3","year"])
    # all_dfs["value_added_gdp_ratio"] = all_dfs["value_added_usd"]/all_dfs["gdp_usd"]
    # all_dfs["value_added_opex_only_gdp_ratio"] = all_dfs["value_added_usd_opex_only"]/all_dfs["gdp_usd"]
    # all_dfs = all_dfs.set_index(["year","scenario","reference_mineral","iso3"])
    # all_dfs.to_excel(writer_t,sheet_name=f"{country_case}_{constraint}")
    # writer_t.close()



if __name__ == '__main__':
    CONFIG = load_config()
    try:
        if len(sys.argv) > 3:
            country_case = str(sys.argv[1])
            constraint = str(sys.argv[2])
            combination = str(sys.argv[3])
            distance_from_origin = float(sys.argv[4])
            environmental_buffer = float(sys.argv[5])
        else:
            country_case = str(sys.argv[1])
            constraint = str(sys.argv[2])
            combination = None
            distance_from_origin = 0.0
            environmental_buffer = 0.0
    except IndexError:
        print("Got arguments", sys.argv)
        exit()
    main(
            CONFIG,
            country_case,
            constraint,
            combination = combination,
            distance_from_origin=distance_from_origin,
            environmental_buffer=environmental_buffer)