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

def add_mines_remaining_tonnages(df,mines_df,year):
    m_df = df[
                (
                    df["initial_processing_location"] == "mine"
                ) & (
                    df["initial_processing_stage"] == 0.0
                )
            ]
    m_df = m_df.groupby(
                        [
                        "reference_mineral",
                        "export_country_code",
                        "initial_processing_stage",
                        "initial_processing_location",
                        "origin_id"]
                        ).agg(dict([(c,"sum") for c in ["initial_stage_production_tons"]])).reset_index()
    mines_df.rename(columns={"id":"origin_id","iso3":"export_country_code"},inplace=True) 
    m_df = pd.merge(
                mines_df[
                            [
                                "origin_id",
                                "export_country_code",
                                "reference_mineral",
                                "initial_processing_location",
                                "metal_content_factor",
                                f"{year}_metal_content"
                            ]
                        ],
                m_df,
                how="left",
                on=[
                    "origin_id",
                    "export_country_code",
                    "reference_mineral",
                    "initial_processing_location"
                    ]
                ).fillna(0)
    m_df["initial_stage_production_tons"] = m_df[f"{year}_metal_content"] - m_df["initial_stage_production_tons"]
    m_df = m_df[m_df["initial_stage_production_tons"] > 0]
    if len(m_df.index) > 0:
        m_df["final_processing_stage"] = 1.0
        m_df["final_stage_production_tons"] = m_df["initial_stage_production_tons"]*m_df["metal_content_factor"]
        m_df["trade_type"] = "Other"
        m_df["final_processing_location"] = "city_demand"
        m_df["import_country_code"] = m_df["export_country_code"]
        m_df.drop([f"{year}_metal_content"],axis=1,inplace=True)
        df = pd.concat([df,m_df],axis=0,ignore_index=True)

    return df

# def add_mines_remaining_tonnages(df,mines_df,year,metal_factor):
#     m_df = df[
#                 (
#                     df["initial_processing_location"] == "mine"
#                 ) & (
#                     df["initial_processing_stage"] == 0.0
#                 )
#             ]
#     m_df = m_df.groupby(
#                         [
#                         "reference_mineral",
#                         "export_country_code",
#                         "initial_processing_stage",
#                         "initial_processing_location",
#                         "origin_id"]
#                         ).agg(dict([(c,"sum") for c in ["initial_stage_production_tons"]])).reset_index() 
#     m_df = pd.merge(
#                 m_df,
#                 mines_df[["id",str(year)]],
#                 how="left",left_on=["origin_id"],
#                 right_on=["id"]).fillna(0)
#     m_df["initial_stage_production_tons"] = m_df[str(year)] - m_df["initial_stage_production_tons"]
#     m_df = m_df[m_df["initial_stage_production_tons"] > 0]
#     if len(m_df.index) > 0:
#         m_df["final_processing_stage"] = 1.0
#         m_df["final_stage_production_tons"] = m_df["initial_stage_production_tons"]/metal_factor
#         m_df["trade_type"] = "Other"
#         m_df["final_processing_location"] = "city_demand"
#         m_df["import_country_code"] = m_df["export_country_code"]
#         m_df.drop(["id",str(year)],axis=1,inplace=True)
#         df = pd.concat([df,m_df],axis=0,ignore_index=True)

#     return df

def main(
            config,
            scenario,
            year,
            percentile,
            efficient_scale,
            country_case,
            constraint,
            combination = None,
            distance_from_origin=0.0,
            environmental_buffer=0.0
        ):
    incoming_data_path = config['paths']['incoming_data']
    processed_data_path = config['paths']['data']
    output_data_path = config['paths']['results']

    baseline_year = 2022
    scenario_rename = scenario.replace(" ","_")
    if year == baseline_year:
        csv_file_name = f"location_totals_{year}_{percentile}"
    else:
        csv_file_name = f"location_totals_{scenario_rename}_{year}_{percentile}_{efficient_scale}"
    if combination is None:
        input_folder = os.path.join(output_data_path,f"flow_optimisation_{country_case}_{constraint}")
        results_file = f"{csv_file_name}_{country_case}_{constraint}.csv"

    else:
        if distance_from_origin > 0.0 or environmental_buffer > 0.0:
            input_folder = os.path.join(
                                output_data_path,
                                f"{combination}_flow_optimisation_{country_case}_{constraint}_op_{distance_from_origin}km_eb_{environmental_buffer}km"
                                )
            ds = str(distance_from_origin).replace('.','p')
            eb = str(environmental_buffer).replace('.','p')
            results_file = f"{combination}_{csv_file_name}_{country_case}_{constraint}_op_{ds}km_eb_{eb}km.csv"
        else:
            input_folder = os.path.join(
                                    output_data_path,
                                    f"{combination}_flow_optimisation_{country_case}_{constraint}"
                                    )
            results_file = f"{combination}_{csv_file_name}_{country_case}_{constraint}.csv"
    
    results_folder = os.path.join(output_data_path,"tonnage_summaries")
    os.makedirs(results_folder,exist_ok=True)

    modified_paths_folder = os.path.join(input_folder,"modified_flow_od_paths")
    """Step 1: Get the input datasets
    """
    reference_minerals = ["graphite","lithium","cobalt","manganese","nickel","copper"]
    trade_ton_columns = [
                            "initial_stage_production_tons",
                            "final_stage_production_tons"
                        ]

    #  Get a number of input dataframes
    data_type = {"initial_refined_stage":"str","final_refined_stage":"str"}
    (
        pr_conv_factors_df, 
        _, 
        ccg_countries,_, _, _
    ) = get_common_input_dataframes(data_type,scenario,year,baseline_year)
    """Step 1: get all the relevant nodes and find their distances 
                to grid and bio-diversity layers 
    """
    all_flows = []
    for reference_mineral in reference_minerals:
        # Find year locations
        if year == baseline_year:
            export_file_path = import_file_path = os.path.join(
                            output_data_path,
                            "flow_od_paths",
                            f"{reference_mineral}_flow_paths_{year}_{percentile}.parquet")
        else:
            export_file_path = os.path.join(
                            modified_paths_folder,
                            f"{reference_mineral}_flow_paths_{scenario_rename}_{year}_{percentile}_{efficient_scale}.parquet")
            import_file_path = os.path.join(
                            output_data_path,
                            "flow_od_paths",
                            f"{reference_mineral}_flow_paths_{scenario_rename}_{year}_{percentile}_{efficient_scale}.parquet")

        metal_factor = pr_conv_factors_df[
                    pr_conv_factors_df["reference_mineral"] == reference_mineral
                    ][["iso3","reference_mineral","metal_content_factor"]]
        metal_factor = metal_factor.drop_duplicates(subset=["iso3","reference_mineral"],keep="first")
        mines_df["year"] = year
        mines_df["reference_mineral"] = reference_mineral
        mines_df["initial_processing_location"] = "mine"
        mines_df = pd.merge(mines_df,metal_factor,how="left",on=["iso3","reference_mineral"])

        mines_df = get_mine_layer(reference_mineral,year,percentile,
                            mine_id_col="id")
        
        export_df = pd.read_parquet(export_file_path)
        import_df = pd.read_parquet(import_file_path)

        export_df = export_df[export_df["trade_type"] != "Import"]
        import_df = import_df[
                                (
                                    import_df["export_country_code"] != import_df["import_country_code"]
                                ) & (
                                    import_df["import_country_code"].isin(ccg_countries)
                                )
                            ]
        if year > baseline_year:
            imp_df = export_df[
                                (
                                    export_df["export_country_code"] != export_df["import_country_code"]
                                ) & (
                                    export_df["import_country_code"].isin(ccg_countries)
                                )
                            ]
            import_df = pd.concat([import_df,imp_df],axis=0,ignore_index=True)
        import_df["trade_type"
            ] = np.where(
                    import_df["export_country_code"].isin(ccg_countries),
                    "Import_CCG",
                    "Import_NonCCG"
                    )
        for idx, (od_type,od_df) in enumerate(zip(["export","import"],[export_df,import_df])):
            if len(od_df.index) > 0:
                od_df["total_gcosts_per_tons"] = od_df.progress_apply(lambda x:sum(x["gcost_usd_tons_path"]),axis=1)
                od_df["total_gcosts_usd"] = od_df["total_gcosts_per_tons"]*od_df["final_stage_production_tons"]
                od_df["trade_type"
                    ] = np.where(
                            od_df["export_country_code"] == od_df["import_country_code"],
                            "Domestic",
                            od_df["trade_type"]
                            )
                sum_cols = trade_ton_columns + ["total_gcosts_usd"]
                if od_type == "export":
                    df = add_mines_remaining_tonnages(od_df,mines_df,year,metal_factor)
                    df = df.groupby(
                                    [
                                    "reference_mineral",
                                    "export_country_code",
                                    "initial_processing_stage",
                                    "final_processing_stage",
                                    "initial_processing_location",
                                    "final_processing_location",
                                    "trade_type"
                                    ]).agg(dict([(c,"sum") for c in sum_cols])).reset_index()
                    df.rename(columns={"export_country_code":"iso3"},inplace=True)
                else:
                    df = od_df.groupby(
                                    [
                                    "reference_mineral",
                                    "import_country_code",
                                    "initial_processing_stage",
                                    "final_processing_stage",
                                    "initial_processing_location",
                                    "final_processing_location",
                                    "trade_type"
                                    ]).agg(dict([(c,"sum") for c in sum_cols])).reset_index()
                    df.rename(columns={"import_country_code":"iso3"},inplace=True)

                all_flows.append(df)

    all_flows = pd.concat(all_flows,axis=0,ignore_index=True)
    all_flows["average_gcost_usd_per_tons"
        ] = np.where(
                all_flows["final_stage_production_tons"] > 0,
                all_flows["total_gcosts_usd"]/all_flows["final_stage_production_tons"],
                0
                )
    all_flows.to_csv(
            os.path.join(
                results_folder,
                results_file),
            index=False)


if __name__ == '__main__':
    CONFIG = load_config()
    try:
        if len(sys.argv) > 7:
            scenario = str(sys.argv[1])
            year = int(sys.argv[2])
            percentile = str(sys.argv[3])
            efficient_scale = str(sys.argv[4])
            country_case = str(sys.argv[5])
            constraint = str(sys.argv[6])
            combination = str(sys.argv[7])
            distance_from_origin = float(sys.argv[8])
            environmental_buffer = float(sys.argv[9])
        else:
            scenario = str(sys.argv[1])
            year = int(sys.argv[2])
            percentile = str(sys.argv[3])
            efficient_scale = str(sys.argv[4])
            country_case = str(sys.argv[5])
            constraint = str(sys.argv[6])
            combination = None
            distance_from_origin = 0.0
            environmental_buffer = 0.0
    except IndexError:
        print("Got arguments", sys.argv)
        exit()
    main(
            CONFIG,
            scenario,
            year,
            percentile,
            efficient_scale,
            country_case,
            constraint,
            combination = combination,
            distance_from_origin=distance_from_origin,
            environmental_buffer=environmental_buffer)