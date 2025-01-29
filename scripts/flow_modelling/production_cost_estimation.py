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

def get_costs_constant_rates(years=[2022,2030,2040]):
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

    costs_df = []
    index_cols = ["year","reference_mineral","processing_stage"]
    for y in years:
        cdf = capex_df[["reference_mineral","processing_stage",y]]
        cdf["year"] = y
        cdf.rename(columns={y:"capex_usd_per_tonne"},inplace=True)

        odf = opex_df[["reference_mineral","processing_stage",y]]
        odf["year"] = y
        odf.rename(columns={y:"opex_usd_per_tonne"},inplace=True)

        pc_df = pd.concat(
                        [
                            cdf.set_index(index_cols),
                            odf.set_index(index_cols)
                        ],
                        axis=1)
        costs_df.append(pc_df.reset_index())

    costs_df = pd.concat(costs_df,axis=0,ignore_index=True)
    costs_df["production_cost_usd_per_tonne"
        ] = costs_df["capex_usd_per_tonne"] + costs_df["opex_usd_per_tonne"]
    return costs_df

def unit_costs_calculations(x,constant_rate_df,curves_df):
    mineral = x.reference_mineral
    stage = x.final_production_stage
    c_df = curves_df[
                        (
                            curves_df["reference_mineral"] == mineral
                        ) & (
                            curves_df["processing_stage"] == stage
                        )
                    ]
    if len(c_df.index) > 0:
        a = c_df["a"].values[0]
        b = c_df["b"].values[0]
        k = c_df["k"].values[0]
        return a*np.exp(-k*x) + b
    else:
        c_df = constant_rate_df[
                        (
                            constant_rate_df["reference_mineral"] == mineral
                        ) & (
                            constant_rate_df["processing_stage"] == stage
                        )
                    ]
        return c_df["production_cost_usd_per_tonne"].values[0]

def add_mines_remaining_tonnages(df,mines_df,year,metal_factor,costs_df,cost_curves_df):
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
    m_df = pd.merge(
                m_df,
                mines_df[["id",str(year)]],
                how="left",left_on=["origin_id"],
                right_on=["id"]).fillna(0)
    m_df["initial_stage_production_tons"] = m_df[str(year)] - m_df["initial_stage_production_tons"]
    m_df = m_df[m_df["initial_stage_production_tons"] > 0]
    if len(m_df.index) > 0:
        m_df["final_processing_stage"] = 1.0
        m_df["final_stage_production_tons"] = m_df["initial_stage_production_tons"]/metal_factor
        m_df["trade_type"] = "Other"
        m_df["final_processing_location"] = "city_demand"
        m_df["import_country_code"] = m_df["export_country_code"]
        m_df["production_cost_usd_per_tonne"
            ] = m_df.progress_apply(lambda x:unit_costs_calculations(x,costs_df,cost_curves_df),axis=1)
        m_df["production_cost_usd"] = m_df["production_cost_usd_per_tonne"]*m_df["final_stage_production_tons"]
        m_df.drop(["id",str(year)],axis=1,inplace=True)
        df = pd.concat([df,m_df],axis=0,ignore_index=True)

    return df


def main(
            year,
            percentile,
            efficient_scale,
            country_case,
            constraint,
            combination = None,
            distance_from_origin=0.0,
            environmental_buffer=0.0
        ):
    
    baseline_year = 2022
    if year == baseline_year:
        csv_file_name = f"location_costs_{year}_{percentile}"
    else:
        csv_file_name = f"location_costs_{year}_{percentile}_{efficient_scale}"
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
    
    results_folder = os.path.join(output_data_path,"production_costs")
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
        metal_content_factors_df, 
        ccg_countries, mine_city_stages, _, _
    ) = get_common_input_dataframes(data_type,year,baseline_year)

    costs_df = get_costs_constant_rates(years=[year])
    cost_curves_df = pd.read_excel(
                        os.path.join(
                            processed_data_path,
                            "production_costs",
                            "cost_curves.xlsx"
                            )
                        )

    """Step 1: get all the relevant nodes and find their distances 
                to grid and bio-diversity layers 
    """
    all_flows = []
    for reference_mineral in reference_minerals:
        # Find year locations
        if year == baseline_year:
            export_file_path = os.path.join(
                            output_data_path,
                            "flow_od_paths",
                            f"{reference_mineral}_flow_paths_{year}_{percentile}.parquet")
        else:
            export_file_path = os.path.join(
                            modified_paths_folder,
                            f"{reference_mineral}_flow_paths_{year}_{percentile}_{efficient_scale}.parquet")
            
        metal_factor = metal_content_factors_df[
                    metal_content_factors_df["reference_mineral"] == reference_mineral
                    ]["metal_content_factor"].values[0]

        mines_df = get_mine_layer(reference_mineral,year,percentile,
                            mine_id_col="id")
        
        export_df = pd.read_parquet(export_file_path)
        od_df = export_df[export_df["trade_type"] == "Export"]
        od_df = export_df[export_df["export_country_code"] != export_df["import_country_code"]]

        od_df["production_cost_usd_per_tonne"
            ] = od_df.progress_apply(lambda x:unit_costs_calculations(x,costs_df,cost_curves_df),axis=1)
        od_df["production_cost_usd"] = od_df["production_cost_usd_per_tonne"]*od_df["final_stage_production_tons"]
        sum_cols = trade_ton_columns + ["production_cost_usd"]
        df = add_mines_remaining_tonnages(od_df,mines_df,year,metal_factor,costs_df,cost_curves_df)
        df = df.groupby(
                        [
                        "reference_mineral",
                        "export_country_code",
                        "final_processing_stage",
                        ]).agg(dict([(c,"sum") for c in sum_cols])).reset_index()
        df.rename(columns={"export_country_code":"iso3"},inplace=True)
        all_flows.append(df)

    all_flows = pd.concat(all_flows,axis=0,ignore_index=True)
    all_flows["production_cost_usd_per_tonne"
        ] = np.where(
                all_flows["final_stage_production_tons"] > 0,
                all_flows["production_cost_usd"]/all_flows["final_stage_production_tons"],
                0
                )
    all_flows.to_csv(
            os.path.join(
                results_folder,
                results_file),
            index=False)


if __name__ == '__main__':
    try:
        if len(sys.argv) > 6:
            year = int(sys.argv[1])
            percentile = str(sys.argv[2])
            efficient_scale = str(sys.argv[3])
            country_case = str(sys.argv[4])
            constraint = str(sys.argv[5])
            combination = str(sys.argv[6])
            distance_from_origin = float(sys.argv[7])
            environmental_buffer = float(sys.argv[8])
        else:
            year = int(sys.argv[1])
            percentile = str(sys.argv[2])
            efficient_scale = str(sys.argv[3])
            country_case = str(sys.argv[4])
            constraint = str(sys.argv[5])
            combination = None
            distance_from_origin = 0.0
            environmental_buffer = 0.0
    except IndexError:
        print("Got arguments", sys.argv)
        exit()
    main(
            year,
            percentile,
            efficient_scale,
            country_case,
            constraint,
            combination = combination,
            distance_from_origin=distance_from_origin,
            environmental_buffer=environmental_buffer)