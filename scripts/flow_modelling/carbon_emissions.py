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

def add_mines_remaining_tonnages(df,mines_df,year,metal_factor):
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
        m_df["import_country_code"] = m_df["export_country_code"]
        m_df.drop(["id",str(year)],axis=1,inplace=True)
        df = pd.concat([df,m_df],axis=0,ignore_index=True)

    return df

def main(config,year,percentile,efficient_scale,country_case,constraint):
    incoming_data_path = config['paths']['incoming_data']
    processed_data_path = config['paths']['data']
    output_data_path = config['paths']['results']

    input_folder = os.path.join(output_data_path,"node_edge_flows")
    results_folder = os.path.join(output_data_path,"carbon_emissions_summaries")
    if os.path.exists(results_folder) == False:
        os.mkdir(results_folder)

    """Step 1: Get the input datasets
    """
    reference_minerals = ["graphite","lithium","cobalt","manganese","nickel","copper"]
    trade_ton_column = "final_stage_production_tons"

    carbon_emission_df = pd.read_excel(
    							os.path.join(
    								processed_data_path,
    								"transport_costs",
    								"carbon_emission_factors.xlsx")
    							)
    country_codes_and_projections = pd.read_excel(
                        os.path.join(
                            processed_data_path,
                            "local_projections.xlsx")
                        )
    countries = country_codes_and_projections["iso3"].values.tolist()

    global_boundaries = gpd.read_file(os.path.join(processed_data_path,
                                    "admin_boundaries",
                                    "gadm36_levels_gpkg",
                                    "gadm36_levels_continents.gpkg"))
    global_boundaries = global_boundaries[global_boundaries["ISO_A3"].isin(countries)]
    
    all_flows = []
    sum_cols = [(c,"sum") for c in [trade_ton_column,"length_km","ton_km"]]
    for reference_mineral in reference_minerals:
        # Find year locations
        if year == 2022:
            layer_name = f"{reference_mineral}_{percentile}"
        else:
            layer_name = f"{reference_mineral}_{percentile}_{efficient_scale}"
        flows_gdf = gpd.read_parquet(
                            os.path.join(
                                input_folder,
                                f"edges_flows_{layer_name}_{year}_{country_case}_{constraint}.geoparquet"))
        flows_gdf = flows_gdf[flows_gdf["mode"].isin(["road","rail"])]
        for row in country_codes_and_projections.itertuples():
            boundary_df = global_boundaries[global_boundaries["ISO_A3"] == row.iso3]
            df = gpd.clip(flows_gdf,boundary_df)

            if len(df.index) > 0:
                df = df.to_crs(epsg=row.projection_epsg)
                df.rename(columns={f"{reference_mineral}_{trade_ton_column}":trade_ton_column},inplace=True)
                df["reference_mineral"] = reference_mineral
                df["iso3"] = row.iso3
                df["length_km"] = 0.001*df.geometry.length
                df["ton_km"] = df[f"{reference_mineral}_{trade_ton_column}"]*df["length_km"]
                df = df.groupby(["reference_mineral","iso3","mode"]).agg(dict(sum_cols)).reset_index()
                all_flows.append(df)
            print (f"Done with {row.iso3} for {reference_mineral}")

    all_flows = pd.concat(all_flows,axis=0,ignore_index=True)
    all_flows = pd.merge(all_flows,carbon_emission_df,how="left",on=["mode"]).fillna(0)
    all_flows["tonsCO2eq"] = 0.001*(
    							all_flows["ton_km"]/all_flows["veh_wt_tons"]
    							)*(0.01*all_flows["spec_energ_consump"]*all_flows["Density_kg"])
    if year == 2022:
        file_name = f"carbon_emission_totals_{year}_{percentile}"
    else:
        file_name = f"carbon_emission_totals_{year}_{percentile}_{efficient_scale}"
    all_flows.to_csv(
            os.path.join(
                results_folder,
                f"{file_name}_{country_case}_{constraint}.csv"),
            index=False)


if __name__ == '__main__':
    CONFIG = load_config()
    try:
        year = int(sys.argv[1])
        percentile = str(sys.argv[2])
        efficient_scale = str(sys.argv[3])
        country_case = str(sys.argv[4])
        constraint = str(sys.argv[5])
    except IndexError:
        print("Got arguments", sys.argv)
        exit()
    main(CONFIG,year,percentile,efficient_scale,country_case,constraint)