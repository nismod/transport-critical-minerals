#!/usr/bin/env python
# coding: utf-8
"""This code finds the route between mines (in Africa) and all ports (outside Africa) 
"""
import sys
import os
import re
import json
import pandas as pd
import igraph as ig
import geopandas as gpd
from utils import *
from transport_cost_assignment import *
from tqdm import tqdm
tqdm.pandas()



def main(config,
        reference_mineral,
        mine_initial_refined_stage,
        mine_final_refined_stage,
        city_initial_refined_stage,
        city_final_refined_stage,percentile):
    incoming_data_path = config['paths']['incoming_data']
    processed_data_path = config['paths']['data']
    output_data_path = config['paths']['results']

    results_folder = os.path.join(output_data_path,"location_outputs")
    if os.path.exists(results_folder) == False:
        os.mkdir(results_folder)

    mine_id_col = "id"
    mine_tons_column = f"{reference_mineral}_output_tons"
    final_columns = [mine_id_col,"reference_mineral",
                        "ISO_A3","final_refined_stage",
                        mine_tons_column,"geometry"]
    """Step 1: Get the input datasets
    """
    pr_conv_factors_df = pd.read_excel(os.path.join(processed_data_path,
                                        "mineral_usage_factors",
                                        "aggregated_stages.xlsx"))[[
                                            "reference_mineral",
                                            "initial_refined_stage",
                                            "final_refined_stage", 
                                            "aggregate_ratio"
                                            ]]
    conversion_factor_column = "aggregate_ratio"

    # Population locations for urban cities
    # Just take the outputs from 2021 and disaggreagte them to the city level
    # We will assume that these outputs do not change till 2040
    un_pop_df = gpd.read_file(os.path.join(processed_data_path,
                                    "admin_boundaries",
                                    "un_urban_population",
                                    "un_pop_df.gpkg"))
    un_pop_df["reference_mineral"] = reference_mineral
    un_pop_df.rename(columns={"city_id":mine_id_col},inplace=True)
    un_pop_crs = un_pop_df.crs
    un_pop_df = un_pop_df[un_pop_df["CONTINENT"] == "Africa"]

    pop_years = [2020,2030,2035]
    years = [2021,2030,2040]
    refined_df = []
    for idx, (year,pop_year) in enumerate(zip(years,pop_years)):
        location_df = []
        if year == 2021:
            mines_df = gpd.read_file(
                            os.path.join(
                                processed_data_path,
                                "minerals",
                                "mine_est_output.gpkg"))
            mines_df = mines_df[~mines_df[f"highest_stage_{reference_mineral}"].isna()]
            mines_df["geometry"] = mines_df.geometry.centroid
            mines_df["reference_mineral"] = reference_mineral
            mines_df.rename(columns={"shapeGroup_primary_admin0":"ISO_A3"},inplace=True)
            mines_df[mine_id_col] = mines_df.progress_apply(lambda x:f"mine_{x['mine_cluster_mini']}",axis=1)
            mines_df["final_refined_stage"] = mines_df[f"highest_stage_{reference_mineral}"]
            mines_df[mine_tons_column] = mines_df[f"{reference_mineral}_processed_ton"] + mines_df[f"{reference_mineral}_unprocessed_ton"]
            refined_df = mines_df[mines_df["final_refined_stage"] == mine_final_refined_stage]
            location_df.append(mines_df[final_columns])
        elif year > 2021:
            mines_df = gpd.read_file(
                            os.path.join(
                                processed_data_path,
                                "minerals",
                                "s_and_p_mines_estimates.gpkg"),layer=f"{reference_mineral}_{percentile}")
            mines_df["reference_mineral"] = reference_mineral
            mines_df.rename(columns={"mine_id":mine_id_col},inplace=True)
            mines_df[mine_tons_column] = mines_df[str(year)]
            mines_df["final_refined_stage"] = mine_initial_refined_stage
            conv_factor_df = pr_conv_factors_df[pr_conv_factors_df["reference_mineral"] == reference_mineral]
            conversion_factor = conv_factor_df[
                                        conv_factor_df["final_refined_stage"] == mine_final_refined_stage
                                        ][conversion_factor_column].values[0]/conv_factor_df[
                                        conv_factor_df["final_refined_stage"] == mine_initial_refined_stage
                                        ][conversion_factor_column].values[0]
            mines_df[mine_tons_column] = np.where(
                                            mines_df["final_refined_stage"] == mine_initial_refined_stage,
                                            1.0*mines_df[mine_tons_column]/conversion_factor,
                                            mines_df[mine_tons_column])
            mines_df["final_refined_stage"] = np.where(
                                            mines_df["final_refined_stage"] == mine_initial_refined_stage,
                                            mine_final_refined_stage,
                                            mines_df["final_refined_stage"])
            location_df.append(mines_df[final_columns])
            location_df.append(refined_df[final_columns])
        trade_df = pd.read_csv(os.path.join(processed_data_path,
                                    "baci",
                                    f"baci_ccg_country_level_trade_{year}.csv"))
        trade_df = trade_df[
                            (
                                trade_df["initial_refined_stage"] == city_initial_refined_stage
                            ) & (
                                trade_df["final_refined_stage"] == city_final_refined_stage
                            ) & (trade_df["reference_mineral"] == reference_mineral)
                            ]
        trade_df = trade_df.groupby(["export_country_code"])["trade_quantity_tons"].sum().reset_index()
        un_pop_df = un_pop_df[un_pop_df["ISO_A3"].isin(trade_df["export_country_code"].values.tolist())]
        # un_pop_df = un_pop_df[["city_id","ISO_A3",str(pop_year),"geometry"]]
        un_pop_df = pd.merge(un_pop_df,trade_df,how="left",left_on=["ISO_A3"],right_on=["export_country_code"])
        un_pop_df[
            mine_tons_column
            ] = un_pop_df[
                    "trade_quantity_tons"
                    ]*un_pop_df[str(pop_year)]/un_pop_df.groupby(["ISO_A3"])[str(pop_year)].transform('sum')
        un_pop_df.drop(["export_country_code","trade_quantity_tons"],axis=1,inplace=True)
        un_pop_df["final_refined_stage"] = city_final_refined_stage
        location_df.append(un_pop_df[final_columns])

        location_df = pd.concat(location_df,axis=0,ignore_index=True)
        if year == 2021:
            gpd.GeoDataFrame(location_df,
                            geometry="geometry",
                            crs=un_pop_crs).to_file(
                                            os.path.join(
                                            results_folder,
                                            f"mine_city_tons_{year}.gpkg"),
                            layer=reference_mineral,driver="GPKG")
        else:
            gpd.GeoDataFrame(location_df,
                            geometry="geometry",
                            crs=un_pop_crs).to_file(
                                            os.path.join(
                                            results_folder,
                                            f"mine_city_tons_{year}.gpkg"),
                            layer=f"{reference_mineral}_{percentile}",driver="GPKG")



if __name__ == '__main__':
    CONFIG = load_config()
    try:
        reference_mineral =  str(sys.argv[1])
        mine_initial_refined_stage = float(sys.argv[2])
        mine_final_refined_stage = float(sys.argv[3])
        city_initial_refined_stage = float(sys.argv[4])
        city_final_refined_stage = float(sys.argv[5])
        percentile = int(sys.argv[6])
    except IndexError:
        print("Got arguments", sys.argv)
        exit()
    main(CONFIG,reference_mineral,
        mine_initial_refined_stage,
        mine_final_refined_stage,
        city_initial_refined_stage,
        city_final_refined_stage,percentile)