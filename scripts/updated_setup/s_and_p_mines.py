#!/usr/bin/env python
# coding: utf-8

import os 
import sys
import pandas as pd
import geopandas as gpd
from collections import defaultdict
from utils import *
from tqdm import tqdm

def add_iso_code(df,df_id_column,global_boundaries):
    # Insert countries' ISO CODE
    # Spatial join
    m = gpd.sjoin(df, 
                    global_boundaries[['geometry', 'ISO_A3','CONTINENT']], 
                    how="left", predicate='within').reset_index()
    m = m[~m["ISO_A3"].isna()]        
    un = df[~df[df_id_column].isin(m[df_id_column].values.tolist())]
    un = gpd.sjoin_nearest(un,
                            global_boundaries[['geometry', 'ISO_A3','CONTINENT']], 
                            how="left").reset_index()
    m = pd.concat([m,un],axis=0,ignore_index=True)
    return m

def identify_future_mine(x,baseline_year,future_year):
    if x[str(baseline_year)] == 0 and x[str(future_year)] > 0:
        return 1
    else:
        return 0

def main(config):
    incoming_data_path = config['paths']['incoming_data']
    processed_data_path = config['paths']['data']

    global_boundaries = gpd.read_file(os.path.join(processed_data_path,
                                    "admin_boundaries",
                                    "gadm36_levels_gpkg",
                                    "gadm36_levels_continents.gpkg")) 
    metal_conversion_df = pd.read_csv(
                                os.path.join(
                                    processed_data_path,
                                    "mineral_usage_factors",
                                    "mine_metal_content_conversion.csv"
                                    )
                                )
    baseline_years = ["2022"]
    future_years = ["2030","2040"]
    reference_minerals = ["graphite","lithium","cobalt","manganese","nickel","copper"]
    # Read the finalised version of the BACI trade data
    ccg_countries = pd.read_csv(
                        os.path.join(processed_data_path,
                                "baci","ccg_country_codes.csv"))
    ccg_countries = ccg_countries[ccg_countries["ccg_country"] == 1]["iso_3digit_alpha"].values.tolist()
    scenarios = [
                    {
                        "scenario":"baseline",
                        "years":["2022"]
                    },
                    {
                        "scenario":"future",
                        "years":["2030","2040"]
                    }
                ]       
    all_years = [str(y) for y in np.arange(1980,2041,1)]
    mineral_dfs = defaultdict()
    all_mines = []
    mine_id_column = "PROP_ID"
    for scenario in scenarios:
        sc = scenario["scenario"]
        years = scenario["years"]
        file_directory = os.path.join(
                                processed_data_path,
                                "minerals",
                                "future prod june 2025",
                                "s_and_p_mine_scenarios_all")
        if sc == "baseline":
            file_end = "_high_all.xlsx"
        else:
            file_end = ".xlsx"
        for root, dirs, files in os.walk(file_directory):
            for file in files:
                if file.endswith(file_end):
                    s_and_p_mines = pd.read_excel(os.path.join(root,file))
                    s_and_p_mines.columns = s_and_p_mines.columns.astype(str)
                    s_and_p_mines[years] = s_and_p_mines[years].fillna(0)

                    mineral = file.split("_")[0]
                    drop_columns = [y for y in all_years if y not in years]
                    s_and_p_mines.drop(drop_columns,axis=1,inplace=True)
                    if sc == "baseline":
                        mineral_dfs[mineral] = s_and_p_mines
                        mineral_scenario = f"{mineral}_baseline"
                    else:
                        mineral_scenario = file.replace("_all.xlsx","")
                        df = mineral_dfs[mineral]
                        s_and_p_mines = pd.merge(
                                            s_and_p_mines,
                                            df[[mine_id_column] + baseline_years],
                                            how="left",on=[mine_id_column])
                        for fy in future_years:
                            s_and_p_mines[
                                f"future_new_mine_{fy}"
                                ] = s_and_p_mines.progress_apply(
                                        lambda x:identify_future_mine(x,baseline_years[0],fy),
                                        axis=1)

                    s_and_p_mines["geometry"] = gpd.points_from_xy(
                                                    s_and_p_mines["LONGITUDE"],s_and_p_mines["LATITUDE"])
                    s_and_p_mines = gpd.GeoDataFrame(s_and_p_mines,geometry="geometry",crs="EPSG:4326")
                    s_and_p_mines = add_iso_code(s_and_p_mines,mine_id_column,global_boundaries)
                    s_and_p_mines.drop(["index","index_right"],axis=1,inplace=True)

                    s_and_p_mines["mine_id"] = s_and_p_mines.progress_apply(
                                                    lambda x:f"s_and_p_{x[mine_id_column]}",
                                                    axis=1)
                    s_and_p_mines = gpd.GeoDataFrame(s_and_p_mines,geometry="geometry",crs="EPSG:4326")
                    s_and_p_mines.to_file(os.path.join(processed_data_path,
                                "minerals",
                                f"s_and_p_mines_current_and_future_estimates_global.gpkg"),
                                layer=mineral_scenario,
                                driver="GPKG")

                    africa_mines = s_and_p_mines[s_and_p_mines["CONTINENT"] == "Africa"]
                    africa_mines.to_file(os.path.join(processed_data_path,
                                "minerals",
                                f"s_and_p_mines_current_and_future_estimates_africa.gpkg"),
                                layer=mineral_scenario,
                                driver="GPKG")

                    ccg_mines = s_and_p_mines[s_and_p_mines["ISO_A3"].isin(ccg_countries)]
                    metal_factors = metal_conversion_df[metal_conversion_df["reference_mineral"] == mineral]
                    ccg_mines = pd.merge(ccg_mines,metal_factors,how="left",on=["ISO_A3"])
                    for y in baseline_years + future_years:
                        ccg_mines[f"{y}_metal_content"] = ccg_mines[y]*ccg_mines["mine_conversion_factor"]

                    ccg_mines.drop(["reference_mineral","mine_conversion_factor"],axis=1,inplace=True)
                    ccg_mines = gpd.GeoDataFrame(ccg_mines,geometry="geometry",crs="EPSG:4326")
                    ccg_mines.to_file(os.path.join(processed_data_path,
                                "minerals",
                                f"s_and_p_mines_current_and_future_estimates.gpkg"),
                                layer=mineral_scenario,
                                driver="GPKG")

if __name__ == '__main__':
    CONFIG = load_config()
    main(CONFIG)


