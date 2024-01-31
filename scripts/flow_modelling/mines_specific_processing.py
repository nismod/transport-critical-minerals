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



def main(config):
    incoming_data_path = config['paths']['incoming_data']
    processed_data_path = config['paths']['data']
    output_data_path = config['paths']['results']

    results_folder = os.path.join(output_data_path,"location_outputs")
    if os.path.exists(results_folder) == False:
        os.mkdir(results_folder)

    mine_id_col = "mine_cluster_mini"
    mine_tons_column = "mine_output_approx_copper"
    copper_conversion_stage = 3.0
    # cargo_type = "Dry bulk"
    # trade_groupby_columns = ["export_country_code", 
    #                         "import_country_code", 
    #                         "export_continent","export_landlocked",
    #                         "import_continent","import_landlocked"]
    # trade_value_columns = ["trade_value_thousandUSD","trade_quantity_tons"]
    # od_columns = ["reference_mineral","process_binary","export_country_code",
    #                 "import_country_code",
    #                 "export_continent",
    #                 "import_continent",
    #                 "mine_output_tons",    
    #                 "mine_output_thousandUSD"]
    """Step 1: Get the input datasets
    """
    # Mine locations in Africa with the copper tonages
    mines_df = gpd.read_file(os.path.join(processed_data_path,
                                    "Minerals",
                                    "copper_mines_tons.gpkg"))
    mines_df[mine_id_col] = mines_df.progress_apply(lambda x:f"mine_{x[mine_id_col]}",axis=1)
    mines_crs = mines_df.crs
    # Select only tonnages > 0
    # mines_df = mines_df[mines_df["mine_output_approx_copper"] > 0]

    mine_tonnages = pd.read_csv(os.path.join(processed_data_path,
                                    "Minerals",
                                    "copper_mine_output_corrected.csv"))
    mines_df = pd.merge(mines_df,
                mine_tonnages[[mine_id_col,"corrected_tons"]],
                how="left",on=[mine_id_col]).fillna(0)

    mines_df[mine_tons_column] = mines_df["corrected_tons"]
    mines_df.drop("corrected_tons",axis=1,inplace=True)
    
    pr_conv_factors_df = pd.read_excel(os.path.join(processed_data_path,
                                        "mineral_usage_factors",
                                        "aggregated_stages.xlsx")) 
    pr_conv_factors_df["initial_stage"] = pr_conv_factors_df.progress_apply(
                                    lambda x:float(str(x["Stage range"]).split(",")[0]),
                                    axis=1)
    pr_conv_factors_df["final_stage"] = pr_conv_factors_df.progress_apply(
                                    lambda x:float(str(x["Stage range"]).split(",")[-1]),
                                    axis=1)
    
    conv_factor = pr_conv_factors_df[
                    pr_conv_factors_df["final_stage"] == copper_conversion_stage
                    ]["aggregate_ratio_normalised"].values[0]
    mines_df[f"{mine_tons_column}_unrefined"] = np.where(mines_df["process_binary"] == 0,mines_df[mine_tons_column],
        conv_factor*mines_df[mine_tons_column])
    mines_df[f"{mine_tons_column}_refined_produced"] = mines_df["process_binary"]*mines_df[mine_tons_column]

    mines_od = pd.read_parquet(os.path.join(output_data_path,"flow_mapping","copper.parquet"))

    mines_od = mines_od[mines_od["destination_id"].isin(mines_df[mine_id_col].values.tolist())]
    unprocessed_input = mines_od.groupby(["destination_id"])["mine_output_tons"].sum().reset_index()

    mines_df = pd.merge(mines_df,unprocessed_input,how="left",
                        left_on=[mine_id_col],right_on=["destination_id"]).fillna(0)
    # mines_df.drop("destination_id",axis=1,inplace=True)

    mines_df[f"{mine_tons_column}_unrefined_produced"] = mines_df[f"{mine_tons_column}_unrefined"] - mines_df["mine_output_tons"]
    mines_df.rename(columns={"mine_output_tons":f"{mine_tons_column}_unrefined_imported"},inplace=True)
    mines_df.drop(["destination_id",f"{mine_tons_column}_unrefined"],axis=1,inplace=True)

    gpd.GeoDataFrame(mines_df,geometry="geometry",crs=mines_crs).to_file(os.path.join(processed_data_path,
                                    "Minerals",
                                    "copper_mines_tons_refined_unrefined.gpkg"))


    







if __name__ == '__main__':
    CONFIG = load_config()
    main(CONFIG)