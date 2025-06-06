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
    """Step 1: Get the input datasets
    """
    # Mine locations in Africa with the copper tonages
    mines_df = gpd.read_file(
                    os.path.join(
                        processed_data_path,
                        "minerals",
                        "copper_mines_tons.gpkg"))
    mines_df[mine_id_col] = mines_df.progress_apply(lambda x:f"mine_{x[mine_id_col]}",axis=1)
    mines_crs = mines_df.crs

    # Correct the tons outputs for copper because they do not match country export values
    # Hopefully this will be avoided in future versions of the data
    correct_tons = True
    if correct_tons is True:
        trade_groupby_columns = ["reference_mineral","export_country_code", 
                                "final_refined_stage"]

        trade_value_columns = ["trade_quantity_tons"]
        # Get the BACI trade linkages between countries 
        trade_df = pd.read_csv(os.path.join(processed_data_path,
                                "baci",
                                "baci_ccg_country_level_trade_2021.csv"))
        # trade_df = trade_df.groupby(trade_groupby_columns)[trade_value_columns].sum().reset_index()
        # trade_df["trade_quantity_tons_fraction"] = trade_df[tons_column
        #                             ]/trade_df.groupby(
        #                                 ["export_country_code"]
        #                             )[tons_column].transform('sum')

        combined_trade_df = []
        processing_stages = list(set(trade_df["final_refined_stage"].values.tolist()))
        for pr_st in processing_stages:
            t_df = trade_df[trade_df["final_refined_stage"] == pr_st]
            t_df = t_df.groupby(trade_groupby_columns)[trade_value_columns].sum().reset_index()
            if pr_st > 1:
                mine_refined_df = mines_df[mines_df["process_binary"] == 1]
            else:
                mine_refined_df = mines_df[mines_df["process_binary"] == 0]

            mine_refined_df = pd.merge(mine_refined_df,
                                        t_df,
                                        how="left",
                                        left_on=["shapeGroup_primary_admin0"],
                                        right_on=["export_country_code"])
            mine_refined_df[
                    "corrected_tons"] = mine_refined_df[
                    "mine_output_approx_copper"]*mine_refined_df["trade_quantity_tons"]/(mine_refined_df.groupby(
                                            ["export_country_code"]
                                        )["mine_output_approx_copper"].transform('sum'))
            mine_refined_df["corrected_tons"] = mine_refined_df["corrected_tons"].fillna(0)
            combined_trade_df.append(mine_refined_df)

        combined_trade_df = pd.concat(combined_trade_df,axis=0,ignore_index=True)

        combined_trade_df["mine_output_approx_copper"] = combined_trade_df["corrected_tons"]

        combined_trade_df.drop(["reference_mineral",
                            "export_country_code",
                            "trade_quantity_tons","corrected_tons"],
                            axis=1,inplace=True)
        gpd.GeoDataFrame(combined_trade_df,
                        geometry="geometry",
                        crs=mines_crs).to_file(
                                os.path.join(processed_data_path,
                                "minerals",
                                "copper_mines_tons_corrected.gpkg"),
                                driver="GPKG")

    mine_tonnages = gpd.read_file(
                        os.path.join(
                            processed_data_path,
                            "minerals",
                            "copper_mines_tons_corrected.gpkg"))
    
    # mines_df = pd.merge(mines_df,
    #             mine_tonnages[[mine_id_col,"corrected_tons"]],
    #             how="left",on=[mine_id_col]).fillna(0)

    # mines_df[mine_tons_column] = mines_df["corrected_tons"]
    # mines_df.drop("corrected_tons",axis=1,inplace=True)
    
    # pr_conv_factors_df = pd.read_excel(
    #                         os.path.join(
    #                             processed_data_path,
    #                             "mineral_usage_factors",
    #                             "aggregated_stages.xlsx")) 
    # pr_conv_factors_df["initial_stage"] = pr_conv_factors_df.progress_apply(
    #                                 lambda x:float(str(x["Stage range"]).split(",")[0]),
    #                                 axis=1)
    # pr_conv_factors_df["final_stage"] = pr_conv_factors_df.progress_apply(
    #                                 lambda x:float(str(x["Stage range"]).split(",")[-1]),
    #                                 axis=1)
    
    # conv_factor = pr_conv_factors_df[
    #                 pr_conv_factors_df["final_stage"] == copper_conversion_stage
    #                 ]["aggregate_ratio_normalised"].values[0]
    # mines_df[f"{mine_tons_column}_unrefined"] = np.where(mines_df["process_binary"] == 0,mines_df[mine_tons_column],
    #     conv_factor*mines_df[mine_tons_column])
    # mines_df[f"{mine_tons_column}_refined_produced"] = mines_df["process_binary"]*mines_df[mine_tons_column]

    # mines_od = pd.read_parquet(os.path.join(output_data_path,"flow_mapping","copper.parquet"))

    # mines_od = mines_od[mines_od["destination_id"].isin(mines_df[mine_id_col].values.tolist())]
    # unprocessed_input = mines_od.groupby(["destination_id"])["mine_output_tons"].sum().reset_index()

    # mines_df = pd.merge(mines_df,unprocessed_input,how="left",
    #                     left_on=[mine_id_col],right_on=["destination_id"]).fillna(0)
    # # mines_df.drop("destination_id",axis=1,inplace=True)

    # mines_df[f"{mine_tons_column}_unrefined_produced"] = mines_df[f"{mine_tons_column}_unrefined"] - mines_df["mine_output_tons"]
    # mines_df.rename(columns={"mine_output_tons":f"{mine_tons_column}_unrefined_imported"},inplace=True)
    # mines_df.drop(["destination_id",f"{mine_tons_column}_unrefined"],axis=1,inplace=True)

    # gpd.GeoDataFrame(mines_df,
    #                 geometry="geometry",
    #                 crs=mines_crs).to_file(
    #                 os.path.join(processed_data_path,
    #                             "minerals",
    #                             "copper_mines_tons_refined_unrefined.gpkg"),
    #                 driver="GPKG")


if __name__ == '__main__':
    CONFIG = load_config()
    main(CONFIG)