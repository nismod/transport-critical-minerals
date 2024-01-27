#!/usr/bin/env python
# coding: utf-8

import os 
import pandas as pd
import sys
import geopandas as gpd
import re
from collections import defaultdict
from utils import *
from tqdm import tqdm

def modify_from_string_to_float(x,column_name):
    value = str(x[column_name]).strip()
    if value != "NA":
        return float(value)
    else:
        return np.nan

def main(config):

    incoming_data_path = config['paths']['incoming_data']
    processed_data_path = config['paths']['data']

    baci_trade_df = pd.read_csv(
                    os.path.join(processed_data_path,
                        "baci","baci_trade_all.csv"))
    print (baci_trade_df[baci_trade_df["trade_quantity_tons"].isna()])
    baci_trade_df[baci_trade_df["product_code"] == 271600].to_csv("blank_code.csv",index=False)

    # refining_stages = ['2. Refined product: battery precursor preparation', 
    #                     '1.1. Mining and early refining (waste)', 
    #                     '2. Refined product: battery precursor', 
    #                     '2. Refined product', 
    #                     '1. Mining and early refining', 
    #                     '2. Refined product: lithium-ion battery precursor', 
    #                     '2. Refined product: other battery precursor', 
    #                     '2.1. Refined product (waste)', 
    #                     '2. Refined product (waste)', 
    #                     '2. Refined product (non-mined): lithium-ion battery precursor']
    # refining_stages = ['2. Refined product: battery precursor preparation', 
    #                     '2. Refined product: battery precursor', 
    #                     '2. Refined product', 
    #                     '1. Mining and early refining', 
    #                     '2. Refined product: lithium-ion battery precursor', 
    #                     '2. Refined product: other battery precursor', 
    #                     '2. Refined product (waste)']
    # df = baci_trade_df[baci_trade_df["refining_stage"].isin(refining_stages)]
    # df = df.drop_duplicates(subset=["i","j","product_code"],keep="first")
    # print (df)
    # df.to_csv("test.csv",index=False)
    # pc = list(set(df["product_code"].values.tolist()))

    # print (len(list(set(df["product_code"].values.tolist()))))

    # samira_baci_df = pd.read_csv(
    #                 os.path.join(processed_data_path,
    #                     "baci","baci_full_clean.csv"))
    # print (len(list(set(samira_baci_df["product_code"].values.tolist()))))
    # sc = list(set(samira_baci_df["product_code"].values.tolist()))
    # extra = [p for p in pc if p not in sc]
    # print (extra)


    # baci_trade_df = pd.read_csv(
    #                 os.path.join(processed_data_path,
    #                     "baci","baci_full_clean.csv"))
    # ccg_trade_df = baci_trade_df[baci_trade_df["ccg_exporter"] == 1]
    # ccg_countries = list(set(ccg_trade_df["export_country_code"].values.tolist()))
    # ccg_trade_df["ccg_importer"] = 0
    # ccg_trade_df.loc[ccg_trade_df["import_country_code"].isin(ccg_countries),"ccg_importer"] = 1
    # ccg_trade_df.to_csv("test3.csv",index=False)

    # # Find the total incoming raw mineral in each country
    # gr_cols = ["product_code","import_country_code",
    #             "product_description","refining_stage",
    #             "refining_stage_cam",
    #             "reference_mineral","processing_level",
    #             "ccg_mineral","ccg_importer"]
    # ccg_import_df = ccg_trade_df[
    #                     ccg_trade_df["ccg_mineral"] == 1
    #                     ].groupby(gr_cols)[["trade_value_thousandUSD",
    #                                 "trade_quantity_tons"]].sum().reset_index() 
    # print (ccg_import_df)
    # ccg_import_df.to_csv("test4.csv",index=False)

    # gr_cols = ["product_code","export_country_code",
    #             "product_description","refining_stage",
    #             "refining_stage_cam",
    #             "reference_mineral","processing_level",
    #             "ccg_mineral","ccg_exporter"]
    # ccg_export_df = ccg_trade_df[
    #                     ccg_trade_df["ccg_mineral"] == 1
    #                     ].groupby(gr_cols)[["trade_value_thousandUSD",
    #                                 "trade_quantity_tons"]].sum().reset_index() 
    # print (ccg_export_df)
    # ccg_export_df.to_csv("test5.csv",index=False)

        



if __name__ == '__main__':
    CONFIG = load_config()
    main(CONFIG)


