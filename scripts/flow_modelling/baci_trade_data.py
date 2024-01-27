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


    # Read files
    status = False
    if status is True:
        global_boundaries = gpd.read_file(
                            os.path.join(processed_data_path,
                                        "Admin_boundaries",
                                        "gadm36_levels_gpkg",
                                        "gadm36_levels_continents.gpkg"))
        global_boundaries = global_boundaries[["GID_0","CONTINENT"]]
        global_boundaries.rename(columns={"GID_0":"iso_3digit_alpha"},inplace=True)
        global_boundaries.columns = map(str.lower, global_boundaries.columns)

        baci_countries = pd.read_csv(os.path.join(processed_data_path,
                            "baci","country_codes_V202301.csv"))
        baci_countries = pd.merge(baci_countries,global_boundaries,
                how="left",on=["iso_3digit_alpha"])
        baci_countries.to_csv(os.path.join(processed_data_path,
                            "baci","country_codes_V202301_v2.csv"),index=False)

    status = False
    if status is True:
        baci_countries = pd.read_csv(os.path.join(processed_data_path,
                            "baci","ccg_country_codes.csv"))
        all_countries = pd.read_csv(os.path.join(processed_data_path,
                            "baci","country_codes_V202301_v2.csv"))
        af_c = list(set(baci_countries["country_code"].values.tolist()))
        all_countries = all_countries[~all_countries["country_code"].isin(af_c)]
        all_countries["ccg_country"] = 0
        baci_countries = pd.concat([baci_countries,all_countries],axis=0,ignore_index=True)
        baci_countries.to_csv(os.path.join(processed_data_path,
                            "baci","ccg_country_codes.csv"),index=False)

    status = True
    if status is True:
        # commodites = pd.read_csv(os.path.join(processed_data_path,
        #                     "baci","ccg_minerals_codes.csv"))
        commodites = pd.read_csv(os.path.join(processed_data_path,
                            "baci","product_codes_HS17_V202301.csv"))
        commodites_codes = list(set(commodites.code.values.tolist()))
        baci_countries = pd.read_csv(os.path.join(processed_data_path,
                            "baci","ccg_country_codes.csv"))
        ccg_country_codes = baci_countries[
                                baci_countries["ccg_country"] == 1
                                ]["country_code"].values.tolist()
        baci_trade = pd.read_csv(os.path.join(processed_data_path,
                            "baci","BACI_HS17_Y2021_V202301.csv"))
        # baci_trade = baci_trade[
        #                 (baci_trade["t"] == 2021
        #                     ) & (baci_trade["i"].isin(ccg_country_codes)
        #                     ) & (baci_trade["k"].isin(commodites_codes))]
        baci_trade["v"] = baci_trade.progress_apply(lambda x: modify_from_string_to_float(x,"v"),axis=1)
        baci_trade["q"] = baci_trade.progress_apply(lambda x: modify_from_string_to_float(x,"q"),axis=1)
        baci_trade["q_v_ratio"] = baci_trade["q"]/baci_trade["v"]
        baci_trade["q_v_ratio_mean"]=baci_trade.groupby(["i","k"])["q_v_ratio"].transform('mean')
        baci_trade["q_mod"] = np.where(baci_trade["q"].isna(),
                                        baci_trade["v"]*baci_trade["q_v_ratio_mean"],
                                        baci_trade["q"])
        if len(baci_trade["q_mod"].isna()) > 0:
            baci_trade["q_v_ratio_mean"]=baci_trade.groupby(["k"])["q_v_ratio"].transform('mean')
            baci_trade["q_mod"] = np.where(baci_trade["q_mod"].isna(),
                                        baci_trade["v"]*baci_trade["q_v_ratio_mean"],
                                        baci_trade["q_mod"])
        baci_trade = baci_trade[["t","i","j","k","v","q_mod"]]
        baci_trade_africa = baci_trade.copy()
        baci_trade_africa.rename(
        				columns={"k":"product_code",
        				"v":"trade_value_thousandUSD",
        				"q_mod":"trade_quantity_tons"},inplace=True)
        # baci_trade_africa = baci_trade[
        #                         baci_trade["j"].isin(
        #                             baci_countries["country_code"].values.tolist()
        #                             )]
        # baci_trade_row = baci_trade[
        #                         ~baci_trade["j"].isin(
        #                             baci_countries["country_code"].values.tolist()
        #                             )]
        # print (baci_trade_row)
        # print (baci_countries)
        # # baci_trade_row = baci_trade_row.groupby(
        # #                                     ["t","i","k"],
        # #                                     dropna=False
        # #                                     ).agg({"v":"sum","q_mod":"sum"}).reset_index()
        # # baci_trade_row["j"] = 9999
        # baci_trade_africa = pd.concat([baci_trade_africa,baci_trade_row],axis=0,ignore_index=True)
        baci_trade_africa = pd.merge(baci_trade_africa,
                                baci_countries,
                                how="left",left_on=["i"],right_on=["country_code"])
        baci_trade_africa.rename(columns = {"country_name_abbreviation":"export_country_name",   
                                            "iso_3digit_alpha":"export_country_code"},inplace=True)

        baci_trade_africa.drop(["country_code","country_name_full",
                        "iso_2digit_alpha","continent","ccg_country"],
                        axis=1,inplace=True)
        baci_trade_africa = pd.merge(baci_trade_africa,
                                baci_countries,
                                how="left",left_on=["j"],right_on=["country_code"])
        baci_trade_africa.rename(columns = {"country_name_abbreviation":"import_country_name",   
                                            "iso_3digit_alpha":"import_country_code"},inplace=True)

        baci_trade_africa.drop(["country_code","country_name_full",
                        "iso_2digit_alpha","continent","ccg_country"],
                        axis=1,inplace=True)
        baci_trade_africa = pd.merge(baci_trade_africa,
                                commodites,
                                how="left",left_on=["product_code"],right_on=["code"])
        baci_trade_africa.drop(["code"],
                        axis=1,inplace=True)
        baci_trade_africa.to_csv(os.path.join(processed_data_path,
                            "baci","baci_trade_all.csv"),
                        index=False)
        



if __name__ == '__main__':
    CONFIG = load_config()
    main(CONFIG)


