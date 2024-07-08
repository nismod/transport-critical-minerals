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

    commodites = pd.read_csv(os.path.join(processed_data_path,
                        "baci","product_codes_HS17_V202401.csv"))
    commodites["description"] = commodites["description"].str.lower()
    baci_countries = pd.read_csv(os.path.join(processed_data_path,
                        "baci","ccg_country_codes.csv"))
    ccg_country_codes = baci_countries[
                            baci_countries["ccg_country"] == 1
                            ]["country_code"].values.tolist()
    baci_trade = pd.read_csv(os.path.join(processed_data_path,
                        "baci","baci_baptiste.csv"))
    
    baci_trade["v_baptiste"] = baci_trade.progress_apply(
                                    lambda x: modify_from_string_to_float(x,"v_baptiste"),
                                    axis=1)
    baci_trade["q_baptiste"] = baci_trade.progress_apply(
                                    lambda x: modify_from_string_to_float(x,"q_baptiste"),
                                    axis=1)

    baci_trade = baci_trade[["i","j","k","q","v_baptiste","q_baptiste"]]
    baci_trade.rename(
                    columns={"k":"product_code",
                    "v_baptiste":"trade_value_thousandUSD",
                    "q_baptiste":"trade_quantity_tons"},inplace=True)
    
    baci_trade = baci_trade[~baci_trade["trade_quantity_tons"].isna()]
    baci_trade = pd.merge(baci_trade,
                            baci_countries,
                            how="left",left_on=["i"],right_on=["country_code"])
    baci_trade.rename(columns = {"country_name_abbreviation":"export_country_name",   
                                "iso_3digit_alpha":"export_country_code",
                                "continent":"export_continent",
                                "ccg_country":"ccg_exporter"
                                        },inplace=True)
    baci_trade.drop(["i","country_code","country_name_full",
                    "iso_2digit_alpha"],
                    axis=1,inplace=True)
    baci_trade = pd.merge(baci_trade,
                            baci_countries,
                            how="left",left_on=["j"],right_on=["country_code"])
    baci_trade.rename(columns = {"country_name_abbreviation":"import_country_name",   
                                        "iso_3digit_alpha":"import_country_code",
                                        "continent":"import_continent",
                                        },
                                        inplace=True)

    baci_trade.drop(["j","country_code","country_name_full",
                    "iso_2digit_alpha","ccg_country"],
                    axis=1,inplace=True)
    print (baci_trade)
    baci_trade = pd.merge(baci_trade,
                            commodites,
                            how="left",left_on=["product_code"],right_on=["code"])
    baci_trade.rename(columns = {"description":"product_description"
                                        },
                                        inplace=True)
    baci_trade.drop(["code"],
                    axis=1,inplace=True)
    product_stages = pd.read_csv(os.path.join(processed_data_path,
                        "baci","productcodes_minerals_refs_updated.csv"))
    baci_trade = pd.merge(baci_trade,
                        product_stages,how="left",
                        on=["product_code","product_description"])
    baci_trade.drop("q",axis=1,inplace=True)
    baci_trade = baci_trade[~baci_trade["refining_stage_cam"].isna()]
    country_conditions = pd.read_csv(os.path.join(processed_data_path,
                        "transport_costs","country_transport_information.csv"))
    baci_trade = pd.merge(baci_trade,country_conditions[["iso3","landlocked"]],
                        how="left",
                        left_on=["export_country_code"],
                        right_on=["iso3"])
    baci_trade.rename(
            columns={"landlocked":"export_landlocked"},
            inplace=True)
    baci_trade.drop("iso3",axis=1,inplace=True)
    baci_trade = pd.merge(baci_trade,country_conditions[["iso3","landlocked"]],
                        how="left",
                        left_on=["import_country_code"],
                        right_on=["iso3"])
    baci_trade.rename(
            columns={"landlocked":"import_landlocked"},
            inplace=True)
    baci_trade.drop("iso3",axis=1,inplace=True)
    baci_trade["ccg_mineral"] = 0
    ccg_minerals = ["copper","cobalt","nickel","graphite","manganese","lithium"]
    baci_trade.loc[baci_trade["reference_mineral"].isin(ccg_minerals),"ccg_mineral"] = 1
    baci_trade.to_csv(os.path.join(processed_data_path,
                        "baci","baci_ccg_minerals_trade_2022_updated.csv"),index=False)
    baci_trade = baci_trade[(baci_trade["ccg_exporter"] == 1) & (baci_trade["ccg_mineral"] == 1)]
    baci_trade = baci_trade.groupby(
                        ["product_code",
                        "product_description",
                        "export_country_code",
                        "reference_mineral",
                        "refining_stage_cam"])[["trade_value_thousandUSD","trade_quantity_tons"]].sum().reset_index()
    baci_trade.to_csv(os.path.join(processed_data_path,
                        "baci","baci_ccg_reference_minerals_exports_2022_updated.csv"),index=False)


if __name__ == '__main__':
    CONFIG = load_config()
    main(CONFIG)