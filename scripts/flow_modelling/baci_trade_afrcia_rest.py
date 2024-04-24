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

    status = True
    if status is True:
        columns = ['i', 'j', 'product_code', 
                    'trade_value_thousandUSD', 
                    'trade_quantity_tons',  
                    'product_description', 
                    ]
        ccg_minerals = ["copper","cobalt","nickel","graphite","manganese","lithium"]
        baci_trade = pd.read_csv(os.path.join(processed_data_path,
                            "baci","baci_trade_all.csv"))[columns]
        baci_trade = baci_trade.drop_duplicates(
                        subset=["product_code","i","j"],
                        keep="first")
        baci_countries = pd.read_csv(os.path.join(processed_data_path,
                            "baci","ccg_country_codes.csv"))
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
        product_stages = pd.read_csv(os.path.join(processed_data_path,
                            "baci","productcodes_minerals_refs.csv"))[
                                                [
                                                "product_code",
                                                "refining_stage",
                                                "refining_stage_cam",
                                                "reference_mineral",
                                                "processing_level"]]
        baci_trade = pd.merge(baci_trade,
                            product_stages,how="left",
                            on=["product_code"])
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
        baci_trade.loc[baci_trade["reference_mineral"].isin(ccg_minerals),"ccg_mineral"] = 1
        baci_trade.to_csv(os.path.join(processed_data_path,
                            "baci","baci_ccg_minerals_trade.csv"),index=False)
        baci_trade = baci_trade[(baci_trade["ccg_exporter"] == 1) & (baci_trade["ccg_mineral"] == 1)]
        baci_trade = baci_trade.groupby(
                            ["product_code",
                            "product_description",
                            "export_country_code",
                            "reference_mineral",
                            "refining_stage_cam"])[["trade_value_thousandUSD","trade_quantity_tons"]].sum().reset_index()
        baci_trade.to_csv(os.path.join(processed_data_path,
                            "baci","baci_ccg_reference_minerals_exports.csv"),index=False)

    status = False
    if status is True:
        baci_countries = pd.read_csv(os.path.join(processed_data_path,
                            "baci","country_codes_V202301_v2.csv"))[["iso_3digit_alpha","continent"]]
        baci_trade = pd.read_csv(os.path.join(processed_data_path,
                            "baci","baci_trade_all.csv"))
        
        baci_trade = pd.merge(baci_trade,baci_countries,
                            how="left",
                            left_on=["export_country_code"],
                            right_on=["iso_3digit_alpha"])
        baci_trade.rename(columns={"continent":"export_continent"},inplace=True)
        baci_trade.drop("iso_3digit_alpha",axis=1,inplace=True)
        baci_trade = pd.merge(baci_trade,baci_countries,
                            how="left",
                            left_on=["import_country_code"],
                            right_on=["iso_3digit_alpha"])
        baci_trade.rename(columns={"continent":"import_continent"},inplace=True)
        baci_trade.drop("iso_3digit_alpha",axis=1,inplace=True)
        
        baci_trade.to_csv(os.path.join(processed_data_path,
                            "baci","baci_trade_all.csv"),index=False)

    status = False
    if status is True:
        baci_trade = pd.read_csv(os.path.join(processed_data_path,
                            "baci","baci_ccg_clean.csv"))
        baci_trade = baci_trade.drop_duplicates(
                        subset=["product_code","export_country_code","import_country_code"],
                        keep="first")
        country_conditions = pd.read_csv(os.path.join(processed_data_path,
                            "transport_costs","country_transport_information.csv"))
        baci_trade = pd.merge(baci_trade,country_conditions[["iso3","continent_name","landlocked"]],
                            how="left",
                            left_on=["export_country_code"],
                            right_on=["iso3"])
        baci_trade.rename(
                columns={"continent_name":"export_continent","landlocked":"export_landlocked"},
                inplace=True)
        baci_trade.drop("iso3",axis=1,inplace=True)
        baci_trade = pd.merge(baci_trade,country_conditions[["iso3","continent_name","landlocked"]],
                            how="left",
                            left_on=["import_country_code"],
                            right_on=["iso3"])
        baci_trade.rename(
                columns={"continent_name":"import_continent","landlocked":"import_landlocked"},
                inplace=True)
        baci_trade.drop("iso3",axis=1,inplace=True)
        
        baci_trade["trade_quantity_tons_fraction"] = baci_trade["trade_quantity_tons"
                                    ]/baci_trade.groupby(["product_code","export_country_code"])["trade_quantity_tons"].transform('sum')
        baci_trade.to_csv(os.path.join(processed_data_path,
                            "baci","baci_ccg_clean_continent_trade.csv"),index=False)
        # df = baci_trade[
        #             baci_trade["import_landlocked"] == 1
        #             ][["import_country_code","import_continent"]]
        
        # df.drop_duplicates(
        #                 subset=["import_country_code","import_continent"],
        #                 keep="first").to_csv(os.path.join(processed_data_path,
        #                     "port_statistics",
        #                     "ccg_importing_landlocked_countries.csv"),index=False)


if __name__ == '__main__':
    CONFIG = load_config()
    main(CONFIG)


