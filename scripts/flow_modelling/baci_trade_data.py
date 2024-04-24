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
        commodites = pd.read_csv(os.path.join(processed_data_path,
                            "baci","product_codes_HS17_V202401.csv"))
        commodites["description"] = commodites["description"].str.lower()
        print (commodites)
        # commodites = pd.read_csv(os.path.join(processed_data_path,
        #                     "baci","productcodes_minerals_refs.csv"))
        # commodites_codes = list(set(commodites.product_code.values.tolist()))
        baci_countries = pd.read_csv(os.path.join(processed_data_path,
                            "baci","ccg_country_codes.csv"))
        ccg_country_codes = baci_countries[
                                baci_countries["ccg_country"] == 1
                                ]["country_code"].values.tolist()
        # baci_trade = pd.read_csv(os.path.join(processed_data_path,
        #                     "baci","BACI_HS17_Y2021_V202301.csv"))
        baci_trade = pd.read_csv(os.path.join(processed_data_path,
                            "baci","BACI_HS17_Y2022_V202401.csv"))
        print (baci_trade)
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
        baci_trade = baci_trade[["t","i","j","k","v","q","q_mod"]]
        baci_trade.rename(
        				columns={"k":"product_code",
        				"v":"trade_value_thousandUSD",
        				"q_mod":"trade_quantity_tons"},inplace=True)
        print (baci_trade)
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
        print (baci_trade)
        product_stages = pd.read_csv(os.path.join(processed_data_path,
                            "baci","productcodes_minerals_refs.csv"))[
                                                [
                                                "product_code",
                                                "product_description",
                                                "refining_stage",
                                                "refining_stage_cam",
                                                "reference_mineral",
                                                "processing_level"]]
        product_stages["product_description"] = product_stages["product_description"].str.lower() 
        product_stages.to_csv(os.path.join(processed_data_path,
                            "baci","productcodes_minerals_refs_updated.csv"))
        baci_trade = pd.merge(baci_trade,
                            product_stages,how="left",
                            on=["product_code","product_description"])
        print (baci_trade)

        baci_trade[baci_trade["q"].isna()].to_csv(os.path.join(processed_data_path,
                            "baci","baci_na_values.csv"),index=False)
        baci_trade.drop("q",axis=1,inplace=True)
        baci_trade.to_csv(os.path.join(processed_data_path,
                            "baci","baci_trade_all_2022.csv"),
                        index=False)
        
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
                            "baci","baci_ccg_minerals_trade_2022.csv"),index=False)
        baci_trade = baci_trade[(baci_trade["ccg_exporter"] == 1) & (baci_trade["ccg_mineral"] == 1)]
        baci_trade = baci_trade.groupby(
                            ["product_code",
                            "product_description",
                            "export_country_code",
                            "reference_mineral",
                            "refining_stage_cam"])[["trade_value_thousandUSD","trade_quantity_tons"]].sum().reset_index()
        baci_trade.to_csv(os.path.join(processed_data_path,
                            "baci","baci_ccg_reference_minerals_exports_2022.csv"),index=False)


if __name__ == '__main__':
    CONFIG = load_config()
    main(CONFIG)


