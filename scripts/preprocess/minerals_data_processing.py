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

def convert_geometries(dataframe,epsg=4326):
    # USGS geometry is wrong, which we will have to convert to Point from Latitude and Longitude values
    dataframe["geom"] = gpd.points_from_xy(
                            dataframe["Longitude"],dataframe["Latitude"])
    if "geometry" in dataframe.columns.values.tolist():
      dataframe.drop("geometry",axis=1,inplace=True)
    dataframe.rename(columns={"geom":"geometry"},inplace=True)
    dataframe.drop(["Latitude","Longitude"],axis=1,inplace=True)
    dataframe = gpd.GeoDataFrame(dataframe,geometry="geometry",crs=f"EPSG:{epsg}")

    return dataframe

def capacity_factor(x):
  if x["shared_capacity"] == "<null>":
    return 1
  else:
    return len(str(x["shared_capacity"]).split(",")) + 1

def convert_units(x):
  y = x['capacity_units']
  if y == 'Metric tons':
    return 1.0
  elif y == 'Kilograms':
    return 1.0/1000.0
  elif y == 'Thousand metric tons':
    return 1000.0
  elif y == '42-gallon barrels per day':
    return 42*264.17*365.0
  elif y == 'Thousand 42-gallon barrels':
    return 42*264.17*1000.0
  elif y == 'Million cubic meters':
    # 1 m3 = 2.41 metric tons
    return 1.0e6*2.41
  elif y == 'Thousand bricks':
    # Assume each brick weighs 3.5kg
    return 3.5
  elif y == 'Thousand carats':
    return 2.0e-7*1.0e3
  else:
    return 1000.0

def main(config):

    incoming_data_path = config['paths']['incoming_data']
    processed_data_path = config['paths']['data']


    # Read files
    USGS_shp = os.path.join(incoming_data_path,
                        "Africa_GIS Supporting Data", 
                        "a. Africa_GIS Shapefiles",
                        "AFR_Mineral_Facilities.shp",
                        "AFR_Mineral_Facilities.shp")
    USGS_mineral = gpd.read_file(USGS_shp)
    USGS_mineral = USGS_mineral[USGS_mineral["FeatureTyp"] == "Mineral Processing Plants"]
    # Step 1: Choose the right columns and find + add  unique values
    column_name_mapping = {
        'Country':'country',
        'FeatureNam':'feature_name',
        'FeatureTyp':'feature_type',
        'OperateNam':'operator_name',
        'LocOpStat': 'status',
        'DsgAttr01': 'commodity_type',
        'DsgAttr02': 'commodity_input',
        'DsgAttr03': 'commodity_products',
        'DsgAttr06': 'year',
        'DsgAttr07': 'annual_production_capacity',
        'DsgAttr08': 'capacity_units',
        'DsgAttr10': 'shared_capacity',
    }
    USGS_mineral = USGS_mineral.rename(columns=column_name_mapping)
    USGS_mineral["ownership"] = USGS_mineral[
                                    ["OwnerName1","OwnerName2",
                                    "OwnerName3","OwnerName4"]].agg('+'.join, axis=1)
    USGS_mineral["ownership"] = USGS_mineral["ownership"].str.replace("+<null>","")
    USGS_mineral["iso3"] = USGS_mineral.progress_apply(lambda x: x["FeatureUID"][:3], axis=1)
    USGS_mineral["annual_production_capacity"] = np.where(
                              USGS_mineral["annual_production_capacity"] == -999,
                              0,USGS_mineral["annual_production_capacity"])
    USGS_mineral["commodity_products"] = np.where(
                              USGS_mineral["commodity_products"] == "<null>",
                              USGS_mineral["commodity_input"],
                              USGS_mineral["commodity_products"])
    USGS_mineral["commodity_products"] = USGS_mineral["commodity_products"].str.replace(",","+")
    USGS_mineral["capacity_factor"] = USGS_mineral.progress_apply(lambda x:capacity_factor(x),axis=1)
    USGS_mineral["capacity_units_conversion"] = USGS_mineral.progress_apply(lambda x:convert_units(x),axis=1)
    USGS_mineral["assigned_capacity"] = USGS_mineral["capacity_units_conversion"]*USGS_mineral["annual_production_capacity"]/USGS_mineral["capacity_factor"]
    USGS_mineral["assigned_capacity_units"] = "tons"

    activity = ["active","inactive"]
    for act in activity:
        if act == "active":
            active_sites = USGS_mineral[USGS_mineral["status"].isin(["Active","Assumed Active"])]
        else:
            active_sites = USGS_mineral[~USGS_mineral["status"].isin(["Active","Assumed Active"])]
        
        index_columns = ["FeatureUID","country",
                              "iso3","year","status","feature_type",
                              "Latitude","Longitude"]
        active_sites = active_sites.groupby(index_columns + [
                "commodity_type","commodity_input"]).agg(['unique']).reset_index()
        active_sites.columns = [c[0] for c in active_sites.columns.values.tolist()]
        print (active_sites)    
        string_columns = ["feature_name","commodity_products","operator_name","ownership","assigned_capacity_units"]

        active_sites["assigned_capacity"] = active_sites.progress_apply(lambda x:sum(x["assigned_capacity"]),axis=1)
        for st in string_columns:
          active_sites[st] = active_sites.progress_apply(lambda x:';'.join(x[st]),axis=1)

        active_sites = active_sites[index_columns + [
                "commodity_type","commodity_input",
                "assigned_capacity"] + string_columns
                ].groupby(index_columns)[
                  ["commodity_input","assigned_capacity"
                  ] + string_columns].agg(list).reset_index()
        string_columns = ["feature_name","operator_name","ownership","assigned_capacity_units"]
        for st in string_columns:
          active_sites[st] = active_sites.progress_apply(lambda x:';'.join(list(set(x[st]))),axis=1)
        # active_sites.to_csv("test.csv",index=False)

        expand_columns = ["commodity_input","commodity_products","assigned_capacity"]
        for ec in expand_columns:
          df = active_sites[ec].apply(pd.Series).add_prefix(f'{ec}_')
          print (df)
          active_sites = active_sites.join(df)

        active_sites.drop(expand_columns,axis=1,inplace=True)
        active_sites = convert_geometries(active_sites)
        # active_sites.to_csv("test0.csv",index=False)

        if act == "active":
            # Read global mineral gpkg
            Global_mineral_path = os.path.join(incoming_data_path,
                                        "global_mines_info_maus",
                                        "facilities.gpkg")


            # os.chdir(Global_mineral_path)
            Global_mineral = gpd.read_file(Global_mineral_path)
            Global_mineral = Global_mineral[~Global_mineral.is_empty]
            Global_mineral = Global_mineral.explode()
            # Global_mineral = Global_mineral[Global_mineral["GID_0"].isin(USGS_isos)]
            print (Global_mineral)

            processing_matches = pd.read_csv(os.path.join(incoming_data_path,
                                          "global_mines_info_maus",
                                          "maus_usgs_processing_matches.csv"))
            processing_matches = processing_matches[processing_matches["matches"] == 'no']
            africa_sites = Global_mineral[
                              Global_mineral["facility_id"
                              ].isin(
                                processing_matches["facility_id"].values.tolist())]
            print (africa_sites)

            save_file = False
            processing_csv_path = os.path.join(incoming_data_path,
                                            "global_mines_info_maus",
                                            "processing.csv")
            processing = pd.read_csv(processing_csv_path)
            processing_columns_strings = ["input",
                                        "output"]
            for col in processing_columns_strings:
                processing[col] = processing[col].fillna("unknown")

            processing_columns_numbers = ["output_value_tonnes"]
            for col in processing_columns_numbers:
                processing[col] = processing[col].fillna(0)

            processing_sorted = processing.sort_values('year', ascending=False)
            processing_unique = processing_sorted.drop_duplicates(
                                subset=['facility_id', 
                                        'facility_type', 
                                        'input', 
                                        'output'])
            processing_unique = processing_unique[
                              processing_unique["facility_id"
                              ].isin(
                                processing_matches["facility_id"].values.tolist())]
            
            processing_unique["input"] = np.where(processing_unique["input"] == 'unknown',
                                          processing_unique["output"],processing_unique["input"])
            processing_unique = processing_unique[["facility_id","facility_type","input","output","output_value_tonnes"]]
            index_columns = ["facility_id","input"]
            processing_unique = processing_unique.groupby(index_columns).agg(['unique']).reset_index()
            processing_unique.columns = [c[0] for c in processing_unique.columns.values.tolist()]
            processing_unique["assigned_capacity"] = processing_unique.progress_apply(lambda x:sum(x["output_value_tonnes"]),axis=1)
            string_columns = ["facility_type","output"]
            for st in string_columns:
              processing_unique[st] = processing_unique.progress_apply(lambda x:';'.join(x[st]),axis=1)

            processing_unique = processing_unique.groupby(["facility_id"])[
                      ["facility_type","input","output","assigned_capacity"
                      ]].agg(list).reset_index()
            string_columns = ["facility_type"]
            for st in string_columns:
              processing_unique[st] = processing_unique.progress_apply(lambda x:';'.join(list(set(x[st]))),axis=1)
            # active_sites.to_csv("test.csv",index=False)

            processing_unique.rename(columns={"input":"commodity_input",
                  "output":"commodity_products"},inplace=True)
            expand_columns = ["commodity_input","commodity_products","assigned_capacity"]
            for ec in expand_columns:
              df = processing_unique[ec].apply(pd.Series).add_prefix(f'{ec}_')
              print (df)
              processing_unique = processing_unique.join(df)

            processing_unique.drop(expand_columns + ["facility_type"],axis=1,inplace=True)
            
            africa_sites = pd.merge(africa_sites,processing_unique,how="left",on=["facility_id"])

            ownership_csv_path = os.path.join(incoming_data_path,
                                            "global_mines_info_maus",
                                            "ownership.csv")
            ownership = pd.read_csv(ownership_csv_path)
            ownership_sorted = ownership.sort_values(by='year',ascending=False)
            ownership_unique = ownership_sorted.drop_duplicates(subset=['facility_id'])
            ownership_unique = ownership_unique[
                              ownership_unique["facility_id"
                              ].isin(
                                ownership_unique["facility_id"].values.tolist())]
            africa_sites = pd.merge(africa_sites,ownership_unique,how="left",on=["facility_id"])
            africa_sites.to_csv("test.csv",index=False)

            africa_sites.rename(columns={"facility_id":"FeatureUID",
                                        "GID_0":"iso3",
                                        "facility_name":"feature_name",
                                        "facility_type":"feature_type",
                                        "operators":"operator_name",
                                        "owners":"ownership"},inplace=True)
            africa_sites["status"] = "Assumed Active"
            africa_sites["year"] = 2018
            africa_sites["assigned_capacity_units"] = "tons"
            africa_sites["commodity_input_0"] = np.where(africa_sites["commodity_input_0"].isna(),
                                    africa_sites["primary_commodity"],africa_sites["commodity_input_0"])
            africa_sites["commodity_products_0"] = np.where(africa_sites["commodity_products_0"].isna(),
                                    africa_sites["commodities_products"],africa_sites["commodity_products_0"])
            africa_sites.to_csv("test.csv",index=False)
            select_cols = [c for c in active_sites.columns.values.tolist() if c in africa_sites.columns.values.tolist()]
            africa_sites = africa_sites[select_cols]

            all_sites = gpd.GeoDataFrame(
                            pd.concat([active_sites,africa_sites],
                                axis=0,ignore_index=True),geometry="geometry",crs="EPSG:4326")
            all_sites.to_file(os.path.join(processed_data_path,
                        "Minerals","africa_mineral_processing_sites_active.gpkg"),layer="nodes")
        else:
            gpd.GeoDataFrame(active_sites,
                            geometry="geometry",
                            crs="EPSG:4326").to_file(
                            os.path.join(processed_data_path,
                                        "Minerals",
                                        "africa_mineral_processing_sites_inactive.gpkg"),layer="nodes")


if __name__ == '__main__':
    CONFIG = load_config()
    main(CONFIG)


