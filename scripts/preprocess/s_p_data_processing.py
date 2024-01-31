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

# def add_iso_code(df,df_id_column,incoming_data_path):
#     # Insert countries' ISO CODE
#     africa_boundaries = gpd.read_file(os.path.join(
#                             incoming_data_path,
#                             "Africa_GIS Supporting Data",
#                             "a. Africa_GIS Shapefiles",
#                             "AFR_Political_ADM0_Boundaries.shp",
#                             "AFR_Political_ADM0_Boundaries.shp"))
#     africa_boundaries.rename(columns={"DsgAttr03":"iso3"},inplace=True)
#     global_boundaries = gpd.read_file(os.path.join(processed_data_path,
#                                     "admin_boundaries",
#                                     "gadm36_levels_gpkg"))
#     # Spatial join
#     m = gpd.sjoin(df, 
#                     africa_boundaries[['geometry', 'iso3']], 
#                     how="left", predicate='within').reset_index()
#     m = m[~m["iso3"].isna()]        
#     un = df[~df[df_id_column].isin(m[df_id_column].values.tolist())]
#     un = gpd.sjoin_nearest(un,
#                             africa_boundaries[['geometry', 'iso3']], 
#                             how="left").reset_index()
#     m = pd.concat([m,un],axis=0,ignore_index=True)
#     return m

def add_iso_code(df,df_id_column,processed_data_path):
    # Insert countries' ISO CODE
    # africa_boundaries = gpd.read_file(os.path.join(
    #                         incoming_data_path,
    #                         "Africa_GIS Supporting Data",
    #                         "a. Africa_GIS Shapefiles",
    #                         "AFR_Political_ADM0_Boundaries.shp",
    #                         "AFR_Political_ADM0_Boundaries.shp"))
    # africa_boundaries.rename(columns={"DsgAttr03":"iso3"},inplace=True)
    global_boundaries = gpd.read_file(os.path.join(processed_data_path,
                                    "admin_boundaries",
                                    "gadm36_levels_gpkg",
                                    "gadm36_levels_continents.gpkg"))
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

def main(config):

    incoming_data_path = config['paths']['incoming_data']
    processed_data_path = config['paths']['data']

    # s_and_p_mines = pd.read_excel(os.path.join(incoming_data_path,
    #                     "mines_spatial_locations",
    #                     "S_P_Southern_Afrca_all_mines_production.xls"),skiprows=[0,1,2],header=[0,1,2])
    # inputs_cols = s_and_p_mines.columns.values.tolist()
    # output_cols = []
    # for idx,(pn,sn,tn) in enumerate(inputs_cols):
    #     oc = pn
    #     if "Unnamed" not in sn:
    #         sn = sn.replace(" Y","")
    #         oc = f"{oc}_{sn}"
    #     if "Unnamed" not in tn:
    #         tn = tn.split("|")
    #         if len(tn) > 1:
    #             oc = f"{oc}_{tn[-1]}"
    #         else:
    #             oc = f"{oc}_total"

    #     output_cols.append(oc)

    # s_and_p_mines.columns = output_cols

    # s_and_p_mines["geometry"] = gpd.points_from_xy(
    #                         s_and_p_mines["LONGITUDE"],s_and_p_mines["LATITUDE"])

    # s_and_p_mines = gpd.GeoDataFrame(s_and_p_mines,geometry="geometry",crs="EPSG:4326")
    # s_and_p_mines = add_iso_code(s_and_p_mines,"PROP_ID",incoming_data_path)
    # s_and_p_mines = gpd.GeoDataFrame(s_and_p_mines,geometry="geometry",crs="EPSG:4326")
    # s_and_p_mines.to_file(os.path.join(processed_data_path,"Minerals","s_and_p_mines.gpkg"),driver="GPKG")

    s_and_p_mines = pd.read_excel(os.path.join(incoming_data_path,
                        "DataSP_New",
                        "GlobalAllMines.xls"))
    inputs_cols = s_and_p_mines.columns.values.tolist()
    print (inputs_cols)
    s_and_p_mines.columns = ['property_name', 'property_ID', 
                            'primary_commodity', 'list_of_commodities', 
                            'development_stage', 'activity_status', 
                            'country_or_region_name', 'latitude', 'longitude', 
                            'owner_name', 'owner_HQ', 'owner_country_or_region', 
                            'processing_methods', 'production_forms', 'ore_processed_mass']
    print (s_and_p_mines.columns.values.tolist())
    s_and_p_mines["geometry"] = gpd.points_from_xy(
                            s_and_p_mines["longitude"],s_and_p_mines["latitude"])

    s_and_p_mines = gpd.GeoDataFrame(s_and_p_mines,geometry="geometry",crs="EPSG:4326")
    s_and_p_mines = add_iso_code(s_and_p_mines,"property_ID",processed_data_path)
    s_and_p_mines = gpd.GeoDataFrame(s_and_p_mines,geometry="geometry",crs="EPSG:4326")
    s_and_p_mines = s_and_p_mines[~s_and_p_mines.geometry.isna()]
    s_and_p_mines.to_file(os.path.join(processed_data_path,
                                    "Minerals",
                                    "s_and_p_mines_global_all.gpkg"),
                                    driver="GPKG")




if __name__ == '__main__':
    CONFIG = load_config()
    main(CONFIG)


