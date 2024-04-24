#!/usr/bin/env python
# coding: utf-8

import os 
import pandas as pd
import sys
import geopandas as gpd
import numpy as np
import re
from collections import defaultdict
from utils import *
from tqdm import tqdm

def add_iso_code(df,df_id_column,global_boundaries):
    # Insert countries' ISO CODE
    # africa_boundaries = gpd.read_file(os.path.join(
    #                         incoming_data_path,
    #                         "Africa_GIS Supporting Data",
    #                         "a. Africa_GIS Shapefiles",
    #                         "AFR_Political_ADM0_Boundaries.shp",
    #                         "AFR_Political_ADM0_Boundaries.shp"))
    # africa_boundaries.rename(columns={"DsgAttr03":"iso3"},inplace=True)
    # global_boundaries = gpd.read_file(os.path.join(processed_data_path,
    #                                 "admin_boundaries",
    #                                 "gadm36_levels_gpkg",
    #                                 "gadm36_levels_continents.gpkg"))
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

    global_boundaries = gpd.read_file(os.path.join(processed_data_path,
                                    "admin_boundaries",
                                    "gadm36_levels_gpkg",
                                    "gadm36_levels_continents.gpkg"))

    process_data = False
    if process_data is True:
        s_and_p_mines = pd.read_excel(os.path.join(incoming_data_path,
                            "mines_spatial_locations",
                            "S_P_Southern_Afrca_all_mines_production.xls"),skiprows=[0,1,2],header=[0,1,2])
        inputs_cols = s_and_p_mines.columns.values.tolist()
        output_cols = []
        for idx,(pn,sn,tn) in enumerate(inputs_cols):
            oc = pn
            if "Unnamed" not in sn:
                sn = sn.replace(" Y","")
                oc = f"{oc}_{sn}"
            if "Unnamed" not in tn:
                tn = tn.split("|")
                if len(tn) > 1:
                    oc = f"{oc}_{tn[-1]}"
                else:
                    oc = f"{oc}_total"

            output_cols.append(oc)

        s_and_p_mines.columns = output_cols

        s_and_p_mines["geometry"] = gpd.points_from_xy(
                                s_and_p_mines["LONGITUDE"],s_and_p_mines["LATITUDE"])

        s_and_p_mines = gpd.GeoDataFrame(s_and_p_mines,geometry="geometry",crs="EPSG:4326")
        s_and_p_mines = add_iso_code(s_and_p_mines,"PROP_ID",global_boundaries)
        s_and_p_mines = gpd.GeoDataFrame(s_and_p_mines,geometry="geometry",crs="EPSG:4326")
        s_and_p_mines.to_file(os.path.join(processed_data_path,"minerals","s_and_p_mines.gpkg"),driver="GPKG")

    process_data = False
    if process_data is True:
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
        s_and_p_mines = add_iso_code(s_and_p_mines,"property_ID",global_boundaries)
        s_and_p_mines = gpd.GeoDataFrame(s_and_p_mines,geometry="geometry",crs="EPSG:4326")
        s_and_p_mines = s_and_p_mines[~s_and_p_mines.geometry.isna()]
        s_and_p_mines.to_file(os.path.join(processed_data_path,
                                        "minerals",
                                        "s_and_p_mines_global_all.gpkg"),
                                        driver="GPKG")

    process_data = False
    if process_data is True:
        file_directory = os.path.join(incoming_data_path,"DataSP_New","mine_mineral_breakdown")
        s_and_p_df = []
        for root, dirs, files in os.walk(file_directory):
            for file in files:
                if file.endswith(".xls"):
                    hazard_file = os.path.join(root, file)
                    s_and_p_mines = pd.read_excel(os.path.join(root,file),skiprows=[0,1,2],header=[0,1,2])
                    inputs_cols = s_and_p_mines.columns.values.tolist()
                    output_cols = []
                    for idx,(pn,sn,tn) in enumerate(inputs_cols):
                        oc = pn
                        if "Unnamed" not in sn:
                            sn = sn.replace(" Y","")
                            oc = f"{oc}_{sn}"
                        if "Unnamed" not in tn:
                            tn = tn.split("|")
                            if len(tn) > 1:
                                oc = f"{oc}_{tn[-1]}"
                            else:
                                oc = f"{oc}_total"

                        output_cols.append(oc)

                    s_and_p_mines.columns = output_cols

                    s_and_p_mines["geometry"] = gpd.points_from_xy(
                                            s_and_p_mines["LONGITUDE"],s_and_p_mines["LATITUDE"])
                    s_and_p_df.append(s_and_p_mines)

        if len(s_and_p_df) > 0:
            s_and_p_mines = pd.concat(s_and_p_df,axis=0,ignore_index=True)
            s_and_p_mines = gpd.GeoDataFrame(s_and_p_mines,geometry="geometry",crs="EPSG:4326")
            s_and_p_mines = add_iso_code(s_and_p_mines,"PROP_ID",global_boundaries)
            s_and_p_mines = gpd.GeoDataFrame(s_and_p_mines,geometry="geometry",crs="EPSG:4326")
            s_and_p_mines.to_file(os.path.join(processed_data_path,
                                        "minerals",
                                        "s_and_p_mines_with_current_mineral_tonnages.gpkg"),driver="GPKG")

    process_data = False
    if process_data is True:
        s_and_p_mines = gpd.read_file(os.path.join(processed_data_path,
                                        "minerals",
                                        "s_and_p_mines_with_current_mineral_tonnages.gpkg"))
        commodity_columns = [c for c in s_and_p_mines.columns.values.tolist() if "COMMODITY_PRODUCTION_TONNE_BY_PERIOD_2022_" in c]
        country_sums = s_and_p_mines.groupby(["ISO_A3"])[commodity_columns].sum().reset_index()
        country_sums.to_csv(os.path.join(processed_data_path,
                                        "minerals",
                                        "s_and_p_mines_country_mineral_tonnages.csv"),index=False)

        baci_df = pd.read_csv(os.path.join(processed_data_path,
                                        "baci",
                                        "baci_full_clean_continent_trade.csv"))
        minerals = ["copper","cobalt","nickel","manganese","lithium","graphite"]
        baci_df = baci_df[baci_df["reference_mineral"].isin(minerals) & baci_df["refining_stage_cam"].isin([1,2,3])]
        baci_m_df = []
        for ref_m in minerals:
            df = baci_df[baci_df["reference_mineral"] == ref_m].groupby(["export_country_code"])["trade_quantity_tons"].sum().reset_index()
            df.rename(columns={"trade_quantity_tons":f"baci_tons_{ref_m}"},inplace=True)
            s_df = country_sums[["ISO_A3",f"COMMODITY_PRODUCTION_TONNE_BY_PERIOD_2022_{ref_m.title()}"]]
            s_df.rename(
                    columns={"ISO_A3":"export_country_code",
                    f"COMMODITY_PRODUCTION_TONNE_BY_PERIOD_2022_{ref_m.title()}":f"s_and_p_tons_{ref_m}"},
                    inplace=True)
            df = pd.merge(df,s_df,how="left",on=["export_country_code"]).fillna(0)
            baci_m_df.append(df.set_index("export_country_code"))

        baci_m_df = pd.concat(baci_m_df,axis=1).fillna(0)
        baci_m_df = baci_m_df.reset_index()
        baci_m_df.to_csv(os.path.join(processed_data_path,
                                        "minerals",
                                        "baci_s_and_p_tonnages_comparison.csv"),
                                    index=False)
        
    process_data = True
    if process_data is True:
        years = [str(y) for y in np.arange(1980,2041,1)]
        file_directory = os.path.join(processed_data_path,"minerals","mine_level_s_and_p_estimates")
        for root, dirs, files in os.walk(file_directory):
            for file in files:
                if file.endswith(".xlsx"):
                    s_and_p_mines = pd.read_excel(os.path.join(root,file))
                    s_and_p_mines["geometry"] = gpd.points_from_xy(
                                s_and_p_mines["LONGITUDE"],s_and_p_mines["LATITUDE"])
                    s_and_p_mines = gpd.GeoDataFrame(s_and_p_mines,geometry="geometry",crs="EPSG:4326")
                    s_and_p_mines = add_iso_code(s_and_p_mines,"PROP_ID",global_boundaries)
                    s_and_p_mines = gpd.GeoDataFrame(s_and_p_mines,geometry="geometry",crs="EPSG:4326")
                    s_and_p_mines.drop(["index","index_right"],axis=1,inplace=True)
                    s_and_p_mines.columns = s_and_p_mines.columns.astype(str)
                    s_and_p_mines[years] = s_and_p_mines[years].fillna(0)
                    s_and_p_mines["mine_id"] = s_and_p_mines.progress_apply(
                                                        lambda x:f"s_and_p_{x['PROP_ID']}",
                                                        axis=1)
                    s_and_p_mines.to_file(os.path.join(processed_data_path,
                                                    "minerals",
                                                    "s_and_p_mines_estimates.gpkg"),
                                                    layer=file.replace(".xlsx",""),
                                                    driver="GPKG")
                    print (file)





if __name__ == '__main__':
    CONFIG = load_config()
    main(CONFIG)


