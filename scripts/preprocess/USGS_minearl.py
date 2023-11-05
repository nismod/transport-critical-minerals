#!/usr/bin/env python
# coding: utf-8

import os 
import pandas as pd
import sys
import geopandas as gpd
import re
from utils import *
from tqdm import tqdm

def main(config):

    incoming_data_path=config['paths']['incoming_data']
    processed_data_path=config['paths']['data']


    # Read files
    USGS_shp=os.path.join(processed_data_path,"Africa_GIS Supporting Data", "a. Africa_GIS Shapefiles","AFR_Mineral_Facilities.shp")
    USGS_mineral=gpd.read_file(USGS_shp)


    # Read global mineral gpkg
    Global_mineral_path=os.path.join(incoming_data_path,'Minerals','global_mines_info_maus')



    os.chdir(Global_mineral_path)
    Global_mineral=gpd.read_file("facilities.gpkg")


    # Check if the 'facility_id' column has only unique values
    is_unique = Global_mineral['facility_id'].is_unique

    print(f"The 'facility_id' column is unique: {is_unique}")



    #Read csv files
    processing= pd.read_csv("processing.csv")
    ownership= pd.read_csv("ownership.csv")


    #1. Replace columns name
    column_name_mapping = {
        'DsgAttr01': 'Commodity Type',
        'DsgAttr02': 'Commodity',
        'DsgAttr03': 'Commodity Product',
        'DsgAttr04': 'Multiple Commodities',
        'DsgAttr05': 'Multiple Products',
        'DsgAttr06': 'Year',
        'DsgAttr07': 'Annual Production Capacity',
        'DsgAttr08': 'Capacity Units',
        'DsgAttr09': 'Capacity Notes',
        'DsgAttr10': 'Shared Capacity',
    }
    gdf = USGS_mineral.rename(columns=column_name_mapping)


    # 2.Filter out the locations which have FeatureTyp = Mineral Processing Plants
    gdf_filtered = gdf[gdf['FeatureTyp'] != 'Mineral Processing Plants']



    # 3. Add the iso3 code
    path_to_shp = os.path.join(incoming_data_path,"ports","ne_110m_admin_0_countries","ne_110m_admin_0_countries.shp")
    world = gpd.read_file(path_to_shp)



    gdf_filtered.columns



    gdf_merged = gdf_filtered.merge(world[['SOVEREIGNT', 'ISO_A3']], left_on='Country', right_on='SOVEREIGNT', how='left')



    gdf_merged.drop(columns='SOVEREIGNT',inplace=True)



    # 4.The modified USGS minerals output
    gdf_merged.to_file(os.path.join(processed_data_path,"africa_minerals_modified_USGS.gpkg"), layer="Mineral points",driver="GPKG")



    # 5.Process the processing file
    processing_sorted = processing.sort_values('year', ascending=False)
    processing_unique= processing_sorted.drop_duplicates(subset=['facility_id', 'facility_type', 'input', 'output', 'incl_purchased', 'source_id'])


    processing_csv_path = os.path.join(processed_data_path,"processing_unique_mineral.csv")
    processing_unique.to_csv(processing_csv_path, index=False)

    # 6.Process the ownership file
    ownership_sorted=ownership.sort_values(by='year',ascending=False)
    ownership_unique=ownership_sorted.drop_duplicates(subset=['facility_id'])
    ownership_csv_path = os.path.join(processed_data_path,"ownership_unique_mineral.csv")
    ownership_unique.to_csv(ownership_csv_path, index=False)

   # 7.1 Merge the processing.csv filtered output (O1) with the ownership.csv filtered output (O2).
    O31=processing_unique.merge(ownership_unique, on='facility_id',how='left')
    O31.to_csv("O31.csv", index=False)

    # 7.2 Merge the processing.csv filtered output (O1) with the ownership.csv filtered output (O2).outer

    O3=processing_unique.merge(ownership_unique, on='facility_id',how='outer')
    O3.to_csv("O3.csv", index=False)
    # 8 O3_with_locations
    O3_with_locations = O3.merge(Global_mineral, on='facility_id', how='left')
    O3_with_locations2=gpd.GeoDataFrame(O3_with_locations, geometry='geometry')
    O3_with_locations2.to_file("O3_with_locations.gpkg",driver="GPKG")

    #9 Extract Africa
    Africa_iso3_path=os.path.join(incoming_data_path,'Minerals','Africa iso3.xlsx')
    Africa_iso3=pd.read_excel(Africa_iso3_path, sheet_name=0)
    africa_data = O3_with_locations[O3_with_locations['GID_0'].isin(Africa_iso3['A 3'])]
    # Assuming 'africa_data' is a DataFrame with a 'geometry' column
    africa_O3 = gpd.GeoDataFrame(africa_data, geometry='geometry')

    # Then, you can save it to a file
    africa_O3.to_file("africaO3_with_locations.gpkg", driver="GPKG")


if __name__ == '__main__':
    CONFIG = load_config()
    main(CONFIG)


