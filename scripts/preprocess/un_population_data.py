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

    process_data = True
    if process_data is True:
        year_cols = np.arange(1950,2040,5)
        print (year_cols)
        pop_df = pd.read_csv(os.path.join(processed_data_path,
                            "admin_boundaries",
                            "un_urban_population",
                            "un_urban_population_estimates.csv"))
        
        for y in year_cols:
            pop_df[str(y)] = pop_df[str(y)].str.replace(" ",'')
            pop_df[str(y)] = pop_df[str(y)].astype(int)
        pop_df.rename(columns={"Index":"city_id"},inplace=True)
        pop_df["city_id"] = pop_df.progress_apply(lambda x:f"city_{x.city_id}",axis=1)
        pop_df["geometry"] = gpd.points_from_xy(
                                pop_df["Longitude"],pop_df["Latitude"])

        pop_df = gpd.GeoDataFrame(pop_df,geometry="geometry",crs="EPSG:4326")
        pop_df = add_iso_code(pop_df,"city_id",processed_data_path)
        pop_df = gpd.GeoDataFrame(pop_df,geometry="geometry",crs="EPSG:4326")
        pop_df.to_file(os.path.join(processed_data_path,
                                    "admin_boundaries",
                                    "un_urban_population",
                                    "un_pop_df.gpkg"),driver="GPKG")





if __name__ == '__main__':
    CONFIG = load_config()
    main(CONFIG)


