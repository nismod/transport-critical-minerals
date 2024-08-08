"""Road network risks and adaptation maps
"""
import os
import sys
import pandas as pd
import geopandas as gpd
from utils import *
from tqdm import tqdm
tqdm.pandas()

def main(config):
    incoming_data_path = config['paths']['incoming_data']
    processed_data_path = config['paths']['data']
    
    # Read the finalised version of the BACI trade data
    ccg_countries = pd.read_csv(
                        os.path.join(processed_data_path,
                                "baci","ccg_country_codes.csv"))
    ccg_countries = ccg_countries[ccg_countries["ccg_country"] == 1]["iso_3digit_alpha"].values.tolist()

    global_boundaries = gpd.read_file(os.path.join(processed_data_path,
                                    "admin_boundaries",
                                    "gadm36_levels_gpkg",
                                    "gadm36_levels_continents.gpkg"))
    global_boundaries = global_boundaries[global_boundaries["ISO_A3"].isin(ccg_countries)]
    layers = ["baseline_annual","future_annual"]
    index_cols = ["ISO_A3","NAME"]
    present_cols = ["bws_raw","bws_score","bws_cat","bws_label"]
    future_cols = []
    for f in ["bau","opt","pes"]:
    	for t in [30,50,80]:
    		for l in ["r","s","c","l"]:
    			future_cols.append(f"{f}{t}_ws_x_{l}")

    columns = [present_cols,future_cols]
    for indx,(l,c) in enumerate(zip(layers,columns)):
        df = gpd.read_file(
                    os.path.join(
                        incoming_data_path,
                        "Aqueduct40_waterrisk_download_Y2023M07D05",
                        "GDB",
                        "Aq40_Y2023D07M05.gdb"),
                    layer=l)
        df_crs = df.crs
        gdf = gpd.sjoin(df,global_boundaries,how="left",predicate='intersects').reset_index()
        gdf = gdf[gdf["ISO_A3"].isin(ccg_countries)]
        gdf = gdf.explode()
        gdf = gdf[index_cols + c + ["geometry"]]
        gpd.GeoDataFrame(gdf,geometry="geometry",crs=df_crs).to_file(
                os.path.join(
                    processed_data_path,
                    "water_stress",
                    "water_stress_data.gpkg"),
                layer = l,driver="GPKG")

    



if __name__ == '__main__':
    CONFIG = load_config()
    main(CONFIG)
