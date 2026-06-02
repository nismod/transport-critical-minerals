"""Mine and processing location volume plots
"""
import os
import sys
from collections import OrderedDict
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import geopandas as gpd
import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from map_plotting_utils import *
from trade_functions import *
from tqdm import tqdm
tqdm.pandas()

config = load_config()
processed_data_path = config['paths']['data']
output_path = config['paths']['results']
figure_path = config['paths']['figures']

def main():
    figures = os.path.join(figure_path,"global_trade")
    os.makedirs(figures,exist_ok=True)
    
    global_df = gpd.read_file(os.path.join(
                    processed_data_path,'admin_boundaries',
                    'ne_10m_admin_0_countries',
                    'ne_10m_admin_0_countries.shp'),encoding="utf-8")
    global_df = global_df[global_df["CONTINENT"] != 'Antarctica']
    continents = list(set(global_df["CONTINENT"].values.tolist()))
    countries = list(set(global_df["ADM0_A3"].values.tolist()))
    centroid_df = pd.read_csv(
                    os.path.join(
                        processed_data_path,
                        "admin_boundaries/centroids",
                        "countries_iso3_code.csv"
                        )
                    )
    centroid_df = centroid_df[["ADM0_A3","longitude_shift","latitude_shift"]]
    # centroid_df["geometry"] = gpd.points_from_xy(
    #                                 centroid_df["longitude_shift"],
    #                                 centroid_df["latitude_shift"])

    _,_,xl,yl = map_background_and_bounds(include_continents=continents)
    dxl = abs(np.diff(xl))[0]
    dyl = abs(np.diff(yl))[0]
    w = 0.003
    dt = 0.05
    panel_span = 2

    trade_df = pd.read_csv(os.path.join(output_path,"baci_trade_matrices","baci_ccg_country_trade_breakdown_2022_baseline.csv"))
    # trade_df = pd.read_csv(os.path.join(output_path,
    #                                     "baci_trade_matrices",
    #                                     "baci_ccg_country_trade_breakdown_precursor_2040_mid_max_threshold_metal_tons.csv"))
    trade_df = trade_df[(trade_df["export_country_code"] == "COD") & (trade_df["reference_mineral"] == "copper")]
    trade_df = trade_df.groupby(["export_country_code","import_country_code"])["final_stage_production_tons"].sum().reset_index()
    trade_df = pd.merge(trade_df,centroid_df,how="left",left_on=["export_country_code"],right_on=["ADM0_A3"])
    trade_df.rename(columns={"longitude_shift":"from_lon","latitude_shift":"from_lat"},inplace=True)
    trade_df.drop("ADM0_A3",axis=1,inplace=True)
    trade_df = pd.merge(trade_df,centroid_df,how="left",left_on=["import_country_code"],right_on=["ADM0_A3"])
    trade_df.rename(columns={"longitude_shift":"to_lon","latitude_shift":"to_lat"},inplace=True)
    trade_df.drop("ADM0_A3",axis=1,inplace=True)
    trade_df = trade_df.drop_duplicates(subset=["import_country_code"],keep="first")
    max_trade = trade_df["final_stage_production_tons"].max()
    trade_df["percentage"] = 100.0*trade_df["final_stage_production_tons"]/trade_df["final_stage_production_tons"].sum()
    print (trade_df.sort_values(by=["percentage"],ascending=False))
     
    # # """Test Map"""

    # figwidth = 12
    # figheight = figwidth/(1+1*w)/dxl*dyl/(1-dt)
    # fig = plt.figure(figsize=(figwidth,figheight))
    # # plt.subplots_adjust(left=0, bottom=0, right=1, top=1-dt,wspace=w)
    # ax = plt.subplot(1,1,1)
    # ax.spines[['top','right','bottom','left']].set_visible(True)
    # ax.set_aspect('equal')
    # ax.set_xticks([])
    # ax.set_yticks([])
    
    # ax = plot_global_basemap(
    #             ax,
    #             include_continents=continents,
    #             include_countries=countries,
    #             facecolor=None
    #             )

    # for row in trade_df.itertuples():
    #     # ax.plot([row.from_lon, row.to_lon], [row.from_lat, row.to_lat],
    #     #      color='coral', linewidth=2, marker='o',transform=ccrs.Geodetic())
    #     ax.plot([row.from_lon, row.to_lon], [row.from_lat, row.to_lat],
    #          color='coral', linewidth=4.0*row.final_stage_production_tons/max_trade)

    #     if row.final_stage_production_tons/max_trade > 0.1:
    #         ax.text(row.to_lon,row.to_lat,row.import_country_code,fontsize=8)
    # plt.tight_layout()
    # # save_fig(os.path.join(figures,"DRC_copper_baseline"))
    # save_fig(os.path.join(figures,"DRC_copper_future"))
    # plt.close()        


if __name__ == '__main__':
    main()
