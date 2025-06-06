#!/usr/bin/env python
# coding: utf-8

import sys
import os
import ast
import pandas as pd
import numpy as np
pd.options.mode.copy_on_write = True
import geopandas as gpd
import shapely
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Circle
from matplotlib.patches import Wedge, Rectangle
from matplotlib.markers import MarkerStyle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from map_plotting_utils import *
from tqdm import tqdm
tqdm.pandas()

config = load_config()
processed_data_path = config['paths']['data']
output_data_path = config['paths']['results']
figure_path = config['paths']['figures']

def draw_pie(dist, xpos, ypos, size, color_map, ax=None):
    """A function to plot pie charts on a map"""
    """https://gis.stackexchange.com/questions/429006/making-spatial-pie-chart-using-two-columns-in-geopandas"""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10,8))

    # for incremental pie slices
    cumsum = np.cumsum(dist)
    cumsum = cumsum/ cumsum[-1]
    pie = [0] + cumsum.tolist()

    xy = []
    s = []
    for r1, r2 in zip(pie[:-1], pie[1:]):
        angles = np.linspace(2 * np.pi * r1 + 0.5*np.pi, 2 * np.pi * r2 + 0.5*np.pi)
        x = [0] + np.cos(angles).tolist()
        y = [0] + np.sin(angles).tolist()

        xy1 = np.column_stack([x, y])
        s1 = np.abs(xy1).max()

        xy.append(xy1)
        s.append(s1)
    for xyi, si in zip(xy,s):
        ax.scatter(
                    [xpos], [ypos], 
                    marker=xyi, 
                    s=size*si**2, 
                    linewidth=0, 
                    edgecolor='none',
                    color=color_map.pop(0),
                    alpha=0.7)

    return ax

def main():
    figures = os.path.join(figure_path,"mine_ownership")
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
    centroid_df["geometry"] = gpd.points_from_xy(
                                    centroid_df["longitude_shift"],
                                    centroid_df["latitude_shift"])
    centroid_df = gpd.GeoDataFrame(centroid_df,geometry="geometry",crs="EPSG:4326")
    
        
    _,_,xl,yl = map_background_and_bounds(include_continents=continents)
    dxl = abs(np.diff(xl))[0]
    dyl = abs(np.diff(yl))[0]
    w = 0.003
    dt = 0.05
    panel_span = 2
    marker_size_max = 3000.0

    """Test Map"""

    bio_diversity_change = pd.read_excel(
                                os.path.join(
                                    processed_data_path,
                                    "investment_index",
                                    "Fig 1 CSIRO_BHI_PARC_INDICES_BY COUNTRY.xlsx"),
                                sheet_name="BHI_habitat_area",
                                skiprows=[0,1,2])
    bio_diversity_change = bio_diversity_change[["ISO3","DIFFERENCE TO PLOT"]]
    bio_diversity_change.rename(columns={"DIFFERENCE TO PLOT":"bio_change"},inplace=True)
    bio_diversity_change = gpd.GeoDataFrame(pd.merge(
                            bio_diversity_change,
                            global_df[["ADM0_A3","geometry"]],
                            how="left",
                            left_on=["ISO3"],
                            right_on=["ADM0_A3"]),
                            geometry="geometry",
                            crs=global_df.crs)
    bio_diversity_change["bio_change"] = bio_diversity_change["bio_change"].fillna(-100.0)
    change_range = [-4.0, -2.0, -1.0, 0.0, 0.5, 1.0]
    change_colors = ["#bf812d","#dfc27d","#f7fcb9","#238443","#004529","#91cf60","#1a9850"]
    investment_df = pd.read_excel(
                        os.path.join(
                            processed_data_path,
                            "investment_index",
                            "Country data infrastructure.xlsx"),
                            sheet_name="Sheet1",skiprows=[0,1])
    investment_df.columns = investment_df.columns.map(str)
    investment_df.rename(
                columns={
                        "Estimates in US$":"country_name",
                        "2020":"investment",
                        "2040":"needs"},
                        inplace=True)
    df1 = investment_df[investment_df["Scenario"]=="Current trends"]
    df2 = investment_df[investment_df["Scenario"]=="Investment need inc. SDGs"]
    investment_df = pd.merge(
                        df1[["country_name","investment"]],
                        df2[["country_name","needs"]],
                        how="left",on=["country_name"]
                        )
    investment_df = gpd.GeoDataFrame(pd.merge(
                        investment_df,
                        centroid_df,
                        how="left",
                        left_on=["country_name"],
                        right_on=["COUNTRY"]),geometry="geometry",crs=centroid_df.crs)
    investment_df.to_csv(os.path.join(
                            processed_data_path,
                            "investment_index",
                            "investment_and_need.csv"),index=False)
    tmax = max(investment_df["needs"].values.tolist() + investment_df["investment"].values.tolist())

    figwidth = 12
    figheight = figwidth/(1+1*w)/dxl*dyl/(1-dt)
    fig = plt.figure(figsize=(figwidth,figheight))
    # plt.subplots_adjust(left=0, bottom=0, right=1, top=1-dt,wspace=w)
    ax = plt.subplot2grid([1,1],[0,0],1,colspan=panel_span)
    ax.spines[['top','right','bottom','left']].set_visible(True)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    
    ax = plot_global_basemap(
                ax,
                include_continents=continents,
                include_countries=countries,
                facecolor=None
                )
    for idx, (lb,ub) in enumerate(zip(change_range[:-1],change_range[1:])):
        df = bio_diversity_change[
                        (
                            bio_diversity_change["bio_change"] >= lb
                        ) & (
                            bio_diversity_change["bio_change"] < ub
                        )]
        df.plot(ax=ax, color=change_colors[idx],edgecolor="black",linewidth=0.2)
    cols = ["investment","needs"]
    colors = ["#0570b0","#d94801"]
    for i,(c,cl) in enumerate(zip(cols,colors)):
        for row in investment_df.itertuples():
            values = [getattr(row,c),getattr(row,c)]
            if i == 0:
                colors = [cl,"none"]
            else:
                colors = ["none",cl]
            draw_pie(dist=values, 
                    xpos=row.geometry.x, 
                    ypos=row.geometry.y, 
                    size=marker_size_max*(getattr(row,c)/tmax)**0.5, 
                    color_map=colors,
                    ax=ax)

    ins = ax.inset_axes([0.60,0.05,0.20,0.12])
    ins.spines[['top','right','bottom','left']].set_visible(False)
    ins.set_xticks([])
    ins.set_yticks([])
    # ins.set_ylim([-3,2])
    # ins.set_xlim([-1,1.5])
    x0 = 0.60
    y0 = 0.08
    x1 = 0.63
    y1 = 0.095
    deltax = 0.03
    for idx, (lb,ub) in enumerate(zip(change_range[:-1],change_range[1:])):
        x0 += deltax
        x1 += deltax 
        box = shapely.geometry.Polygon(((x0, y0), (x0, y1), (x1, y1), (x1, y0), (x0, y0)))
        df = gpd.GeoDataFrame(pd.DataFrame([box],columns=["geometry"]),geometry="geometry")
        df.plot(ax=ins,color=change_colors[idx],edgecolor="black",linewidth=0.2)
        ins.text(x0 - 0.005,0.90*y0,lb,va='center',fontsize=8)
        ins.text(0.635,0.105,'Biodiversity change (%)',weight='bold',va='center',fontsize=10)
    ins.text(x0 + deltax - 0.005,0.90*y0,change_range[-1],va='center',fontsize=8)

    ins = ax.inset_axes([0.575,0.02,0.10,0.04])
    ins.spines[['top','right','bottom','left']].set_visible(False)
    ins.set_xticks([])
    ins.set_yticks([])
    x0 = 0.575
    y0 = 0.02
    x1 = 0.605
    y1 = 0.035
    box = shapely.geometry.Polygon(((x0, y0), (x0, y1), (x1, y1), (x1, y0), (x0, y0)))
    df = gpd.GeoDataFrame(pd.DataFrame([box],columns=["geometry"]),geometry="geometry")
    df.plot(ax=ins,color='lightgrey',edgecolor="black",linewidth=0.2)
    ins.text(0.610,0.028,"No value",va='center',fontsize=8)

    ins = ax.inset_axes([0.35,0.01,0.2,0.2])
    ins.spines[['top','right','bottom','left']].set_visible(False)
    ins.set_xticks([])
    ins.set_yticks([])
    draw_pie(dist=[0.5,0.5], 
            xpos=0.40, 
            ypos=0.05, 
            size=marker_size_max*(0.5)**0.5, 
            color_map=["#0570b0","#d94801"],
            ax=ins)
    ins.text(0.388,0.0508,"Stock",va='center',fontsize=8)
    ins.text(0.407,0.0508,"Investment\nneed",va='center',fontsize=8)
    ins.set_facecolor('none')

    ins = ax.inset_axes([-0.05,-0.05,0.2,0.8])
    ins.spines[['top','right','bottom','left']].set_visible(False)
    ins.set_xticks([])
    ins.set_yticks([])
    ins.set_ylim([-1.5,1.5])
    ins.set_xlim([0,2])
    ins.set_facecolor('none')
    xk = 0.9
    xt = 0.7
    tonnage_key = 1.0e9*np.array([2,10,50,100,1000])
    tonnage_key = tonnage_key[::-1]
    Nk = tonnage_key.size
    yk = np.linspace(-0.5,0.8,Nk)
    yt = 1.1
    size_key = marker_size_max*(tonnage_key/tmax)**0.5
    # points_radius = 2 * radius / 1.0 * points_whole_ax
    # size = points_radius**2

    points_radius = size_key**0.5
    radius = 0.5*points_radius/(0.8*72.0)
    yk = yk - radius
    
    key = gpd.GeoDataFrame(geometry=gpd.points_from_xy(np.ones(Nk)*xk, yk))
    key.geometry.plot(ax=ins,marker=MarkerStyle('o', fillstyle='left'),markersize=size_key,color='none',edgecolor="black",linewidth=0.5)
    key.geometry.plot(ax=ins,marker=MarkerStyle('o', fillstyle='right'),markersize=size_key,color='none',edgecolor="white",linewidth=0.5)

    ins.text(xt,yt,'Stock/Investment\n(US$ billion)',weight='bold',va='center',fontsize=10)
    for k in range(Nk):
        ins.text(xk,yk[k],'       {:,.0f}'.format(tonnage_key[k]/1.0e9),va='center',fontsize=10)
    plt.tight_layout()
    save_fig(os.path.join(figures,"global_basemap_test"))
    plt.close()

    

if __name__ == '__main__':
    main()