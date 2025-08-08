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
import matplotlib as mpl
mpl.rcParams['hatch.linewidth'] = 0.3  # previous svg hatch linewidth
from map_plotting_utils import *
from trade_functions import *
from tqdm import tqdm
tqdm.pandas()

config = load_config()
processed_data_path = config['paths']['data']
output_path = config['paths']['results']
figure_path = config['paths']['figures']

def main():
    figures = os.path.join(figure_path,"regional_figures")
    os.makedirs(figures,exist_ok=True)
    
    figures = os.path.join(figure_path,"regional_figures","mine_and_processing_locations")
    os.makedirs(figures,exist_ok=True)
    
    ccg_countries = pd.read_csv(os.path.join(processed_data_path,"admin_boundaries","ccg_country_codes.csv"))
    ccg_isos = ccg_countries[ccg_countries["ccg_country"] == 1]["iso_3digit_alpha"].values.tolist()

    _,_,xl,yl = map_background_and_bounds(include_countries=ccg_isos)
    dxl = abs(np.diff(xl))[0]
    dyl = abs(np.diff(yl))[0]
    w = 0.03
    dt = 0.05
    panel_span = 2
    marker_size_max = 600
    key_info = ["key",pd.DataFrame(),0,1]
    reference_minerals = ["cobalt", "copper", "graphite", "lithium", "manganese", "nickel"]
    reference_mineral_colors = ["#3288bd", "#fee08b", "#66c2a5", "#c2a5cf", "#fdae61", "#f46d43"]
    plot_descriptions = [
                            {
                                "type":"initial_stage_production_tons",
                                "stage_type":["Metal content"],
                                "scenarios":["country_unconstrained"],
                                "scenario_names":["country"],
                                "years":[2022],
                                "layers":["2022_baseline"],
                                "layers_names":["2022 - Baseline"]
                            },
                            {
                                "type":"initial_stage_production_tons",
                                "stage_type":["Metal content"],
                                "scenarios":["country_unconstrained","country_constrained"],
                                "scenario_names":["country","country"],
                                "years":[2040,2040],
                                "layers":[
                                            "bau_2040_mid_min_threshold_metal_tons",
                                            "bau_2040_mid_min_threshold_metal_tons"],
                                "layers_names":["2040 - No Environmental constraints",
                                                "2040 - Environmental constraints"]
                            },
                        ]
    result_type = ["combined"]
    for rt in result_type:
        for plot in plot_descriptions:
            ton_type = plot["type"]
            st_type = plot["stage_type"]
            scenarios = plot["scenarios"]
            scenario_names = plot["scenario_names"]
            years = plot["years"]
            layers = plot["layers"]
            layers_names = plot["layers_names"]
            combos = enumerate(zip(years,scenarios,scenario_names,layers,layers_names))
            sc_dfs = []
            tmax = []
            for idx, (y, sc, sc_nm, lyr, lyr_nm) in combos:
                if rt == "combined":
                    fname = f"{rt}_node_locations_for_energy_conversion_{sc}.gpkg"
                else:
                    fname = f"node_locations_for_energy_conversion_{sc}.gpkg"
                mine_sites_df = gpd.read_file(
                                        os.path.join(
                                            output_path,
                                            "optimised_processing_locations",
                                            fname),
                                        layer=lyr)
                mine_city_stages = modify_mineral_usage_factors(future_year=y)
                dfs = []
                for kdx,(rf,rc) in enumerate(zip(reference_minerals,reference_mineral_colors)):
                    if ton_type == "initial_stage_production_tons":
                        cols = [f"{rf}_{ton_type}_0.0_in_{sc_nm}"]
                    else:
                        stages = mine_city_stages[
                                    mine_city_stages["reference_mineral"] == rf
                                    ]["final_refined_stage"].values.tolist()
                        cols = [f"{rf}_{ton_type}_{float(st)}_in_{sc_nm}" for st in stages]
                        cols = [c for c in cols if c in mine_sites_df.columns.values.tolist()]

                    mine_sites_df["total_tons"] = mine_sites_df[cols].sum(axis=1)
                    df = mine_sites_df[mine_sites_df["total_tons"] > 0]
                    df["env_filter"
                    ] = np.where(
                                    ((
                                        df["distance_to_keybiodiversityareas_km"] == 0
                                    ) | (
                                        df["distance_to_lastofwild_km"] == 0
                                    ) | (
                                        df["distance_to_protectedareas_km"] == 0
                                    )),1,0
                                    )
                    df["water_filter"
                    ] = np.where(
                                    df["distance_to_waterstress_km"] == 0,1,0
                                )
                    df["total_filter"
                    ] = np.where(
                                    (df["env_filter"] == 1) & (df["water_filter"] == 0),1,0
                                )
                    df = df[["total_tons","env_filter","water_filter","total_filter","geometry"]]
                    df["reference_mineral"] = rf
                    df["color"] = rc
                    dfs.append(df)
                    tmax += df["total_tons"].values.tolist()
                dfs = pd.concat(dfs,axis=0,ignore_index=True)
                sc_dfs.append((lyr_nm,dfs,panel_span*idx + 1,panel_span))
        
            tmax = max(tmax)
            tmax = 2e6
            tonnage_key = 10**np.arange(1,np.ceil(np.log10(tmax)),1)
            sc_dfs.append(tuple(key_info))
            if len(scenarios) == 1:
                figwidth = 8
                figheight = figwidth/(2+len(layers_names)*w)/dxl*dyl/(1-dt)
                # figheight = 5
                textfontsize = 6.5
                textfontsize_heading = 7.5
                textfontsize_title = 12
                textfontsize_legend = 8
            else:
                figwidth = 16
                figheight = figwidth/(2.5+len(layers_names)*w)/dxl*dyl/(1-dt)
                # figheight = 8
                textfontsize = 9
                textfontsize_heading = 10
                textfontsize_title = 16
                textfontsize_legend = 10
            fig = plt.figure(figsize=(figwidth,figheight))
            plt.subplots_adjust(left=0, bottom=0, right=1, top=1-dt,wspace=w)
            for jdx, (sc_n,df,pos,span) in enumerate(sc_dfs):
                ax = plt.subplot2grid([1,2*len(scenarios)+1],[0,pos],1,colspan=span)
                ax.spines[['top','right','bottom','left']].set_visible(False)
                ax.set_aspect('equal')
                ax.set_xticks([])
                ax.set_yticks([])
                if sc_n == "key":
                    ax.set_ylim(yl)
                    ax.set_xlim(xl[0]+0.58*dxl,xl[1])
                    xk = xl[0] + 0.63*dxl
                    xt = xk-0.04*dxl
                    keys = ['tonnage','filter_type','mineral']
                    for ky in range(len(keys)):
                        key = keys[ky]
                        if key == 'tonnage':
                            tonnage_key = tonnage_key[::-1]
                            Nk = tonnage_key.size
                            yk = yl[0] + np.linspace(0.15*dyl,0.45*dyl,Nk) + 0.4*ky*dyl
                            yt = yk[-1]+np.diff(yk[-3:-1])
                            size_key = marker_size_max*(tonnage_key/tmax)**0.5
                            key = gpd.GeoDataFrame(geometry=gpd.points_from_xy(np.ones(Nk)*xk, yk))
                            key.geometry.plot(ax=ax,markersize=size_key,color='k')
                            ax.text(xt,yt,'Mine annual output (tonnes)',weight='bold',va='center')
                            for k in range(Nk):
                                ax.text(xk,yk[k],'     {:,.0f}'.format(tonnage_key[k]),
                                        fontsize=textfontsize_legend,va='center')
                        elif key == 'filter_type':
                            ftyp = [
                                    "Protected areas only",
                                    "Water stress areas only",
                                    "Protected and water stress areas"
                                    ]
                            htyp = ["||||||","++++++","xxxxxx"]
                            Nk = len(ftyp)
                            yk = yl[0] + np.linspace(0.15*dyl,0.25*dyl,Nk) + 0.4*ky*dyl
                            yt = yk[-1]+np.diff(yk[-3:-1])
                            ax.text(xt,yt,'Mine and area overlaps',weight='bold',va='center')
                            for k in range(Nk):
                                ax.text(xk,yk[k],'   '+ftyp[k].capitalize(),
                                    fontsize=textfontsize_legend,va='center')
                                ax.scatter(xk,yk[k],marker='s',
                                        facecolor='none',
                                        edgecolor='k',
                                        hatch=htyp[k],
                                        s=100)
                        else:
                            Nk = len(reference_minerals)
                            yk = yl[0] + np.linspace(0.15*dyl,0.3*dyl,Nk) + 0.3*ky*dyl
                            yt = yk[-1]+np.diff(yk[-3:-1])
                            ax.text(xt,yt,'Mineral produced',weight='bold',va='center')
                            for k in range(Nk):
                                ax.text(xk,yk[k],'   '+reference_minerals[k].capitalize(),
                                        fontsize=textfontsize_legend,va='center')
                                ax.plot(xk,yk[k],'s',
                                        mfc=reference_mineral_colors[k],
                                        mec=reference_mineral_colors[k],
                                        ms=10)
                else:
                    ax = plot_ccg_basemap(
                                ax,
                                include_continents=["Africa"],
                                include_countries=ccg_isos,
                                include_labels=True
                                )
                    ax.set_title(sc_n,fontsize=textfontsize_title,fontweight="bold")
                    df["markersize"] = marker_size_max*(df["total_tons"]/tmax)**0.5
                    df = df.sort_values(by="total_tons",ascending=False)
                    ax.text(
                        xl[0]+0.57*dxl,yl[0]+0.35*dyl,
                        'Annual Production (million tonnes)',
                        fontsize=textfontsize_heading,weight='bold',ha='left')
                    ax.text(
                        xl[0]+0.58*dxl,yl[0]+0.32*dyl,
                        'Total = {:.1f}'.format(df["total_tons"].sum()/1e6),
                        fontsize=textfontsize,weight='bold',ha='left') 
                    total_filter_df = df[df["total_filter"] == 1]
                    total_filter_df.geometry.plot(
                        ax=ax, 
                        color=df["color"], 
                        edgecolor='none',
                        markersize=df["markersize"],
                        linewidth=0.5,
                        hatch = "xxxxxx",
                        alpha=0.7)
                    ax.text(
                        xl[0]+0.58*dxl,yl[0]+0.29*dyl,
                        'Protected and water stress areas = {:.1f}'.format(total_filter_df["total_tons"].sum()/1e6),
                        fontsize=textfontsize,weight='bold',ha='left')
                    env_filter_df = df[(df["total_filter"] != 1) & (df["env_filter"] == 1)]
                    env_filter_df.geometry.plot(
                        ax=ax, 
                        color=df["color"], 
                        edgecolor='none',
                        markersize=df["markersize"],
                        linewidth=0.5,
                        hatch = "||||||",
                        alpha=0.7)
                    ax.text(
                        xl[0]+0.58*dxl,yl[0]+0.26*dyl,
                        'Protected areas only = {:.1f}'.format(env_filter_df["total_tons"].sum()/1e6),
                        fontsize=textfontsize,weight='bold',ha='left')
                    water_filter_df = df[(df["total_filter"] != 1) & (df["water_filter"] == 1)]
                    water_filter_df.geometry.plot(
                        ax=ax, 
                        color=df["color"], 
                        edgecolor='none',
                        markersize=df["markersize"],
                        linewidth=0.5,
                        hatch = "++++++",
                        alpha=0.7)
                    ax.text(
                        xl[0]+0.58*dxl,yl[0]+0.23*dyl,
                        'Water stress areas only = {:.1f}'.format(water_filter_df["total_tons"].sum()/1e6),
                        fontsize=textfontsize,weight='bold',ha='left')
                    df = df[(df["env_filter"] == 0) & (df["water_filter"] == 0)]
                    df.geometry.plot(
                        ax=ax, 
                        color=df["color"], 
                        edgecolor='none',
                        markersize=df["markersize"],
                        alpha=0.7)
            fig_nm = '_'.join(list(set(layers))).replace("_min_threshold_metal_tons","").replace("_max_threshold_metal_tons","")
            if ton_type == "initial_stage_production_tons":
                fig_file = f"mine_metal_content_maps_{fig_nm}_with_filters.png"
            else:
                fig_nm = fig_nm + '_' + '_'.join(list(set(scenario_names)))
                fig_file = f"{rt}_processing_locations_maps_{fig_nm}_with_filters.png"
            plt.tight_layout()
            save_fig(os.path.join(figures,fig_file))
            plt.close()          


if __name__ == '__main__':
    main()
