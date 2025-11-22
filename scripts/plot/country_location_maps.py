"""Mine and processing location volume plots
"""
import os
import sys
from collections import OrderedDict
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import geopandas as gpd
import ast
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

def main(country_codes,offsets,x_text,include_labels=True):
    figures = os.path.join(figure_path,f"{'_'.join(country_codes)}_figures")
    os.makedirs(figures,exist_ok=True)

    figures = os.path.join(figure_path,f"{'_'.join(country_codes)}_figures","mine_and_processing_locations")
    os.makedirs(figures,exist_ok=True)

    xmin_offset = offsets[0]
    ymin_offset = offsets[1]
    xmax_offset = offsets[2]
    ymax_offset = offsets[3]
    _,_,xl,yl = map_background_and_bounds(include_countries=country_codes,
                                            xmin_offset = xmin_offset,
                                            xmax_offset = xmax_offset,
                                            ymin_offset = ymin_offset,
                                            ymax_offset = ymax_offset)
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
                            {
                                "type":"final_stage_production_tons",
                                "stage_type":["BAU"],
                                "scenarios":["country_unconstrained","country_constrained"],
                                "scenario_names":["country","country"],
                                "years":[2040,2040],
                                "layers":[
                                            "bau_2040_mid_min_threshold_metal_tons",
                                            "bau_2040_mid_min_threshold_metal_tons"],
                                "layers_names":["BAU - No Environmental constraints",
                                                "BAU - Environmental constraints"]
                            },
                            {
                                "type":"final_stage_production_tons",
                                "stage_type":["Early refining"],
                                "scenarios":["country_unconstrained","country_constrained"],
                                "scenario_names":["country","country"],
                                "years":[2040,2040],
                                "layers":[
                                            "early_refining_2040_mid_min_threshold_metal_tons",
                                            "early_refining_2040_mid_min_threshold_metal_tons"],
                                "layers_names":["Early Refining - No Environmental constraints",
                                                "Early Refining - Environmental constraints"]
                            },
                            {
                                "type":"final_stage_production_tons",
                                "stage_type":["Precursor related product"],
                                "scenarios":["country_unconstrained","country_constrained"],
                                "scenario_names":["country","country"],
                                "years":[2040,2040],
                                "layers":[
                                            "precursor_2040_mid_min_threshold_metal_tons",
                                            "precursor_2040_mid_min_threshold_metal_tons"],
                                "layers_names":["Precursor - No Environmental constraints",
                                                "Precursor - Environmental constraints"]
                            },
                            {
                                "type":"final_stage_production_tons",
                                "stage_type":["Early refining"],
                                "scenarios":["region_unconstrained","region_constrained"],
                                "scenario_names":["region","region"],
                                "years":[2040,2040],
                                "layers":[
                                            "early_refining_2040_mid_max_threshold_metal_tons",
                                            "early_refining_2040_mid_max_threshold_metal_tons"],
                                "layers_names":["Early Refining - No Environmental constraints",
                                                "Early Refining - Environmental constraints"]
                            },
                            {
                                "type":"final_stage_production_tons",
                                "stage_type":["Precursor related product"],
                                "scenarios":["region_unconstrained","region_constrained"],
                                "scenario_names":["region","region"],
                                "years":[2040,2040],
                                "layers":[
                                            "precursor_2040_mid_max_threshold_metal_tons",
                                            "precursor_2040_mid_max_threshold_metal_tons"],
                                "layers_names":["Precursor - No Environmental constraints",
                                                "Precursor - Environmental constraints"]
                            },
                        ]
    result_type = ["combined"]
    stage_mapping_df = pd.read_excel(
                                os.path.join(
                                    processed_data_path,
                                    "mineral_usage_factors",
                                    "stage_mapping.xlsx"),
                                sheet_name='stage_maps')
    tmax = 2000000
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
                mine_sites_df = mine_sites_df[mine_sites_df["iso3"].isin(country_codes)]
                mine_city_stages = modify_mineral_usage_factors(sc,future_year=y)
                dfs = []
                for kdx,(rf,rc) in enumerate(zip(reference_minerals,reference_mineral_colors)):
                    if ton_type == "initial_stage_production_tons":
                        cols = [f"{rf}_{ton_type}_0.0_in_{sc_nm}"]
                    else:
                        stages = list(set(mine_city_stages[
                                    mine_city_stages["reference_mineral"] == rf
                                    ]["final_refined_stage"].values.tolist()))
                        cols = [f"{rf}_{ton_type}_{float(st)}_in_{sc_nm}" for st in stages]
                        cols = [c for c in cols if c in mine_sites_df.columns.values.tolist()]

                    mine_sites_df["total_tons"] = mine_sites_df[cols].sum(axis=1)
                    df = mine_sites_df[mine_sites_df["total_tons"] > 0]
                    df = df[["total_tons","geometry"]]
                    df["reference_mineral"] = rf
                    df["color"] = rc
                    dfs.append(df)
                    # tmax += df["total_tons"].values.tolist()
                dfs = pd.concat(dfs,axis=0,ignore_index=True)
                sc_dfs.append((lyr_nm,dfs,panel_span*idx + 1,panel_span))
        
            # tmax = max(tmax)
            # print (tmax)
            # tmax = 3250000.0
            tonnage_key = 10**np.arange(1,np.ceil(np.log10(tmax)),1)
            sc_dfs.append(tuple(key_info))
            if len(scenarios) == 1:
                figwidth = 12
                figheight = figwidth/(2+len(layers_names)*w)/dxl*dyl/(1-dt)
                # figheight = 5
                textfontsize = 8
            else:
                figwidth = 16
                figheight = figwidth/(2.5+len(layers_names)*w)/dxl*dyl/(1-dt)
                # figheight = 8
                textfontsize = 10
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
                    ax.set_xlim(xl[0]+0.5*dxl,xl[1])
                    xk = xl[0] + 0.65*dxl
                    xt = xk-0.04*dxl
                    keys = ['tonnage','mineral']
                    for ky in range(len(keys)):
                        key = keys[ky]
                        if key == 'tonnage':
                            tonnage_key = tonnage_key[::-1]
                            Nk = tonnage_key.size
                            yk = yl[0] + np.linspace(0.15*dyl,0.4*dyl,Nk) + 0.4*ky*dyl
                            yt = yk[-1]+np.diff(yk[-3:-1])
                            size_key = marker_size_max*(tonnage_key/tmax)**0.5
                            key = gpd.GeoDataFrame(geometry=gpd.points_from_xy(np.ones(Nk)*xk, yk))
                            key.geometry.plot(ax=ax,markersize=size_key,color='k')
                            if ton_type == "initial_stage_production_tons":
                                hd = 'Mine annual output\n(tonnes)'
                            else:
                                hd = 'Processed annual output\n(tonnes)'
                            ax.text(xt,yt,hd,weight='bold',fontsize=textfontsize,ha='left',va='center')
                            for k in range(Nk):
                                ax.text(xk,yk[k],'     {:,.0f}'.format(tonnage_key[k]),fontsize=textfontsize,va='center')
                        else:
                            Nk = len(reference_minerals)
                            yk = yl[0] + np.linspace(0.15*dyl,0.4*dyl,Nk) + 0.4*ky*dyl
                            yt = yk[-1]+np.diff(yk[-3:-1])
                            ax.text(xt,yt,'Mineral produced',weight='bold',fontsize=textfontsize,va='center')
                            for k in range(Nk):
                                ax.text(xk,yk[k],'   '+reference_minerals[k].capitalize(),va='center')
                                ax.plot(xk,yk[k],'s',
                                        mfc=reference_mineral_colors[k],
                                        mec=reference_mineral_colors[k],
                                        ms=10)
                else:
                    ax = plot_ccg_country_basemap(
                                        ax,
                                        include_continents=["Africa"],
                                        include_countries=country_codes,
                                        include_labels=include_labels,
                                        xmin_offset = xmin_offset,
                                        xmax_offset = xmax_offset,
                                        ymin_offset = ymin_offset,
                                        ymax_offset = ymax_offset
                                        )
                    ax.set_title(sc_n,fontsize=16,fontweight="bold")
                    df["markersize"] = marker_size_max*(df["total_tons"]/tmax)**0.5
                    df = df.sort_values(by="total_tons",ascending=False)
                    if len(df.index) > 0:
                        df.geometry.plot(
                            ax=ax, 
                            color=df["color"], 
                            edgecolor='none',
                            markersize=df["markersize"],
                            alpha=0.7)
                        total_tons = df["total_tons"].sum()/1e3
                    else:
                        total_tons = 0.0
                    ax.text(
                        xl[0]+x_text*dxl,yl[0]+0.05*dyl,
                        "Total = {:,.1f} kilotonnes".format(total_tons),
                        fontsize=textfontsize,weight='bold',ha='center')  
            fig_nm = '_'.join(list(set(layers))).replace("_min_threshold_metal_tons","").replace("_max_threshold_metal_tons","")
            if ton_type == "initial_stage_production_tons":
                fig_file = f"mine_metal_content_maps_{fig_nm}.png"
            else:
                fig_nm = fig_nm + '_' + '_'.join(list(set(scenario_names)))
                fig_file = f"{rt}_processing_locations_maps_{fig_nm}.png"
            plt.tight_layout()
            save_fig(os.path.join(figures,fig_file))
            plt.close()          


if __name__ == '__main__':
    # ccg_countries = ["AGO","BDI","BWA","COD","KEN","MDG","MOZ","MWI","NAM","TZA","UGA","ZAF","ZMB","ZWE"]
    ccg_countries = ["ZMB"]
    default_offset = [-0.2,-0.2,1.0,1.0]
    zaf_offset = [0.0,10.0,0.0,0.0]
    right_text = 0.72
    left_text = 0.30 
    for ccg in ccg_countries:
        country_codes = [ccg]
        if ccg in ["UGA"]:
            include_labels = False
        else:
            include_labels = True
        if ccg == "ZAF":
            offsets = zaf_offset
        elif ccg in ["UGA","MWI","BDI"]:
            offsets = [-0.2,-0.2,0.0,0.0]
        else:
            offsets = default_offset
        if ccg in ["COD","TZA","AGO","ZWE","MWI","KEN"]:
            x_text = left_text
        else:
            x_text = right_text
        main(country_codes,offsets,x_text,include_labels=include_labels)
