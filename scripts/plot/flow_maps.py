#!/usr/bin/env python
# coding: utf-8

import sys
import os
import ast
import pandas as pd
import numpy as np
pd.options.mode.copy_on_write = True
import geopandas as gpd
from shapely.geometry import LineString
from map_plotting_utils import *
from mapping_properties import *
from tqdm import tqdm
tqdm.pandas()
def assign_processing_type(x):
    if x["mode"] == "mine":
        return "Mine"
    else:
        return "Processing location"

def set_geometry_buffer(x,value_column,width_by_range):
    for (i, ((nmin, nmax), width)) in enumerate(width_by_range.items()):
        if nmin <= x[value_column] and x[value_column] < nmax:
            return width

def main(
        config,
        reference_mineral,
        scenarios,
        years,
        percentiles,
        efficient_scales,
        country_cases,
        constraints,
        combination = None,
        distance_from_origin=0.0,
        environmental_buffer=0.0
        ):
    processed_data_path = config['paths']['data']
    output_data_path = config['paths']['results']
    figure_path = config['paths']['figures']


    figures = os.path.join(figure_path,"regional_figures")
    os.makedirs(figures,exist_ok=True)

    figures = os.path.join(figure_path,"regional_figures","flow_figures")
    os.makedirs(figures,exist_ok=True)

    flow_data_folder = os.path.join(output_data_path,"node_edge_flows")
    node_data_folder = os.path.join(output_data_path,"optimised_processing_locations")
    make_plot = True
    

    mp = mineral_properties()[reference_mineral]
    flow_column = f"{reference_mineral}_final_stage_production_tons"
    ccg_countries = pd.read_csv(os.path.join(processed_data_path,"admin_boundaries","ccg_country_codes.csv"))
    ccg_isos = ccg_countries[ccg_countries["ccg_country"] == 1]["iso_3digit_alpha"].values.tolist()
    processing_types = ["Mine","Processing location"]
    processing_colors = mp["node_colors"][:-1]
    link_color = mp["edge_color"]
    
    _,_,xl,yl = map_background_and_bounds(include_countries=ccg_isos)
    dxl = abs(np.diff(xl))[0]
    dyl = abs(np.diff(yl))[0]
    w = 0.03
    dt = 0.05
    panel_span = 2
    marker_size_max = 300
    line_width_max = 0.4
    line_steps = 6
    width_step = 0.08
    interpolation='fisher-jenks'
    key_info = ["key",pd.DataFrame(),pd.DataFrame(),0,1]

    fig_scenario = [
                    scenarios,
                    percentiles,
                    country_cases
                ]
    fig_st = ""
    for fsc in fig_scenario:
        fig_st += "_" + '_'.join(list(set(map(str,fsc))))

    figure_result_file = f"{reference_mineral}{fig_st}"
    ds = str(distance_from_origin).replace('.','p')
    eb = str(environmental_buffer).replace('.','p')
    if combination is None:
        figure_result_file = f"noncombined_{figure_result_file}_scenarios.png"
    else:
        if distance_from_origin > 0.0 or environmental_buffer > 0.0:
            figure_result_file = f"{combination}_{figure_result_file}_op_{ds}km_eb_{eb}km_scenarios.png"
        else:
            figure_result_file = f"{combination}_{figure_result_file}_scenarios.png"

    combinations = list(zip(scenarios,years,percentiles,efficient_scales,country_cases,constraints))
    sc_dfs = []
    nodes_range = []
    edges_range = []
    for idx, (scn,y,p,e,cnt,con) in enumerate(combinations):
        scn_rename = scn.replace(" ","_")
        if scn == "bau":
            scn_title = "BAU"
        else:
            scn_title = scn.title()
        title_name = f"{reference_mineral.title()}: {scn_title} - {p.title()}"
        if y == 2022:
            layer_name = f"{reference_mineral}_{p}"
        else:
            layer_name = f"{reference_mineral}_{p}_{e}"
            if con == "unconstrained":
                title_name = f"{title_name} - No Environmental constraints"
            else:
                title_name = f"{title_name} - Environmental constraints"
        if combination is None:
            results_gpq = f"flows_{layer_name}_{scn_rename}_{y}_{cnt}_{con}.geoparquet"
            optimisation_gpq = f"node_locations_for_energy_conversion_{cnt}_{con}.gpkg"
        else:
            if distance_from_origin > 0.0 or environmental_buffer > 0.0:
                results_gpq = f"{combination}_flows_{layer_name}_{scn_rename}_{y}_{cnt}_{con}_op_{ds}km_eb_{eb}km.geoparquet"
                optimisation_gpq = f"{combination}_node_locations_for_energy_conversion_{cnt}_{con}_op_{ds}km_eb_{eb}km.gpkg"
            else:
                results_gpq = f"{combination}_flows_{layer_name}_{scn_rename}_{y}_{cnt}_{con}.geoparquet"
                optimisation_gpq = f"{combination}_node_locations_for_energy_conversion_{cnt}_{con}.gpkg"

        edge_file_path = os.path.join(flow_data_folder,
                            f"edges_{results_gpq}")
        if os.path.exists(edge_file_path):
            edges_flows_df = gpd.read_parquet(os.path.join(flow_data_folder,
                                f"edges_{results_gpq}"))
            edges_flows_df = edges_flows_df[~edges_flows_df.geometry.isna()]
            nodes_flows_df = gpd.read_parquet(os.path.join(flow_data_folder,
                                f"nodes_{results_gpq}"))
            nodes = nodes_flows_df[
                            (
                                nodes_flows_df["iso3"].isin(ccg_isos)
                            ) & (
                                nodes_flows_df["mode"].isin(["road","rail"])
                            )
                            ]["id"].values.tolist()
            del nodes_flows_df
            edges_flows_df = edges_flows_df[
                                    (
                                        edges_flows_df["from_id"].isin(nodes)
                                    ) | (
                                        edges_flows_df["to_id"].isin(nodes)
                                    )]
            del nodes
            edges_flows_df[flow_column] = edges_flows_df[f"{flow_column}_export"] + edges_flows_df[f"{flow_column}_inter"]
            edges_flows_df = edges_flows_df[edges_flows_df[flow_column] > 0]
            edges_range += edges_flows_df[flow_column].values.tolist()   
            if y == 2022:
                layer_name = f"{y}_{p}"
            else:
                layer_name = f"{scn}_{y}_{p}_{e}"          
            nodes_flows_df = gpd.read_file(
                                os.path.join(
                                    node_data_folder,
                                    optimisation_gpq),
                                layer=layer_name)
            nodes_flows_df = nodes_flows_df[~nodes_flows_df.geometry.isna()]
            nodes_flows_df["processing_type"
                ] = nodes_flows_df.progress_apply(lambda x:assign_processing_type(x),axis=1)
            nodes_flows_df["color"] = np.where(
                                        nodes_flows_df["processing_type"] == "Mine",
                                        processing_colors[0],
                                        processing_colors[1]
                                        )
            fcols = [c for c in nodes_flows_df.columns.values.tolist() if flow_column in c]
            nodes_flows_df[flow_column] = nodes_flows_df[fcols].sum(axis=1)
            nodes_flows_df = nodes_flows_df[nodes_flows_df[flow_column]>0]
            nodes_range += nodes_flows_df[flow_column].values.tolist()

            sc_dfs.append((title_name,edges_flows_df,nodes_flows_df,panel_span*idx + 1,panel_span))
        else:
            make_plot = False

    if make_plot is True:
        n_tmax = max(nodes_range)
        e_tmax = max(edges_range)
        n_tonnage_key = 10**np.arange(1,np.ceil(np.log10(n_tmax)),1)
        e_tonnage_key = 10**np.arange(1,np.ceil(np.log10(e_tmax)),1)
        e_tonnage_weights = generate_weight_bins(edges_range, 
                                width_step=width_step, 
                                n_steps=line_steps,
                                interpolation=interpolation)
        sc_dfs.append(tuple(key_info))
        
        sc_l = len(sc_dfs) - 1 
        if sc_l == 1:
            figwidth = 8
            figheight = figwidth/(2+sc_l*w)/dxl*dyl/(1-dt)
            textfontsize = 9
        else:
            figwidth = 16
            figheight = figwidth/(2.5+sc_l*w)/dxl*dyl/(1-dt)
            textfontsize = 9
        fig = plt.figure(figsize=(figwidth,figheight))
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1-dt,wspace=w)
        for jdx, (sc_n,e_df,n_df,pos,span) in enumerate(sc_dfs):
            ax = plt.subplot2grid([1,2*sc_l+1],[0,pos],1,colspan=span)
            ax.spines[['top','right','bottom','left']].set_visible(False)
            ax.set_aspect('equal')
            ax.set_xticks([])
            ax.set_yticks([])
            if sc_n == "key":
                ax.set_ylim(yl)
                # ax.set_xlim(xl[0]+0.2*dxl,xl[1])
                ax.set_xlim(xl)
                xk = xl[0] + 0.10*dxl
                xt = xk-0.04*dxl
                keys = ['edge_tonnage','node_tonnage','location']
                for ky in range(len(keys)):
                    key = keys[ky]
                    if key == 'node_tonnage':
                        n_tonnage_key = n_tonnage_key[::-1]
                        Nk = n_tonnage_key.size
                        yk = yl[0] + np.linspace(0.05*dyl,0.3*dyl,Nk) + 0.4*ky*dyl
                        yt = yk[-1]+np.diff(yk[-3:-1])
                        size_key = marker_size_max*(n_tonnage_key/n_tmax)**0.5
                        key = gpd.GeoDataFrame(geometry=gpd.points_from_xy(np.ones(Nk)*xk, yk))
                        key.geometry.plot(ax=ax,markersize=size_key,color='k')
                        ax.text(xt,yt,'Locations annual output (tonnes)',weight='bold',fontsize=10,va='center')
                        for k in range(Nk):
                            ax.text(xk,yk[k],'     {:,.0f}'.format(n_tonnage_key[k]),va='center')
                    elif key == 'edge_tonnage':
                        widths = []
                        min_max_vals = []
                        for (i, ((nmin, nmax), w)) in enumerate(e_tonnage_weights.items()):
                            widths.append(w)
                            min_max_vals.append((nmin,nmax))
                        min_max_vals = min_max_vals[::-1]
                        Nk = len(e_tonnage_weights)
                        yk = yl[0] + np.linspace(0.05*dyl,0.3*dyl,Nk) + 0.3*ky*dyl
                        yt = yk[-1]+np.diff(yk[-3:-1])
                        key_1 = gpd.GeoDataFrame(geometry=gpd.points_from_xy(np.ones(Nk)*xk, yk))
                        key_1["id"] = key_1.index.values.tolist()
                        key_2 = gpd.GeoDataFrame(geometry=gpd.points_from_xy(1.1*np.ones(Nk)*xk, yk))
                        key_2["id"] = key_2.index.values.tolist()
                        key = pd.concat([key_1,key_2],axis=0,ignore_index=False)
                        key = key.groupby(['id'])['geometry'].apply(lambda x: LineString(x.tolist())).reset_index()
                        key = gpd.GeoDataFrame(key, geometry='geometry')
                        key["buffersize"] = widths[::-1]
                        key["geometry"] = key.progress_apply(lambda x:x.geometry.buffer(x.buffersize),axis=1)
                        key = gpd.GeoDataFrame(key, geometry='geometry')
                        key.geometry.plot(ax=ax,linewidth=0,facecolor='k',edgecolor='none')
                        ax.text(xt,yt,'Links annual output (tonnes)',weight='bold',fontsize=10,va='center')
                        for k in range(Nk):
                            ax.text(xk,yk[k],'     {:,.0f} - {:,.0f}'.format(min_max_vals[k][0],min_max_vals[k][1]),va='center')
                    else:
                        Nk = len(processing_types)
                        yk = yl[0] + np.linspace(0.05*dyl,0.12*dyl,Nk) + 0.4*ky*dyl
                        yt = yk[-1]+np.diff(yk)[0]
                        ax.text(xt,yt,'Location type',weight='bold',fontsize=10,va='center')
                        for k in range(Nk):
                            ax.text(xk,yk[k],'   '+processing_types[k].capitalize(),va='center')
                            ax.plot(xk,yk[k],'s',
                                    mfc=processing_colors[k],
                                    mec=processing_colors[k],
                                    ms=10)
            else:
                ax = plot_ccg_basemap(
                            ax,
                            include_continents=["Africa"],
                            include_countries=ccg_isos,
                            include_labels=True
                            )
                ax.set_title(sc_n,fontsize=textfontsize,fontweight="bold")
                # e_df["linewidth"] = line_width_max*(np.log10(e_df[flow_column])/np.log10(e_tmax))
                e_df["linewidth"] = e_df.progress_apply(
                                        lambda x:set_geometry_buffer(
                                            x,flow_column,e_tonnage_weights),
                                        axis=1)
                e_df["geometry"] = e_df.progress_apply(lambda x:x.geometry.buffer(x.linewidth),axis=1)
                e_df.geometry.plot(ax=ax,facecolor=link_color,edgecolor='none',linewidth=0,alpha=0.7)
                n_df["markersize"] = marker_size_max*(n_df[flow_column]/n_tmax)**0.5
                n_df = n_df.sort_values(by=flow_column,ascending=False)
                n_df.geometry.plot(
                    ax=ax, 
                    color=n_df["color"], 
                    edgecolor='none',
                    markersize=n_df["markersize"],
                    alpha=0.7)

        plt.tight_layout()
        save_fig(os.path.join(figures,figure_result_file))
        plt.close()


if __name__ == '__main__':
    CONFIG = load_config()
    try:
        if len(sys.argv) > 8:
            reference_mineral = str(sys.argv[1])
            scenarios = ast.literal_eval(str(sys.argv[2]))
            years = ast.literal_eval(str(sys.argv[3]))
            percentiles = ast.literal_eval(str(sys.argv[4]))
            efficient_scales = ast.literal_eval(str(sys.argv[5]))
            country_cases = ast.literal_eval(str(sys.argv[6]))
            constraints = ast.literal_eval(str(sys.argv[7]))
            combination = str(sys.argv[8])
            distance_from_origin = float(sys.argv[9])
            environmental_buffer = float(sys.argv[10])
        else:
            reference_mineral = str(sys.argv[1])
            scenarios = ast.literal_eval(str(sys.argv[2]))
            years = ast.literal_eval(str(sys.argv[3]))
            percentiles = ast.literal_eval(str(sys.argv[4]))
            efficient_scales = ast.literal_eval(str(sys.argv[5]))
            country_cases = ast.literal_eval(str(sys.argv[6]))
            constraints = ast.literal_eval(str(sys.argv[7]))
            combination = None
            distance_from_origin = 0.0
            environmental_buffer = 0.0
    except IndexError:
        print("Got arguments", sys.argv)
        exit()
    main(
            CONFIG,
            reference_mineral,
            scenarios,
            years,
            percentiles,
            efficient_scales,
            country_cases,
            constraints,
            combination = combination,
            distance_from_origin=distance_from_origin,
            environmental_buffer=environmental_buffer
            )