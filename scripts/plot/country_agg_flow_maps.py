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

def get_plotting_layers(
        sc_dataframe,
        e_range,
        make_plot,
        scn,y,p,e,cnt,con,
        flow_data_folder,
        flow_column,
        ccg_isos,
        idx = 0,
        combination = None,
        distance_from_origin=0.0,
        environmental_buffer=0.0,
        panel_span=2
        ):
    ds = str(distance_from_origin).replace('.','p')
    eb = str(environmental_buffer).replace('.','p')
    scn_rename = scn.replace(" ","_")
    if scn == "bau":
        scn_title = "BAU"
    else:
        scn_title = scn.title()
    title_name = f"{scn_title} - {p.title()}"
    if y == 2022:
        layer_name = f"{p}"
    else:
        layer_name = f"{p}_{e}_{scn_rename}"
        if con == "unconstrained":
            title_name = f"{title_name} - No Environmental constraints"
        else:
            title_name = f"{title_name} - Environmental constraints"
    if combination is None:
        results_gpq = f"flows_{layer_name}_{y}_{cnt}_{con}.geoparquet"
    else:
        if distance_from_origin > 0.0 or environmental_buffer > 0.0:
            results_gpq = f"{combination}_flows_{layer_name}_{y}_{cnt}_{con}_op_{ds}km_eb_{eb}km.geoparquet"
        else:
            results_gpq = f"{combination}_flows_{layer_name}_{y}_{cnt}_{con}.geoparquet"
            
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
        if len(edges_flows_df.index) > 0:
            all_cols = edges_flows_df.columns.values.tolist()
            o_cols = []
            for cd in country_codes:
                o_cols += [c for c in all_cols if c == f"{flow_column}_{cd}_export"]
        
            cols = list(set(o_cols))
            if len(cols) > 0:
                edges_flows_df[flow_column] = edges_flows_df[cols].sum(axis=1)
                edges_flows_df = edges_flows_df[edges_flows_df[flow_column] > 0]
                if len(edges_flows_df.index) > 0:
                    make_plot = True
                    e_range += edges_flows_df[flow_column].values.tolist()   
                    sc_dataframe.append((title_name,edges_flows_df,panel_span*idx + 1,panel_span))
    return e_range,sc_dataframe,make_plot

def main(
        config,
        country_codes,
        offsets,
        scenarios,
        years,
        percentiles,
        efficient_scales,
        country_cases,
        constraints,
        combination = None,
        distance_from_origin=0.0,
        environmental_buffer=0.0,
        include_labels=True
        ):
    processed_data_path = config['paths']['data']
    output_data_path = config['paths']['results']
    figure_path = config['paths']['figures']


    figures = os.path.join(figure_path,f"{'_'.join(country_codes)}_figures")
    os.makedirs(figures,exist_ok=True)

    figures = os.path.join(figure_path,f"{'_'.join(country_codes)}_figures","aggregated_flow_figures")
    os.makedirs(figures,exist_ok=True)

    flow_data_folder = os.path.join(output_data_path,"aggregated_node_edge_flows")
    make_plot = False
    

    flow_column = "final_stage_production_tons"
    ccg_isos = country_codes
    link_color = "#525252"
    
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
    marker_size_max = 300
    line_width_max = 0.4
    line_steps = 6
    width_step = 0.03
    interpolation='fisher-jenks'
    key_info = ["key",pd.DataFrame(),0,1]
    
    fig_scenario = [
                    scenarios,
                    percentiles,
                    country_cases
                ]
    fig_st = ""
    for fsc in fig_scenario:
        fig_st += "_" + '_'.join(list(set(map(str,fsc))))

    figure_result_file = f"{fig_st}"
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
    edges_range = []
    for scn in ["baseline","bau","early refining","precursor"]:
        for y in [2022,2040]:
            for e in ["min_threshold_metal_tons","max_threshold_metal_tons"]:
                for cnt in ["country","region"]:
                    for con in ["unconstrained","constrained"]:
                        edges_range, _ , _ = get_plotting_layers(
                                                    sc_dfs,
                                                    edges_range,
                                                    make_plot,
                                                    scn,y,"mid",e,cnt,con,
                                                    flow_data_folder,
                                                    flow_column,
                                                    ccg_isos,
                                                    combination=combination,
                                                    distance_from_origin=distance_from_origin,
                                                    environmental_buffer=environmental_buffer)

    sc_dfs = []
    er = []
    make_plot = False
    for idx, (scn,y,p,e,cnt,con) in enumerate(combinations):
        _, sc_dfs , make_plot = get_plotting_layers(
                                                sc_dfs,
                                                er,
                                                make_plot,
                                                scn,y,p,e,cnt,con,
                                                flow_data_folder,
                                                flow_column,
                                                ccg_isos,
                                                idx = idx,
                                                combination=combination,
                                                distance_from_origin=distance_from_origin,
                                                environmental_buffer=environmental_buffer)
        

    if make_plot is True:
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
            legendfontsize = 10
        else:
            figwidth = 16
            figheight = figwidth/(2.5+sc_l*w)/dxl*dyl/(1-dt)
            textfontsize = 10
            legendfontsize = 12
        fig = plt.figure(figsize=(figwidth,figheight))
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1-dt,wspace=w)
        for jdx, (sc_n,e_df,pos,span) in enumerate(sc_dfs):
            ax = plt.subplot2grid([1,2*sc_l+1],[0,pos],1,colspan=span)
            ax.spines[['top','right','bottom','left']].set_visible(False)
            ax.set_aspect('equal')
            ax.set_xticks([])
            ax.set_yticks([])
            if sc_n == "key":
                ax.set_ylim(yl)
                ax.set_xlim(xl[0]+0.5*dxl,xl[1])
                # ax.set_xlim(xl)
                xk = xl[0] + 0.65*dxl
                xt = xk-0.04*dxl
                Nk = len(e_tonnage_weights)
                yk = yl[0] + np.linspace(0.15*dyl,0.4*dyl,Nk)
                yt = yk[-1]+np.diff(yk[-3:-1])
                
                widths = []
                min_max_vals = []
                for (i, ((nmin, nmax), w)) in enumerate(e_tonnage_weights.items()):
                    widths.append(w)
                    min_max_vals.append((nmin,nmax))
                min_max_vals = min_max_vals[::-1]
                key_1 = gpd.GeoDataFrame(geometry=gpd.points_from_xy(np.ones(Nk)*xk, yk))
                key_1["id"] = key_1.index.values.tolist()
                key_2 = gpd.GeoDataFrame(geometry=gpd.points_from_xy(1.008*np.ones(Nk)*xk, yk))
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
                    ax.text(xk,yk[k],'      {:,.0f} - {:,.0f}'.format(min_max_vals[k][0],min_max_vals[k][1]),va='center')
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
                ax.set_title(sc_n,fontsize=textfontsize,fontweight="bold")
                if len(e_df.index) > 0:
                    e_df["linewidth"] = e_df.progress_apply(
                                            lambda x:set_geometry_buffer(
                                                x,flow_column,e_tonnage_weights),
                                            axis=1)
                    print (e_df)
                    e_df["geometry"] = e_df.progress_apply(lambda x:x.geometry.buffer(x.linewidth),axis=1)
                    e_df.geometry.plot(ax=ax,facecolor=link_color,edgecolor='none',linewidth=0,alpha=0.7)

        plt.tight_layout()
        save_fig(os.path.join(figures,figure_result_file))
        plt.close()


if __name__ == '__main__':
    CONFIG = load_config()
    try:
        if len(sys.argv) > 7:
            scenarios = ast.literal_eval(str(sys.argv[1]))
            years = ast.literal_eval(str(sys.argv[2]))
            percentiles = ast.literal_eval(str(sys.argv[3]))
            efficient_scales = ast.literal_eval(str(sys.argv[4]))
            country_cases = ast.literal_eval(str(sys.argv[5]))
            constraints = ast.literal_eval(str(sys.argv[6]))
            combination = str(sys.argv[7])
            distance_from_origin = float(sys.argv[8])
            environmental_buffer = float(sys.argv[9])
        else:
            scenarios = ast.literal_eval(str(sys.argv[1]))
            years = ast.literal_eval(str(sys.argv[2]))
            percentiles = ast.literal_eval(str(sys.argv[3]))
            efficient_scales = ast.literal_eval(str(sys.argv[4]))
            country_cases = ast.literal_eval(str(sys.argv[5]))
            constraints = ast.literal_eval(str(sys.argv[6]))
            combination = None
            distance_from_origin = 0.0
            environmental_buffer = 0.0
    except IndexError:
        print("Got arguments", sys.argv)
        exit()

    # ccg_countries = ["AGO","BDI","BWA","COD","KEN","MDG","MOZ","MWI","NAM","TZA","UGA","ZAF","ZMB","ZWE"]
    ccg_countries = ["ZMB"]
    default_offset = [-0.2,-0.2,1.0,1.0]
    zaf_offset = [0.0,10.0,0.0,0.0]
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
        
        main(
                CONFIG,
                country_codes,
                offsets,
                scenarios,
                years,
                percentiles,
                efficient_scales,
                country_cases,
                constraints,
                combination = combination,
                distance_from_origin=distance_from_origin,
                environmental_buffer=environmental_buffer,
                include_labels=include_labels
                )