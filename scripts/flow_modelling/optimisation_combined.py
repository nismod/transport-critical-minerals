#!/usr/bin/env python
# coding: utf-8
"""This code finds the optimal locations of processing in Africa 
"""
import sys
import os
import re
import json
import pandas as pd
import ast
import numpy as np
pd.options.mode.copy_on_write = True
import igraph as ig
import geopandas as gpd
from collections import defaultdict
from utils import *
from transport_cost_assignment import *
from trade_functions import * 
from tqdm import tqdm
tqdm.pandas()

def get_mine_conversion_factors(mcf_df,pcf_df,ref_min,mine_st,exp_st,cf_column="aggregate_ratio"):
    cf_df = pcf_df[pcf_df["reference_mineral"] == ref_min]
    if mine_st == 0:
        mc = mcf_df[mcf_df["reference_mineral"] == ref_min
                    ]["metal_content_factor"].values[0]
        cf_val = mc*cf_df[cf_df["final_refined_stage"] == str(exp_st).replace(".0","")
                        ][cf_column].values[0]/cf_df[
                        cf_df["final_refined_stage"] == '1'
                        ][cf_column].values[0]
    else:
        cf_val = cf_df[cf_df["final_refined_stage"] == str(exp_st).replace(".0","")
                        ][cf_column].values[0]/cf_df[
                        cf_df["final_refined_stage"] == str(mine_st).replace(".0","")
                        ][cf_column].values[0]
    return cf_val

def find_processing_locations(x,all_columns,rm,tr,ogns,tc="initial_stage_production_tons"):
    origin = [o for o in ogns if o == x['iso3']]
    c = 0
    r = 0
    thr = f"{rm}_{tr}"
    if len(origin) > 0: 
        col = f"{rm}_{tc}_0.0_origin_{origin[0]}"
        if col in all_columns:
            if x[col] >= x[thr]:
                c = 1
    if x[f"{rm}_{tc}_0.0"] >= x[thr]:
        r = 1
    if x["infra"] == "city":
        c = 1

    return c,r

def filter_out_future_mines(od_dataframe,points_dataframe,
                            mines_dataframe,year,
                            criteria_columns,criteria_thresholds):
    mines_dataframe = mines_dataframe[mines_dataframe[str(year)] > 0]
    new_mines = mines_dataframe[mines_dataframe[f"future_new_mine_{year}"] == 1]["id"].values.tolist()
    points_dataframe = points_dataframe[points_dataframe["id"].isin(new_mines)]
    criteria_columns = [f"distance_to_{l}_km" for l in criteria_columns]
    for idx, (c,v) in enumerate(zip(criteria_columns,criteria_thresholds)):
        points_dataframe = points_dataframe[points_dataframe[c] > v]

    remaining_mines = points_dataframe["id"].values.tolist()
    excluded_mines = list(set(new_mines) - set(remaining_mines))
    if len(excluded_mines) > 0:
        od_dataframe = od_dataframe[~od_dataframe["origin_id"].isin(excluded_mines)]
        mines_dataframe = mines_dataframe[~mines_dataframe["id"].isin(excluded_mines)]

    return od_dataframe,mines_dataframe

def filter_out_offgrid_locations(points_dataframe,
                            grid_column,grid_threshold,
                            locations_include=["mine","city"]):
    p_df = points_dataframe.copy()
    p_df = p_df[
                (
                    ~p_df["mode"].isin(locations_include)
                ) & (
                    p_df[f"distance_to_{grid_column}_km"] > grid_threshold
                )
                ]

    excluded_points = list(set(p_df["id"].values.tolist()))
    if len(excluded_points) > 0:
        points_dataframe = points_dataframe[~points_dataframe["id"].isin(excluded_points)]

    return points_dataframe

def filter_out_processing_locations(points_dataframe,
                            criteria_columns,criteria_thresholds):
    p_df = points_dataframe.copy()
    all_points = set(points_dataframe["id"].values.tolist())
    criteria_columns = [f"distance_to_{l}_km" for l in criteria_columns]
    for idx, (c,v) in enumerate(zip(criteria_columns,criteria_thresholds)):
        p_df = p_df[p_df[c] > v]

    remaining_points = set(p_df["id"].values.tolist())
    excluded_points = list(all_points - remaining_points)
    if len(excluded_points) > 0:
        points_dataframe = points_dataframe[~points_dataframe["id"].isin(excluded_points)]

    return points_dataframe

def assign_node_flows(od_dataframe,trade_ton_columns,reference_mineral,additional_columns=[]):
    sum_dict = dict([(f,[]) for f in trade_ton_columns + additional_columns])
    nodes_flows_df = []
    origin_isos = list(set(od_dataframe["export_country_code"].values.tolist()))
    stages = list(
                    set(
                        zip(
                            od_dataframe["initial_processing_stage"].values.tolist(),
                            od_dataframe["final_processing_stage"].values.tolist()
                            )
                        )
                    )
    for o_iso in origin_isos:
        for idx,(i_st,f_st) in enumerate(stages):
            dataframe = od_dataframe[
                        (
                            od_dataframe["export_country_code"] == o_iso
                        ) & (
                            od_dataframe["initial_processing_stage"] == i_st
                        ) & (
                            od_dataframe["final_processing_stage"] == f_st
                        )]
            if len(dataframe.index) > 0:
                st_tons = list(zip(trade_ton_columns,[i_st,f_st]))
                if len(additional_columns) > 0:
                    st_tons += list(zip(additional_columns,[f_st]*len(additional_columns)))
                for jdx, (flow_column,st) in enumerate(st_tons):
                    sum_dict[flow_column].append(f"{reference_mineral}_{flow_column}_{st}_origin_{o_iso}")                    
                    dataframe.rename(
                            columns={flow_column:f"{reference_mineral}_{flow_column}_{st}_origin_{o_iso}"},
                            inplace=True)
                nodes_flows_df.append(dataframe)

    sum_add = []
    for k,v in sum_dict.items():
        sum_add += list(zip(list(set(v)),["sum"]*len(v)))

    flows_df = pd.concat(nodes_flows_df,axis=0,ignore_index=True).fillna(0)
    flows_df.rename(columns={"origin_id":"id"},inplace=True)
    flows_df = flows_df.groupby(["id"]).agg(dict(sum_add)).reset_index()

    # for flow_column in [trade_ton_column,trade_usd_column]:
    for flow_column,stages in sum_dict.items():
        flow_sums = []
        stage_sums = defaultdict(list)
        for stage in stages:
            stage_sums[stage.split("_origin")[0]].append(stage)
        for k,v in stage_sums.items():
            flows_df[k] = flows_df[list(set(v))].sum(axis=1)
            flow_sums.append(k)

        flows_df[f"{reference_mineral}_{flow_column}"] = flows_df[list(set(flow_sums))].sum(axis=1) 
    
    return flows_df

def find_optimal_locations_combined(flow_dataframe,
                            nodes_dataframe,
                            iso_list,
                            year_column,
                            reference_mineral_column,
                            initial_tons_column,
                            final_tons_column,
                            gcost_column,
                            distance_column,
                            time_column,
                            production_size_column,
                            country_case,
                            grid_column,
                            grid_threshold,
                            non_grid_columns,
                            non_grid_thresholds,
                            optimisation="constrained"):
    flow_dataframe = pd.merge(
                            flow_dataframe,
                            nodes_dataframe,
                            how="left",
                            on=["id"]
                            )
    flow_dataframe = filter_out_offgrid_locations(flow_dataframe,
                            grid_column,grid_threshold)

    if optimisation == "constrained":
        flow_dataframe = filter_out_processing_locations(flow_dataframe,
                            non_grid_columns,non_grid_thresholds)

    if country_case == "country":
        flow_dataframe = flow_dataframe[flow_dataframe["export_country_code"] == flow_dataframe["iso3"]]
    else:
        flow_dataframe = flow_dataframe[flow_dataframe["iso3"].isin(iso_list)]
    columns = [
                initial_tons_column,
                final_tons_column,
                gcost_column,
                distance_column,
                time_column
                ]
    # for c in columns:
    #     flow_dataframe[c] = flow_dataframe.groupby(["id"])[c].transform('sum')
    
    # c_df_flows = flow_dataframe[
    #                     (
    #                         flow_dataframe["initial_processing_stage"] == 0
    #                     ) & (
    #                         flow_dataframe[initial_tons_column] >= production_size
    #                     )]
    opt_list = []
    opt_list_y_rf_df = []
    c_df_flows = flow_dataframe[flow_dataframe["initial_processing_stage"] == 0]
    while len(c_df_flows.index) > 0:
        optimal_locations = defaultdict()
        # First get the sums by years and minerals
        for c in columns:
                c_df_flows[f"combined_{c}"
                    ] = c_df_flows.groupby(
                                ["id"]
                                )[c].transform('sum')
                c_df_flows[f"total_{c}"
                        ] = c_df_flows.groupby(
                                    ["id",year_column,reference_mineral_column]
                                    )[c].transform('sum')
        c_df_flows = c_df_flows[c_df_flows[
                            f"total_{initial_tons_column}"
                            ] >= c_df_flows[production_size_column]
                            ]
        if len(c_df_flows.index) > 0:
            c_df_flows = c_df_flows.sort_values(
                                        by=[f"combined_{c}" for c in columns],
                                        ascending=[False,False,True,True,True])
            id_value = c_df_flows["id"].values[0]
            optimal_locations["iso3"] = c_df_flows["iso3"].values[0]
            optimal_locations["id"] = id_value
            optimal_locations["processing_location"] = c_df_flows["mode"].values[0]
            # for c in columns:
            #     optimal_locations[f"total_{c}"] = c_df_flows[f"total_{c}"].values[0]
            pth_idx = list(set(c_df_flows[c_df_flows["id"] == id_value]["path_index"].values.tolist()))
            optimal_locations["node_paths"] = pth_idx 
            opt_list.append(optimal_locations)
            y_rf_df = c_df_flows[c_df_flows["id"] == id_value].drop_duplicates(
                                    subset=[year_column,reference_mineral_column],keep="first")
            y_rf_df.rename(columns={"mode":"processing_location"},inplace=True)
            opt_list_y_rf_df.append(
                                    y_rf_df[
                                            [
                                                "id","iso3","processing_location",
                                                year_column,
                                                reference_mineral_column,
                                                "production_size"
                                            ] + [f"total_{c}" for c in columns]
                                        ]
                                    )
            c_df_flows = c_df_flows[~c_df_flows["path_index"].isin(pth_idx)]


    return opt_list, opt_list_y_rf_df

def add_mines_remaining_tonnages(df,mines_df,year,metal_factor):
    m_df = df[
                (
                    df["initial_processing_location"] == "mine"
                ) & (
                    df["initial_processing_stage"] == 0.0
                )
            ]
    m_df = m_df.groupby(
                        [
                        "reference_mineral",
                        "export_country_code",
                        "initial_processing_stage",
                        "initial_processing_location",
                        "origin_id"]
                        ).agg(dict([(c,"sum") for c in ["initial_stage_production_tons"]])).reset_index() 
    m_df = pd.merge(
                m_df,
                mines_df[["id",str(year)]],
                how="left",left_on=["origin_id"],
                right_on=["id"]).fillna(0)
    m_df["initial_stage_production_tons"] = m_df[str(year)] - m_df["initial_stage_production_tons"]
    m_df = m_df[m_df["initial_stage_production_tons"] > 0]
    if len(m_df.index) > 0:
        m_df["final_processing_stage"] = 1.0
        m_df["final_stage_production_tons"] = m_df["initial_stage_production_tons"]/metal_factor
        m_df.drop(["id",str(year)],axis=1,inplace=True)
        df = pd.concat([df,m_df],axis=0,ignore_index=True)

        sum_cols = ["initial_stage_production_tons","final_stage_production_tons"]
        df = df.groupby(
                        [
                        "reference_mineral",
                        "export_country_code",
                        "initial_processing_stage",
                        "final_processing_stage",
                        "initial_processing_location",
                        "origin_id"]
                        ).agg(dict([(c,"sum") for c in sum_cols])).reset_index()

    return df



def update_od_dataframe(initial_df,optimal_df,modify_columns):
    u_df = []
    initial_df["stage_1_tons"] = initial_df["initial_stage_production_tons"]/initial_df["metal_factor"]
    initial_df["node_path"] = initial_df.progress_apply(
                                    lambda x:[n.replace("_land","") for n in x["node_path"]],axis=1)
    modified_paths = []
    for row in optimal_df.itertuples():
        pth_idx = row.node_paths
        modified_paths += pth_idx
        id_value = row.id
        s_df = initial_df[initial_df["path_index"].isin(pth_idx)]
        s_df["nidx"] = s_df.progress_apply(lambda x:x["node_path"].index(id_value),axis=1)
        u_df.append(s_df[s_df["nidx"] == 0])
        s_df = s_df[s_df["nidx"] > 0]
        if len(s_df.index) > 0:
            f_df = s_df.copy()
            f_df["destination_id"] = id_value
            f_df["import_country_code"] = row.iso3
            f_df["final_stage_production_tons"
                ] = np.where(f_df["final_processing_stage"] < f_df["mine_final_refined_stage"],
                        f_df["final_stage_production_tons"],
                        f_df["stage_1_tons"]
                        )
            f_df["final_processing_stage"
                ] = np.where(f_df["final_processing_stage"] < f_df["mine_final_refined_stage"],
                        f_df["final_processing_stage"],1.0
                        )
            f_df["final_processing_location"] = row.processing_location
            for m in modify_columns:
                if m in ["node_path","full_node_path"]:
                    f_df[m] = f_df.progress_apply(lambda x:x[m][:x["nidx"]+1],axis=1)
                else:        
                    f_df[m] = f_df.progress_apply(lambda x:x[m][:x["nidx"]],axis=1)
            
            s_df["origin_id"] = id_value
            s_df["export_country_code"] = row.iso3
            # s_df["initial_stage_production_tons"] = s_df["stage_1_tons"]
            # s_df["initial_processing_stage"] = 1.0

            s_df["initial_stage_production_tons"
                ] = np.where(s_df["final_processing_stage"] < s_df["mine_final_refined_stage"],
                        s_df["final_stage_production_tons"],
                        s_df["stage_1_tons"]
                        )
            s_df["initial_processing_stage"
                ] = np.where(s_df["final_processing_stage"] < s_df["mine_final_refined_stage"],
                        s_df["final_processing_stage"],1.0
                        )
            s_df["initial_processing_location"] = row.processing_location
            s_df["final_stage_production_tons"
                ] =  np.where(s_df["final_processing_stage"] < s_df["mine_final_refined_stage"],
                        s_df["final_stage_production_tons"]/s_df["stage_factor"],
                        s_df["final_stage_production_tons"]
                        )
            s_df["final_processing_stage"] = s_df["mine_final_refined_stage"]
            for m in modify_columns:
                s_df[m] = s_df.progress_apply(lambda x:x[m][x["nidx"]:],axis=1)

            u_df.append(f_df)
            u_df.append(s_df)

    modified_paths = list(set(modified_paths))
    remaining_df = initial_df[~initial_df["path_index"].isin(modified_paths)]
    remaining_df["final_stage_production_tons"
        ] = np.where(remaining_df["final_processing_stage"] < remaining_df["mine_final_refined_stage"],
                remaining_df["final_stage_production_tons"],
                remaining_df["stage_1_tons"]
                )
    remaining_df["final_processing_stage"
        ] = np.where(remaining_df["final_processing_stage"] < remaining_df["mine_final_refined_stage"],
                remaining_df["final_processing_stage"],1.0)
    u_df.append(remaining_df)
    u_df = pd.concat(u_df,axis=0,ignore_index=True).fillna(0)
    u_df.drop(["stage_1_tons","nidx"],axis=1,inplace=True)

    return u_df

def main(config,years,reference_minerals,percentile,efficient_scale,country_case,constraint,baseline_year=2022):
    incoming_data_path = config['paths']['incoming_data']
    processed_data_path = config['paths']['data']
    output_data_path = config['paths']['results']

    input_folder = os.path.join(output_data_path,"flow_od_paths")
    results_folder = os.path.join(
                            output_data_path,
                            f"combined_flow_optimisation_{country_case}_{constraint}"
                            )
    if os.path.exists(results_folder) == False:
        os.mkdir(results_folder)

    flows_folder = os.path.join(results_folder,"processed_flows")
    if os.path.exists(flows_folder) == False:
        os.mkdir(flows_folder)

    modified_paths_folder = os.path.join(results_folder,"modified_flow_od_paths")
    if os.path.exists(modified_paths_folder) == False:
        os.mkdir(modified_paths_folder)
    """Step 1: Get the input datasets
    """
    trade_ton_columns = [
                            "initial_stage_production_tons",
                            "final_stage_production_tons"
                        ]
    location_types = ["mine","city_process"]
    modify_columns = [
                        "node_path",
                        "full_node_path",
                        "edge_path",
                        "full_edge_path",
                        "gcost_usd_tons_path",
                        "distance_km_path",
                        "time_hr_path"
                    ]
    grid_column = "grid"
    grid_threshold = 5.0
    non_grid_columns = ["keybiodiversityareas","lastofwild","protectedareas","waterstress"]
    non_grid_thresholds = [0.0,0.0,0.0,0.0]

    #  Get a number of input dataframes
    data_type = {"initial_refined_stage":"str","final_refined_stage":"str"}
    # (
    #     pr_conv_factors_df, 
    #     metal_content_factors_df, 
    #     ccg_countries, mine_city_stages, _,_
    # ) = get_common_input_dataframes(data_type,year,baseline_year)
    production_size_df = pd.read_excel(
                                        os.path.join(
                                            processed_data_path,
                                            "production_costs",
                                            "scales.xlsx"),
                                        sheet_name="efficient_scales"
                                        )
    """Step 1: get all the relevant nodes and find their distances 
                to grid and bio-diversity layers 
    """
    node_location_path = os.path.join(
                                    output_data_path,
                                    "location_filters",
                                    "nodes_with_location_identifiers.geoparquet"
                                    )
    nodes = gpd.read_parquet(node_location_path)
    nodes["mode"] = np.where(nodes["mode"] == "city","city_process",nodes["mode"])
    if baseline_year in years:
        # 2022 scenario is baseline. It should not have future years 
        years = [baseline_year]

    df = []
    l_dfs = []
    country_df_flows_combined = []
    all_optimal_locations = []
    mines_dfs = []
    optimise = True
    for year in years:
        (
            pr_conv_factors_df, 
            metal_content_factors_df, 
            ccg_countries, mine_city_stages, _,_
        ) = get_common_input_dataframes(data_type,year,baseline_year)
        for reference_mineral in reference_minerals:
            # Find year locations
            production_size = production_size_df[
                                        production_size_df[
                                            "reference_mineral"] == reference_mineral
                                            ][efficient_scale].values[0]
            metal_factor = metal_content_factors_df[
                        metal_content_factors_df["reference_mineral"] == reference_mineral
                        ]["metal_content_factor"].values[0]
            mines_df = get_mine_layer(reference_mineral,year,percentile,
                                    mine_id_col="id")
            mines_df["year"] = year
            mines_df["reference_mineral"] = reference_mineral
            if year == baseline_year:
                file_name = f"{reference_mineral}_flow_paths_{year}_{percentile}"
                production_size = 0
            else:
                file_name = f"{reference_mineral}_flow_paths_{year}_{percentile}_{efficient_scale}"

            od_df = pd.read_parquet(
                            os.path.join(
                                input_folder,
                                f"{file_name}.parquet"
                                )
                            )
            od_df = od_df[od_df["trade_type"] != "Import"]
            od_df["year"] = year
            od_df["path_index"] = od_df.apply(lambda x:f"{x.reference_mineral}_{x.year}_{x.name}",axis=1)
            od_df["metal_factor"] = metal_factor
            od_df = pd.merge(od_df,mine_city_stages,how="left",on=["reference_mineral"])
            od_df["stage_factor"] = od_df.progress_apply(
                                    lambda x:get_mine_conversion_factors(
                                        metal_content_factors_df,
                                        pr_conv_factors_df,
                                        reference_mineral,x["final_processing_stage"],
                                        x["mine_final_refined_stage"]),axis=1)
            if year == baseline_year:
                optimise = False
                df.append(od_df.copy())
                del od_df
                mines_dfs.append(mines_df)
            else:
                if constraint == "constrained":
                    od_df, mines_df = filter_out_future_mines(od_df,nodes,
                                                    mines_df,year,
                                                    non_grid_columns,
                                                    non_grid_thresholds)
                    # print (od_df)

                mines_dfs.append(mines_df)
                for lt in location_types:
                    l_df = od_df[od_df["initial_processing_location"] == lt]
                    # od_df["metal_factor"] = metal_factor
                    if lt == "mine":
                        l_dfs.append(l_df)
                        country_df_flows = []
                        for row in l_df.itertuples():
                            in_st = row.initial_processing_stage
                            f_st = row.final_processing_stage
                            m_st = row.mine_final_refined_stage
                            if (in_st == 0.0) and (f_st == m_st):
                                o_iso = row.export_country_code
                                pidx = row.path_index
                                in_tons = row.initial_stage_production_tons
                                f_tons = row.final_stage_production_tons
                                node_path = row.node_path
                                node_path = [n.replace("_land","") for n in node_path]
                                gcosts = [0] + list(np.cumsum(row.gcost_usd_tons_path))
                                dist = [0] + list(np.cumsum(row.distance_km_path))
                                time = [0] + list(np.cumsum(row.time_hr_path))                                
                                fst_list = [f_st]*len(node_path)
                                ftons_list = [f_tons]*len(node_path)
                                country_df_flows += list(zip([o_iso]*len(node_path),
                                                [pidx]*len(node_path),
                                                node_path,
                                                [in_st]*len(node_path),
                                                fst_list,
                                                [in_tons]*len(node_path),
                                                ftons_list,
                                                gcosts,
                                                dist,
                                                time
                                                ))
                        if len(country_df_flows) > 0:
                            country_df_flows = pd.DataFrame(country_df_flows,
                                                columns=["export_country_code",
                                                "path_index",
                                                "id",
                                                "initial_processing_stage",
                                                "final_processing_stage",
                                                "initial_stage_production_tons",
                                                "final_stage_production_tons",
                                                "gcosts","distance_km","time_hr"])
                            country_df_flows["year"] = year
                            country_df_flows["reference_mineral"] = reference_mineral
                            country_df_flows["production_size"] = production_size
                            country_df_flows_combined.append(country_df_flows)
                    else:
                        df.append(l_df)

    if optimise is True:
        country_df_flows = pd.concat(country_df_flows_combined,axis=0,ignore_index=True)
        print (country_df_flows)
        l_df = pd.concat(l_dfs,axis=0,ignore_index=True)
        optimal_df, year_ref_min_df = find_optimal_locations_combined(
                        country_df_flows,
                        nodes,
                        ccg_countries,
                        "year","reference_mineral",
                        "initial_stage_production_tons",
                        "final_stage_production_tons",
                        "gcosts","distance_km",
                        "time_hr",
                        "production_size",
                        country_case,
                        grid_column,
                        grid_threshold,
                        non_grid_columns,
                        non_grid_thresholds,
                        optimisation=constraint)
        if len(optimal_df) > 0:
            optimal_df = pd.DataFrame(optimal_df)
            all_optimal_locations += year_ref_min_df
            l_df = update_od_dataframe(l_df,optimal_df,modify_columns)

            df.append(l_df)

    df = pd.concat(df,axis=0,ignore_index=True).fillna(0)
    mines_dfs = pd.concat(mines_dfs,axis=0,ignore_index=True)

    if len(all_optimal_locations) > 0:
        all_optimal_locations = pd.concat(all_optimal_locations,axis=0,ignore_index=True)
    for year in years:
        all_flows = []
        for reference_mineral in reference_minerals:
            df_year_rf = df[(df["year"] == year) & (df["reference_mineral"] == reference_mineral)]
            mines_df = mines_dfs[(mines_dfs["year"] == year) & (mines_dfs["reference_mineral"] == reference_mineral)]
            metal_factor = df_year_rf["metal_factor"].values[0]
            if year > baseline_year:
                df_year_rf.to_parquet(
                    os.path.join(
                        modified_paths_folder,
                        f"{file_name}.parquet"),
                    index=False)
            df_year_rf = df_year_rf.groupby(
                            [
                            "reference_mineral",
                            "export_country_code",
                            "initial_processing_stage",
                            "final_processing_stage",
                            "initial_processing_location",
                            "origin_id"]).agg(dict([(c,"sum") for c in trade_ton_columns])).reset_index()
            df_year_rf = add_mines_remaining_tonnages(df_year_rf,mines_df,year,metal_factor)
            all_flows.append(df_year_rf)
            flows_df = assign_node_flows(df_year_rf,trade_ton_columns,reference_mineral)
            flows_df = pd.merge(flows_df,nodes,how="left",on=["id"])
            if year > baseline_year:
                flows_df[f"{reference_mineral}_{efficient_scale}"] = production_size

            flows_df = gpd.GeoDataFrame(flows_df,
                                    geometry="geometry",
                                    crs="EPSG:4326")
            if year == baseline_year:
                layer_name = f"{reference_mineral}_{percentile}"
            else:
                layer_name = f"{reference_mineral}_{percentile}_{efficient_scale}"
            
            flows_df.to_parquet(os.path.join(flows_folder,
                                f"{layer_name}_{year}_{country_case}.geoparquet"))

        all_flows = pd.concat(all_flows,axis=0,ignore_index=True)
        if year == baseline_year:
            file_name = f"location_totals_{year}_{percentile}"
            production_size = 0
        else:
            file_name = f"location_totals_{year}_{percentile}_{efficient_scale}"
        all_flows.to_csv(
                os.path.join(
                    results_folder,
                    f"{file_name}_{country_case}_{constraint}.csv"),
                index=False)
        
        all_opt_loc = all_optimal_locations[all_optimal_locations["year"] == year]
        if len(all_opt_loc.index) > 0:
            all_opt_loc.to_csv(
                            os.path.join(
                                results_folder,
                                f"{file_name}_{country_case}_{constraint}_optimal_locations.csv"),
                            index=False)



if __name__ == '__main__':
    CONFIG = load_config()
    try:
        years = ast.literal_eval(str(sys.argv[1]))
        minerals = ast.literal_eval(str(sys.argv[2]))
        percentile = str(sys.argv[3])
        efficient_scale = str(sys.argv[4])
        country_case = str(sys.argv[5])
        constraint = str(sys.argv[6])
        baseline_year = int(sys.argv[7])
    except IndexError:
        print("Got arguments", sys.argv)
        exit()
    main(CONFIG,years,minerals,percentile,efficient_scale,country_case,constraint,baseline_year=baseline_year)