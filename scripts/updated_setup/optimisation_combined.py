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

def get_mine_conversion_factors(x,pcf_df,ini_st_column,fnl_st_column,cf_column="aggregate_ratio"):
    ref_min = x["reference_mineral"]
    exp_st = x[fnl_st_column]
    if exp_st == 0:
        exp_st = 1.0
    mine_st = x[ini_st_column]
    cf_df = pcf_df[
                    (
                        pcf_df["reference_mineral"] == ref_min
                    ) & (
                        pcf_df["iso3"] == x["export_country_code"])
                    ]
    mc_exp = cf_df["metal_content_factor"].values[0]
    
    if mine_st == 0:
        cf_val = cf_df[cf_df["final_refined_stage"] == exp_st
                        ][cf_column].values[0]
    else:
        cf_val = cf_df[cf_df["final_refined_stage"] == exp_st
                        ][cf_column].values[0]/cf_df[
                        cf_df["final_refined_stage"] == mine_st
                        ][cf_column].values[0]

    return 1.0/cf_val, 1.0/mc_exp

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
    mines_dataframe = mines_dataframe[mines_dataframe[f"{year}_metal_content"] > 0]
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

def existing_processing(od_dataframe,baseline_dataframe):
    opt_df = []
    res_df = []
    b_df = baseline_dataframe[
                                (
                                    baseline_dataframe["initial_processing_location"] == "mine"
                                ) & (
                                    baseline_dataframe["final_processing_stage"] > 1.0
                                )
                                ]
    b_df["baseline_stage"] = b_df["final_processing_stage"].transform("max")                          
    b_df = b_df.groupby(
                    [
                        "origin_id",
                        "baseline_stage"
                    ]
                    )["initial_stage_production_tons"].sum().reset_index()
    b_df.rename(columns={"initial_stage_production_tons":"baseline_metal_tons"},inplace=True)
    od_dataframe = pd.merge(
                            od_dataframe,
                            b_df,
                            how="left",
                            on=["origin_id"]).fillna(0)
    o_df = od_dataframe[od_dataframe["baseline_metal_tons"] <= 0]
    o_df.drop(["baseline_metal_tons"],axis=1,inplace=True)
    opt_df.append(o_df)

    m_df = od_dataframe[od_dataframe["baseline_metal_tons"] > 0]
    m_df["scenario_metal_tons"] = m_df.groupby("origin_id")["initial_stage_production_tons"].transform("sum")
    m_df["extra_tons"] = m_df["scenario_metal_tons"] - m_df["baseline_metal_tons"]

    ms_df = m_df[
                    (
                        m_df["extra_tons"] <= 0
                    ) & (
                    m_df["final_processing_stage"] <= m_df["baseline_stage"]
                )]
    res_df.append(ms_df)
    ms_df = m_df[
                    (
                        m_df["extra_tons"] > 0
                    ) & (
                    m_df["final_processing_stage"] <= m_df["baseline_highest_stage"]
                )]
    ext_df = ms_df.copy()
    ext_df["initial_stage_production_tons"
        ] = ext_df["initial_stage_production_tons"
        ]*ext_df["extra_tons"]/ext_df["scenario_metal_tons"]
    ext_df["final_stage_production_tons"
        ] = ext_df["final_stage_production_tons"
        ]*ext_df["extra_tons"]/ext_df["scenario_metal_tons"]
    ext_df.drop(
                [
                    "baseline_metal_tons",
                    "scenario_metal_tons",
                    "extra_tons"
                ],axis=1,inplace=True
            )
    ext_df["baseline_stage"] = 1.0
    opt_df.append(ext_df)
    ms_df["initial_stage_production_tons"
        ] = ms_df["initial_stage_production_tons"
        ]*ms_df["baseline_metal_tons"]/ms_df["scenario_metal_tons"]
    ms_df["final_stage_production_tons"
        ] = ms_df["final_stage_production_tons"
        ]*ms_df["baseline_metal_tons"]/ms_df["scenario_metal_tons"]
    res_df.append(ms_df)

    ms_df = m_df[m_df["final_processing_stage"] > m_df["baseline_highest_stage"]]
    ms_df.drop(
                [
                    "baseline_metal_tons",
                    "scenario_metal_tons",
                    "extra_tons"
                ],axis=1,inplace=True
            )
    opt_df.append(ms_df)
    opt_df = pd.concat(opt_df,axis=0,ignore_index=True)

    res_df = pd.concat(res_df,axis=0,ignore_index=True)
    res_df.drop(
                [
                    "baseline_stage",
                    "baseline_metal_tons",
                    "scenario_metal_tons",
                    "extra_tons"
                ],axis=1,inplace=True
            )

    return opt_df, res_df



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
                            distance_from_origin=0.0,
                            processing_buffer=0.0,
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
                            non_grid_columns,[processing_buffer]*len(non_grid_columns))

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
    # opt_list_y_rf_df = []
    c_df_flows = flow_dataframe[flow_dataframe["initial_processing_stage"] == 0]
    if distance_from_origin > 0.0:
        c_df_flows = c_df_flows[c_df_flows[
                        distance_column
                        ] <= distance_from_origin
                        ]
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
            # y_rf_df = c_df_flows[c_df_flows["id"] == id_value].drop_duplicates(
            #                         subset=[year_column,reference_mineral_column],keep="first")
            # y_rf_df.rename(columns={"mode":"processing_location"},inplace=True)
            # opt_list_y_rf_df.append(
            #                         y_rf_df[
            #                                 [
            #                                     "id","iso3","processing_location",
            #                                     year_column,
            #                                     reference_mineral_column,
            #                                     "production_size"
            #                                 ] + [f"total_{c}" for c in columns]
            #                             ]
            #                         )
            c_df_flows = c_df_flows[~c_df_flows["path_index"].isin(pth_idx)]


    # return opt_list, opt_list_y_rf_df
    return opt_list

def add_mines_remaining_tonnages(df,mines_df,year):
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
                        "metal_factor",
                        "origin_id"]
                        ).agg(dict([(c,"sum") for c in ["initial_stage_production_tons"]])).reset_index() 
    m_df = pd.merge(
                m_df,
                mines_df[["id",f"{year}_metal_content"]],
                how="left",left_on=["origin_id"],
                right_on=["id"]).fillna(0)
    m_df["initial_stage_production_tons"] = m_df[f"{year}_metal_content"] - m_df["initial_stage_production_tons"]
    m_df = m_df[m_df["initial_stage_production_tons"] > 0]
    if len(m_df.index) > 0:
        m_df["final_processing_stage"] = 1.0
        m_df["final_stage_production_tons"] = m_df["initial_stage_production_tons"]/m_df["metal_factor"]
        m_df.drop(["id",f"{year}_metal_content"],axis=1,inplace=True)
        df = pd.concat([df,m_df],axis=0,ignore_index=True)

        sum_cols = ["initial_stage_production_tons","final_stage_production_tons"]
        df = df.groupby(
                        [
                        "reference_mineral",
                        "export_country_code",
                        "initial_processing_stage",
                        "final_processing_stage",
                        "initial_processing_location",
                        "metal_factor",
                        "origin_id"]
                        ).agg(dict([(c,"sum") for c in sum_cols])).reset_index()

    return df

def get_final_stage_values(x):
    final_tons = x["final_stage_production_tons"]
    stage_1_tons = x["stage_1_tons"]
    base_stage_tons = x["baseline_tons"]

    final_stage = x["final_processing_stage"]
    mine_stage = x["mine_final_refined_stage"]
    baseline_stage = x["baseline_stage"]

    if final_stage < mine_stage:
        return final_stage, final_tons
    elif baseline_stage > 1.0:
        return baseline_stage, base_stage_tons
    else:
        return 1.0, stage_1_tons


def update_od_dataframe(initial_df,optimal_df,pcf_df,modify_columns):
    u_df = []
    modified_paths = []
    initial_df["stage_1_tons"] = initial_df["initial_stage_production_tons"]/initial_df["metal_factor"]
    initial_df["baseline_stage_factors"
        ] = initial_df.progress_apply(
                            lambda x:get_mine_conversion_factors(
                                            x,pcf_df,
                                            "initial_processing_stage",
                                            "baseline_stage"),axis=1
                            )
    initial_df[["baseline_factor","mf"]] = od_df["baseline_stage_factors"].apply(pd.Series)
    initial_df["baseline_tons"] = initial_df["initial_stage_production_tons"]/initial_df["baseline_factor"]
    initial_df.drop(["baseline_factor","mf","baseline_stage_factors"],axis=1,inplace=True)

    initial_df["node_path"] = initial_df.progress_apply(
                                    lambda x:[n.replace("_land","") for n in x["node_path"]],axis=1)
    if len(optimal_df.index) > 0:
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
                f_df["final_stage_values"
                    ] = f_df.progress_apply(lambda x:get_final_stage_values(x),axis=1)
                f_df[
                        [
                            "final_processing_stage",
                            "final_stage_production_tons"
                        ]
                    ] = f_df["final_stage_values"].apply(pd.Series)
                f_df.drop("final_stage_values",axis=1,inplace=True)
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
                s_df["initial_stage_values"
                   ] = s_df.progress_apply(lambda x:get_final_stage_values(x),axis=1)
                s_df[
                       [
                           "initial_processing_stage",
                           "initial_stage_production_tons"
                       ]
                   ] = s_df["initial_stage_values"].apply(pd.Series)
                s_df.drop("initial_stage_values",axis=1,inplace=True)
                s_df["initial_processing_location"] = row.processing_location
                s_df["final_stage_production_tons"
                    ] =  np.where(s_df["final_processing_stage"] < s_df["mine_final_refined_stage"],
                            s_df["final_stage_production_tons"]/s_df["stage_factor"],
                            s_df["final_stage_production_tons"]
                            )
                # s_df["final_processing_stage"] = s_df["mine_final_refined_stage"]
                for m in modify_columns:
                    s_df[m] = s_df.progress_apply(lambda x:x[m][x["nidx"]:],axis=1)

                u_df.append(f_df)
                u_df.append(s_df)

    modified_paths = list(set(modified_paths))
    if len(modified_paths) > 0:
        remaining_df = initial_df[~initial_df["path_index"].isin(modified_paths)]
    else:
        remaining_df = initial_df.copy()
        remaining_df["nidx"] = 1
    
    remaining_df["final_stage_values"
                    ] = remaining_df.progress_apply(lambda x:get_final_stage_values(x),axis=1)
    remaining_df[
            [
                "final_processing_stage",
                "final_stage_production_tons"
            ]
        ] = remaining_df["final_stage_values"].apply(pd.Series)
    remaining_df.drop("final_stage_values",axis=1,inplace=True)
    u_df.append(remaining_df)
    u_df = pd.concat(u_df,axis=0,ignore_index=True).fillna(0)
    u_df.drop(["stage_1_tons","baseline_tons","baseline_stage","nidx"],axis=1,inplace=True)

    return u_df

def main(
        config,
        reference_minerals,
        scenario,
        years,
        percentile,
        efficient_scale,
        country_case,
        constraint,
        baseline_year=2022,
        distance_from_origin=0.0,
        environmental_buffer=0.0
        ):
    incoming_data_path = config['paths']['incoming_data']
    processed_data_path = config['paths']['data']
    output_data_path = config['paths']['results']

    scenario = scenario.replace(" ","_")
    input_folder = os.path.join(output_data_path,"flow_od_paths")
    if distance_from_origin > 0.0 or environmental_buffer > 0.0:
        results_folder = os.path.join(
                                output_data_path,
                                f"combined_flow_optimisation_{country_case}_{constraint}_op_{distance_from_origin}km_eb_{environmental_buffer}km"
                                )
    else:
        results_folder = os.path.join(
                                output_data_path,
                                f"combined_flow_optimisation_{country_case}_{constraint}"
                                )
    os.makedirs(results_folder,exist_ok=True)

    flows_folder = os.path.join(results_folder,"processed_flows")
    os.makedirs(flows_folder,exist_ok=True)

    modified_paths_folder = os.path.join(results_folder,"modified_flow_od_paths")
    os.makedirs(modified_paths_folder,exist_ok=True)
    
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
    non_grid_thresholds = [environmental_buffer]*len(non_grid_columns)

    #  Get a number of input dataframes
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
                                    "nodes_with_location_identifiers_regional.geoparquet"
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
            _, 
            ccg_countries,_,_,_
        ) = get_common_input_dataframes("none",scenario,year,baseline_year)
        mine_city_stages = modify_mineral_usage_factors(scenario,future_year=year)
        mine_city_stages["mine_final_refined_stage"
            ] = mine_city_stages.groupby(
                                ["export_country_code","reference_mineral"]
                                )["final_refined_stage"].transform("min")
        mine_city_stages = mine_city_stages[
                                ["export_country_code","reference_mineral","mine_final_refined_stage"]
                                ]
        mine_city_stages = mine_city_stages.drop_duplicates(
                            [
                                "export_country_code",
                                "reference_mineral",
                                "mine_final_refined_stage"
                            ],
                            keep="first")

        for reference_mineral in reference_minerals:
            mines_df = get_mine_layer(reference_mineral,year,percentile,
                                    mine_id_col="id")
            mines_df["year"] = year
            mines_df["reference_mineral"] = reference_mineral
            if year == baseline_year:
                file_name = f"{reference_mineral}_flow_paths_{year}_{percentile}"
                production_size = production_size_df[
                                            production_size_df[
                                                "reference_mineral"] == reference_mineral
                                                ]["min_threshold_metal_tons"].values[0]
            else:
                file_name = f"{reference_mineral}_flow_paths_{scenario}_{year}_{percentile}_{efficient_scale}"
                production_size = production_size_df[
                                            production_size_df[
                                                "reference_mineral"] == reference_mineral
                                                ][efficient_scale].values[0]

            od_df = pd.read_parquet(
                            os.path.join(
                                input_folder,
                                f"{file_name}.parquet"
                                )
                            )
            od_df = od_df[od_df["trade_type"] != "Import"]
            od_df["year"] = year
            od_df["path_index"] = od_df.apply(lambda x:f"{x.reference_mineral}_{x.year}_{x.name}",axis=1)
            od_df = pd.merge(od_df,mine_city_stages,
                                how="left",
                                on=["export_country_code","reference_mineral"]
                                )
            od_df["stage_metal_factors"] = od_df.progress_apply(
                                    lambda x:get_mine_conversion_factors(
                                        x,
                                        pr_conv_factors_df,
                                        "final_processing_stage",
                                        "mine_final_refined_stage"),axis=1)
            od_df[["stage_factor","metal_factor"]] = od_df["stage_metal_factors"].apply(pd.Series)
            od_df.drop("stage_metal_factors",axis=1,inplace=True)
            if year == baseline_year or scenario == "bau":
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

                mines_dfs.append(mines_df)
                for lt in location_types:
                    l_df = od_df[od_df["initial_processing_location"] == lt]
                    if lt == "mine":
                        baseline_df = pd.read_parquet(
                            os.path.join(
                                input_folder,
                                f"{reference_mineral}_flow_paths_{baseline_year}_baseline.parquet"
                                )
                            )
                        l_df, mod_df = existing_processing(l_df,baseline_df)
                        df.append(mod_df)
                        l_dfs.append(l_df)
                        country_df_flows = []
                        for row in l_df.itertuples():
                            in_st = row.initial_processing_stage
                            f_st = row.final_processing_stage
                            m_st = row.mine_final_refined_stage
                            if (in_st == 0.0) and (f_st >= m_st):
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
        l_df = pd.concat(l_dfs,axis=0,ignore_index=True)
        optimal_df = find_optimal_locations_combined(
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
                        distance_from_origin=distance_from_origin,
                        optimisation=constraint)
        if len(optimal_df) > 0:
            optimal_df = pd.DataFrame(optimal_df)
        else:
            optimal_df = pd.DataFrame()
        
        l_df = update_od_dataframe(l_df,optimal_df,pr_conv_factors_df,modify_columns)
        df.append(l_df)

    df = pd.concat(df,axis=0,ignore_index=True).fillna(0)
    mines_dfs = pd.concat(mines_dfs,axis=0,ignore_index=True)

    # if len(all_optimal_locations) > 0:
    #     all_optimal_locations = pd.concat(all_optimal_locations,axis=0,ignore_index=True)

    for year in years:
        all_flows = []
        for reference_mineral in reference_minerals:
            df_year_rf = df[(df["year"] == year) & (df["reference_mineral"] == reference_mineral)]
            mines_df = mines_dfs[(mines_dfs["year"] == year) & (mines_dfs["reference_mineral"] == reference_mineral)]
            # metal_factor = df_year_rf["metal_factor"].values[0]
            if year > baseline_year:
                file_name = f"{reference_mineral}_flow_paths_{scenario}_{year}_{percentile}_{efficient_scale}"
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
                            "metal_factor",
                            "origin_id"]).agg(dict([(c,"sum") for c in trade_ton_columns])).reset_index()
            df_year_rf = add_mines_remaining_tonnages(df_year_rf,mines_df,year)
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
        reference_mineral_string = "_".join(reference_minerals)
        if year == baseline_year:
            file_name = f"location_totals_{year}_{percentile}"
        else:
            file_name = f"location_totals_{scenario}_{year}_{percentile}_{efficient_scale}"

        all_flows_file = os.path.join(
                                    results_folder,
                                    f"{file_name}_{country_case}_{constraint}.csv"
                                    )
        if os.path.isfile(all_flows_file) is True:
            all_flows.to_csv(all_flows_file,mode='a',header=False,index=False)
        else:
            all_flows.to_csv(all_flows_file,index=False)

if __name__ == '__main__':
    CONFIG = load_config()
    try:
        if len(sys.argv) > 8:
            minerals = ast.literal_eval(str(sys.argv[1]))
            scenario = str(sys.argv[2])
            years = ast.literal_eval(str(sys.argv[3]))
            percentile = str(sys.argv[4])
            efficient_scale = str(sys.argv[5])
            country_case = str(sys.argv[6])
            constraint = str(sys.argv[7])
            baseline_year = int(sys.argv[8])
            distance_from_origin = float(sys.argv[9])
            environmental_buffer = float(sys.argv[10])
        else:
            minerals = ast.literal_eval(str(sys.argv[1]))
            scenario = str(sys.argv[2])
            years = ast.literal_eval(str(sys.argv[3]))
            percentile = str(sys.argv[4])
            efficient_scale = str(sys.argv[5])
            country_case = str(sys.argv[6])
            constraint = str(sys.argv[7])
            baseline_year = 2022
            distance_from_origin = 0.0
            environmental_buffer = 0.0

    except IndexError:
        print("Got arguments", sys.argv)
        exit()
    main(
            CONFIG,
            minerals,scenario,years,percentile,
            efficient_scale,country_case,constraint,
            baseline_year=baseline_year,
            distance_from_origin=distance_from_origin,
            environmental_buffer=environmental_buffer
        )