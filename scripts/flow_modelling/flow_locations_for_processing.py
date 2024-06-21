#!/usr/bin/env python
# coding: utf-8
"""This code finds the route between mines (in Africa) and all ports (outside Africa) 
"""
import sys
import os
import re
import json
import pandas as pd
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

def assign_node_flows(od_dataframe,trade_ton_columns,reference_mineral):
    sum_dict = dict([(f,[]) for f in trade_ton_columns])
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

def main(config,year,percentile,efficient_scale):
    incoming_data_path = config['paths']['incoming_data']
    processed_data_path = config['paths']['data']
    output_data_path = config['paths']['results']

    results_folder = os.path.join(output_data_path,"flow_mapping")
    if os.path.exists(results_folder) == False:
        os.mkdir(results_folder)

    """Step 1: Get the input datasets
    """
    epsg_meters = 3395
    reference_minerals = ["graphite","lithium","cobalt","manganese","nickel","copper"]
    trade_ton_columns = [
                            "initial_stage_production_tons",
                            "final_stage_production_tons"
                        ]
    location_types = ["mine","city_process"]

    #  Get a number of input dataframes
    baseline_year = 2022
    data_type = {"initial_refined_stage":"str","final_refined_stage":"str"}
    (
        pr_conv_factors_df, 
        metal_content_factors_df, 
        ccg_countries, mine_city_stages, _
    ) = get_common_input_dataframes(data_type,year,baseline_year)
    all_flows = []
    for reference_mineral in reference_minerals:
        # Find year locations
        if year == 2022:
            file_name = f"{reference_mineral}_flow_paths_{year}.parquet"
            production_size = 0
        else:
            file_name = f"{reference_mineral}_flow_paths_{year}_{percentile}_{efficient_scale}.parquet"
            # Read data on production scales
            production_size_df = pd.read_excel(
                                        os.path.join(
                                            processed_data_path,
                                            "production_costs",
                                            "scales.xlsx"),
                                        sheet_name="efficient_scales"
                                        )
            # print (production_size_df)
            production_size = production_size_df[
                                        production_size_df[
                                            "reference_mineral"] == reference_mineral
                                            ][efficient_scale].values[0]

        od_df = pd.read_parquet(
                        os.path.join(results_folder,
                            file_name)
                        )
        od_df = od_df[od_df["trade_type"] != "Import"]
        od_df = pd.merge(od_df,mine_city_stages,how="left",on=["reference_mineral"])
        if year == 2022:
            df = od_df.copy()
        else:
            for lt in location_types:
                df = od_df[od["initial_processing_location"] == lt]
                if lt == "mine":
                    country_df_flows = []
                    for row in df.itertuples():
                        o_iso = row.export_country_code
                        in_st = row.initial_processing_stage
                        node_path = row.full_node_path
                        f_st = row.initial_processing_stage
                        m_st = row.mine_final_refined_stage
                        in_tons = row.initial_stage_production_tons
                        f_tons = row.final_stage_production_tons
                        if f_st < m_st:
                            fst_list = [f_st] + [m_st]*(len(node_path) - 1)
                            cf = get_mine_conversion_factors(
                                            metal_content_factors_df,
                                            pr_conv_factors_df,
                                            reference_mineral,
                                            in_st,m_st)
                            ftons_list = [f_tons] + [in_tons/cf]*(len(node_path) - 1)
                        else:
                            fst_list = [f_st]*len(node_path)
                            ftons_list = [f_tons]*len(node_path)
                        

                        country_df_flows += list(zip([o_iso]*len(node_path),
                                        node_path
                                        [in_st]*len(node_path),
                                        fst_list,
                                        [in_tons]*len(node_path),
                                        ftons_list
                                        ))
                    df = pd.DataFrame(country_df_flows,
                                        columns=["export_country_code",
                                        "origin_id","initial_processing_stage",
                                        "final_processing_stage",
                                        "initial_stage_production_tons",
                                        "final_stage_production_tons"])
                    df["reference_mineral"] = reference_mineral

        df = df.groupby(
                        [
                        "reference_mineral",
                        "export_country_code",
                        "initial_processing_stage",
                        "final_processing_stage",
                        "origin_id"]).agg(dict([(c,"sum") for c in trade_ton_columns])).reset_index()
        
        flows_df = assign_node_flows(df,trade_ton_columns,reference_mineral)
        print (flows_df)
        flows_df = add_geometries_to_flows(flows_df,
                                merge_column="id",
                                modes=["rail","sea","road","mine","city"],
                                layer_type="nodes")
        if year > 2022:
            flows_df[f"{reference_mineral}_{efficient_scale}"] = production_size

        flows_df = gpd.GeoDataFrame(flows_df,
                                geometry="geometry",
                                crs="EPSG:4326")
        if year == 2022:
            layer_name = f"{reference_mineral}"
        else:
            layer_name = f"{reference_mineral}_{percentile}_{efficient_scale}"
        
        flows_df.to_file(os.path.join(results_folder,
                            f"processing_nodes_flows_{year}.gpkg"),
                            layer=layer_name,driver="GPKG")

        


if __name__ == '__main__':
    CONFIG = load_config()
    try:
        year = int(sys.argv[1])
        percentile = int(sys.argv[2])
        efficient_scale = str(sys.argv[3])
    except IndexError:
        print("Got arguments", sys.argv)
        exit()
    main(CONFIG,year,percentile,efficient_scale)