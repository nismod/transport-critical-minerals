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

# def find_optimal_locations(opt_list,flows_df,path_df,
#                             reference_mineral,
#                             loc_type,
#                             columns,
#                             mine_stage):
#     if len(flows_df.index) > 0:
#         path_df = path_df[path_df["final_processing_stage"] == mine_stage]
#         node_path_indexes = get_flow_paths_indexes_and_edges_dataframe(path_df,"full_node_path")

#         all_ids = flows_df["id"].values.tolist()
#         node_path_indexes = node_path_indexes[node_path_indexes["id"].isin(all_ids)]
#         c_df_flows = pd.merge(node_path_indexes,flows_df,how="left",on=["id"]).fillna(0)

#         while len(c_df_flows.index) > 0:
#             optimal_locations = defaultdict()
#             c_df_flows = c_df_flows.sort_values(
#                                         by=columns,
#                                         ascending=[False,True,True,True])
#             nid = c_df_flows["id"].values[0]
#             optimal_locations["iso3"] = c_df_flows["iso3"].values[0]
#             optimal_locations["id"] = c_df_flows["id"].values[0]
#             optimal_locations[
#                 f"{reference_mineral}_{loc_type}"
#                 ] = c_df_flows[f"{reference_mineral}_{loc_type}"].values[0]
#             for c in columns:
#                 optimal_locations[c] = c_df_flows[c].values[0]

#             opt_list.append(optimal_locations)
#             pth_idx = list(set(c_df_flows[c_df_flows["id"] == nid]["path_index"].values.tolist()))
#             c_df_flows = c_df_flows[~c_df_flows["path_index"].isin(pth_idx)]

#     return opt_list

def find_optimal_locations(flow_dataframe,nodes_dataframe,iso_list,
                            initial_tons_column,
                            final_tons_column,
                            gcost_column,
                            distance_column,
                            time_column,
                            production_size,
                            country_case):
    flow_dataframe = pd.merge(
                            flow_dataframe,
                            nodes_dataframe[["id","iso3","mode"]],
                            how="left",
                            on=["id"]
                            )
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
    for c in columns:
        flow_dataframe[{c}] = flow_dataframe.groupby(["id"])[c].transform('sum')
    
    c_df_flows = flow_dataframe[
                        (
                            flow_dataframe["initial_processing_stage"] == 0
                        ) & (
                            flow_dataframe[initial_tons_column] >= production_size
                        )]
    opt_list = []
    while len(c_df_flows.index) > 0:
        optimal_locations = defaultdict()
        c_df_flows = c_df_flows.sort_values(
                                    by=c,
                                    ascending=[False,False,True,True,True])
        id_value = c_df_flows["id"].values[0]
        optimal_locations["iso3"] = c_df_flows["iso3"].values[0]
        optimal_locations["id"] = id_value
        for c in columns:
            optimal_locations[c] = c_df_flows[c].values[0]

        pth_idx = list(set(c_df_flows[c_df_flows["id"] == id_value]["path_index"].values.tolist()))
        optimal_locations["node_paths"] = pth_idx 
        all_paths += pth_idx
        c_df_flows = c_df_flows[~c_df_flows["path_index"].isin(pth_idx)]
        opt_list.append(optimal_locations)

    return opt_list

def update_od_dataframe(initial_df,optimal_df,metal_factor,modify_columns):
    u_df = []
    initial_df["stage_1_tons"] = initial_df["initial_stage_production_tons"]/metal_factor
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
        f_df = s_df.copy()
        f_df["destination_id"] = id_value
        f_df["import_country_code"] = row.iso3
        f_df["final_stage_production_tons"] = f_df["stage_1_tons"]
        f_df["final_processing_stage"] = 1.0
        for m in modify_columns:
            if m in ["node_path","full_node_path"]:
                f_df[m] = f_df.progress_apply(lambda x:x[m][:x["nidx"]+1],axis=1)
            else:        
                f_df[m] = f_df.progress_apply(lambda x:x[m][:x["nidx"]],axis=1)
        
        s_df["origin_id"] = id_value
        s_df["export_country_code"] = row.iso3
        s_df["initial_stage_production_tons"] = s_df["stage_1_tons"]
        s_df["initial_processing_stage"] = 1.0
        for m in modify_columns:
            s_df[m] = s_df.progress_apply(lambda x:x[m][x["nidx"]:],axis=1)

        u_df.append(f_df)
        u_df.append(s_df)

    modified_paths = list(set(modified_paths))
    u_df.append(initial_df[~initial_df["path_index"].isin(modified_paths)])
    u_df = pd.concat(u_df,axis=0,ignore_index=True).fillna(0)
    u_df.drop(["stage_1_tons","nidx"],axis=1,ignore_index=True)

    return u_df

def main(config,year,percentile,efficient_scale,country_case):
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
    modify_columns = [
                        "node_path",
                        "full_node_path",
                        "edge_path",
                        "full_edge_path",
                        "gcost_usd_tons_path",
                        "distance_km_path",
                        "time_hr"
                    ]

    #  Get a number of input dataframes
    baseline_year = 2022
    data_type = {"initial_refined_stage":"str","final_refined_stage":"str"}
    (
        pr_conv_factors_df, 
        metal_content_factors_df, 
        ccg_countries, mine_city_stages, _
    ) = get_common_input_dataframes(data_type,year,baseline_year)
    nodes = add_geometries_to_flows([],
                                modes=["rail","sea","road","mine","city"],
                                layer_type="nodes",merge=False)

    all_flows = []
    for reference_mineral in reference_minerals:
        # Find year locations
        if year == 2022:
            file_name = f"{reference_mineral}_flow_paths_{year}"
            production_size = 0
        else:
            file_name = f"{reference_mineral}_flow_paths_{year}_{percentile}_{efficient_scale}"
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
            metal_factor = metal_content_factors_df[
                        metal_content_factors_df["reference_mineral"] == reference_mineral
                        ]["metal_content_factor"].values[0]

        od_df = pd.read_parquet(
                        os.path.join(results_folder,
                            f"{file_name}.parquet")
                        )
        od_df["path_index"] = od_df.index.values.tolist()
        od_df = od_df[od_df["trade_type"] != "Import"]
        od_df = pd.merge(od_df,mine_city_stages,how="left",on=["reference_mineral"])
        if year == 2022:
            df = od_df.copy()
        else:
            df = []
            for lt in location_types:
                l_df = od_df[od_df["initial_processing_location"] == lt]
                if lt == "mine":
                    country_df_flows = []
                    for row in l_df.itertuples():
                        o_iso = row.export_country_code
                        pidx = row.path_index
                        in_st = row.initial_processing_stage
                        f_st = row.final_processing_stage
                        m_st = row.mine_final_refined_stage
                        in_tons = row.initial_stage_production_tons
                        f_tons = row.final_stage_production_tons
                        node_path = row.node_path
                        node_path = [n.replace("_land","") for n in node_path]
                        gcosts = [0] + list(np.cumsum(row.gcost_usd_tons_path))
                        dist = [0] + list(np.cumsum(row.distance_km_path))
                        time = [0] + list(np.cumsum(row.time_hr_path))
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
                    country_df_flows = pd.DataFrame(country_df_flows,
                                        columns=["export_country_code",
                                        "path_index",
                                        "id",
                                        "initial_processing_stage",
                                        "final_processing_stage",
                                        "initial_stage_production_tons",
                                        "final_stage_production_tons",
                                        "gcosts","distance_km","time_hr"])
                    optimal_df = find_optimal_locations(
                                    country_df_flows,
                                    nodes,
                                    ccg_countries,
                                    "initial_stage_production_tons",
                                    "final_stage_production_tons",
                                    "gcosts","distance_km",
                                    "time_hr",
                                    production_size,
                                    country_case)
                    if len(optimal_df) > 0:
                        optimal_df = pd.DataFrame(optimal_df)
                        l_df = update_od_dataframe(l_df,optimal_df,metal_factor,modify_columns)

                df.append(l_df)

            df = pd.concat(df,axis=0,ignore_index=True).fillna(0)

        df.to_parquet(
            os.path.join(
                results_folder,
                f"{file_name}_modified_od_{country_case}.parquet"),
            index=False)

        df = df.groupby(
                        [
                        "reference_mineral",
                        "export_country_code",
                        "initial_processing_stage",
                        "final_processing_stage",
                        "origin_id"]).agg(dict([(c,"sum") for c in trade_ton_columns])).reset_index()
        flows_df = assign_node_flows(df,trade_ton_columns,reference_mineral)
        flows_df = pd.merge(flows_df,nodes,how="left",on=["id"])
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
                            f"processing_nodes_flows_{year}_{country_case}.gpkg"),
                            layer=layer_name,driver="GPKG")

        


if __name__ == '__main__':
    CONFIG = load_config()
    try:
        year = int(sys.argv[1])
        percentile = int(sys.argv[2])
        efficient_scale = str(sys.argv[3])
        country_case = str(sys.argv[4])
    except IndexError:
        print("Got arguments", sys.argv)
        exit()
    main(CONFIG,year,percentile,efficient_scale,country_case)