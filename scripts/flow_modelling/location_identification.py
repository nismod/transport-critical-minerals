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
from tqdm import tqdm
tqdm.pandas()

def find_country_edges(edges,network_edges):
    weights = defaultdict(int)
    for i, e in enumerate(edges):
        weights[e] += i

    result = sorted(set(edges) & set(network_edges), key=lambda i: weights[i])
    return result
    # items = set(edges) & set(network_edges)
    # result = sorted(items, key=lambda element: edges.index(element))
    # return result
    # return [e for e in edges if e in network_edges]

def main(config):
    incoming_data_path = config['paths']['incoming_data']
    processed_data_path = config['paths']['data']
    output_data_path = config['paths']['results']

    results_folder = os.path.join(output_data_path,"flow_mapping")
    if os.path.exists(results_folder) == False:
        os.mkdir(results_folder)

    """Step 1: Get the input datasets
    """
    reference_mineral = "copper"
    final_refined_stage = 3.0
    trade_ton_column = "mine_output_tons"
    trade_usd_column = "mine_output_thousandUSD"
    years = [2021,2030]
    find_flows = False
    if find_flows is True:
        for year in years:
            od_df = pd.read_parquet(
                                os.path.join(results_folder,
                                    f"{reference_mineral}_flow_paths_{year}.parquet")
                                )
            origin_isos = list(set(od_df["export_country_code"].values.tolist()))
            stages = list(set(od_df["final_refined_stage"].values.tolist()))
            country_df = []
            edges_flows_df = []
            nodes_flows_df = []
            sum_dict = []    
            for o_iso in origin_isos:
                for stage in stages:
                    df = od_df[(od_df["export_country_code"] == o_iso) & (od_df["final_refined_stage"] == stage)]
                    for flow_column in [trade_ton_column,trade_usd_column]:
                        sum_dict.append((f"{reference_mineral}_{flow_column}_stage_{stage}_origin_{o_iso}","sum"
                                            ))
                        for path_type in ["full_edge_path","full_node_path"]:
                            f_df = get_flow_on_edges(
                                           df,
                                            "id",path_type,
                                            flow_column)
                            f_df.rename(columns={flow_column:f"{reference_mineral}_{flow_column}_stage_{stage}_origin_{o_iso}"},
                                inplace=True)
                            if path_type == "full_edge_path":
                                edges_flows_df.append(f_df)
                            else:
                                nodes_flows_df.append(f_df)
                print ("* Done with:",o_iso)

            degree_df = pd.DataFrame()
            for path_type in ["edges","nodes"]:
                if path_type == "edges":
                    flows_df = pd.concat(edges_flows_df,axis=0,ignore_index=True).fillna(0)
                else:
                    flows_df = pd.concat(nodes_flows_df,axis=0,ignore_index=True).fillna(0)
                flows_df = flows_df.groupby(["id"]).agg(dict(sum_dict)).reset_index()

                for flow_column in [trade_ton_column,trade_usd_column]:
                    flow_sums = []
                    for stage in stages:
                        stage_sums = []
                        for o_iso in origin_isos:
                            stage_sums.append(f"{reference_mineral}_{flow_column}_stage_{stage}_origin_{o_iso}")

                        flows_df[f"{reference_mineral}_{flow_column}_stage_{stage}"] = flows_df[stage_sums].sum(axis=1)
                        flow_sums.append(f"{reference_mineral}_{flow_column}_stage_{stage}")

                    flows_df[f"{reference_mineral}_{flow_column}"] = flows_df[flow_sums].sum(axis=1) 

                flows_df = add_geometries_to_flows(flows_df,
                                        merge_column="id",
                                        modes=["rail","sea","road"],
                                        layer_type=path_type)
                if path_type == "edges":
                    degree_df = flows_df[["from_id","to_id"]].stack().value_counts().rename_axis('id').reset_index(name='degree')
                elif path_type == "nodes" and len(degree_df.index) > 0:
                    flows_df = pd.merge(flows_df,degree_df,how="left",on=["id"])

                gpd.GeoDataFrame(flows_df,
                        geometry="geometry",
                        crs="EPSG:4326").to_file(os.path.join(results_folder,
                                f"{reference_mineral}_flows_{year}.gpkg"),
                                layer=path_type,driver="GPKG")


    # Find 2030 locations
    flows_df = gpd.read_file(os.path.join(results_folder,
                            f"{reference_mineral}_flows_2030.gpkg"),
                            layer="nodes",driver="GPKG")
    # origin_isos = list(set(od_df["export_country_code"].values.tolist()))
    # for o_iso in origin_isos:
    #     df = od_df[(od_df["export_country_code"] == o_iso) & (od_df["final_refined_stage"] == final_refined_stage)]
    #     f_df = flows_df[flows_df["iso3"] == o_iso]
    #     if len(df.index) > 0 and len(f_df.index) > 0:
    #         f_df = f_df.sort_values(by=[f"{reference_mineral}_{trade_ton_column}_stage_{final_refined_stage}_origin_{o_iso}","degree"],ascending=False)
    #         print(f_df[["id",f"{reference_mineral}_{trade_ton_column}_stage_{final_refined_stage}_origin_{o_iso}","degree"]])

    od_df = pd.read_parquet(
                                os.path.join(results_folder,
                                    f"{reference_mineral}_flow_paths_2030.parquet")
                                )
    od_df["path_index"] = od_df.index.values.tolist()
    origin_isos = list(set(od_df["export_country_code"].values.tolist()))
    max_flow_min_cost_locations = []
    for o_iso in origin_isos:
        df = od_df[(od_df["export_country_code"] == o_iso) & (od_df["final_refined_stage"] == final_refined_stage)]
        # df["total_mine_tons"] = df.groupby(["origin_id"])[trade_ton_column].transform('sum')
        if len(df.index) > 0:
            country_df_flows = []
            f_df = flows_df[(flows_df["iso3"] == o_iso) & (flows_df["mode"].isin(["road","rail"]))]
            for row in df.itertuples():
                origin = row.origin_id
                pidx = row.path_index
                node_path = row.node_path[1:]
                gcosts = np.cumsum(row.gcost_usd_tons_path)
                dist = np.cumsum(row.distance_km_path)
                time = np.cumsum(row.time_hr_path)
                flow = getattr(row,trade_ton_column)
                values_usd = getattr(row,trade_usd_column)
                # total_flow = getattr(row,"total_mine_tons")
                country_df_flows += list(zip([pidx]*len(node_path),
                                        [origin]*len(node_path),
                                        [o_iso]*len(node_path),
                                        node_path,gcosts,dist,time,
                                        [flow]*len(node_path),
                                        [values_usd]*len(node_path)
                                        ))
            
            country_df_flows = pd.DataFrame(country_df_flows,
                                        columns=["path_index",
                                        "origin_id","export_country_code",
                                        "id","gcost","distance_km","time_hr",
                                        trade_ton_column,trade_usd_column])
            country_df_flows = pd.merge(country_df_flows,f_df[["id","iso3","mode"]],how="left",on=["id"])
            country_df_flows["total_network_tons"] = country_df_flows.groupby(["id"])[trade_ton_column].transform('sum')
            country_df_flows["total_gcost"] = country_df_flows.groupby(["id"])["gcost"].transform('sum')
            country_df_flows["total_distance_km"] = country_df_flows.groupby(["id"])["distance_km"].transform('sum')
            country_df_flows["total_time_hr"] = country_df_flows.groupby(["id"])["time_hr"].transform('sum')
            # origin_tons = country_df_flows.drop_duplicates(subset=["origin_id","path_index"],keep="first")
            # origin_tons["total_mine_tons"] = origin_tons.groupby(["origin_id","id"])[trade_ton_column].transform('sum')
            # country_df_flows = pd.merge(country_df_flows,origin_tons[["origin_id"]],how="left",on=["origin_id","path_index"])
            c_df_flows = country_df_flows[~country_df_flows["iso3"].isna()]
            while len(c_df_flows.index) > 0:
                c_df_flows = c_df_flows.sort_values(
                							by=["total_network_tons",
                								"total_gcost",
                								"total_distance_km",
                								"total_time_hr"],
                							ascending=[False,True,True,True])
                id_value = c_df_flows["id"].values[0]
                max_flow_min_cost_locations.append(
                                (o_iso,id_value,
                                c_df_flows["total_network_tons"].values[0],
                                c_df_flows["total_gcost"].values[0],
                                c_df_flows["total_distance_km"].values[0],
                                c_df_flows["total_time_hr"].values[0]))
                pth_idx = list(set(c_df_flows[c_df_flows["id"] == id_value]["path_index"].values.tolist()))
                c_df_flows = c_df_flows[~c_df_flows["path_index"].isin(pth_idx)]

    max_flow_min_cost_locations = pd.DataFrame(max_flow_min_cost_locations,
                                    columns=["iso3","id",
                                            "total_network_tons",
                                            "total_gcost",
                                            "total_distance_km",
                                            "total_time_hr"])
    print (max_flow_min_cost_locations)

if __name__ == '__main__':
    CONFIG = load_config()
    main(CONFIG)