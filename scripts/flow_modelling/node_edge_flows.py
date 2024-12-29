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

def main(
            config,
            reference_mineral,
            year,
            percentile,
            efficient_scale,
            country_case,
            constraint,
            combination = None,
            distance_from_origin=0.0,
            environmental_buffer=0.0
            ):
    incoming_data_path = config['paths']['incoming_data']
    processed_data_path = config['paths']['data']
    output_data_path = config['paths']['results']

    baseline_year = 2022
    # Read the finalised version of the BACI trade data
    ccg_countries = pd.read_csv(
                        os.path.join(processed_data_path,
                                "baci","ccg_country_codes.csv"))
    ccg_countries = ccg_countries[ccg_countries["ccg_country"] == 1]["iso_3digit_alpha"].values.tolist()

    modified_paths_folder = os.path.join(
                                output_data_path,
                                f"flow_optimisation_{country_case}_{constraint}",
                                "modified_flow_od_paths")
    
    if year == baseline_year:
        layer_name = f"{reference_mineral}_{percentile}"
    else:
        layer_name = f"{reference_mineral}_{percentile}_{efficient_scale}"

    if combination is None:
        modified_paths_folder = os.path.join(
                                    output_data_path,
                                    f"flow_optimisation_{country_case}_{constraint}",
                                    "modified_flow_od_paths")
        results_gpq = f"flows_{layer_name}_{year}_{country_case}_{constraint}.geoparquet"
    else:
        if distance_from_origin > 0.0 or environmental_buffer > 0.0:
            modified_paths_folder = os.path.join(
                                output_data_path,
                                f"{combination}_flow_optimisation_{country_case}_{constraint}_op_{distance_from_origin}km_eb_{environmental_buffer}km",
                                "modified_flow_od_paths"
                                )
            ds = str(distance_from_origin).replace('.','p')
            eb = str(environmental_buffer).replace('.','p')
            results_gpq = f"{combination}_flows_{layer_name}_{year}_{country_case}_{constraint}_op_{ds}km_eb_{eb}km.geoparquet"
        else:
            modified_paths_folder = os.path.join(
                                    output_data_path,
                                    f"{combination}_flow_optimisation_{country_case}_{constraint}",
                                    "modified_flow_od_paths"
                                    )
            results_gpq = f"{combination}_flows_{layer_name}_{year}_{country_case}_{constraint}.geoparquet"
    
    results_folder = os.path.join(output_data_path,"node_edge_flows")
    if os.path.exists(results_folder) == False:
        os.mkdir(results_folder)

    """Step 1: Get the input datasets
    """
    trade_ton_columns = [
                            "initial_stage_production_tons",
                            "final_stage_production_tons"
                        ]

    if year == baseline_year:
        file_path = os.path.join(
                        output_data_path,
                        "flow_od_paths",
                        f"{reference_mineral}_flow_paths_{year}_{percentile}.parquet")
        # production_size = 0
        od_df = pd.read_parquet(file_path)
    else:
        export_file_path = os.path.join(
                        modified_paths_folder,
                        f"{reference_mineral}_flow_paths_{year}_{percentile}_{efficient_scale}.parquet")
        export_df = pd.read_parquet(export_file_path)
        export_df = export_df[export_df["trade_type"] != "Import"]
        import_file_path = os.path.join(
                        output_data_path,
                        "flow_od_paths",
                        f"{reference_mineral}_flow_paths_{year}_{percentile}_{efficient_scale}.parquet")
        import_df = pd.read_parquet(import_file_path)
        import_df = import_df[import_df["trade_type"] == "Import"]
        od_df = pd.concat([export_df,import_df],axis=0,ignore_index=True)
    
    edges_flows_df = []
    nodes_flows_df = []
    sum_dict = dict([(f,[]) for f in trade_ton_columns])
    inter_country_df = od_df[od_df["trade_type"] != "Import"]
    inter_country_df = inter_country_df[
                            (
                                inter_country_df["export_country_code"] != inter_country_df["import_country_code"]
                            ) & (
                                inter_country_df["import_country_code"].isin(ccg_countries)
                            )
                            ]
    for ty in ["export","import","inter"]:
        if ty == "export":
            gdf = od_df[od_df["trade_type"] != "Import"]
            gdf = gdf[~gdf.index.isin(inter_country_df.index.values.tolist())]
        elif ty == "import":
            gdf = od_df[od_df["trade_type"] == "Import"]
        else:
            gdf = inter_country_df
            gdf["inter_country_code"] = gdf.progress_apply(
                                            lambda x:f"{x.export_country_code}_{x.import_country_code}",
                                            axis=1)

        origin_isos = list(set(gdf[f"{ty}_country_code"].values.tolist()))
        stages = list(
                        set(
                            zip(
                                gdf["initial_processing_stage"].values.tolist(),
                                gdf["final_processing_stage"].values.tolist()
                                )
                            )
                        )
        for o_iso in origin_isos:
            for idx,(i_st,f_st) in enumerate(stages):
                df = gdf[
                            (
                                gdf[f"{ty}_country_code"] == o_iso
                            ) & (
                                gdf["initial_processing_stage"] == i_st
                            ) & (
                                gdf["final_processing_stage"] == f_st
                            )]
                if len(df.index) > 0:
                    st_tons = list(zip(trade_ton_columns,[i_st,f_st]))
                    for jdx, (flow_column,st) in enumerate(st_tons):
                        if ty == "export":
                            rename_column = f"{reference_mineral}_{flow_column}_{st}_origin_{o_iso}"
                            sum_dict[flow_column].append(rename_column)
                        elif ty == "import":
                            rename_column = f"{reference_mineral}_{flow_column}_{st}_destination_{o_iso}"
                            sum_dict[flow_column].append(rename_column)
                        else:
                            rename_column = f"{reference_mineral}_{flow_column}_{st}_inter_{o_iso}"
                            sum_dict[flow_column].append(rename_column)
                        for path_type in ["full_edge_path","full_node_path"]:
                            f_df = get_flow_on_edges(
                                           df,
                                            "id",path_type,
                                            flow_column)
                            f_df.rename(columns={flow_column:rename_column},inplace=True)
                            if path_type == "full_edge_path":
                                edges_flows_df.append(f_df)
                            else:
                                nodes_flows_df.append(f_df)
            print ("* Done with:",o_iso)

    # print (sum_dict)
    sum_add = []
    for k,v in sum_dict.items():
        sum_add += list(zip(v,["sum"]*len(v)))
    # print ([list(zip(v,["sum"]*len(v))) for k,v in sum_dict.items()])
    degree_df = pd.DataFrame()
    for path_type in ["edges","nodes"]:
        if path_type == "edges":
            flows_df = pd.concat(edges_flows_df,axis=0,ignore_index=True).fillna(0)
        else:
            flows_df = pd.concat(nodes_flows_df,axis=0,ignore_index=True).fillna(0)
        flows_df = flows_df.groupby(
                        ["id"]).agg(dict(sum_add)).reset_index()

        # for flow_column in [trade_ton_column,trade_usd_column]:
        for flow_column,stages in sum_dict.items():
            flow_sums = []
            stage_sums = defaultdict(list)
            for stage in stages:
                if "_origin_" in stage:
                    sn = stage.split("_origin_")[0] + "_export"
                elif "_destination_" in stage:
                    sn = stage.split("_destination_")[0] + "_import"
                else:
                    sn = stage.split("_inter_")[0] + "_inter"
                stage_sums[sn].append(stage)
            for k,v in stage_sums.items():
                flows_df[k] = flows_df[list(set(v))].sum(axis=1)
                flow_sums.append(k)

            import_columns = list(set([c for c in flow_sums if "_import" in c]))
            export_columns = list(set([c for c in flow_sums if "_export" in c]))
            inter_columns = list(set([c for c in flow_sums if "_inter" in c]))
            flows_df[f"{reference_mineral}_{flow_column}_export"] = flows_df[export_columns].sum(axis=1)
            flows_df[f"{reference_mineral}_{flow_column}_import"] = flows_df[import_columns].sum(axis=1)
            flows_df[f"{reference_mineral}_{flow_column}_inter"] = flows_df[inter_columns].sum(axis=1)
            flows_df[f"{reference_mineral}_{flow_column}"
                    ] = flows_df[export_columns + import_columns + inter_columns].sum(axis=1)

        flows_df = add_geometries_to_flows(flows_df,
                                merge_column="id",
                                modes=["rail","sea","road","mine","city"],
                                layer_type=path_type)
        if path_type == "edges":
            degree_df = flows_df[["from_id","to_id"]].stack().value_counts().rename_axis('id').reset_index(name='degree')
        elif path_type == "nodes" and len(degree_df.index) > 0:
            flows_df = pd.merge(flows_df,degree_df,how="left",on=["id"])
            # if year > 2022:
            #     flows_df[f"{reference_mineral}_{efficient_scale}"] = production_size
            # flows_df["min_production_size_global_tons"] = min_production_size_global

        flows_df = gpd.GeoDataFrame(flows_df,
                                geometry="geometry",
                                crs="EPSG:4326")
        if year == 2022:
            layer_name = f"{reference_mineral}_{percentile}"
        else:
            layer_name = f"{reference_mineral}_{percentile}_{efficient_scale}"
        # flows_df.to_file(os.path.join(results_folder,
        #                     f"{path_type}_flows_{year}_{country_case}_{constraint}.gpkg"),
        #                     layer=layer_name,driver="GPKG")
        flows_df.to_parquet(os.path.join(results_folder,
                            f"{path_type}_{results_gpq}"))


if __name__ == '__main__':
    CONFIG = load_config()
    try:
        if len(sys.argv) > 7:
            reference_mineral = str(sys.argv[1])
            year = int(sys.argv[2])
            percentile = str(sys.argv[3])
            efficient_scale = str(sys.argv[4])
            country_case = str(sys.argv[5])
            constraint = str(sys.argv[6])
            combination = str(sys.argv[7])
            distance_from_origin = float(sys.argv[8])
            environmental_buffer = float(sys.argv[9])
        else:
            reference_mineral = str(sys.argv[1])
            year = int(sys.argv[2])
            percentile = str(sys.argv[3])
            efficient_scale = str(sys.argv[4])
            country_case = str(sys.argv[5])
            constraint = str(sys.argv[6])
            combination = None
            distance_from_origin = 0.0
            environmental_buffer = 0.0
    except IndexError:
        print("Got arguments", sys.argv)
        exit()
    main(
            CONFIG,
            reference_mineral,
            year,
            percentile,
            efficient_scale,
            country_case,
            constraint,
            combination = combination,
            distance_from_origin=distance_from_origin,
            environmental_buffer=environmental_buffer
            )