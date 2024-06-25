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
    # Read the finalised version of the BACI trade data
    ccg_countries = pd.read_csv(
                        os.path.join(processed_data_path,
                                "baci","ccg_country_codes.csv"))
    ccg_countries = ccg_countries[ccg_countries["ccg_country"] == 1]["iso_3digit_alpha"].values.tolist()
    # Read the data on the highest stages at the mines
    # This will help identify which stage goes to mine and which outside
    mine_city_stages = pd.read_csv(os.path.join(processed_data_path,"baci","mine_city_stages.csv"))
    mine_city_stages = mine_city_stages[
                            mine_city_stages["year"] == year
                            ][["reference_mineral","mine_final_refined_stage"]]
    all_flows = []
    for reference_mineral in reference_minerals:
        include_columns = ["id","iso3","mode","geometry"]
        # Find year locations
        if year == 2022:
            layer_name = f"{reference_mineral}"
        else:
            layer_name = f"{reference_mineral}_{percentile}_{efficient_scale}"

        flows_df = gpd.read_file(os.path.join(results_folder,
                                f"processing_nodes_flows_{year}.gpkg"),
                                layer=layer_name)
        flows_df = flows_df[flows_df["iso3"].isin(ccg_countries)]
        flows_df[f"{reference_mineral}_mine_highest_stage"
        ] = mine_city_stages[mine_city_stages["reference_mineral"] == reference_mineral
        ]["mine_final_refined_stage"].values[0]
        include_columns += [f"{reference_mineral}_mine_highest_stage"]
        origin_cols = [c for c in flows_df.columns.values.tolist() if "_origin_" in c]
        origins = list(set([o.split("_origin_")[1] for o in origin_cols]))
        r_cols = list(set([o.split("_origin_")[0] for o in origin_cols]))
        # print (flows_df.columns.values.tolist())
        # print (r_cols)
        stages = list(set([float(r.split("_")[-1]) for r in r_cols]))
        if year == 2022:
            flows_df[f"{reference_mineral}_conversion_location_in_country"
            ] = np.where(
                    flows_df["infra"].isin(["mine","city"]),
                    1,
                    0
                )
            flows_df[f"{reference_mineral}_conversion_location_in_region"
            ] = 0
        else:
            all_columns = [c for c in flows_df.columns.values.tolist()]
            flows_df["conversion_locations"
            ] = flows_df.progress_apply(
                lambda x:find_processing_locations(x,all_columns,reference_mineral,
                                                efficient_scale,origins),axis=1)
            flows_df[
            [
                f"{reference_mineral}_conversion_location_in_country",
                f"{reference_mineral}_conversion_location_in_region"
            ]] = flows_df["conversion_locations"].apply(pd.Series)
            flows_df.drop("conversion_locations",axis=1,inplace=True)

        include_columns += [f"{reference_mineral}_conversion_location_in_country",
                            f"{reference_mineral}_conversion_location_in_region"]
        flows_df = flows_df[
                    (
                        flows_df[f"{reference_mineral}_conversion_location_in_country"] == 1
                    ) | (
                        flows_df[f"{reference_mineral}_conversion_location_in_region"] == 1
                    )]
        origin_isos = list(set(flows_df["iso3"].values.tolist()))
        for origin in origin_isos:
            f_df = flows_df[flows_df["iso3"] == origin]
            if len(f_df.index) > 0:
                c_cols = [c for c in f_df.columns.values.tolist() if f"_{origin}" in c]
                replace_c_cols = [c.replace(f"_{origin}","_in_country") for c in c_cols]
                replace_r_cols =[f"{r}_in_region" for r in r_cols]
                f_df.rename(columns=
                        dict(
                                [
                                        (c,r) for i,(c,r) in enumerate(zip(c_cols,replace_c_cols))
                                ]
                            ),inplace=True
                        )
                f_df.rename(columns=
                        dict(
                                [
                                        (c,r) for i,(c,r) in enumerate(zip(r_cols,replace_r_cols))
                                ]
                            ),inplace=True
                        )

                all_flows.append(f_df[include_columns + replace_c_cols + replace_r_cols])

    all_flows = pd.concat(all_flows,axis=0,ignore_index=True).fillna(0)
    all_geoms = all_flows[["id","geometry"]].drop_duplicates(subset=["id"],keep="first")
    all_flows.drop("geometry",axis=1,inplace=True)
    # include_columns = [c for c in include_columns if c != "geometry"]
    add_columns = [c for c in all_flows.columns.values.tolist() if c not in ["id","iso3","mode"]]

    all_flows = all_flows.groupby(["id","iso3","mode"]).agg(dict([(c,"sum") for c in add_columns])).reset_index()
    grid_network = get_electricity_grid_lines()
    flows_grid_intersects = gpd.sjoin_nearest(
                                all_geoms[["id","geometry"]].to_crs(epsg=epsg_meters),
                                grid_network[["grid_id","geometry"]].to_crs(epsg=epsg_meters),
                                how="left",distance_col="distance_to_grid_meters").reset_index()
    flows_grid_intersects = flows_grid_intersects.to_crs(epsg=4326)
    all_flows = pd.merge(all_flows,flows_grid_intersects[["id","distance_to_grid_meters","geometry"]],how="left",on=["id"])
    all_flows["on_grid"
    ] = np.where(all_flows["distance_to_grid_meters"] <= 2000.0,1,0)

    all_flows = gpd.GeoDataFrame(
                    all_flows,
                    geometry="geometry",
                    crs="EPSG:4326")
    all_flows = all_flows.drop_duplicates(subset=["id"],keep="first")

    if year == 2022:
        layer_name = f"{year}"
    else:
        layer_name = f"{year}_{percentile}_{efficient_scale}"
    all_flows.to_file(os.path.join(results_folder,
                        f"node_locations_for_energy_conversion.gpkg"),
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