#!/usr/bin/env python
# coding: utf-8
"""This code finds the route between mines (in Africa) and all ports (outside Africa) 
"""
import sys
import os
import pandas as pd
import fiona
pd.options.mode.copy_on_write = True
import geopandas as gpd
from utils import *
from tqdm import tqdm
tqdm.pandas()
    
def main(config,year,percentile,efficient_scale,country_case,constraint):
    incoming_data_path = config['paths']['incoming_data']
    processed_data_path = config['paths']['data']
    output_data_path = config['paths']['results']

    input_folder = os.path.join(
                        output_data_path,
                        f"flow_optimisation_{country_case}_{constraint}"
                        )
    flows_folder = os.path.join(
                        output_data_path,
                        f"flow_optimisation_{country_case}_{constraint}",
                        "processed_flows")

    results_folder = os.path.join(output_data_path,"optimised_processing_locations")
    if os.path.exists(results_folder) == False:
        os.mkdir(results_folder)

    """Step 1: Get the input datasets
    """
    reference_minerals = ["graphite","lithium","cobalt","manganese","nickel","copper"]
    id_column = "id"
    
    """Step 1: get all the relevant nodes and find their distances 
                to grid and bio-diversity layers 
    """
    node_location_path = os.path.join(
                                    output_data_path,
                                    "location_filters",
                                    "nodes_with_location_identifiers.geoparquet"
                                    )
    nodes = gpd.read_parquet(node_location_path)
    include_columns = nodes.columns.values.tolist()
    all_flows = []
    add_columns = []
    for reference_mineral in reference_minerals:
        # Find year locations
        if year == 2022:
            layer_name = f"{reference_mineral}_{percentile}"
        else:
            layer_name = f"{reference_mineral}_{percentile}_{efficient_scale}"
        
        # gpkg_file = os.path.join(
        #                     input_folder,
        #                     f"processing_nodes_flows_{year}_{country_case}.gpkg"
        #                     )
        # layers = fiona.listlayers(gpkg_file)
        # if layer_name in layers:
            # flows_df = gpd.read_file(os.path.join(input_folder,
            #                     f"processing_nodes_flows_{year}_{country_case}.gpkg"),
            #                     layer=layer_name)
        pq_file = os.path.join(flows_folder,
                            f"{layer_name}_{year}_{country_case}.geoparquet")
        if os.path.exists(pq_file):
            flows_df = gpd.read_parquet(pq_file)

            origin_cols = [c for c in flows_df.columns.values.tolist() if "_origin_" in c]
            r_cols = list(set([o.split("_origin_")[0] for o in origin_cols]))
            replace_r_cols = [f"{r}_in_{country_case}" for r in r_cols]
            add_columns += replace_r_cols
            flows_df.rename(columns=
                    dict(
                            [
                                    (c,r) for i,(c,r) in enumerate(zip(r_cols,replace_r_cols))
                            ]
                        ),inplace=True
                    )

            all_flows.append(flows_df[[id_column] + replace_r_cols])

    add_columns = list(set(add_columns))
    all_flows = pd.concat(all_flows,axis=0,ignore_index=True).fillna(0)
    all_flows = all_flows.groupby([id_column]).agg(dict([(c,"sum") for c in add_columns])).reset_index()
    all_flows = pd.merge(all_flows,nodes,how="left",on=[id_column])

    all_flows = gpd.GeoDataFrame(
                    all_flows[include_columns + add_columns],
                    geometry="geometry",
                    crs="EPSG:4326")
    all_flows = all_flows.drop_duplicates(subset=["id"],keep="first")

    if year == 2022:
        layer_name = f"{year}_{percentile}"
    else:
        layer_name = f"{year}_{percentile}_{efficient_scale}"
    all_flows.to_file(os.path.join(results_folder,
                        f"node_locations_for_energy_conversion_{country_case}_{constraint}.gpkg"),
                        layer=layer_name,driver="GPKG")


if __name__ == '__main__':
    CONFIG = load_config()
    try:
        year = int(sys.argv[1])
        percentile = str(sys.argv[2])
        efficient_scale = str(sys.argv[3])
        country_case = str(sys.argv[4])
        constraint = str(sys.argv[5])
    except IndexError:
        print("Got arguments", sys.argv)
        exit()
    main(CONFIG,year,percentile,efficient_scale,country_case,constraint)