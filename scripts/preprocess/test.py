#!/usr/bin/env python
# coding: utf-8
import sys
import os
import re
import json
import pandas as pd
import geopandas as gpd
import snkit
from shapely.geometry import LineString
from updated_utils import *
from tqdm import tqdm
tqdm.pandas()

def get_line_status(x):
    if "abandon" in x:
        return "abandoned"
    else:
        return "open"

def add_attributes(dataframe,columns_attributes):
    for column_name,attribute_value in columns_attributes.items():
        dataframe[column_name] = attribute_value

    return dataframe


def main(config):
    incoming_data_path = config['paths']['incoming_data']
    processed_data_path = config['paths']['data']
    output_data_path = config['paths']['results']

    ccg_totals = pd.read_csv(
                        os.path.join(
                            output_data_path,
                            "baci_trade_matrices",
                            "baci_ccg_country_metal_content_production_2022.csv"))
    print (ccg_totals)

    bgs_totals = pd.read_excel(
                        os.path.join(
                            processed_data_path,
                            "baci","BGS_SnP_comparison.xlsx"),
                        index_col=[0,1])
    bgs_totals = bgs_totals.reset_index()
    print (bgs_totals)
    bgs_totals.rename(columns={"level_0":"reference_mineral","level_1":"export_country_code"},inplace=True)
    bgs_totals["reference_mineral"] = bgs_totals["reference_mineral"].str.lower()

    ccg_totals = pd.merge(ccg_totals,bgs_totals,how="left",on=["reference_mineral","export_country_code"]).fillna(0)
    print (ccg_totals)
    ccg_totals.to_csv("baci_bgs_snp_comparison.csv",index=False)

    # baci = pd.read_csv(os.path.join(processed_data_path,"baci","baci_ccg_country_level_trade_2040.csv"))
    # baci = baci.groupby(["export_country_code","final_refined_stage"])["trade_quantity_tons"].sum().reset_index()
    # baci.to_csv("test.csv")

    # road_edges = gpd.read_parquet(os.path.join(
    #                         processed_data_path,
    #                         "infrastructure",
    #                         "africa_roads_edges.geoparquet"))
    # print (road_edges.columns.values.tolist())

    # road_nodes = gpd.read_parquet(os.path.join(
    #                         processed_data_path,
    #                         "infrastructure",
    #                         "africa_roads_nodes.geoparquet"))
    # # road_nodes.rename(columns={"road_id":"id","iso_a3":"iso3"},inplace=True)
    # # road_nodes.to_parquet(os.path.join(
    #                         # processed_data_path,
    #                         # "infrastructure",
    #                         # "africa_roads_nodes.geoparquet"))
    # print (road_nodes.columns.values.tolist())

    # extract_countires = ["ZMB","COD","ZWE","MOZ"]
    # country_nodes = road_nodes[road_nodes["iso_a3"].isin(extract_countires)]
    # print (country_nodes)

    # country_edges = road_edges[(road_edges["from_iso_a3"].isin(extract_countires)) | (road_edges["to_iso_a3"].isin(extract_countires))]
    # print (country_edges)

    # country_nodes.to_parquet(os.path.join(
    #                         processed_data_path,
    #                         "infrastructure",
    #                         "country_roads_nodes.geoparquet"),index=False)
    # country_edges.to_parquet(os.path.join(
    #                         processed_data_path,
    #                         "infrastructure",
    #                         "country_roads_edges.geoparquet"),index=False)

    # gdf = gpd.read_file(os.path.join(processed_data_path,"minerals","ccg_mines_est_production.gpkg"))
    # print (gdf.columns.values.tolist())
    # gdf = gdf.groupby(["country_code"])[['copper_unprocessed_ton','copper_processed_ton']].sum().reset_index()
    # print (gdf)
    # gdf.to_csv("test.csv",index=False)

    # country_outputs = []
    # for year in [2022,2030,2040]:
    #     for reference_mineral in ["copper","cobalt","manganese","lithium","graphite","nickel"]:
    #         if year == 2022:
    #             mines_df = gpd.read_file(os.path.join(output_data_path,
    #                                 "location_outputs",
    #                                f"mine_city_tons_{year}.gpkg"),
    #                             layer=reference_mineral)
    #             mines_df = mines_df.groupby(
    #                                         ["ISO_A3",
    #                                         "reference_mineral"]
    #                                         )[
    #                                     [f"{reference_mineral}_initial_tons",
    #                                     f"{reference_mineral}_final_tons"]].sum().reset_index()
    #             mines_df["year"] = year
    #             mines_df["percentile"] = 50
    #             country_outputs.append(mines_df)
    #         else:
    #             for percentile in [25,50,75]:
    #                 mines_df = gpd.read_file(os.path.join(output_data_path,
    #                                 "location_outputs",
    #                                f"mine_city_tons_{year}.gpkg"),
    #                             layer=f"{reference_mineral}_{percentile}")
    #                 mines_df = mines_df.groupby(
    #                                         ["ISO_A3",
    #                                         "reference_mineral"]
    #                                         )[
    #                                     [f"{reference_mineral}_initial_tons",
    #                                     f"{reference_mineral}_final_tons"]].sum().reset_index()
    #                 mines_df["year"] = year
    #                 mines_df["percentile"] = percentile
    #                 country_outputs.append(mines_df)

    # country_outputs = pd.concat(country_outputs,axis=0,ignore_index=True)
    # country_outputs.to_csv("country_totals_over_time.csv",index=False)

    # results_folder = os.path.join(output_data_path,"flow_mapping")
    # od_df = pd.read_csv(os.path.join(results_folder,
    #                         f"mining_city_node_level_ods_2021.csv"))
    # print (od_df)

    # path_df = pd.read_parquet(
    #                     os.path.join(results_folder,f"copper_flow_paths_2021.parquet"))
    # print (path_df)
    # path_df["edge_path"] = path_df["edge_path"].map(tuple)
    # path_df = path_df.drop_duplicates(subset=["origin_id","destination_id",
    #                         "export_country_code","import_country_code","edge_path"],keep="first")
    # print (path_df)
    # path_df[["origin_id","destination_id",
    #                         "export_country_code","import_country_code","mine_output_tons"]].to_csv("test0.csv")
    # path_df = path_df.groupby(["origin_id","destination_id",
    #                         "export_country_code","import_country_code"])["mine_output_tons"].sum().reset_index()
    # print (path_df)

    # pd.merge(od_df,path_df,how="left",on=["origin_id","destination_id","export_country_code","import_country_code"]).to_csv("test.csv")
    # results_folder = os.path.join(output_data_path,"flow_mapping")
    
    # global_df = pd.read_parquet(
    #                     os.path.join(results_folder,
    #                     "global_network_2021.parquet"))
    # print (global_df)
    
    # """change the format of Karla's conversion dataset
    # """
    # df = pd.read_csv("process_summary.csv")
    # print (df)

    # index_columns = ["Reference mineral","Activity name",
    #                     "Reference product","Unit","Type","Processes","Stage number"]
    # pivot_columns = ["Input/output?","Value","Exchanges"]

    # t_df = df[df["Type"] == "Materials"]
    # input_output = t_df[index_columns].drop_duplicates(subset=index_columns,keep="first")

    # inpt = t_df[t_df["Input/output?"] == "Input"] 
    # opt = t_df[t_df["Input/output?"] == "Output"] 

    # inpt.rename(columns={"Input/output?":"input","Value":"input_kg","Exchanges":"input_exchanges"},inplace=True)
    # opt.rename(columns={"Input/output?":"output","Value":"output_kg","Exchanges":"output_exchanges"},inplace=True)

    # input_output = pd.merge(input_output,inpt,how="left",on=index_columns)
    # input_output = pd.merge(input_output,opt,how="left",on=index_columns)
    # print (input_output)
    # input_output.to_csv("process_summary_materials.csv",index=False)
    
    # epsg_meters = 3395 # To convert geometries to measure distances in meters
    # distance_threshold = 1000 # Join point within 50cm of each other
    # Read a number of rail network files and convert them to networks first
    # rail_paths = os.path.join(incoming_data_path,"africa_corridor_developments")

    # road_edges = gpd.read_parquet(os.path.join(
    #                         processed_data_path,
    #                         "infrastructure",
    #                         "africa_roads_edges.geoparquet"))
    # print (road_edges.columns.values.tolist())

    # road_nodes = gpd.read_parquet(os.path.join(
    #                         processed_data_path,
    #                         "infrastructure",
    #                         "africa_roads_nodes.geoparquet"))
    # road_nodes.rename(columns={"road_id":"id","iso_a3":"iso3"},inplace=True)
    # road_nodes.to_parquet(os.path.join(
    #                         processed_data_path,
    #                         "infrastructure",
    #                         "africa_roads_nodes.geoparquet"))
    # print (road_nodes.columns.values.tolist())


    # condition = False
    # if condition is True:
    #     del_lines = ["SGR Phases 3 - 5","Morogoro - Makutupora SGR","Dar es Salaam - Morogoro SGR"]
    #     # Read the rail edges data for Africa
    #     rail_edges = json.load(open(os.path.join(incoming_data_path,
    #                             "africa_rail_network",
    #                             "network_data",
    #                             "africa_rail_network.geojson")))
    #     rail_edges = convert_json_geopandas(rail_edges)
    #     rail_edges = rail_edges.to_crs(epsg=4326)
    #     tza_lines = rail_edges[rail_edges["line"].isin(del_lines)]

    #     rail_nodes = json.load(open(os.path.join(incoming_data_path,
    #                             "africa_rail_network",
    #                             "network_data",
    #                             "africa_rail_stops.geojson")))
    #     rail_nodes = convert_json_geopandas(rail_nodes)
    #     rail_nodes = rail_nodes.to_crs(epsg=4326)

    #     tza_node_ids = list(set(tza_lines["source"].values.tolist() + tza_lines["target"].values.tolist()))
    #     tza_nodes = rail_nodes[rail_nodes["oid"].isin(tza_node_ids)]

    #     rail_edges[~rail_edges["line"].isin(del_lines)].to_file(os.path.join(incoming_data_path,
    #                             "africa_rail_network",
    #                             "network_data",
    #                             "africa_railways.gpkg"),layer="edges",driver="GPKG")

    #     rail_nodes[~rail_nodes["oid"].isin(tza_node_ids)].to_file(os.path.join(incoming_data_path,
    #                             "africa_rail_network",
    #                             "network_data",
    #                             "africa_railways.gpkg"),layer="nodes",driver="GPKG")

    #     tza_lines = tza_lines[tza_lines["line"] != "SGR Phases 3 - 5"]
    #     tza_node_ids = list(set(tza_lines["source"].values.tolist() + tza_lines["target"].values.tolist()))
    #     tza_nodes = tza_nodes[tza_nodes["oid"].isin(tza_node_ids)]

    #     tza_l = gpd.read_file(os.path.join(rail_paths,"tanzania_sgr_v2.gpkg"))
    #     tza_lines = pd.concat([tza_lines,tza_l],axis=0,ignore_index=True)
    #     tza_lines["gauge"] = 1435
    #     tza_lines["mode"] = "mixed"
    #     tza_lines["country"] = "Tanzania"
    #     tza_lines["oid"] = tza_lines.index.values.tolist()
    #     tza_lines.drop("objectid",axis=1,inplace=True)
    #     gpd.GeoDataFrame(tza_lines,
    #                     geometry="geometry",
    #                     crs="EPSG:4326").to_file(os.path.join(rail_paths,
    #                         "tanzania_sgr_lines.gpkg"),layer="lines",driver="GPKG")
    #     gpd.GeoDataFrame(tza_nodes,
    #                     geometry="geometry",
    #                     crs="EPSG:4326").to_file(os.path.join(rail_paths,
    #                         "tanzania_sgr_lines.gpkg"),layer="nodes",driver="GPKG")
    # if condition is True:
    #     del_lines = ["Lomé to Lomé Port",
    #                     "Lomé to Blitta",
    #                     "Diamond Cement Alfao",
    #                     "Dalavé-Adétikopé",
    #                     "Tabligbo-Dalavé",
    #                     "Route into Lomé station"]
    #     # Read the rail edges data for Africa
    #     rail_edges = gpd.read_file(os.path.join(incoming_data_path,
    #                             "africa_rail_network",
    #                             "network_data",
    #                             "africa_railways.gpkg"),layer="edges")
    #     togo_lines = rail_edges[rail_edges["line"].isin(del_lines)]

    #     rail_nodes = gpd.read_file(os.path.join(incoming_data_path,
    #                             "africa_rail_network",
    #                             "network_data",
    #                             "africa_railways.gpkg"),layer="nodes")

    #     togo_node_ids = list(set(togo_lines["source"].values.tolist() + togo_lines["target"].values.tolist()))
    #     togo_nodes = rail_nodes[rail_nodes["oid"].isin(togo_node_ids)]

    #     rail_edges[~rail_edges["line"].isin(del_lines)].to_file(os.path.join(incoming_data_path,
    #                             "africa_rail_network",
    #                             "network_data",
    #                             "africa_railways_v2.gpkg"),layer="edges",driver="GPKG")

    #     rail_nodes[~rail_nodes["oid"].isin(togo_node_ids)].to_file(os.path.join(incoming_data_path,
    #                             "africa_rail_network",
    #                             "network_data",
    #                             "africa_railways_v2.gpkg"),layer="nodes",driver="GPKG")


    #     togo_l = gpd.read_file(os.path.join(rail_paths,"lome_cinkasse_railway.gpkg"),layer="edges")
    #     togo_n = gpd.read_file(os.path.join(rail_paths,"lome_cinkasse_railway.gpkg"),layer="nodes")
        
    #     togo_lines = togo_lines[["country","line","status","mode","gauge","comment","geometry"]]
    #     togo_lines = pd.concat([togo_lines,togo_l],axis=0,ignore_index=True)
    #     gpd.GeoDataFrame(togo_lines,
    #                     geometry="geometry",
    #                     crs="EPSG:4326").to_file(os.path.join(rail_paths,
    #                         "togo_lines.gpkg"),layer="lines",driver="GPKG")

    #     togo_nodes = togo_nodes[["country","type","name","facility","status","comment","geometry"]]
    #     togo_nodes = pd.concat([togo_nodes,togo_n],axis=0,ignore_index=True)
    #     gpd.GeoDataFrame(togo_nodes,
    #                     geometry="geometry",
    #                     crs="EPSG:4326").to_file(os.path.join(rail_paths,
    #                         "togo_lines.gpkg"),layer="nodes",driver="GPKG")
    # condition = True
    # if condition is True:
    #     del_country = ["Guinea"]
    #     # Read the rail edges data for Africa
    #     rail_edges = gpd.read_file(os.path.join(incoming_data_path,
    #                             "africa_rail_network",
    #                             "network_data",
    #                             "africa_railways_v0.gpkg"),layer="edges")
    #     guinea_lines = rail_edges[rail_edges["country"].isin(del_country)]

    #     rail_nodes = gpd.read_file(os.path.join(incoming_data_path,
    #                             "africa_rail_network",
    #                             "network_data",
    #                             "africa_railways_v0.gpkg"),layer="nodes")

    #     guinea_node_ids = list(set(guinea_lines["source"].values.tolist() + guinea_lines["target"].values.tolist()))
    #     guinea_nodes = rail_nodes[rail_nodes["oid"].isin(guinea_node_ids)]

    #     rail_edges[~rail_edges["country"].isin(del_country)].to_file(os.path.join(incoming_data_path,
    #                             "africa_rail_network",
    #                             "network_data",
    #                             "africa_railways_v2.gpkg"),layer="edges",driver="GPKG")

    #     rail_nodes[~rail_nodes["oid"].isin(guinea_node_ids)].to_file(os.path.join(incoming_data_path,
    #                             "africa_rail_network",
    #                             "network_data",
    #                             "africa_railways_v2.gpkg"),layer="nodes",driver="GPKG")


    #     guinea_l = gpd.read_file(os.path.join(rail_paths,"conakry-kankan_railway.gpkg"),layer="edges")
    #     guinea_n = gpd.read_file(os.path.join(rail_paths,"conakry-kankan_railway.gpkg"),layer="nodes")
    #     guinea_l.drop(["edge_id","from_node","to_node","component"],axis=1,inplace=True)
    #     guinea_n.drop(["node_id","component"],axis=1,inplace=True)

        
    #     guinea_lines = pd.concat([guinea_lines,guinea_l],axis=0,ignore_index=True)
    #     guinea_nodes = pd.concat([guinea_nodes,guinea_n],axis=0,ignore_index=True)

    #     network = create_network_from_nodes_and_edges(
    #                                         guinea_nodes,
    #                                         guinea_lines,"",
    #                                         geometry_precision=False)
    #     edges, nodes = components(network.edges,network.nodes,"node_id")
    #     gpd.GeoDataFrame(edges,
    #                     geometry="geometry",
    #                     crs="EPSG:4326").to_file(os.path.join(rail_paths,
    #                         "guinea_lines.gpkg"),layer="edges",driver="GPKG")

    #     gpd.GeoDataFrame(nodes,
    #                     geometry="geometry",
    #                     crs="EPSG:4326").to_file(os.path.join(rail_paths,
    #                         "guinea_lines.gpkg"),layer="nodes",driver="GPKG")




if __name__ == '__main__':
    CONFIG = load_config()
    main(CONFIG)