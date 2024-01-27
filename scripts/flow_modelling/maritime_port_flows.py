#!/usr/bin/env python
# coding: utf-8
import sys
import os
import re
import json
import pandas as pd
import igraph as ig
import geopandas as gpd
from utils import *
from transport_cost_assignment import *
from tqdm import tqdm
tqdm.pandas()



def main(config):
    incoming_data_path = config['paths']['incoming_data']
    processed_data_path = config['paths']['data']
    results_data_path = config['paths']['results']
    
    select_isos = ["MOZ","ZMB","TZA"]
    road_df = gpd.read_parquet(os.path.join(
                            processed_data_path,
                            "infrastructure",
                            "africa_roads_edges.geoparquet"))
    road_df.rename(columns={"road_id":"id"},inplace=True)
    road_df = transport_cost_assignment_function(road_df,"road")
    print (road_df)
    df = road_df[(road_df["from_iso_a3"].isin(select_isos)) | (road_df["to_iso_a3"].isin(select_isos))]
    print (df)
    gpd.GeoDataFrame(df,geometry="geometry",crs="EPSG:4326").to_file(os.path.join(
                            processed_data_path,
                            "infrastructure",
                            "select_roads_edges.gpkg"),layer="edges",driver="GPKG")


    # copper_routing = pd.read_parquet(
    #                         os.path.join(
    #                             results_data_path,
    #                             "flow_mapping",
    #                             "copper_unrefined.parquet"))
    # print (copper_routing[["edge_path","full_edge_path"]])
    # print (copper_routing[["node_path","full_node_path"]])
    # port_to_port_drybulk = pd.read_parquet(
    #                     os.path.join(
    #                         processed_data_path,
    #                         "shipping_network",
    #                         "maritime_base_network_dry_bulk.parquet"))
    # drybulk_country_pairs = list(
    #                             set(
    #                                 zip(
    #                                     port_to_port_drybulk["from_iso3"].values.tolist(),
    #                                     port_to_port_drybulk["to_iso3"].values.tolist()
    #                                     )
    #                                 )
    #                             )
    # port_to_port_general_cargo = pd.read_parquet(
    #                     os.path.join(
    #                         processed_data_path,
    #                         "shipping_network",
    #                         "maritime_base_network_general_cargo.parquet"))
    # general_cargo_country_pairs = list(
    #                             set(
    #                                 zip(
    #                                     port_to_port_general_cargo["from_iso3"].values.tolist(),
    #                                     port_to_port_general_cargo["to_iso3"].values.tolist()
    #                                     )
    #                                 )
    #                             )
    # extra_pairs = [p for p in general_cargo_country_pairs if p not in drybulk_country_pairs]
    # print (len(extra_pairs))

    # extra_origins = [p for p in list(set(port_to_port_general_cargo["from_iso3"].values.tolist())) if p not in list(set(port_to_port_drybulk["from_iso3"].values.tolist()))]
    # print (extra_origins)

    # extra_destinations = [p for p in list(set(port_to_port_general_cargo["to_iso3"].values.tolist())) if p not in list(set(port_to_port_drybulk["to_iso3"].values.tolist()))]
    # print (extra_destinations)

    files = [
            "maritime_base_network_dry_bulk.parquet",
            # "maritime_base_network_general_cargo.parquet"
            ]

    for f in files:
        port_to_port = pd.read_parquet(
                        os.path.join(
                            processed_data_path,
                            "shipping_network",
                            f))
        port_to_port.rename(columns={"from_iso3":"from_iso_a3","to_iso3":"to_iso_a3"},inplace=True)
        port_to_port = transport_cost_assignment_function(port_to_port,"sea")
        # port_to_port.to_csv(f"{f.replace('.parquet','.csv')}",index=False)
        port_to_port["id"] = port_to_port.index.values.tolist()
        G = ig.Graph.TupleList(port_to_port[["from_id","to_id","id","gcost_usd_tons"]].itertuples(index=False), 
                            edge_attrs=["id","gcost_usd_tons"],
                            directed=True)
        _,c =  network_od_path_estimations(G,"port278_out", ["port1195_in"],"gcost_usd_tons","id")
        print ("Dar es Salaam to China costs in USD per ton:",c[0])
        _,c =  network_od_path_estimations(G,"port137_out", ["port1195_in"],"gcost_usd_tons","id")
        print ("Beira to China costs in USD per ton:",c[0])
    
    # edges = gpd.read_file(os.path.join(processed_data_path,
    #                         "infrastructure",
    #                         "global_maritime_network.gpkg"),layer="edges")
    # edges.drop("geometry",axis=1,inplace=True)
    
    # nodes = gpd.read_file(os.path.join(processed_data_path,
    #                         "infrastructure",
    #                         "global_maritime_network.gpkg"),layer="nodes")
    # nodes.rename(columns={"id":"node_id"},inplace=True)

    # edges = pd.merge(edges,nodes[["node_id","iso3"]],
    #                 how="left",left_on=["from_id"],
    #                 right_on=["node_id"]).fillna("maritime")
    # edges.rename(columns={"iso3":"from_iso_a3"},inplace=True)
    # edges.drop("node_id",axis=1,inplace=True)
    # edges = pd.merge(edges,nodes[["node_id","iso3"]],
    #                 how="left",left_on=["to_id"],
    #                 right_on=["node_id"]).fillna("maritime")
    # edges.rename(columns={"iso3":"to_iso_a3"},inplace=True)
    # edges.drop("node_id",axis=1,inplace=True)
    # edges = transport_cost_assignment_function(edges,"sea")
    # edges["time_h"] = edges["distance"]/edges["drybulk_max_speed_kmh"]
    # print (edges)
    # edges.to_csv("test1.csv",index=False)

    # G = ig.Graph.TupleList(edges.itertuples(index=False), 
    #                         edge_attrs=list(edges.columns)[2:],
    #                         directed=True)
    # # print (G)

    # port_to_port = port_to_port[["from_port","to_port",
    #                     "from_id","to_id","from_infra",
    #                     "to_infra","mode",
    #                     "from_iso3","to_iso3"]]
    # df = network_od_paths_assembly_multiattributes(port_to_port.head(10), G,
    #                             "distance","id",
    #                             "from_port","to_port",
    #                             attribute_list=["time_h",
    #                                         "drybulk_cost_US_tonnes",
    #                                         "sea_cost_tonne_h_shipper"],store_edge_path=False)
    # df["shipper_cost"] = df["time_h"]*df["sea_cost_tonne_h_shipper"]
    # print (df)
    # for row in port_to_port.head(10).itertuples():
    #     origin = row.from_port
    #     destination = row.to_port
    #     _,c = network_od_path_estimations(G,origin,destination,"time_h","id")
    #     network_od_paths_assembly_multiattributes(points_dataframe, graph,
    #                             cost_criteria,path_id_column,
    #                             origin_id_column,destination_id_column,
    #                             attribute_list=None,store_edge_path=True)
    #     print (origin,destination,c)







if __name__ == '__main__':
    CONFIG = load_config()
    main(CONFIG)