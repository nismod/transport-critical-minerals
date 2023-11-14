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
from tqdm import tqdm
tqdm.pandas()



def main(config):
    incoming_data_path = config['paths']['incoming_data']
    processed_data_path = config['paths']['data']
    
    epsg_meters = 3395 # To convert geometries to measure distances in meters
    mine_id_column = "mine_id"
    road_id_column = "id"
    road_type_column = "tag_highway"
    main_road_types = ["trunk","motorway","primary","secondary"]
    # Read the mining data from the global extract
    mines = gpd.read_file(os.path.join(incoming_data_path,
                            "mines_spatial_locations",
                            "all_mines_adm.gpkg"))
    mines = mines[mines["continent"] == "Africa"]
    mines = mines.to_crs(epsg=epsg_meters)
    mine_countries = list(set(mines["shapeGroup_primary_admin0"].values.tolist()))
    # Read the road edges data for Africa
    road_edges = gpd.read_parquet(os.path.join(incoming_data_path,
                            "africa_roads",
                            "edges_with_topology.geoparquet"))
    road_edges = road_edges.to_crs(epsg=epsg_meters) 
    main_roads = road_edges[
                        road_edges[road_type_column].isin(main_road_types)
                        ][road_id_column].values.tolist() 

    # We assume all the mines intersect the networks of the countries they are within
    # Seems like only a few mines are border mines, so our assumption is fine
    nearest_roads = []
    for m_c in mine_countries:
        country_roads = road_edges[(
                    road_edges["from_iso_a3"] == m_c
                    ) & (road_edges["to_iso_a3"] == m_c)]
        country_mines = mines[mines["shapeGroup_primary_admin0"] == m_c]
        # intersect mines with roads first to find which mines have roads on them
        mine_intersects = gpd.sjoin_nearest(country_mines[[mine_id_column,"geometry"]],
                            country_roads[[road_id_column,road_type_column,"geometry"]],
                            how="left").reset_index()
        # get the intersected roads which are not the main roads
        intersected_roads_df = mine_intersects[~mine_intersects[road_type_column].isin(main_road_types)]
        selected_edges = list(set(intersected_roads_df[road_id_column].values.tolist()))
        mining_roads = country_roads[country_roads[road_id_column].isin(selected_edges)]
        targets = list(set(mining_roads.from_id.values.tolist() + mining_roads.to_id.values.tolist()))
        del intersected_roads_df, mining_roads

        # We just need access to one road in the main roud network, since the rest are connected
        source = country_roads[country_roads[road_type_column].isin(main_road_types)].from_id.values[0] 
        
        graph = create_igraph_from_dataframe(
                country_roads[["from_id","to_id",road_id_column,"length_m"]])
        n_r, _ = network_od_path_estimations(graph,source, targets,"length_m",road_id_column)
        connected_roads = list(set([item for sublist in n_r + [selected_edges] for item in sublist]))
        
        # nearest_roads.append(country_roads[country_roads[road_id_column].isin(connected_roads)])
        nearest_roads += connected_roads

        print (f"* Done with country - {m_c}")

    # print (nearest_roads)
    nearest_roads = list(set(connected_roads + main_roads))
    nearest_roads = road_edges[
                        road_edges[road_id_column].isin(nearest_roads)
                        ]
    # gpd.GeoDataFrame(
    #                 pd.concat(nearest_roads,axis=0,ignore_index=True),
    #                 geometry="geometry",
    #                 crs=f"EPSG:{epsg_meters}")
    nearest_roads = nearest_roads.to_crs(epsg=4326)
    nearest_roads.to_file(os.path.join(incoming_data_path,
                            "africa_roads",
                            "africa_main_roads.gpkg"),
                        layer="edges",driver="GPKG")





if __name__ == '__main__':
    CONFIG = load_config()
    main(CONFIG)