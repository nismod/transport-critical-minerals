#!/usr/bin/env python
# coding: utf-8
import sys
import os
import re
import json
import pandas as pd
import geopandas as gpd
from utils import *
from tqdm import tqdm
tqdm.pandas()

def main(config):
    incoming_data_path = config['paths']['incoming_data']
    processed_data_path = config['paths']['data']
    
    # Read the rail edges data for Africa
    rail_edges = json.load(open(os.path.join(incoming_data_path,
                            "africa_rail_network",
                            "network_data",
                            "africa_rail_network.geojson")))
    rail_edges = convert_json_geopandas(rail_edges)
    rail_edges = rail_edges.to_crs(epsg=3587)

    line_name = list(set(rail_edges["line"].values.tolist()))
    pd.DataFrame(line_name,columns=["line"]).to_csv(os.path.join(
                                    incoming_data_path,
                                    "africa_corridor_developments",
                                    "africa_network_lines.csv"))

    # rail_developments = gpd.read_file(os.path.join(incoming_data_path,
    #                                 "africa_corridor_developments",
    #                                 "AfricanDevelopmentCorridorDatabase2022.gpkg"),layer="line")
    # rail_developments = rail_developments[
    #                             rail_developments[
    #                                 "Infrastructure_development_type"
    #                                 ].isin(
    #                                     ["Freight railway",
    #                                     "Passenger and freight railway"]
    #                                     )
    #                             ]
    # rail_developments = rail_developments.to_crs(epsg=3587)
    # rail_developments["geometry"] = rail_developments.geometry.buffer(10)

    # matches = gpd.sjoin(rail_developments,rail_edges,how='inner', predicate='intersects')
    # print (matches)

    # print (len(list(set(matches["Project_name"].values.tolist()))))

    # oids = matches["oid"].values.tolist()
    # print (len(oids))
    # print (len(list(set(oids))))


if __name__ == '__main__':
    CONFIG = load_config()
    main(CONFIG)