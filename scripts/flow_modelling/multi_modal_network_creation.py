#!/usr/bin/env python
# coding: utf-8
import os
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString
from utils import *
from tqdm import tqdm
tqdm.pandas()

config = load_config()
incoming_data_path = config['paths']['incoming_data']
processed_data_path = config['paths']['data']

def create_edges_from_nearest_node_joins(from_df,to_df,
                    from_id_column,to_id_column,
                    from_iso_column,to_iso_column,
                    from_mode,to_mode,
                    distance_threshold=2000):
    from_df.rename(columns={from_id_column:"from_id",from_iso_column:"from_iso_a3"},inplace=True)
    to_df.rename(columns={to_id_column:"to_id",to_iso_column:"to_iso_a3"},inplace=True)
    from_df["from_infra"] = from_mode
    to_df["to_infra"] = to_mode

    from_to_df = ckdnearest(from_df[["from_id","from_iso_a3","from_infra","geometry"]],
                            to_df[["to_id","to_iso_a3","to_infra","geometry"]])
    from_to_df["link_type"] = f"{from_mode}-{to_mode}"
    from_to_df = from_to_df[from_to_df["dist"] <= distance_threshold]

    if len(from_to_df.index) > 0:
        from_to_df.rename(columns = {"geometry":"from_geometry","dist":"length_m"},inplace=True)
        to_df.rename(columns = {"geometry":"to_geometry"},inplace=True)
        from_to_df = pd.merge(from_to_df,to_df[["to_id","to_geometry"]],how="left",on=["to_id"])
        from_to_df["geometry"] = from_to_df.progress_apply(
                                    lambda x:LineString([x.from_geometry,x.to_geometry]),
                                    axis=1)
        from_to_df.drop(["from_geometry","to_geometry"],axis=1,inplace=True)
    
    return from_to_df

def get_mode_dataframe(mode,rail_status=["open"],rail_to_road_connection=False):
    if mode == "sea":
        nodes =  gpd.read_file(os.path.join(
                                processed_data_path,
                                "infrastructure",
                                "africa_maritime_network.gpkg"
                                    ), layer="nodes"
                                )  
        nodes = nodes[nodes["infra"] == "port"]
    elif mode == "IWW":
        nodes = gpd.read_file(os.path.join(
                                processed_data_path,
                                "infrastructure",
                                "africa_iww_network.gpkg"
                                    ), layer="nodes"
                                ) 
        nodes = nodes[nodes["infra"] == "IWW port"]
    elif mode == "rail":
        rail_edges = gpd.read_file(os.path.join(
                                processed_data_path,
                                "infrastructure",
                                "africa_railways_network.gpkg"
                                    ), layer="edges"
                        )
        rail_edges = rail_edges[rail_edges["status"].isin(rail_status)]
        rail_node_ids = list(set(rail_edges["from_id"].values.tolist() + rail_edges["to_id"].values.tolist()))
        nodes = gpd.read_file(os.path.join(
                                processed_data_path,
                                "infrastructure",
                                "africa_railways_network.gpkg"
                                    ), layer="nodes"
                        )
        nodes = nodes[(nodes["id"].isin(rail_node_ids)) & (nodes["infra"].isin(['stop','station']))]

        degree_df = rail_edges[["from_id","to_id"]].stack().value_counts().rename_axis('id').reset_index(name='degree')
        nodes = pd.merge(nodes,degree_df,how="left",on=["id"])

        if rail_to_road_connection is True:
            freight_facility_types = ["port","port (dry)",
                                        "port (inland)",
                                        "port (river)",
                                        "road-rail transfer",
                                        "container terminal",
                                        "freight terminal",
                                        "freight marshalling yard",
                                        "mining",
                                        "refinery"]
            nodes = nodes[
                                            (
                                                nodes["facility"].isin(freight_facility_types)
                                            ) | (
                                                nodes["degree"] == 1
                                            )
                                        ]
    elif mode == "road": 
        nodes = gpd.read_parquet(os.path.join(
                                processed_data_path,
                                "infrastructure",
                                "africa_roads_nodes.geoparquet"))
        nodes.rename(columns={"iso_a3":"iso3"},inplace=True)

    return nodes

def main():
    epsg_meters = 3395 # To convert geometries to measure distances in meters
    from_modes = ["sea","sea","IWW","IWW","rail"]
    to_modes = ["rail","road","rail","road","road"]

    multi_df = []
    for idx,(f_m,t_m) in enumerate(zip(from_modes,to_modes)):
        if f_m == "rail" and t_m == "road":
            f_df = get_mode_dataframe(f_m,rail_to_road_connection=True)
        else:
            f_df = get_mode_dataframe(f_m)

        t_df = get_mode_dataframe(t_m)
        f_t_df = create_edges_from_nearest_node_joins(
                            f_df.to_crs(epsg=epsg_meters),
                            t_df.to_crs(epsg=epsg_meters),
                            "id","id",
                            "iso3","iso3",
                            f_m,t_m)
        if len(f_t_df) > 0:
            multi_df.append(f_t_df)
            c_t_df = f_t_df[["from_id","to_id","from_infra",
                        "to_infra","from_iso_a3","to_iso_a3",
                        "link_type","length_m","geometry"]].copy()
            c_t_df.columns = ["to_id","from_id",
                        "to_infra","from_infra",
                        "to_iso_a3",
                        "from_iso_a3",
                        "link_type","length_m","geometry"]
            multi_df.append(c_t_df)

    multi_df = gpd.GeoDataFrame(
                    pd.concat(multi_df,axis=0,ignore_index=True),
                    geometry="geometry",crs=f"EPSG:{epsg_meters}")
    multi_df = multi_df.to_crs(epsg=4326)
    multi_df["id"] = multi_df.index.values.tolist()
    multi_df["id"] = multi_df.progress_apply(lambda x:f"intermodale_{x.id}",axis=1)
    multi_df.to_file(os.path.join(
                            processed_data_path,
                            "infrastructure",
                            "africa_multimodal.gpkg"
                                ), 
                            layer="edges",
                            driver="GPKG"
                            )

if __name__ == '__main__':
    main()