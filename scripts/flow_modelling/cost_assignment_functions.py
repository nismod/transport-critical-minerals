#!/usr/bin/env python
# coding: utf-8
import sys
import os
import json
import pandas as pd
import geopandas as gpd
import itertools
from utils import *
from tqdm import tqdm
tqdm.pandas()
epsg_meters = 3395
config = load_config()
incoming_data_path = config['paths']['incoming_data']
processed_data_path = config['paths']['data']
output_data_path = config['paths']['results']

def assign_road_speeds(x):
    if float(x.tag_maxspeed) > 0:
        return x['tag_maxspeed'],x['tag_maxspeed'] 
    # elif x.tag_highway in ('motorway','trunk','primary'):
    elif x.tag_highway in ('motorway','trunk'):
        return x["Highway_min"],x["Highway_max"]
    elif x.paved == "road_paved":
        return max(x["Urban_min"],x["Rural_min"]),max(x["Urban_max"],x["Rural_max"])
    else:
        return min(x["Urban_min"],x["Rural_min"]),min(x["Urban_max"],x["Rural_max"])


def transport_cost_assignment_function_destination_based(network_edges,transport_mode,destination_iso):    
    costs_df = pd.read_csv(
                        os.path.join(
                            processed_data_path,
                            "transport_costs",
                            "country_transport_information.csv"))
    costs_df = costs_df[costs_df["iso3"] == destination_iso]
    """Tariffs and cost of time
    """
    if len(costs_df.index) > 0:
        if transport_mode in ("road","rail","IWW"):
            network_edges = pd.merge(network_edges,
                                costs_df[["iso3",
                                f"{transport_mode}_cost_tonnes_km",
                                f"{transport_mode}_cost_tonne_h_shipper"]],
                                how="left",left_on=["to_iso_a3"],right_on=["iso3"])
            network_edges[f"{transport_mode}_cost_tonnes_km"] = network_edges[f"{transport_mode}_cost_tonnes_km"].fillna(0)
        elif transport_mode == "air":
            network_edges = pd.merge(network_edges,
                                costs_df[["iso3",
                                f"{transport_mode}_cost_US_tonnes",
                                f"{transport_mode}_cost_tonne_h_shipper"]],
                                how="left",left_on=["to_iso_a3"],right_on=["iso3"])
            network_edges[f"{transport_mode}_cost_US_tonnes"] = network_edges[f"{transport_mode}_cost_US_tonnes"].fillna(0)
        elif transport_mode == "sea":
            network_edges = pd.merge(network_edges,
                                costs_df[["iso3",
                                f"{transport_mode}_cost_tonne_h_shipper"]],
                                how="left",left_on=["to_iso_a3"],right_on=["iso3"])
        elif transport_mode == "intermodal":
            network_edges = pd.merge(network_edges,
                                costs_df[["iso3",
                                "road_cost_tonne_h_shipper",
                                "rail_cost_tonne_h_shipper",
                                "sea_cost_tonne_h_shipper",
                                "IWW_cost_tonne_h_shipper"]],
                                how="left",left_on=["to_iso_a3"],right_on=["iso3"])
            network_edges[
                "intermodal_cost_tonne_h_shipper"
                ] = network_edges.progress_apply(lambda x:x[f"{x.to_infra}_cost_tonne_h_shipper"],axis=1)
            network_edges.drop(["road_cost_tonne_h_shipper",
                                "rail_cost_tonne_h_shipper",
                                "sea_cost_tonne_h_shipper",
                                "IWW_cost_tonne_h_shipper"],axis=1,inplace=True)   
        network_edges[f"{transport_mode}_cost_tonne_h_shipper"] = network_edges[f"{transport_mode}_cost_tonne_h_shipper"].fillna(0)
        network_edges.drop("iso3",axis=1,inplace=True)
    else:
        network_edges[f"{transport_mode}_cost_tonne_h_shipper"] = 1.0
        network_edges["intermodal_cost_tonne_h_shipper"] = 1.0
        network_edges[f"{transport_mode}_cost_US_tonnes"] = 1.0
        network_edges[f"{transport_mode}_cost_tonnes_km"] = 1.0


    if transport_mode in ("road","rail"):
        inter_country_costs_df = pd.read_csv(
                                    os.path.join(
                                            processed_data_path,
                                            "transport_costs",
                                            "OD_trade_information.csv"))
        """Border dwell times
        """
        network_edges = pd.merge(network_edges,
                            inter_country_costs_df[[
                            "from_iso3","to_iso3",
                            "border_dwell","border_USD_t"]],
                            how="left",
                            left_on=["from_iso_a3","to_iso_a3"],
                            right_on=["from_iso3","to_iso3"])
        network_edges["border_dwell"] = 24.0*network_edges["border_dwell"].fillna(0)
        network_edges["border_USD_t"] = network_edges["border_USD_t"].fillna(0)
        network_edges.drop(["from_iso3","to_iso3"],axis=1,inplace=True)
        del inter_country_costs_df

    if transport_mode == "intermodal":
        intermodal_costs_df = pd.read_excel(
                                os.path.join(
                                processed_data_path,
                                "transport_costs",
                                "intermodal.xlsx"),
                            sheet_name="Sheet1")
        network_edges = pd.merge(network_edges,
                            intermodal_costs_df,
                        how="left",on=["from_infra","to_infra"])


    if transport_mode == "road":
        speeds_df = pd.read_excel(
                        os.path.join(
                            processed_data_path,
                            "transport_costs",
                            "speed_tables.xlsx"),
                        sheet_name="road")
        network_edges = pd.merge(network_edges,
                            speeds_df,
                            how="left",
                            left_on=["from_iso_a3"],
                            right_on=["ISO_A3"]
                        )
        # infer a likely min and max road speed
        network_edges["min_max_speed"] = network_edges.apply(
            lambda x: assign_road_speeds(x), axis=1
        )

        # assign_road_speeds returned two values, unpack these into two columns
        network_edges[["min_speed_kmh", "max_speed_kmh"]] = network_edges[
                    "min_max_speed"
                    ].apply(pd.Series)

        # drop the intermediate columns
        network_edges.drop(
                    ["min_max_speed"] + speeds_df.columns.values.tolist(),
                    axis=1,
                    inplace=True,
                )
        del speeds_df
    elif transport_mode == "rail":
        speeds_df = pd.read_excel(
                        os.path.join(
                            processed_data_path,
                            "transport_costs",
                            "speed_tables.xlsx"),
                        sheet_name="rail")
        speeds_df["gauge"] = speeds_df["gauge"].astype(str)
        network_edges["gauge"] = network_edges["gauge"].astype(str)
        network_edges = pd.merge(network_edges,
                            speeds_df,
                            how="left",
                            on=["gauge"]
                        )
        del speeds_df
    elif transport_mode == "IWW":
        speeds_df = pd.read_excel(
                        os.path.join(
                            processed_data_path,
                            "transport_costs",
                            "speed_tables.xlsx"),
                        sheet_name="iww")
        # print (speeds_df)
        network_edges["min_speed_kmh"] = speeds_df["min_speed_kmh"].values[0]
        network_edges["max_speed_kmh"] = speeds_df["max_speed_kmh"].values[0]
        del speeds_df
    
    network_edges["land_border_cost_usd_tons"] = 0
    if transport_mode in ("road","rail"):
        # voc = 1.5/20.0
        # network_edges["gcost_usd_tons"] = 0.001*network_edges[
        #                                         "length_m"
        #                                         ]*(network_edges[
        #                                 f"{transport_mode}_cost_tonnes_km"
        #                                 ]) + network_edges[
        #                                         f"{transport_mode}_cost_tonne_h_shipper"
        #                                         ]*(network_edges["border_dwell"] + 0.001*network_edges[
        #                                         "length_m"
        #                                         ]/network_edges[
        #                                         "max_speed_kmh"
        #                                         ]) + network_edges["border_USD_t"]
        network_edges["gcost_usd_tons"] = 0.001*network_edges[
                                                "length_m"
                                                ]*(network_edges[
                                        f"{transport_mode}_cost_tonnes_km"
                                        ]) + network_edges[
                                                f"{transport_mode}_cost_tonne_h_shipper"
                                                ]*(0.001*network_edges[
                                                "length_m"
                                                ]/network_edges[
                                                "max_speed_kmh"
                                                ])
        network_edges["land_border_cost_usd_tons"] = network_edges[
                                                        f"{transport_mode}_cost_tonne_h_shipper"
                                                        ]*network_edges["border_dwell"
                                                        ] + network_edges["border_USD_t"]
        network_edges["distance_km"] = 0.001*network_edges[
                                                "length_m"
                                                ]
        # network_edges["time_hr"] = network_edges["border_dwell"] + 0.001*network_edges[
        #                                         "length_m"
        #                                         ]/network_edges[
        #                                         "max_speed_kmh"
        #                                         ]
        network_edges["time_hr"] = 0.001*network_edges[
                                                "length_m"
                                                ]/network_edges[
                                                "max_speed_kmh"
                                                ]
    elif transport_mode == "sea":
        network_edges["gcost_usd_tons"] = network_edges[
                                                "distance_km"
                                                ]*network_edges["cost_USD_t_km"
                                                    ] + network_edges[
                                                f"{transport_mode}_cost_tonne_h_shipper"
                                                ]*(network_edges["time_h"] + network_edges[
                                                "handling_h"
                                                ]) + network_edges["handling_USD_t"]
        network_edges["distance_km"] = network_edges[
                                                "distance_km"
                                                ]
        network_edges["time_hr"] = network_edges["time_h"] + network_edges[
                                                "handling_h"
                                                ]
    elif transport_mode == "IWW":
        network_edges["gcost_usd_tons"] = 0.001*network_edges[
                                                "length_m"
                                                ]*network_edges[
                                        f"{transport_mode}_cost_tonnes_km"
                                        ] + network_edges[
                                                f"{transport_mode}_cost_tonne_h_shipper"
                                                ]*(0.001*network_edges[
                                                "length_m"
                                                ]/network_edges[
                                                "max_speed_kmh"
                                                ])
        network_edges["distance_km"] = 0.001*network_edges[
                                                "length_m"
                                                ]
        network_edges["time_hr"] = 0.001*network_edges[
                                                "length_m"
                                                ]/network_edges[
                                                "max_speed_kmh"
                                                ]

    else:
        network_edges["gcost_usd_tons"] = network_edges[
                                                f"{transport_mode}_cost_tonne_h_shipper"
                                                ]*network_edges[
                                                "dwell_time_h"
                                                ] + network_edges["handling_cost_usd_t"]
        network_edges["distance_km"] = 0
        network_edges["time_hr"] = network_edges[
                                                "dwell_time_h"
                                                ]

    return network_edges


"""
Modified cost function above
"""

def transport_cost_assignment_function(network_edges,transport_mode):    
    costs_df = pd.read_csv(
                        os.path.join(
                            processed_data_path,
                            "transport_costs",
                            "country_transport_information.csv"))
    """Tariffs and cost of time
    """
    if transport_mode in ("road","rail","IWW"):
        network_edges = pd.merge(network_edges,
                            costs_df[["iso3",
                            f"{transport_mode}_cost_tonnes_km",
                            f"{transport_mode}_cost_tonne_h_shipper"]],
                            how="left",left_on=["to_iso_a3"],right_on=["iso3"])
        network_edges[f"{transport_mode}_cost_tonnes_km"] = network_edges[f"{transport_mode}_cost_tonnes_km"].fillna(0)
    elif transport_mode == "air":
        network_edges = pd.merge(network_edges,
                            costs_df[["iso3",
                            f"{transport_mode}_cost_US_tonnes",
                            f"{transport_mode}_cost_tonne_h_shipper"]],
                            how="left",left_on=["to_iso_a3"],right_on=["iso3"])
        network_edges[f"{transport_mode}_cost_US_tonnes"] = network_edges[f"{transport_mode}_cost_US_tonnes"].fillna(0)
    elif transport_mode == "sea":
        network_edges = pd.merge(network_edges,
                            costs_df[["iso3",
                            f"{transport_mode}_cost_tonne_h_shipper"]],
                            how="left",left_on=["to_iso_a3"],right_on=["iso3"])
    elif transport_mode == "intermodal":
        network_edges = pd.merge(network_edges,
                            costs_df[["iso3",
                            "road_cost_tonne_h_shipper",
                            "rail_cost_tonne_h_shipper",
                            "sea_cost_tonne_h_shipper",
                            "IWW_cost_tonne_h_shipper"]],
                            how="left",left_on=["to_iso_a3"],right_on=["iso3"])
        network_edges[
            "intermodal_cost_tonne_h_shipper"
            ] = network_edges.progress_apply(lambda x:x[f"{x.to_infra}_cost_tonne_h_shipper"],axis=1)
        network_edges.drop(["road_cost_tonne_h_shipper",
                            "rail_cost_tonne_h_shipper",
                            "sea_cost_tonne_h_shipper",
                            "IWW_cost_tonne_h_shipper"],axis=1,inplace=True)   
    network_edges[f"{transport_mode}_cost_tonne_h_shipper"] = network_edges[f"{transport_mode}_cost_tonne_h_shipper"].fillna(0)
    network_edges.drop("iso3",axis=1,inplace=True)
    if transport_mode in ("road","rail"):
        inter_country_costs_df = pd.read_csv(
                                    os.path.join(
                                            processed_data_path,
                                            "transport_costs",
                                            "OD_trade_information.csv"))
        """Border dwell times
        """
        network_edges = pd.merge(network_edges,
                            inter_country_costs_df[[
                            "from_iso3","to_iso3",
                            "border_dwell","border_USD_t"]],
                            how="left",
                            left_on=["from_iso_a3","to_iso_a3"],
                            right_on=["from_iso3","to_iso3"])
        network_edges["border_dwell"] = 24.0*network_edges["border_dwell"].fillna(0)
        network_edges["border_USD_t"] = network_edges["border_USD_t"].fillna(0)
        network_edges.drop(["from_iso3","to_iso3"],axis=1,inplace=True)
        del inter_country_costs_df

    if transport_mode == "intermodal":
        intermodal_costs_df = pd.read_excel(
                                os.path.join(
                                processed_data_path,
                                "transport_costs",
                                "intermodal.xlsx"),
                            sheet_name="Sheet1")
        network_edges = pd.merge(network_edges,
                            intermodal_costs_df,
                        how="left",on=["from_infra","to_infra"])


    if transport_mode == "road":
        speeds_df = pd.read_excel(
                        os.path.join(
                            processed_data_path,
                            "transport_costs",
                            "speed_tables.xlsx"),
                        sheet_name="road")
        network_edges = pd.merge(network_edges,
                            speeds_df,
                            how="left",
                            left_on=["from_iso_a3"],
                            right_on=["ISO_A3"]
                        )
        # infer a likely min and max road speed
        network_edges["min_max_speed"] = network_edges.apply(
            lambda x: assign_road_speeds(x), axis=1
        )

        # assign_road_speeds returned two values, unpack these into two columns
        network_edges[["min_speed_kmh", "max_speed_kmh"]] = network_edges[
                    "min_max_speed"
                    ].apply(pd.Series)

        # drop the intermediate columns
        network_edges.drop(
                    ["min_max_speed"] + speeds_df.columns.values.tolist(),
                    axis=1,
                    inplace=True,
                )
        del speeds_df
    elif transport_mode == "rail":
        speeds_df = pd.read_excel(
                        os.path.join(
                            processed_data_path,
                            "transport_costs",
                            "speed_tables.xlsx"),
                        sheet_name="rail")
        speeds_df["gauge"] = speeds_df["gauge"].astype(str)
        network_edges["gauge"] = network_edges["gauge"].astype(str)
        network_edges = pd.merge(network_edges,
                            speeds_df,
                            how="left",
                            on=["gauge"]
                        )
        del speeds_df
    elif transport_mode == "IWW":
        speeds_df = pd.read_excel(
                        os.path.join(
                            processed_data_path,
                            "transport_costs",
                            "speed_tables.xlsx"),
                        sheet_name="iww")
        # print (speeds_df)
        network_edges["min_speed_kmh"] = speeds_df["min_speed_kmh"].values[0]
        network_edges["max_speed_kmh"] = speeds_df["max_speed_kmh"].values[0]
        del speeds_df

        # port_costs_df = pd.read_excel(
        #                         os.path.join(
        #                         incoming_data_path,
        #                         "transport_costs",
        #                         "intermodal.xlsx"),
        #                     sheet_name="Sheet1")
        # port_costs_df = port_costs_df[(port_costs_df["from_mode"] == "IWW") & (port_costs_df["to_mode"] == "IWW")]
        # port_costs_df["infa"] = "IWW port"
        # network_edges = pd.merge(network_edges,
        #                     port_costs_df[["infra","dwell_time_h","handling_cost_usd_t"]],
        #                 how="left",left_on=["from_infra"],right_on=["infra"])
        # network_edges.drop(["infra"],axis=1,inplace=True)
    # elif transport_mode == "sea":
    #     speeds_df = pd.read_excel(
    #                     os.path.join(
    #                         incoming_data_path,
    #                         "transport_costs",
    #                         "speed_tables.xlsx"),
    #                     sheet_name="maritime")[["cargo_type","min_speed_kmh","max_speed_kmh"]]
    #     for sp in speeds_df.itertuples():
    #         network_edges[f"{sp.cargo_type}_min_speed_kmh"] = sp.min_speed_kmh
    #         network_edges[f"{sp.cargo_type}_max_speed_kmh"] = sp.max_speed_kmh


    #     """Sea port costs
    #     """
    #     cost_types = ["container_cost_US_TEU",
    #                     "drybulk_cost_US_tonnes",
    #                     "breakbulk_cost_US_tonnes",
    #                     "liquidbulk_cost_US_tonnes",
    #                     "vehicle_cost_US_tonnes",
    #                     "container_dwell"]
    #     sea_port_costs_df = costs_df[["iso3"] + cost_types]
    #     # sea_port_costs_df["infra"] = "port"
    #     # network_edges = pd.merge(network_edges,
    #     #                     sea_port_costs_df[["iso3","infra"] + cost_types],
    #     #                 how="left",left_on=["from_iso_a3","from_infra"],
    #     #                 right_on=["iso3","infra"]).fillna(0)
    #     network_edges[cost_types] = 0
    #     for row in sea_port_costs_df.itertuples():
    #         network_edges.loc[
    #             network_edges["from_iso_a3"] == row.iso3,cost_types
    #             ] = [getattr(row,c) for c in cost_types]
    #         network_edges.loc[
    #             network_edges["to_iso_a3"] == row.iso3,cost_types
    #             ] = [getattr(row,c) for c in cost_types]
        
    #     # network_edges.drop(["iso3","infra"],axis=1,inplace=True)
    
    network_edges["land_border_cost_usd_tons"] = 0
    if transport_mode in ("road","rail"):
        # voc = 1.5/20.0
        # network_edges["gcost_usd_tons"] = 0.001*network_edges[
        #                                         "length_m"
        #                                         ]*(network_edges[
        #                                 f"{transport_mode}_cost_tonnes_km"
        #                                 ]) + network_edges[
        #                                         f"{transport_mode}_cost_tonne_h_shipper"
        #                                         ]*(network_edges["border_dwell"] + 0.001*network_edges[
        #                                         "length_m"
        #                                         ]/network_edges[
        #                                         "max_speed_kmh"
        #                                         ]) + network_edges["border_USD_t"]
        network_edges["gcost_usd_tons"] = 0.001*network_edges[
                                                "length_m"
                                                ]*(network_edges[
                                        f"{transport_mode}_cost_tonnes_km"
                                        ]) + network_edges[
                                                f"{transport_mode}_cost_tonne_h_shipper"
                                                ]*(0.001*network_edges[
                                                "length_m"
                                                ]/network_edges[
                                                "max_speed_kmh"
                                                ])
        network_edges["land_border_cost_usd_tons"] = network_edges[
                                                        f"{transport_mode}_cost_tonne_h_shipper"
                                                        ]*network_edges["border_dwell"
                                                        ] + network_edges["border_USD_t"]
        network_edges["distance_km"] = 0.001*network_edges[
                                                "length_m"
                                                ]
        # network_edges["time_hr"] = network_edges["border_dwell"] + 0.001*network_edges[
        #                                         "length_m"
        #                                         ]/network_edges[
        #                                         "max_speed_kmh"
        #                                         ]
        network_edges["time_hr"] = 0.001*network_edges[
                                                "length_m"
                                                ]/network_edges[
                                                "max_speed_kmh"
                                                ]
    elif transport_mode == "sea":
        network_edges["gcost_usd_tons"] = network_edges[
                                                "distance_km"
                                                ]*network_edges["cost_USD_t_km"
                                                    ] + network_edges[
                                                f"{transport_mode}_cost_tonne_h_shipper"
                                                ]*(network_edges["time_h"] + network_edges[
                                                "handling_h"
                                                ]) + network_edges["handling_USD_t"]
        network_edges["distance_km"] = network_edges[
                                                "distance_km"
                                                ]
        network_edges["time_hr"] = network_edges["time_h"] + network_edges[
                                                "handling_h"
                                                ]
    elif transport_mode == "IWW":
        network_edges["gcost_usd_tons"] = 0.001*network_edges[
                                                "length_m"
                                                ]*network_edges[
                                        f"{transport_mode}_cost_tonnes_km"
                                        ] + network_edges[
                                                f"{transport_mode}_cost_tonne_h_shipper"
                                                ]*(0.001*network_edges[
                                                "length_m"
                                                ]/network_edges[
                                                "max_speed_kmh"
                                                ])
        network_edges["distance_km"] = 0.001*network_edges[
                                                "length_m"
                                                ]
        network_edges["time_hr"] = 0.001*network_edges[
                                                "length_m"
                                                ]/network_edges[
                                                "max_speed_kmh"
                                                ]

    else:
        network_edges["gcost_usd_tons"] = network_edges[
                                                f"{transport_mode}_cost_tonne_h_shipper"
                                                ]*network_edges[
                                                "dwell_time_h"
                                                ] + network_edges["handling_cost_usd_t"]
        network_edges["distance_km"] = 0
        network_edges["time_hr"] = network_edges[
                                                "dwell_time_h"
                                                ]

    return network_edges

def multimodal_network_costs_destination_based(multi_modal_edges,destination_iso,
                        modes=["IWW","rail","road","sea","intermodal"],
                        network_columns=["from_id","to_id","id",
                                        "mode","capacity","distance_km",
                                        "time_hr","land_border_cost_usd_tons",
                                        "gcost_usd_tons"]):
    multi_modal_df = []
    for mode in modes:
        edges = transport_cost_assignment_function_destination_based(
                                multi_modal_edges[multi_modal_edges["mode"] ==  mode],
                                mode,destination_iso)
        multi_modal_df.append(edges[network_columns])

    multi_modal_df.append(multi_modal_edges[~multi_modal_edges["mode"].isin(modes)])

    return pd.concat(multi_modal_df,axis=0,ignore_index=True).fillna(0)



    

def multimodal_network_assembly(modes=["IWW","rail","road","sea","intermodal"],
                            rail_status=["open"],intermodal_ports="all",cargo_type="general_cargo",
                            default_capacity=1e10,port_to_land_capacity=None):    
    multi_modal_df = []
    for mode in modes:
        if mode == "IWW":
            edges = gpd.read_file(os.path.join(
                                    processed_data_path,
                                    "infrastructure",
                                    "africa_iww_network.gpkg"
                                        ), layer="edges"
                            )
            edges['capacity'] = default_capacity
        elif mode == "rail":
            edges = gpd.read_file(os.path.join(
                            processed_data_path,
                            "infrastructure",
                            "africa_railways_network.gpkg"
                                ), layer="edges"
                    )
            edges = edges[edges["status"].isin(rail_status)]
            edges['capacity'] = default_capacity
        elif mode == "road":
            edges = gpd.read_parquet(os.path.join(
                            processed_data_path,
                            "infrastructure",
                            "africa_roads_edges.geoparquet"))
            edges['capacity'] = default_capacity
        elif mode == "sea":
            if cargo_type != "general_cargo":
                c_df = pd.read_parquet(os.path.join(
                        processed_data_path,
                        "shipping_network",
                        f"maritime_base_network_{cargo_type}.parquet"))
                g_df = pd.read_parquet(os.path.join(
                    processed_data_path,
                    "shipping_network",
                    "maritime_base_network_general_cargo.parquet")) 
                edges = pd.concat([c_df,g_df],axis=0,ignore_index=True)
                edges = edges.drop_duplicates(subset=["from_id","to_id"],keep="first")
            else:
                edges = pd.read_parquet(os.path.join(
                        processed_data_path,
                        "shipping_network",
                        "maritime_base_network_general_cargo.parquet"))
            edges.rename(columns={"from_iso3":"from_iso_a3","to_iso3":"to_iso_a3"},inplace=True)
            edges["id"] = edges.index.values.tolist()
            edges["id"] = edges.apply(lambda x:f"maritimeroute{x.id}",axis=1)
            edges['capacity'] = default_capacity
        elif mode == "intermodal":
            edges = gpd.read_file(os.path.join(
                            processed_data_path,
                            "infrastructure",
                            "africa_multimodal.gpkg"
                                ), layer="edges")
            link_types = [f"{fm}-{tm}" for idx,(fm,tm) in enumerate(itertools.permutations(modes,2))]
            edges = edges[edges["link_type"].isin(link_types)]
            if intermodal_ports != "all":
                port_edges = edges[(edges["from_infra"] == "sea") | (edges["to_infra"] == "sea")]
                port_edges = port_edges[~((port_edges["from_id"].isin(intermodal_ports)) | (port_edges["to_id"].isin(intermodal_ports)))]
                edges = edges[~edges["id"].isin(port_edges.id.values.tolist())]

            edges["from_id"] = edges.progress_apply(
                                lambda x: f"{x.from_id}_land" if "port" in str(x.from_id) else x.from_id,
                                axis=1)
            edges["to_id"] = edges.progress_apply(
                                lambda x: f"{x.to_id}_land" if "port" in str(x.to_id) else x.to_id,
                                axis=1)
            edges['capacity'] = default_capacity
            if port_to_land_capacity is not None:
                for idx,(port,capacity) in enumerate(port_to_land_capacity):
                    cap_factor = 0.5*len(edges[((edges["from_id"] == f"{port}_land") | (edges["to_id"] == f"{port}_land"))].index)
                    edges.loc[((edges["from_id"] == f"{port}_land") | (edges["to_id"] == f"{port}_land")),"capacity"] = 1.0*capacity/cap_factor
        edges["mode"] = mode
        # edges = transport_cost_assignment_function(edges,mode)
        # multi_modal_df.append(edges[["from_id","to_id","id",
        #                                 "mode","capacity","distance_km",
        #                                 "time_hr",
        #                                 "gcost_usd_tons"]])
        # multi_modal_df.append(edges[["from_id","to_id","id",
        #                                 "mode","capacity","distance_km",
        #                                 "time_hr","land_border_cost_usd_tons",
        #                                 "gcost_usd_tons"]])
        multi_modal_df.append(edges[["from_id","to_id","id",
                                        "mode","capacity"]])
        if mode in ("road","rail"):
            # add_edges = edges[["from_id","to_id","id","mode","capacity","distance_km","time_hr","gcost_usd_tons"]].copy()
            # add_edges.columns = ["to_id","from_id","id","mode","capacity","distance_km","time_hr","gcost_usd_tons"]
            add_edges = edges[["from_id","to_id","id",
                                "mode","capacity"]].copy()
            add_edges.columns = ["to_id","from_id",
                                "id","mode","capacity"]
            multi_modal_df.append(add_edges)

        # if mode in ("road","rail"):
        #     # add_edges = edges[["from_id","to_id","id","mode","capacity","distance_km","time_hr","gcost_usd_tons"]].copy()
        #     # add_edges.columns = ["to_id","from_id","id","mode","capacity","distance_km","time_hr","gcost_usd_tons"]
        #     add_edges = edges[["from_id","to_id","id",
        #                         "mode","capacity","distance_km",
        #                         "time_hr","land_border_cost_usd_tons",
        #                         "gcost_usd_tons"]].copy()
        #     add_edges.columns = ["to_id","from_id",
        #                         "id","mode","capacity",
        #                         "distance_km","time_hr",
        #                         "land_border_cost_usd_tons",
        #                         "gcost_usd_tons"]
        #     multi_modal_df.append(add_edges)


    return multi_modal_df

def add_geometries_to_flows(flows_dataframe,merge_column="id",modes=["rail","sea","road","IWW"],layer_type="edges"):    
    flow_edges = []
    for mode in modes:
        if mode == "IWW":
            edges = gpd.read_file(os.path.join(
                                    processed_data_path,
                                    "infrastructure",
                                    "africa_iww_network.gpkg"
                                        ), layer=layer_type
                            )
        elif mode == "rail":
            edges = gpd.read_file(os.path.join(
                            processed_data_path,
                            "infrastructure",
                            "africa_railways_network.gpkg"
                                ), layer=layer_type
                    )
        elif mode == "road":
            if layer_type == "edges":
                edges = gpd.read_parquet(os.path.join(
                                processed_data_path,
                                "infrastructure",
                                "africa_roads_edges.geoparquet"))
            else:
                edges = gpd.read_parquet(os.path.join(
                                processed_data_path,
                                "infrastructure",
                                "africa_roads_nodes.geoparquet"))
                edges.rename(columns={"road_id":merge_column,"iso_a3":"iso3"},inplace=True)
                edges["infra"] = "road"
        elif mode == "sea":
            edges = gpd.read_file(
                    os.path.join(processed_data_path,
                        "infrastructure",
                        "global_maritime_network.gpkg"
                    ),layer=layer_type)
        edges["mode"] = mode
        if layer_type == "edges":
            edges = edges[[merge_column,"from_id","to_id","mode","geometry"]]
        else:
            edges = edges[[merge_column,"iso3","infra","mode","geometry"]]
        flow_edges.append(
            edges[
                edges[merge_column].isin(flows_dataframe[merge_column].values.tolist())
                ]
            )

    flow_edges = pd.concat(flow_edges,axis=0,ignore_index=True)
    return gpd.GeoDataFrame(
                pd.merge(
                        flows_dataframe,flow_edges,
                        how="left",on=[merge_column]
                        ),
                    geometry="geometry",crs="EPSG:4326")

def add_node_degree_to_flows(nodes_flows_dataframe,mineral_class):
    edge_flows_df = gpd.read_file(os.path.join(output_data_path,
                                    "flow_mapping",
                                    f"{mineral_class}_flows.gpkg"),
                                    layer="edges")

    degree_df = edge_flows_df[["from_id","to_id"]].stack().value_counts().rename_axis('id').reset_index(name='degree')
    nodes_flows_dataframe = pd.merge(nodes_flows_dataframe,degree_df,how="left",on=["id"])

    return gpd.GeoDataFrame(nodes_flows_dataframe,geometry="geometry",crs="EPSG:4326")

def connect_points_to_transport(points_df,points_id_col,
                            points_type,
                            rail_status=["open"],
                            distance_threshold=50,
                            default_capacity=1e10):
    points_transport_df = []
    for mode in ["rail","road"]:
        if mode == "rail":
            edges = gpd.read_file(os.path.join(
                                    processed_data_path,
                                    "infrastructure",
                                    "africa_railways_network.gpkg"
                                        ), layer="edges"
                            )
            edges = edges[edges["status"].isin(rail_status)]
            node_ids = list(set(edges["from_id"].values.tolist() + edges["to_id"].values.tolist()))
            nodes = gpd.read_file(os.path.join(
                                    processed_data_path,
                                    "infrastructure",
                                    "africa_railways_network.gpkg"
                                        ), layer="nodes"
                            )
            nodes = nodes[(nodes["id"].isin(node_ids)) & (nodes["infra"].isin(['stop','station']))]
        else:
            nodes = gpd.read_parquet(os.path.join(
                            processed_data_path,
                            "infrastructure",
                            "africa_roads_nodes.geoparquet"))
            nodes.rename(columns={"road_id":"id"},inplace=True)


        """Find mines attached to rail and roads
        """
        points_node_intersects = gpd.sjoin_nearest(points_df[[points_id_col,"geometry"]].to_crs(epsg=epsg_meters),
                                nodes[["id","geometry"]].to_crs(epsg=epsg_meters),
                                how="left",distance_col="distance").reset_index()
        if mode == "rail":
            points_node_intersects = points_node_intersects[points_node_intersects["distance"] <= distance_threshold]
        
        points_node_intersects = points_node_intersects.drop_duplicates(subset=[points_id_col],keep="first")
        points_node_intersects = points_node_intersects[[points_id_col,"id"]]
        points_node_intersects.columns = ["from_id","to_id"]
        points_transport_df.append(points_node_intersects)
        points_dup = points_node_intersects.copy()
        points_dup.columns = ["to_id","from_id"]
        points_transport_df.append(points_dup)
    
    if len(points_transport_df) > 0:
        points_transport_df = pd.concat(points_transport_df,axis=0,ignore_index=True)
        points_transport_df["id"] = points_transport_df.index.values.tolist()
        points_transport_df["id"] = points_transport_df.progress_apply(
                                                lambda x:f"{points_type}transporte{x.id}",
                                                axis=1)
        points_transport_df["capacity"] = default_capacity
        # points_transport_df["gcost_usd_tons"] = 0
        # points_transport_df["distance_km"] = 0
        # points_transport_df["time_hr"] = 0
        # points_transport_df["land_border_cost_usd_tons"] = 0
        points_transport_df["mode"] = points_type


        return [points_transport_df]
    else:
        return points_transport_df

def create_mines_to_port_network(mines_df,mine_id_col,
                    modes=["IWW","rail","road","sea","intermodal"],
                    rail_status=["open"],distance_threshold=50,
                    # network_columns=["from_id","to_id","id","mode","capacity","distance_km","time_hr","gcost_usd_tons"],
                    network_columns=["from_id","to_id","id","mode","capacity"],
                    intermodal_ports="all",cargo_type="general_cargo",port_to_land_capacity=None):
    multi_modal_df = multimodal_network_assembly(modes=modes,
                    rail_status=rail_status,
                    intermodal_ports=intermodal_ports,
                    cargo_type=cargo_type,port_to_land_capacity=port_to_land_capacity)
    # mine_transport_df = connect_mines_to_transport(mines_df,mine_id_col,
    #                                 rail_status=rail_status,
    #                                 distance_threshold=distance_threshold)
    mine_transport_df = connect_points_to_transport(mines_df,mine_id_col,
                            "mine",
                            rail_status=rail_status,
                            distance_threshold=distance_threshold)

    network_df = pd.concat(mine_transport_df+multi_modal_df,axis=0,ignore_index=True)
    # gpd.GeoDataFrame(network_df,
    #     geometry="geometry",crs="EPSG_4326").to_file(
    #     os.path.join(processed_data_path,
    #         "infrastructure",
    #         "mines_to_port_complete_network.gpkg"),driver="GPKG")
    # return create_igraph_from_dataframe(network_df[network_columns],directed=True)
    return network_df[network_columns]

def create_mines_and_cities_to_port_network(mines_df,mine_id_col,
                    cities_df,city_id_col,
                    modes=["IWW","rail","road","sea","intermodal"],
                    rail_status=["open"],distance_threshold=50,
                    # network_columns=["from_id","to_id","id",
                    #                 "mode","capacity","distance_km",
                    #                 "time_hr","land_border_cost_usd_tons",
                    #                 "gcost_usd_tons"],
                    network_columns=["from_id","to_id","id",
                                    "mode","capacity"],
                    intermodal_ports="all",cargo_type="general_cargo",port_to_land_capacity=None):
    multi_modal_df = multimodal_network_assembly(modes=modes,
                    rail_status=rail_status,
                    intermodal_ports=intermodal_ports,
                    cargo_type=cargo_type,port_to_land_capacity=port_to_land_capacity)
    # mine_transport_df = connect_mines_to_transport(mines_df,mine_id_col,
    #                                 rail_status=rail_status,
    #                                 distance_threshold=distance_threshold)
    mine_transport_df = connect_points_to_transport(mines_df,mine_id_col,
                            "mine",
                            rail_status=rail_status,
                            distance_threshold=distance_threshold)
    city_transport_df = connect_points_to_transport(cities_df,city_id_col,
                            "city",
                            rail_status=rail_status,
                            distance_threshold=distance_threshold)

    network_df = pd.concat(mine_transport_df+multi_modal_df+city_transport_df,axis=0,ignore_index=True)
    # gpd.GeoDataFrame(network_df,
    #     geometry="geometry",crs="EPSG_4326").to_file(
    #     os.path.join(processed_data_path,
    #         "infrastructure",
    #         "mines_to_port_complete_network.gpkg"),driver="GPKG")
    # return create_igraph_from_dataframe(network_df[network_columns],directed=True)
    return network_df[network_columns]  

def separate_land_and_sea_routes(edges,nodes,network_dataframe,
                            land_modes=["road","rail","IWW","intermodal"]):
    if len(edges) > 0:
        land_edges = []
        land_nodes = []
        sea_edges = []
        sea_nodes = []
        for e in range(len(edges)):
            edge = edges[e]
            for lm in land_modes:
                if lm in edge:
                    land_edges.append(edge)
                    if e == 0:
                        land_nodes += [nodes[e],nodes[e+1]]
                    else:
                        land_nodes += [nodes[e+1]]
                    break
                elif "maritime" in edge:
                    sea_edges.append(edge)
                    if e == 0:
                        sea_nodes += [nodes[e],nodes[e+1]]
                    else:
                        sea_nodes += [nodes[e+1]]
                    break

        land_costs = network_dataframe[network_dataframe["id"].isin(land_edges)]["gcost_usd_tons"].sum()
        sea_costs = network_dataframe[network_dataframe["id"].isin(sea_edges)]["gcost_usd_tons"].sum()

        return land_edges, land_nodes, sea_edges, sea_nodes, land_costs, sea_costs
    else:
        return edges, nodes, edges, nodes,0,0

def get_land_and_sea_routes_costs(route_dataframe,network_dataframe,
                                    edge_path_column,node_path_column):
    route_dataframe["land_sea_paths_costs"] = route_dataframe.progress_apply(
                            lambda x:separate_land_and_sea_routes(x[edge_path_column],
                                        x[node_path_column],
                                        network_dataframe),
                            axis=1)

    route_dataframe[
            [f"land_{edge_path_column}",
            f"land_{node_path_column}",
            f"sea_{edge_path_column}",
            f"sea_{node_path_column}",
            "land_gcost_usd_tons",
            f"sea_gcost_usd_tons"]
            ] = route_dataframe["land_sea_paths_costs"].apply(pd.Series)
    route_dataframe.drop("land_sea_paths_costs",axis=1,inplace=True)
    return route_dataframe

def separate_land_and_sea_costs(edges,network_dataframe,
                            land_modes=["road","rail","IWW","intermodal"]):
    if len(edges) > 0:
        land_costs_df = network_dataframe[
                                        (
                                            network_dataframe["id"].isin(edges)
                                        ) & (
                                            network_dataframe["mode"].isin(land_modes)
                                        )]
        land_costs_df = land_costs_df.drop_duplicates(subset=["id"],keep="first")

        sea_costs_df = network_dataframe[
                                        (
                                            network_dataframe["id"].isin(edges)
                                        ) & (
                                            network_dataframe["mode"] == "sea"
                                        )]
        sea_costs_df = sea_costs_df.drop_duplicates(subset=["id"],keep="first")

        return land_costs_df["gcost_usd_tons"].sum(), sea_costs_df["gcost_usd_tons"].sum()
    else:
        return 0,0

def get_land_and_sea_costs(route_dataframe,network_dataframe,
                                    edge_path_column,node_path_column):
    route_dataframe["land_sea_costs"] = route_dataframe.progress_apply(
                            lambda x:separate_land_and_sea_costs(x[edge_path_column],
                                        network_dataframe),
                            axis=1)

    route_dataframe[
            [
            "land_gcost_usd_tons",
            "sea_gcost_usd_tons"]
            ] = route_dataframe["land_sea_costs"].apply(pd.Series)
    route_dataframe.drop("land_sea_costs",axis=1,inplace=True)
    return route_dataframe

def add_port_path(edges,nodes,G):
    # print ("edges:",edges)
    # print ("nodes:",nodes)
    if len(edges) > 0:
        new_edges = []
        new_nodes = [nodes[0]]
        for e in range(len(edges)):
            source = nodes[e]
            target = nodes[e+1]
            if ("port" in source) and ("port" in target):
                source = source.split("_")[0]
                target = target.split("_")[0]
                if source != target:
                    route,points,_,_,_,_ = network_od_node_edge_path_estimations(G,
                                    source, target,"distance","distance","distance","id")
                    new_edges += route[0]
                    new_nodes += points[0]
            else:
                new_edges.append(edges[e])
                new_nodes += [target] 

        return new_edges,new_nodes
    else:
        return edges, nodes

def convert_port_routes(route_dataframe,edge_path_column,node_path_column):
    global_port_network = gpd.read_file(
                    os.path.join(processed_data_path,
                        "infrastructure",
                        "global_maritime_network.gpkg"
                    ),layer="edges")
    port_graph = create_igraph_from_dataframe(global_port_network[["from_id","to_id","id","distance"]])
    route_dataframe["full_paths"] = route_dataframe.progress_apply(
                            lambda x:add_port_path(x[edge_path_column],x[node_path_column],port_graph),axis=1)

    route_dataframe[
            [f"full_{edge_path_column}",f"full_{node_path_column}"]
            ] = route_dataframe["full_paths"].apply(pd.Series)
    route_dataframe.drop("full_paths",axis=1,inplace=True)
    return route_dataframe