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

def assign_road_speeds(x):
    if float(x.tag_maxspeed) > 0:
        return x['tag_maxspeed'],x['tag_maxspeed'] 
    if x.tag_highway in ('motorway','trunk','primary'):
        return x["Highway_min"],x["Highway_max"]
    elif x.paved == "road_paved":
        return max(x["Urban_min"],x["Rural_min"]),max(x["Urban_max"],x["Rural_max"])
    else:
        return min(x["Urban_min"],x["Rural_min"]),min(x["Urban_max"],x["Rural_max"])


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
        network_edges["border_dwell"] = network_edges["border_dwell"].fillna(0)
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
        network_edges["min_speed_kmh"] = speeds_df["min_speed_kmh"].values[0]
        network_edges["max_speed_kmh"] = speeds_df["max_speed_kmh"].values[0]
        del speeds_df

    if transport_mode in ("road","rail"):
        network_edges["gcost_usd_tons"] = 0.001*network_edges[
                                                "length_m"
                                                ]*network_edges[
                                        f"{transport_mode}_cost_tonnes_km"
                                        ] + network_edges[
                                                f"{transport_mode}_cost_tonne_h_shipper"
                                                ]*(network_edges["border_dwell"] + 0.001*network_edges[
                                                "length_m"
                                                ]/network_edges[
                                                "max_speed_kmh"
                                                ]) + network_edges["border_USD_t"]
    elif transport_mode == "sea":
        network_edges["gcost_usd_tons"] = network_edges[
                                                "distance_km"
                                                ]*network_edges["cost_USD_t_km"
                                                    ] + network_edges[
                                                f"{transport_mode}_cost_tonne_h_shipper"
                                                ]*(network_edges["time_h"] + network_edges[
                                                "handling_h"
                                                ]) + network_edges["handling_USD_t"]
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
    else:
        network_edges["gcost_usd_tons"] = network_edges[
                                                f"{transport_mode}_cost_tonne_h_shipper"
                                                ]*network_edges[
                                                "dwell_time_h"
                                                ] + network_edges["handling_cost_usd_t"]

    return network_edges