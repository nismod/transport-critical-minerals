#!/usr/bin/env python
# coding: utf-8
"""This code finds the route between mines (in Africa) and all ports (outside Africa) 
"""
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
    output_data_path = config['paths']['results']

    results_folder = os.path.join(output_data_path,"flow_mapping")
    if os.path.exists(results_folder) == False:
        os.mkdir(results_folder)

    mine_id_col = "mine_cluster_mini"
    cargo_type = "Dry bulk"
    trade_groupby_columns = ["export_country_code", 
                            "import_country_code", 
                            "export_continent","export_landlocked",
                            "import_continent","import_landlocked"]
    trade_value_columns = ["trade_value_thousandUSD","trade_quantity_tons"]
    od_columns = ["reference_mineral","process_binary","export_country_code",
                    "import_country_code",
                    "export_continent",
                    "import_continent",
                    "mine_output_tons",    
                    "mine_output_thousandUSD"]
    """Step 1: Get the input datasets
    """
    # Mine locations in Africa with the copper tonages
    mines_df = gpd.read_file(os.path.join(processed_data_path,
                                    "Minerals",
                                    "copper_mines_tons.gpkg"))
    mines_df[mine_id_col] = mines_df.progress_apply(lambda x:f"mine_{x[mine_id_col]}",axis=1)
    # Select only tonnages > 0
    mines_df = mines_df[mines_df["mine_output_approx_copper"] > 0]

    # Get the BACI trade linkages between countries 
    trade_df = pd.read_csv(os.path.join(processed_data_path,
                            "baci",
                            "baci_ccg_clean_continent_trade.csv"))

    # Get the global port network data for Dry Bulk transport
    # We assume CCG critical minerals are transported as Dry Bulk Cargo (General Cargo maybe)
    port_df = pd.read_csv(os.path.join(
                            processed_data_path,
                            "port_statistics",
                            "port_vessel_types_and_capacities.csv"
                                )
                            )
    port_df = port_df[port_df["vessel_type_main"] == cargo_type]
    # Get the max capacity port in each country
    port_df = port_df.sort_values(by=["annual_vessel_capacity_tons"],ascending=False)
    port_df = port_df.drop_duplicates(subset=["iso3"],keep="first")

    # Get the perferred port mapping of the landlocked countries outside Africa
    landlocked_port_df = pd.read_csv(
                            os.path.join(
                                processed_data_path,
                                "port_statistics",
                                "ccg_importing_landlocked_countries.csv"))
    # Create a list of all ports with unique port assumed for each country
    all_ports_df = pd.concat(
                    [port_df[["id","iso3"]],landlocked_port_df[["id","iso3"]]],
                    axis=0,ignore_index=True)
    del port_df, landlocked_port_df

    port_commodities_df = pd.read_csv(os.path.join(processed_data_path,
                                    "port_statistics",
                                    "port_known_commodities_traded.csv"))
    """Step 2: Identify all the countries outside Africa 
    """
    baci_codes_types = pd.read_csv(os.path.join(processed_data_path,
                            "baci",
                            "commodity_codes_refined_unrefined.csv"))
    set_port_capacity = [("port137",0.5*1e6)] # Amount of copper restricted for Biera port

    """Create the reference commodity level OD first
    """
    combined_trade_df = []
    for row in baci_codes_types.itertuples():
        refined_type = row.product_type
        product_codes = [int(r) for r in str(row.product_codes).split(",")]
        mineral_class = row.reference_mineral
        t_df = trade_df[
                    trade_df["product_code"
                    ].isin(product_codes)
                ].groupby(trade_groupby_columns)[trade_value_columns].sum().reset_index()
        t_df["trade_quantity_tons_fraction"] = t_df["trade_quantity_tons"
                                    ]/t_df.groupby(
                                        ["export_country_code"]
                                    )["trade_quantity_tons"].transform('sum')

        if refined_type == "refined":
            mine_refined_df = mines_df[mines_df["process_binary"] == 1]
        else:
            mine_refined_df = mines_df[mines_df["process_binary"] == 0]

        mine_refined_df = pd.merge(mine_refined_df,
                                    t_df,
                                    how="left",
                                    left_on=["shapeGroup_primary_admin0"],
                                    right_on=["export_country_code"])
        mine_refined_df["mine_output_tons"
            ] = mine_refined_df["mine_output_approx_copper"
            ]*mine_refined_df["trade_quantity_tons_fraction"]
        mine_refined_df["mine_output_thousandUSD"
            ] = mine_refined_df["mine_output_tons"
            ]*mine_refined_df["trade_value_thousandUSD"]/mine_refined_df["trade_quantity_tons"]
        mine_refined_df.drop("geometry",axis=1,inplace=True)
        # mine_refined_df.to_csv(f"{mineral_class}_{refined_type}_mine_level.csv",index=False)
        mine_refined_df.rename(columns={mine_id_col:"origin_id"},inplace=True)
        mine_refined_df["reference_mineral"] = mineral_class
        """Get the flows being exported out of Africa to ports around the world
        """
        port_export_df = mine_refined_df[mine_refined_df["import_continent"] != "Africa"]
        if len(port_export_df.index) > 0:
            port_export_df = pd.merge(port_export_df,
                                all_ports_df,
                                how="left",left_on=["import_country_code"],right_on=["iso3"])
            port_export_df["destination_id"] = port_export_df.progress_apply(lambda x:f"{x.id}_land",axis=1)
            # port_export_df.drop("geometry",axis=1,inplace=True)
            # port_export_df.to_csv(f"{mineral_class}_{refined_type}.csv",index=False)
            combined_trade_df.append(port_export_df[["origin_id","destination_id"]+od_columns])

        land_export_df = mine_refined_df[
                            mine_refined_df["import_continent"] == "Africa"]
        if len(land_export_df.index) > 0 and refined_type == "unrefined":
            land_refined_df = mines_df[mines_df["process_binary"] == 1]
            land_refined_df["mine_processed_quantity_tons_fraction"] = land_refined_df["mine_output_approx_copper"
                                    ]/land_refined_df.groupby(
                                        ["shapeGroup_primary_admin0"]
                                    )["mine_output_approx_copper"].transform('sum')
            land_refined_df.rename(
                    columns={mine_id_col:"destination_id",
                            "shapeGroup_primary_admin0":"import_country_code"},
                    inplace=True)
            land_export_df = pd.merge(land_export_df,land_refined_df[["import_country_code",
                                    "destination_id","mine_processed_quantity_tons_fraction"]],
                                    how="left",on=["import_country_code"]).fillna(0)
            land_export_df["mine_output_tons"
                    ] = land_export_df["mine_output_tons"]*land_export_df["mine_processed_quantity_tons_fraction"]
            land_export_df["mine_output_thousandUSD"
                    ] = land_export_df["mine_output_thousandUSD"]*land_export_df["mine_processed_quantity_tons_fraction"]
            # land_export_df.to_csv(f"{mineral_class}_{refined_type}_mine_to_mine.csv",index=False)
            combined_trade_df.append(land_export_df[land_export_df["destination_id"] != 0][["origin_id","destination_id"]+od_columns])

    combined_trade_df = pd.concat(combined_trade_df,axis=0,ignore_index=True)
    combined_trade_df.to_csv("copper_ods.csv",index=False)

    origin_id = "origin_id"
    destination_id = "destination_id"
    trade_ton_column = "mine_output_tons"
    trade_usd_column = "mine_output_thousandUSD"
    mineral_classes = list(set(combined_trade_df.reference_mineral.values.tolist()))
    for mineral_class in mineral_classes:
        export_ports_africa = port_commodities_df[port_commodities_df[f"{mineral_class}_export_binary"] == 1]
        if len(export_ports_africa.index) == 0:
            export_ports_africa = port_commodities_df[
                                port_commodities_df[
                                f"{cargo_type.lower().replace(' ','_')}_annual_vessel_capacity_tons"] > 0
                                ]
        if set_port_capacity is not None:
            for idx,(prt,cap) in enumerate(set_port_capacity):
                export_ports_africa.loc[
                        export_ports_africa["id"] == prt,
                        f"{cargo_type.lower().replace(' ','_')}_annual_vessel_capacity_tons"] = cap

        export_port_ids = list(zip(export_ports_africa["id"].values.tolist(),
                        export_ports_africa[
                            f"{cargo_type.lower().replace(' ','_')}_annual_vessel_capacity_tons"]))
        network_graph = create_mines_to_port_network(mines_df,mine_id_col,
                    modes=["sea","intermodal","road","rail"],
                    intermodal_ports=export_ports_africa["id"].values.tolist(),
                    cargo_type=f"{cargo_type.lower().replace(' ','_')}",
                    port_to_land_capacity=export_port_ids
                    )

        network_graph[trade_ton_column] = 0
        mine_routes, unassinged_routes = od_flow_allocation_capacity_constrained(
                                            combined_trade_df,network_graph,
                                            trade_ton_column,"gcost_usd_tons",
                                            "id",origin_id,
                                            destination_id)
        if len(mine_routes) > 0:
            mine_routes = pd.concat(mine_routes,axis=0,ignore_index=True)
            c_df = combined_trade_df.copy()
            c_df.rename(columns={trade_ton_column:"initial_tonnage"},inplace=True)
            mine_routes = pd.merge(mine_routes,
                            c_df[[origin_id,destination_id,"initial_tonnage"]],
                            how="left",on=[origin_id,destination_id])
            mine_routes[trade_usd_column] = mine_routes[
                                                        trade_usd_column]*mine_routes[
                                                        trade_ton_column]/mine_routes["initial_tonnage"]
            mine_routes.drop("initial_tonnage",axis=1,inplace=True)
            del c_df
            mine_routes[combined_trade_df.columns.values.tolist()].to_csv("copper_ods_assigned.csv",index=False)

            print (mine_routes[[origin_id,destination_id,trade_ton_column,"gcost_usd_tons"]])
            if "geometry" in mine_routes.columns.values.tolist():
                mine_routes.drop("geometry",axis=1,inplace=True)
            # mine_routes = add_node_paths(mine_routes,network_graph,"id","edge_path")
            # mine_routes.to_csv("test.csv")
            mine_routes = convert_port_routes(mine_routes,"edge_path","node_path")
            mine_routes[[origin_id,destination_id] + od_columns + [
                                    "edge_path","node_path","full_edge_path",
                                    "full_node_path","gcost_usd_tons"]].to_parquet(
                    os.path.join(results_folder,f"{mineral_class}.parquet"),
                    index=False)

            flows_df = []
            for flow_column in [trade_ton_column,trade_usd_column]: 
                f_df = get_flow_on_edges(mine_routes,"id","full_edge_path",
                                flow_column)
                # f_df = f_df.set_index("id")
                flows_df.append(f_df)

            flows_df = pd.concat(flows_df,axis=0,ignore_index=True).fillna(0)
            flows_df = flows_df.groupby(["id"]).agg(
                        dict(
                                [(trade_ton_column,"sum"),(trade_usd_column,"sum")]
                            )
                        ).reset_index()
            # flows_df = flows_df.reset_index()
            flows_df = add_geometries_to_flows(flows_df,merge_column="id",modes=["rail","sea","road"])
            flows_df.to_file(os.path.join(results_folder,
                                    f"{mineral_class}_flows.gpkg"),
                                    layer="edges",driver="GPKG")









if __name__ == '__main__':
    CONFIG = load_config()
    main(CONFIG)