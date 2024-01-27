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
    # port_df = gpd.read_file(os.path.join(
    #                         processed_data_path,
    #                         "infrastructure",
    #                         "global_port_utilisation.gpkg"
    #                             ), layer="nodes"
    #                         )
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
    # mine_countries = list(set(mines_df["shapeGroup_primary_admin0"].values.tolist()))
    # product_trade_df = trade_df[(
    #                 trade_df["export_country_code"].isin(mine_countries)
    #                 ) & (trade_df["product_code"].isin(baci_codes_unrefined + baci_codes_refined))]

    """Step 2: Identify all the countries outside Africa 
    """
    # non_africa_exports_df = product_trade_df[
    #                               product_trade_df["export_continent"] != "Africa"
    #                               ]
    baci_codes_types = pd.read_csv(os.path.join(processed_data_path,
                            "baci",
                            "commodity_codes_refined_unrefined.csv"))
    # network_graph = create_mines_to_port_network(mines_df,mine_id_col,
    #                 modes=["rail","sea","intermodal","road"],
    #                 )
    # if "geometry" in mines_df.columns.values.tolist():
    #     mines_df.drop("geometry",axis=1,inplace=True)
    set_port_capacity = [("port137",0.5*1e6)] # Amount of copper restricted for Biera port
    for row in baci_codes_types.itertuples():
        refined_type = row.product_type
        product_codes = [int(r) for r in str(row.product_codes).split(",")]
        mineral_class = row.reference_mineral
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
        mine_refined_df["mine_output_approx_copper_export"
            ] = mine_refined_df["mine_output_approx_copper"
            ]*mine_refined_df["trade_quantity_tons_fraction"]
        
        port_export_df = mine_refined_df[mine_refined_df["import_continent"] != "Africa"]
        port_export_df = pd.merge(port_export_df,
                            all_ports_df,
                            how="left",left_on=["import_country_code"],right_on=["iso3"])
        port_export_df["id"] = port_export_df.progress_apply(lambda x:f"{x.id}_land",axis=1)
        port_export_df.drop("geometry",axis=1,inplace=True)
        port_export_df.to_csv(f"{mineral_class}_{refined_type}.csv",index=False)
        # mine_to_port_routes = network_od_node_edge_paths_assembly(
        #                             port_export_df,network_graph,
        #                             "gcost_usd_tons","id",
        #                             mine_id_col,"id")
        network_graph["trade_quantity_tons"] = 0
        mine_to_port_routes, unassinged_routes = od_flow_allocation_capacity_constrained(
                                            port_export_df,network_graph,
                                            "trade_quantity_tons","gcost_usd_tons",
                                            "id",mine_id_col,
                                            "id")
        if len(mine_to_port_routes) > 0:
            mine_to_port_routes = pd.concat(mine_to_port_routes,axis=0,ignore_index=True)
            print (mine_to_port_routes[[mine_id_col,"id","trade_quantity_tons","gcost_usd_tons"]])
            if "geometry" in mine_to_port_routes.columns.values.tolist():
                mine_to_port_routes.drop("geometry",axis=1,inplace=True)
            # mine_to_port_routes = add_node_paths(mine_to_port_routes,network_graph,"id","edge_path")
            mine_to_port_routes.to_csv("test.csv")
            mine_to_port_routes = convert_port_routes(mine_to_port_routes,"edge_path","node_path")
            mine_to_port_routes[[mine_id_col,"id",
                        "export_country_code","import_country_code",
                        "trade_value_thousandUSD","trade_quantity_tons",
                        "edge_path","node_path","full_edge_path","full_node_path","gcost_usd_tons"]].to_parquet(
                    os.path.join(results_folder,f"{mineral_class}_{refined_type}.parquet"),
                    index=False)

            flows_df = []
            for flow_column in ["trade_value_thousandUSD","trade_quantity_tons"]: 
                f_df = get_flow_on_edges(mine_to_port_routes,"id","full_edge_path",
                                flow_column)
                # f_df = f_df.set_index("id")
                flows_df.append(f_df)

            flows_df = pd.concat(flows_df,axis=0,ignore_index=True).fillna(0)
            flows_df = flows_df.groupby(["id"]).agg(
                        dict(
                                [("trade_value_thousandUSD","sum"),("trade_quantity_tons","sum")]
                            )
                        ).reset_index()
            # flows_df = flows_df.reset_index()
            flows_df = add_geometries_to_flows(flows_df,merge_column="id",modes=["rail","sea","road"])
            flows_df.to_file(os.path.join(results_folder,
                                    f"{mineral_class}_{refined_type}_flows.gpkg"),
                                    layer="edges",driver="GPKG")









if __name__ == '__main__':
    CONFIG = load_config()
    main(CONFIG)