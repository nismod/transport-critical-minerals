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



def main(config,reference_mineral,year,percentile):
    incoming_data_path = config['paths']['incoming_data']
    processed_data_path = config['paths']['data']
    output_data_path = config['paths']['results']

    results_folder = os.path.join(output_data_path,"flow_mapping")
    if os.path.exists(results_folder) == False:
        os.mkdir(results_folder)

    # cargo_type = "Dry bulk"
    cargo_type = "General cargo"
    # trade_groupby_columns = ["export_country_code", 
    #                         "import_country_code", 
    #                         "export_continent","export_landlocked",
    #                         "import_continent","import_landlocked"]
    # trade_value_columns = ["trade_value_thousandUSD","trade_quantity_tons"]
    od_columns = ["reference_mineral",
                    "initial_refined_stage",
                    "final_refined_stage",
                    "export_country_code",
                    "import_country_code",
                    "export_continent",
                    "import_continent",
                    "mine_output_tons"]
    data_type = {"initial_refined_stage":"str","final_refined_stage":"str"}
    """Step 1: Get the OD matrix
    """
    # print (year,percentile)
    generate_graph = False
    if generate_graph is True:
        if year == 2022:
            od_file_name = f"mining_city_node_level_ods_{year}.csv"
            mine_layer = f"{reference_mineral}"
        else:
            od_file_name = f"mining_city_node_level_ods_{year}_{percentile}.csv"
            mine_layer = f"{reference_mineral}_{percentile}"

        # print (od_file_name)

        combined_trade_df = pd.read_csv(os.path.join(
                                    results_folder,
                                    od_file_name),dtype=data_type)
        od_locations = list(
                            set(
                                combined_trade_df["origin_id"].values.tolist() + combined_trade_df["destination_id"].values.tolist()
                                )
                            )

        # Mine locations in Africa with the mineral tonages
        mine_id_col = "mine_id"
        mines_df = gpd.read_file(os.path.join(output_data_path,
                                        "location_outputs",
                                       f"mine_city_tons_{year}.gpkg"),layer=mine_layer)
        mines_df = mines_df[(mines_df["id"].isin(od_locations)) & (mines_df["location_type"] == "mine")]
        mines_df.rename(columns={"id":mine_id_col},inplace=True)
        mines_df = mines_df[[mine_id_col,"geometry"]]

        # Population locations for urban cities
        pop_id_col = "city_id"
        un_pop_df = gpd.read_file(os.path.join(processed_data_path,
                                        "admin_boundaries",
                                        "un_urban_population",
                                        "un_pop_df.gpkg"))
        # un_pop_df = un_pop_df[un_pop_df["CONTINENT"] == "Africa"]
        un_pop_df = un_pop_df[un_pop_df[pop_id_col].isin(od_locations)]
        un_pop_df = un_pop_df[[pop_id_col,"geometry"]]

        port_commodities_df = pd.read_csv(os.path.join(processed_data_path,
                                        "port_statistics",
                                        "port_known_commodities_traded.csv"))
        """Step 2: Identify all the countries outside Africa 
        """
        # Amount of copper restricted for Biera and to disallow transhipment via Nacala ports
        # set_port_capacity = [("port137",0.27*1e6),("port784",0.0)]
        origin_id = "origin_id"
        destination_id = "destination_id"
        trade_ton_column = "mine_output_tons"
        trade_usd_column = "mine_output_thousandUSD"
        # combined_trade_df = pd.read_csv(os.path.join(
        #                                         results_folder,
        #                                         f"mining_city_node_level_ods_{year}.csv"))
        # reference_minerales = list(set(combined_trade_df.reference_mineral.values.tolist()))
        edges_flows_df = []
        nodes_flows_df = []
        flow_column_types = []  
        c_t_df = combined_trade_df[combined_trade_df.reference_mineral == reference_mineral]
        export_ports_africa = port_commodities_df[port_commodities_df[f"{reference_mineral}_export_binary"] == 1]
        if len(export_ports_africa.index) == 0:
            export_ports_africa = port_commodities_df[
                                port_commodities_df[
                                "total_annual_vessel_capacity_tons"] > 0
                                ]
        export_port_ids = list(zip(export_ports_africa["id"].values.tolist(),
                        export_ports_africa[
                            "total_annual_vessel_capacity_tons"]))
        network_graph = create_mines_and_cities_to_port_network(
                                mines_df,mine_id_col,
                                un_pop_df,pop_id_col,
                                modes=["sea","intermodal","road","rail"],
                                intermodal_ports=export_ports_africa["id"].values.tolist(),
                                cargo_type=f"{cargo_type.lower().replace(' ','_')}",
                                port_to_land_capacity=export_port_ids,
                                distance_threshold=1500
                                )
        network_graph.to_parquet(os.path.join(
                                processed_data_path,
                                "infrastructure",
                                "network_graph.parquet"),index=False)
    write_file = False
    if write_file is True:
	    roads = gpd.read_parquet(os.path.join(
	                            processed_data_path,
	                            "infrastructure",
	                            "country_roads_edges.geoparquet"))
	    network_graph = pd.read_parquet(os.path.join(
                            processed_data_path,
                            "infrastructure",
                            "network_graph.parquet"))
	    network_graph = network_graph.drop_duplicates(subset=["id"],keep="first")
	    roads = pd.merge(roads,
	    				network_graph[["id","distance_km",
	    								"time_hr","land_border_cost_usd_tons",
	    								"gcost_usd_tons"]],
	    				how="left",on=["id"])
	    roads["speed_kmh"] = roads["distance_km"]/roads["time_hr"]
	    gpd.GeoDataFrame(roads,geometry="geometry",crs="EPSG:4326").to_file(os.path.join(
	                            processed_data_path,
	                            "infrastructure",
	                            "country_roads.gpkg"),layer="edges",driver="GPKG")

    od_matrix = pd.read_parquet(os.path.join(
                            output_data_path,
                            "flow_mapping",
                            "copper_flow_paths_2022_0.parquet"))
    print (od_matrix)
    od_matrix["find_node"] = od_matrix.progress_apply(lambda x:1 if "mine_58" in x["node_path"] else 0,axis=1)
    print (od_matrix[od_matrix["find_node"] == 1])

    # od_matrix = od_matrix[(od_matrix["origin_id"] == "mine_58") | (od_matrix["destination_id"] == "mine_58")] 
    # print (od_matrix[["mine_output_tons"]])
    # od_matrix.to_csv("test.csv",index=False)
    # network_graph = pd.read_parquet(os.path.join(
    #                         processed_data_path,
    #                         "infrastructure",
    #                         "network_graph.parquet"))
    # network_graph = network_graph[network_graph["mode"] == "mine"]
    # print (network_graph)
    # network_graph.to_csv("test.csv",index=False)
    # # network_graph["land_border_cost_usd_tons"] = network_graph["land_border_cost_usd_tons"].fillna(0)
    # # distance_factor = 0.103384681
    # # distance_factor = 0.12001543
    # # distance_factor = 0.12001543
    # # time_factor = 0.437579871
    # # time_factor = 0.057013382
    # # time_factor = 0.104254953
    # # network_graph["distance_time_cost"] = distance_factor*network_graph["distance_km"] + time_factor*network_graph["time_hr"]
    # # network_graph["gcost_usd_tons"] += network_graph["land_border_cost_usd_tons"]
    # # network_graph = network_graph[network_graph["mode"] != "rail"]
    # print (network_graph)
    # roads = gpd.read_parquet(os.path.join(
    #                         processed_data_path,
    #                         "infrastructure",
    #                         "africa_roads_edges.geoparquet"))
    # # print (roads)
    # rails = gpd.read_file(os.path.join(
    #                         processed_data_path,
    #                         "infrastructure",
    #                         "africa_railways_network.gpkg"
    #                             ), layer="edges"
    #                 )
    # # print (rails)
    # road_rail = gpd.GeoDataFrame(pd.concat(
    #                     [
    #                         roads[["id","geometry"]],
    #                         rails[["id","geometry"]]
    #                     ], axis=0,ignore_index=True
    #                     ),geometry="geometry",crs=roads.crs)

    # # print (road_rail)
    # network_graph = create_igraph_from_dataframe(network_graph, directed=True)
    # source = "mine_80"
    # target = "port311_land"
    # target = "port137_land"
    # target = "port278_land"
    # path = network_graph.get_shortest_paths(source, target, weights="gcost_usd_tons", output="epath")
    # edge_path = [network_graph.es[n]["id"] for n in path[0]]
    # route_df = road_rail[road_rail["id"].isin(edge_path)]
    # print ("Transport costs:",sum([network_graph.es[n]["gcost_usd_tons"] for n in path[0]]))
    # print ("Distance:",sum([network_graph.es[n]["distance_km"] for n in path[0]]))
    # print ("Time:",sum([network_graph.es[n]["time_hr"] for n in path[0]]))
    # # print ("Distance-Time Cost:",sum([network_graph.es[n]["distance_time_cost"] for n in path[0]]))
    # route_df.to_file(os.path.join(
    #                         processed_data_path,
    #                         "infrastructure",
    #                         "network_route.gpkg"
    #                             ),driver="GPKG")


    # print (edge_path)
    # network_graph.to_parquet(os.path.join(results_folder,
    #             f"global_network_{year}.parquet"),
    #             index=False)

    # network_graph[trade_ton_column] = 0
    # mine_routes, unassinged_routes, network_graph = od_flow_allocation_capacity_constrained(
    #                                     c_t_df,network_graph,
    #                                     trade_ton_column,"gcost_usd_tons",
    #                                     "distance_km","time_hr",
    #                                     "id",origin_id,
    #                                     destination_id)
    # network_graph = network_graph[network_graph[trade_ton_column] > 0]
    # network_graph.to_csv(os.path.join(results_folder,
    #             f"{reference_mineral}_total_flows_{year}_{percentile}.csv"),
    #             index=False)
    # del network_graph
    # if len(unassinged_routes) > 0:
    #     unassinged_routes = pd.concat(unassinged_routes,axis=0,ignore_index=True)
    #     if "geometry" in unassinged_routes.columns.values.tolist():
    #         unassinged_routes.drop("geometry",axis=1,inplace=True)
    #     unassinged_routes.to_csv(
    #             os.path.join(results_folder,
    #             f"{reference_mineral}_unassigned_flow_paths_{year}_{percentile}.csv"),
    #             index=False)
    # del unassinged_routes

    # # mine_routes = pd.read_parquet(
    # #             os.path.join(results_folder,f"{reference_mineral}.parquet"))
    # if len(mine_routes) > 0:
    #     mine_routes = pd.concat(mine_routes,axis=0,ignore_index=True)
    #     c_t_df.rename(columns={trade_ton_column:"initial_tonnage"},inplace=True)
    #     mine_routes = pd.merge(mine_routes,
    #                     c_t_df[[origin_id,destination_id,"initial_tonnage"]],
    #                     how="left",on=[origin_id,destination_id])
    #     # mine_routes[trade_usd_column] = mine_routes[
    #     #                                             trade_usd_column]*mine_routes[
    #     #                                             trade_ton_column]/mine_routes["initial_tonnage"]
    #     mine_routes.drop("initial_tonnage",axis=1,inplace=True)
    #     # mine_routes[c_t_df.columns.values.tolist()].to_csv("copper_ods_assigned.csv",index=False)
    #     del c_t_df

    #     # print (mine_routes[[origin_id,destination_id,trade_ton_column,"gcost_usd_tons"]])
    #     if "geometry" in mine_routes.columns.values.tolist():
    #         mine_routes.drop("geometry",axis=1,inplace=True)
    #     # mine_routes = add_node_paths(mine_routes,network_graph,"id","edge_path")
    #     # mine_routes.to_csv("test.csv")
    #     # mine_routes = get_land_and_sea_costs(mine_routes,
    #     #                                         network_graph.drop_duplicates(subset=["id"],keep="first"),
    #     #                                     "edge_path","node_path")
    #     # mine_routes.to_parquet("test.parquet",index=False)
    #     mine_routes["edge_path"] = mine_routes["edge_path"].map(tuple)
    #     mine_routes = mine_routes.drop_duplicates(
    #                             subset=["origin_id",
    #                                     "destination_id",
    #                                     "export_country_code",
    #                                     "reference_mineral",
    #                                     "initial_refined_stage",
    #                                     "final_refined_stage",
    #                                     "import_country_code",
    #                                     "edge_path"],
    #                             keep="first")
    #     mine_routes['edge_path'] = mine_routes['edge_path'].map(list)
    #     mine_routes = convert_port_routes(mine_routes,"edge_path","node_path")
    #     mine_routes[[origin_id,destination_id] + od_columns + [
    #                             "edge_path",
    #                             "node_path",
    #                             "full_edge_path",
    #                             "full_node_path",
    #                             "gcost_usd_tons_path",
    #                             "distance_km_path",
    #                             "time_hr_path",
    #                             "gcost_usd_tons"]].to_parquet(
    #             os.path.join(results_folder,f"{reference_mineral}_flow_paths_{year}_{percentile}.parquet"),
    #             index=False)


if __name__ == '__main__':
    CONFIG = load_config()
    try:
        reference_mineral = str(sys.argv[1])
        year = int(sys.argv[2])
        percentile = int(sys.argv[3])
    except IndexError:
        print("Got arguments", sys.argv)
        exit()
    main(CONFIG,reference_mineral,year,percentile)