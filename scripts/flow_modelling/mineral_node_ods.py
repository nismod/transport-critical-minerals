#!/usr/bin/env python
# coding: utf-8
"""This code finds the route between mines (in Africa) and all ports (outside Africa) 
"""
import sys
import os
import re
import json
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import igraph as ig
import geopandas as gpd
from utils import *
from transport_cost_assignment import *
from trade_functions import *
from tqdm import tqdm
tqdm.pandas()



def main(config,
        year,
        percentile,
        efficient_scale):
    incoming_data_path = config['paths']['incoming_data']
    processed_data_path = config['paths']['data']
    output_data_path = config['paths']['results']

    results_folder = os.path.join(output_data_path,"flow_mapping")
    if os.path.exists(results_folder) == False:
        os.mkdir(results_folder)

    mine_id_col = "id"
    od_combine_cols = [mine_id_col,"iso3","weight"]
    reference_minerals = ["copper","cobalt","manganese","lithium","graphite","nickel"]
    # reference_minerals = ["copper"]
    # cargo_type = "Dry bulk"
    cargo_type = "General cargo"
    trade_groupby_columns = [
                                "reference_mineral",
                                "export_country_code", 
                                "import_country_code", 
                                "export_continent",
                                "import_continent",
                                "initial_refined_stage",
                                "final_refined_stage"
                            ]

    # trade_value_columns = ["trade_value_thousandUSD","trade_quantity_tons"]
    trade_value_columns = ["trade_quantity_tons"]
    tons_column = "trade_quantity_tons"
    value_column = "trade_value_thousandUSD"
    od_columns = [
                    "reference_mineral",
                    "export_country_code",
                    "import_country_code",
                    "export_continent",
                    "import_continent",
                    "trade_type",
                    "initial_processing_stage",
                    "final_processing_stage",
                    "initial_processing_location",
                    "final_processing_location",
                    "initial_stage_production_tons",    
                    "final_stage_production_tons"
                ]
    ccg_countries = pd.read_csv(
                        os.path.join(processed_data_path,
                                "baci","ccg_country_codes.csv"))
    ccg_countries = ccg_countries[ccg_countries["ccg_country"] == 1]["iso_3digit_alpha"].values.tolist()

    # Population locations for urban cities
    if year == 2040:
        pop_year = 2035
    elif year == 2030:
        pop_year = 2030
    else:
        pop_year = 2020
    un_pop_df = gpd.read_file(os.path.join(processed_data_path,
                                    "admin_boundaries",
                                    "un_urban_population",
                                    "un_pop_df.gpkg"))
    un_pop_df = un_pop_df[un_pop_df["CONTINENT"] == "Africa"][["city_id","ISO_A3",str(pop_year)]]
    un_pop_df.rename(
                        columns={
                            "city_id":"id",
                            "ISO_A3":"iso3",
                            str(pop_year):"weight"},
                        inplace=True)
    un_pop_df["weight"
        ] = un_pop_df["weight"
        ]/un_pop_df.groupby(["iso3"])["weight"].transform("sum")

    un_pop_df["initial_processing_location"
        ] = un_pop_df["final_processing_location"
        ] = "city_demand"
    
    un_pop_df_copy = un_pop_df.copy()
    un_pop_df_copy["initial_processing_location"
        ] = un_pop_df_copy["final_processing_location"
        ] = "city_process"
    city_df = pd.concat([un_pop_df,un_pop_df_copy],axis=0,ignore_index=True)


    pop_years = [2020,2030,2035]

    # Get the global port network data for General Cargo transport
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
    all_ports_df["id"] = all_ports_df.progress_apply(lambda x:f"{x.id}_land",axis=1)
    all_ports_df["weight"] = 1
    all_ports_df["initial_processing_location"
        ] = all_ports_df["final_processing_location"
        ] = "port"
    del port_df, landlocked_port_df

    years = [2022,2030,2040]
    if year > 2022:
        file_name = f"baci_ccg_country_trade_breakdown_{year}_{percentile}_{efficient_scale}.csv"
    else:
        file_name = f"baci_ccg_country_trade_breakdown_{year}_{percentile}.csv"
    
    trade_df = pd.read_csv(os.path.join(
                            output_data_path,
                            "baci_trade_matrices",
                            file_name)
                            )
    combined_trade_df = []
    for reference_mineral in reference_minerals:
        t_df = trade_df[trade_df["reference_mineral"] == reference_mineral]
        # layer = f"{reference_mineral}_{percentile}"
        mines_df = gpd.read_file(
                            os.path.join(
                                processed_data_path,
                                "minerals",
                                "s_and_p_mines_current_and_future_estimates.gpkg"),
                            layer=f"{reference_mineral}_{percentile}")
        # mines_df["reference_mineral"] = reference_mineral
        mines_df.rename(columns={"ISO_A3":"iso3","mine_id":mine_id_col},inplace=True)
        mines_df["weight"] = mines_df[str(year)]
        # if year > 2022:
        #     layer = f"{reference_mineral}_{percentile}"
        # else:
        #     layer = f"{reference_mineral}"
        # Mine locations in Africa with the mineral tonnages
        # if year == 2022:
        #     mines_df = gpd.read_file(
        #                     os.path.join(
        #                         processed_data_path,
        #                         "minerals",
        #                         "ccg_mines_est_production.gpkg"))
        #     if mine_id_col not in mines_df.columns.values.tolist():
        #         mines_df[mine_id_col] = mines_df.index.values.tolist()
        #         mines_df[mine_id_col] = mines_df.progress_apply(lambda x:f"mine_{x[mine_id_col]}",axis=1)
        #     if f"{reference_mineral}_processed_ton" not in mines_df.columns.values.tolist():
        #         mines_df[f"{reference_mineral}_processed_ton"] = 0
        #     if f"{reference_mineral}_unprocessed_ton" not in mines_df.columns.values.tolist():
        #         mines_df[f"{reference_mineral}_unprocessed_ton"] = 0

        #     mines_df[f"{reference_mineral}"] = mines_df[f"{reference_mineral}"].astype(int)
        #     mines_df = mines_df[mines_df[f"{reference_mineral}"] == 1]
        #     # mines_df["reference_mineral"] = reference_mineral
        #     mines_df.rename(columns={"country_code":"iso3"},inplace=True)
            
        #     # mines_df["initial_processing_location"
        #     #     ] = mines_df["final_processing_location"] = "mine"
        #     # mines_df["final_refined_stage"] = mines_df[f"highest_stage_{reference_mineral}"]
        #     # mines_df["initial_refined_stage"] = np.where(
        #     #                                     mines_df[f"{reference_mineral}_processed_ton"] > 0,
        #     #                                     mine_final_refined_stage,
        #     #                                     mine_initial_refined_stage)
        #     # mines_df["final_refined_stage"] = np.where(
        #     #                                     mines_df[f"{reference_mineral}_processed_ton"] > 0,
        #     #                                     mine_final_refined_stage,
        #     #                                     mine_initial_refined_stage)
        #     mines_df["weight"] = mines_df[f"{reference_mineral}_processed_ton"] + mines_df[f"{reference_mineral}_unprocessed_ton"]
        #     # mines_df[original_tons_column] = mines_df[mine_tons_column].copy()
        #     # refined_df = mines_df[mines_df["final_refined_stage"] == mine_final_refined_stage]
        #     # mines_df["location_type"] = "mine"
        #     # location_df.append(mines_df[final_columns])
        # elif year > 2022:
        #     mines_df = gpd.read_file(
        #                     os.path.join(
        #                         processed_data_path,
        #                         "minerals",
        #                         "s_and_p_mines_estimates.gpkg"),
        #                     layer=f"{reference_mineral}_{percentile}")
        #     # mines_df["reference_mineral"] = reference_mineral
        #     mines_df.rename(columns={"ISO_A3":"iso3","mine_id":mine_id_col},inplace=True)
        #     mines_df["weight"] = mines_df[str(year)]
        
        mines_df = mines_df[mines_df["weight"] > 0]
        mines_df["weight"
            ] = mines_df["weight"
            ]/mines_df.groupby(["iso3"])["weight"].transform("sum")
        mines_df["initial_processing_location"
                ] = mines_df["final_processing_location"] = "mine"
        # mines_df = mines_df[mines_df["weight"] > 0]
        
        origins = pd.concat(
                        [city_df,all_ports_df,mines_df],
                        axis=0,ignore_index=True)[[mine_id_col,"iso3",
                                    "initial_processing_location","weight"]]
        origins.rename(
                        columns={
                            mine_id_col:"origin_id",
                            "iso3":"export_country_code",
                            "weight":"origin_weight"
                            },
                        inplace=True)
        origins = origins[origins["origin_weight"] > 0]

        destinations = origins.copy()
        destinations.rename(
                        columns={
                            "origin_id":"destination_id",
                            "export_country_code":"import_country_code",
                            "initial_processing_location":"final_processing_location",
                            "origin_weight":"destination_weight"
                            },
                        inplace=True)
        destinations = destinations[destinations["destination_weight"] > 0]
        t_df = pd.merge(t_df,
                        origins,
                        how="left",
                        on=["export_country_code","initial_processing_location"]).fillna(0)
        t_df = pd.merge(t_df,
                        destinations,
                        how="left",
                        on=["import_country_code","final_processing_location"]).fillna(0)

        t_df["initial_stage_production_tons"
            ] = t_df["initial_stage_production_tons"
            ]*t_df["origin_weight"]*t_df["destination_weight"]

        t_df["final_stage_production_tons"
            ] = t_df["final_stage_production_tons"
            ]*t_df["origin_weight"]*t_df["destination_weight"]

        combined_trade_df.append(
            t_df[["origin_id","destination_id"]+od_columns])
        

    if year > 2022:
        file_name_full = f"mining_city_node_level_ods_full_{year}_{percentile}_{efficient_scale}.csv"
        file_name = f"mining_city_node_level_ods_{year}_{percentile}_{efficient_scale}.csv"
    else:
        file_name_full = f"mining_city_node_level_ods_full_{year}_{percentile}.csv"
        file_name = f"mining_city_node_level_ods_{year}_{percentile}.csv"

    combined_trade_df = pd.concat(combined_trade_df,axis=0,ignore_index=True)
    combined_trade_df = combined_trade_df[combined_trade_df["final_stage_production_tons"]>0]
    combined_trade_df.to_csv(os.path.join(
                            results_folder,
                            file_name_full),index=False)

    tons_total = combined_trade_df["final_stage_production_tons"].sum()
    combined_trade_df = truncate_by_threshold(combined_trade_df,
                                        flow_column="final_stage_production_tons",
                                        threshold=0.9999)
    tons_truncated = combined_trade_df["final_stage_production_tons"].sum()
    # value_truncated = combined_trade_df["mine_output_thousandUSD"].sum()

    print (f"{tons_total:,.0f} tons before and {tons_truncated:,.0f} after with difference:",tons_total - tons_truncated)
    # print (f"{1000*value_total:,.0f} USD before and {1000*value_truncated:,.0f} after with difference:",1000*(value_total - value_truncated))
    
    combined_trade_df.to_csv(os.path.join(
                            results_folder,
                            file_name),index=False)


if __name__ == '__main__':
    CONFIG = load_config()
    try:
        year = int(sys.argv[1])
        percentile = str(sys.argv[2])
        efficient_scale = str(sys.argv[3])
    except IndexError:
        print("Got arguments", sys.argv)
        exit()
    main(CONFIG,
        year,
        percentile,
        efficient_scale)