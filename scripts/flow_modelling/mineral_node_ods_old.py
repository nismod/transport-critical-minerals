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
    reference_minerals = ["copper","cobalt","manganese","lithium","graphite","nickel"]
    # reference_minerals = ["copper"]
    data_type = {"initial_refined_stage":"str","final_refined_stage":"str"}
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
                    "initial_refined_stage",
                    "final_refined_stage",
                    "export_country_code",
                    "import_country_code",
                    "export_continent",
                    "import_continent",
                    "mine_output_tons",    
                    # "mine_output_thousandUSD"
                ]
    """Step 1: Get the input datasets
    """
    # # Mine locations in Africa with the copper tonages
    # mines_df = gpd.read_file(os.path.join(processed_data_path,
    #                                 "minerals",
    #                                 "copper_mines_tons.gpkg"))
    # mines_df[mine_id_col] = mines_df.progress_apply(lambda x:f"mine_{x[mine_id_col]}",axis=1)
    # # Select only tonnages > 0
    # mines_df = mines_df[mines_df["mine_output_approx_copper"] > 0]
    # mine_isos = list(set(mines_df["shapeGroup_primary_admin0"].values.tolist()))

    # island_isos = pd.read_csv(os.path.join(
    #                   processed_data_path,
    #                   "baci",
    #                   "ccg_country_codes.csv")
    #               )
    # island_isos = island_isos[
    #               island_isos["continent"] == "Africa_island"
    #               ]["iso_3digit_alpha"].values.tolist()
    # Read the finalised version of the BACI trade data
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
        pop_year = 2022
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
    all_ports_df["weight"] = 1
    all_ports_df["initial_processing_location"
        ] = all_ports_df["final_processing_location"
        ] = "port"
    del port_df, landlocked_port_df

    years = [2022,2030,2040]
    # years = [2022]
    # pop_years = [2020]
    # trade_df = pd.read_csv(os.path.join(output_data_path,
    #                             "baci_trade_matrices",
    #                             f"baci_ccg_country_level_trade_{year}.csv"),
    #                         dtype=data_type)
    if year > 2022:
        file_name = f"baci_ccg_country_trade_breakdown_{year}_{percentile}_{efficient_scale}.csv"
    else:
        file_name = f"baci_ccg_country_trade_breakdown_{year}.csv"
    
    trade_df = pd.read_csv(os.path.join(
                            output_data_path,
                            "baci_trade_matrices",
                            file_name)
                            )
    intermediate_trade_df = []
    combined_trade_df = []
    for reference_mineral in reference_minerals:
        t_df = trade_df[trade_df["reference_mineral"] == reference_mineral]
        # t_df = t_df.groupby(trade_groupby_columns)[trade_value_columns].sum().reset_index()
        # print (t_df)
        # t_df["trade_quantity_tons_fraction"] = t_df[tons_column
        #                             ]/t_df.groupby(
        #                                 ["export_country_code",
        #                                 "initial_refined_stage",
        #                                 "final_refined_stage"]
        #                             )[tons_column].transform('sum')
        if year > 2022:
            layer = f"{reference_mineral}_{percentile}"
        else:
            layer = f"{reference_mineral}"
        # Mine locations in Africa with the mineral tonnages
        if year == 2022:
            mines_df = gpd.read_file(
                            os.path.join(
                                processed_data_path,
                                "minerals",
                                "ccg_mines_est_production.gpkg"))
            if mine_id_col not in mines_df.columns.values.tolist():
                mines_df[mine_id_col] = mines_df.index.values.tolist()
                mines_df[mine_id_col] = mines_df.progress_apply(lambda x:f"mine_{x[mine_id_col]}",axis=1)
            if f"{reference_mineral}_processed_ton" not in mines_df.columns.values.tolist():
                mines_df[f"{reference_mineral}_processed_ton"] = 0
            if f"{reference_mineral}_unprocessed_ton" not in mines_df.columns.values.tolist():
                mines_df[f"{reference_mineral}_unprocessed_ton"] = 0

            mines_df[f"{reference_mineral}"] = mines_df[f"{reference_mineral}"].astype(int)
            mines_df = mines_df[mines_df[f"{reference_mineral}"] == 1]
            # mines_df["reference_mineral"] = reference_mineral
            mines_df.rename(columns={"country_code":"iso3"},inplace=True)
            
            # mines_df["initial_processing_location"
            #     ] = mines_df["final_processing_location"] = "mine"
            # mines_df["final_refined_stage"] = mines_df[f"highest_stage_{reference_mineral}"]
            # mines_df["initial_refined_stage"] = np.where(
            #                                     mines_df[f"{reference_mineral}_processed_ton"] > 0,
            #                                     mine_final_refined_stage,
            #                                     mine_initial_refined_stage)
            # mines_df["final_refined_stage"] = np.where(
            #                                     mines_df[f"{reference_mineral}_processed_ton"] > 0,
            #                                     mine_final_refined_stage,
            #                                     mine_initial_refined_stage)
            mines_df["weight"] = mines_df[f"{reference_mineral}_processed_ton"] + mines_df[f"{reference_mineral}_unprocessed_ton"]
            # mines_df[original_tons_column] = mines_df[mine_tons_column].copy()
            # refined_df = mines_df[mines_df["final_refined_stage"] == mine_final_refined_stage]
            # mines_df["location_type"] = "mine"
            # location_df.append(mines_df[final_columns])
        elif year > 2022:
            mines_df = gpd.read_file(
                            os.path.join(
                                processed_data_path,
                                "minerals",
                                "s_and_p_mines_estimates.gpkg"),
                            layer=f"{reference_mineral}_{percentile}")
            # mines_df["reference_mineral"] = reference_mineral
            mines_df.rename(columns={"mine_id":mine_id_col},inplace=True)
            mines_df["weight"] = mines_df[str(year)]
            # mines_df["initial_refined_stage"] = mine_initial_refined_stage
            # mines_df["final_refined_stage"] = mine_final_refined_stage
            # mines_df["location_type"] = "mine"
            # mines_df[original_tons_column] = mines_df[mine_tons_column].copy()
            # conv_factor_df = pr_conv_factors_df[pr_conv_factors_df["reference_mineral"] == reference_mineral]
            # conversion_factor = conv_factor_df[
            #                             conv_factor_df["final_refined_stage"] == mine_final_refined_stage
            #                             ][conversion_factor_column].values[0]/conv_factor_df[
            #                             conv_factor_df["final_refined_stage"] == mine_initial_refined_stage
            #                             ][conversion_factor_column].values[0]

            # mines_df[mine_tons_column] = np.where(
            #                                 mines_df["final_refined_stage"] != mines_df["initial_refined_stage"],
            #                                 1.0*mines_df[mine_tons_column]/conversion_factor,
            #                                 mines_df[mine_tons_column])
            # mines_df["final_refined_stage"] = np.where(
            #                                 mines_df["final_refined_stage"] == mine_initial_refined_stage,
            #                                 mine_final_refined_stage,
            #                                 mines_df["final_refined_stage"])
            # location_df.append(mines_df[final_columns])
            # location_df.append(refined_df[final_columns])
        mines_df["weight"
            ] = mines_df["weight"
            ]/mines_df.groupby(["iso3"])["weight"].transform("sum")
        mines_df["initial_processing_location"
                ] = mines_df["final_processing_location"] = "mine"
        mines_df = gpd.read_file(
                                os.path.join(output_data_path,
                                "location_outputs",
                                f"mine_city_tons_{year}.gpkg"),
                                layer=layer,dtype=data_type)
        # mines_df["final_refined_stage"] = mines_df["final_refined_stage"].astype(str)
        # print (mines_df)
        # Select only tonnages > 0
        mines_df = mines_df[mines_df[f"{reference_mineral}_final_tons"] > 0]
        mine_isos = list(set(mines_df["ISO_A3"].values.tolist()))
        africa_island_isos = list(
                                    set(
                                        t_df[
                                                t_df["import_continent"] == "Africa_island"
                                                ]["import_country_code"].values.tolist()
                                        )
                                )
        islands_with_mines = list(set(africa_island_isos).intersection(mine_isos))
        islands_without_mines = [mn for mn in island_isos if mn not in islands_with_mines]
        print (reference_mineral,year)
        print (islands_with_mines)
        print (islands_without_mines)
        
        mine_refined_df = pd.merge(
                            mines_df,
                            t_df,
                            how="left",
                            left_on=["ISO_A3",
                                    "initial_refined_stage",
                                    "final_refined_stage",
                                    "reference_mineral"],
                            right_on=["export_country_code",
                                    "initial_refined_stage",
                                    "final_refined_stage",
                                    "reference_mineral"]
                            )
        mine_refined_df.rename(columns={mine_id_col:"origin_id"},inplace=True)
        mine_refined_df["mine_output_tons"
                ] = mine_refined_df[f"{reference_mineral}_final_tons"
                ]*mine_refined_df["trade_quantity_tons_fraction"]
        mine_refined_df.drop("geometry",axis=1,inplace=True)
        intermediate_trade_df.append(mine_refined_df)
        # mine_refined_df.to_csv("text.csv",index=False)

        if len(islands_with_mines) == 0:
            port_export_df = mine_refined_df[mine_refined_df["import_continent"] != "Africa"]
        else:
            port_export_df = mine_refined_df[
                                    (
                                        ~mine_refined_df["import_continent"].isin(["Africa","Africa_island"])
                                    ) | (mine_refined_df["import_country_code"].isin(islands_without_mines))
                                    ]

        if len(port_export_df.index) > 0:
            port_export_df = pd.merge(port_export_df,
                                all_ports_df,
                                how="left",left_on=["import_country_code"],right_on=["iso3"])
            # port_export_df.to_csv("test.csv",index=False)
            port_export_df["destination_id"] = port_export_df.progress_apply(lambda x:f"{x.id}_land",axis=1)
            combined_trade_df.append(port_export_df[["origin_id","destination_id"]+od_columns])

        if len(islands_with_mines) > 0:
            land_export_df = mine_refined_df[
                                (
                                    mine_refined_df["import_continent"] == "Africa") | (
                                    mine_refined_df["import_country_code"].isin(islands_with_mines)
                                )
                            ]
        else:
            land_export_df = mine_refined_df[mine_refined_df["import_continent"] == "Africa"]

        if len(land_export_df.index) > 0:
            # land_refined_df = mines_df[mines_df["initial_refined_stage"] != '1']
            # land_refined_df[
            #     "mine_processed_quantity_tons_fraction"
            #     ] = land_refined_df[f"{reference_mineral}_final_tons"
            #                         ]/land_refined_df.groupby(
            #                             [["ISO_A3","initial_refined_stage"]]
            #                         )[f"{reference_mineral}_final_tons"].transform('sum')
            # land_refined_df.rename(
            #         columns={mine_id_col:"destination_id",
            #                 "ISO_A3":"import_country_code"},
            #         inplace=True)
            # land_export_df = pd.merge(land_export_df,land_refined_df[["import_country_code",
            #                     "destination_id","mine_processed_quantity_tons_fraction"]],
            #                     how="left",on=["import_country_code"]).fillna(0)

            land_refined_df = un_pop_df[["city_id","ISO_A3",str(pop_year)]]
            land_refined_df["mine_processed_quantity_tons_fraction"] = land_refined_df[str(pop_year)
                                    ]/land_refined_df.groupby(
                                        ["ISO_A3"]
                                    )[str(pop_year)].transform('sum')
            land_refined_df.rename(
                    columns={"city_id":"destination_id",
                            "ISO_A3":"import_country_code"},
                    inplace=True)

            land_export_df = pd.merge(land_export_df,land_refined_df[["import_country_code",
                                    "destination_id","mine_processed_quantity_tons_fraction"]],
                                    how="left",on=["import_country_code"]).fillna(0)
            land_export_df["mine_output_tons"
                    ] = land_export_df["mine_output_tons"]*land_export_df["mine_processed_quantity_tons_fraction"]
            combined_trade_df.append(land_export_df[land_export_df["destination_id"] != 0][["origin_id","destination_id"]+od_columns])


        """Create the reference commodity level OD first
        """
        # processing_stages = list(set(trade_df["final_refined_stage"].values.tolist()))
        # if idx > 0:
        #     print (reference_mineral,processing_stages)
        #     print ("mines:",list(set(mines_df["final_refined_stage"].values.tolist())))
        # for pr_st in processing_stages:
        #     t_df = trade_df[trade_df["final_refined_stage"] == pr_st]
        #     t_df["trade_quantity_tons_fraction"] = t_df[tons_column
        #                             ]/t_df.groupby(
        #                                 ["export_country_code"]
        #                             )[tons_column].transform('sum')
        #     mine_refined_df = mines_df[mines_df["final_refined_stage"] == pr_st]
        #     # city_refined_df = cities_df[cities_df["final_refined_stage"] == pr_st]
        #     mine_refined_df = pd.merge(mine_refined_df,
        #                                     t_df,
        #                                     how="left",
        #                                     left_on=["ISO_A3","initial_refined_stage","final_refined_stage","reference_mineral"],
        #                                     right_on=["export_country_code","initial_refined_stage","final_refined_stage","reference_mineral"])
        #     mine_refined_df.rename(columns={mine_id_col:"origin_id"},inplace=True)
        #     # if len(mine_refined_df) > 0:
        #     #     mine_refined_df = pd.merge(mine_refined_df,
        #     #                                 t_df,
        #     #                                 how="left",
        #     #                                 left_on=["ISO_A3","final_refined_stage"],
        #     #                                 right_on=["export_country_code","final_refined_stage"])
        #     #     mine_refined_df.rename(columns={mine_id_col:"origin_id"},inplace=True)
        #     # else:
        #     #     mine_refined_df = pd.merge(city_refined_df,
        #     #                                 t_df,
        #     #                                 how="left",
        #     #                                 left_on=["ISO_A3","final_refined_stage"],
        #     #                                 right_on=["export_country_code","final_refined_stage"])
        #     #     mine_refined_df.rename(columns={"city_id":"origin_id",
        #     #                                 "city_output_approx_copper":"mine_output_approx_copper"},
        #     #                                 inplace=True)

        #     # mine_refined_df[
        #     #         "corrected_tons"] = mine_refined_df[
        #     #         "mine_output_approx_copper"]*mine_refined_df.groupby(
        #     #                                 ["export_country_code"]
        #     #                             )["trade_quantity_tons"].transform('sum')/(mine_refined_df.groupby(
        #     #                                 ["export_country_code"]
        #     #                             )["mine_output_approx_copper"].transform('sum'))

        #     # mine_refined_df.drop("geometry",axis=1,inplace=True)
        #     # mine_refined_df.to_csv(f"test_{year}.csv")
        #     mine_refined_df["mine_output_tons"
        #         ] = mine_refined_df[f"{reference_mineral}_final_tons"
        #         ]*mine_refined_df["trade_quantity_tons_fraction"]
        #     # mine_refined_df["mine_output_thousandUSD"
        #     #     ] = mine_refined_df["mine_output_tons"
        #     #     ]*mine_refined_df["trade_value_thousandUSD"]/mine_refined_df["trade_quantity_tons"]
        #     mine_refined_df.drop("geometry",axis=1,inplace=True)
        #     # mine_refined_df.to_csv(f"{mineral_class}_{refined_type}_mine_level.csv",index=False)
        #     # mine_refined_df.drop_duplicates(
        #     #     subset=mine_id_col,keep="first").to_csv(f"{mineral_class}_{refined_type}_mine_output.csv",index=False)
        #     # mine_refined_df.rename(columns={mine_id_col:"origin_id"},inplace=True)
        #     # mine_refined_df["reference_mineral"] = mineral_class
        #     """Get the flows being exported out of Africa to ports around the world
        #     """
        #     if len(islands_with_mines) == 0:
        #         port_export_df = mine_refined_df[mine_refined_df["import_continent"] != "Africa"]
        #     else:
        #         port_export_df = mine_refined_df[
        #                                 (
        #                                     ~mine_refined_df["import_continent"].isin(["Africa","Africa_island"])
        #                                 ) | (mine_refined_df["import_country_code"].isin(islands_without_mines))
        #                                 ]

        #     if len(port_export_df.index) > 0:
        #         port_export_df = pd.merge(port_export_df,
        #                             all_ports_df,
        #                             how="left",left_on=["import_country_code"],right_on=["iso3"])
        #         port_export_df["destination_id"] = port_export_df.progress_apply(lambda x:f"{x.id}_land",axis=1)
        #         # port_export_df.drop("geometry",axis=1,inplace=True)
        #         # port_export_df.to_csv(f"{mineral_class}_{refined_type}.csv",index=False)
        #         combined_trade_df.append(port_export_df[["origin_id","destination_id"]+od_columns])

        #     if len(islands_with_mines) > 0:
        #         land_export_df = mine_refined_df[
        #                             (
        #                                 mine_refined_df["import_continent"] == "Africa") | (
        #                                 mine_refined_df["import_country_code"].isin(islands_with_mines)
        #                             )
        #                         ]
        #     else:
        #         land_export_df = mine_refined_df[mine_refined_df["import_continent"] == "Africa"]

        #     if len(land_export_df.index) > 0:
        #         if pr_st == 1:
        #             land_refined_df = mines_df[mines_df["final_refined_stage"] == 3]
        #             land_refined_df["mine_processed_quantity_tons_fraction"] = land_refined_df["mine_output_approx_copper"
        #                                     ]/land_refined_df.groupby(
        #                                         ["shapeGroup_primary_admin0"]
        #                                     )["mine_output_approx_copper"].transform('sum')
        #             land_refined_df.rename(
        #                     columns={mine_id_col:"destination_id",
        #                             "shapeGroup_primary_admin0":"import_country_code"},
        #                     inplace=True)
        #         else:
        #             land_refined_df = un_pop_df[["city_id","ISO_A3",str(pop_year)]]
        #             land_refined_df["mine_processed_quantity_tons_fraction"] = land_refined_df[str(pop_year)
        #                                     ]/land_refined_df.groupby(
        #                                         ["ISO_A3"]
        #                                     )[str(pop_year)].transform('sum')
        #             land_refined_df.rename(
        #                     columns={"city_id":"destination_id",
        #                             "ISO_A3":"import_country_code"},
        #                     inplace=True)

        #         land_export_df = pd.merge(land_export_df,land_refined_df[["import_country_code",
        #                                 "destination_id","mine_processed_quantity_tons_fraction"]],
        #                                 how="left",on=["import_country_code"]).fillna(0)
        #         land_export_df["mine_output_tons"
        #                 ] = land_export_df["mine_output_tons"]*land_export_df["mine_processed_quantity_tons_fraction"]
        #         # land_export_df["mine_output_thousandUSD"
        #         #         ] = land_export_df["mine_output_thousandUSD"]*land_export_df["mine_processed_quantity_tons_fraction"]
        #         # land_export_df.to_csv(f"{mineral_class}_{refined_type}_mine_to_mine.csv",index=False)
        #         combined_trade_df.append(land_export_df[land_export_df["destination_id"] != 0][["origin_id","destination_id"]+od_columns])

    if year > 2022:
        file_name_full = f"mining_city_node_level_ods_full_{year}_{percentile}.csv"
        file_name = f"mining_city_node_level_ods_{year}_{percentile}.csv"
    else:
        file_name_full = f"mining_city_node_level_ods_full_{year}.csv"
        file_name = f"mining_city_node_level_ods_{year}.csv"

    combined_trade_df = pd.concat(combined_trade_df,axis=0,ignore_index=True)
    intermediate_trade_df = pd.concat(intermediate_trade_df,axis=0,ignore_index=True)

    intermediate_trade_df.to_csv(os.path.join(
                            results_folder,
                            f"intermediate_{file_name_full}"),index=False)
    combined_trade_df.to_csv(os.path.join(
                            results_folder,
                            file_name_full),index=False)

    tons_total = combined_trade_df["mine_output_tons"].sum()
    # value_total = combined_trade_df["mine_output_thousandUSD"].sum()

    combined_trade_df = truncate_by_threshold(combined_trade_df,
                                        flow_column="mine_output_tons",
                                        threshold=0.9999)
    tons_truncated = combined_trade_df["mine_output_tons"].sum()
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
        percentile = int(sys.argv[2])
        efficient_scale = str(sys.argv[3])
    except IndexError:
        print("Got arguments", sys.argv)
        exit()
    main(CONFIG,
        year,
        percentile,
        efficient_scale)