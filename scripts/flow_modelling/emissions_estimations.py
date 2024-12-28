#!/usr/bin/env python
# coding: utf-8
"""This code finds the optimal locations of processing in Africa 
"""
import sys
import os
import re
import json
import pandas as pd
import numpy as np
pd.options.mode.copy_on_write = True
import igraph as ig
import geopandas as gpd
from collections import defaultdict
from utils import *
from transport_cost_assignment import *
from trade_functions import * 
from tqdm import tqdm
tqdm.pandas()

def add_mines_remaining_tonnages(df,mines_df,year,metal_factor):
    m_df = df[
                (
                    df["initial_processing_location"] == "mine"
                ) & (
                    df["initial_processing_stage"] == 0.0
                )
            ]
    m_df = m_df.groupby(
                        [
                        "reference_mineral",
                        "export_country_code",
                        "initial_processing_stage",
                        "initial_processing_location",
                        "origin_id"]
                        ).agg(dict([(c,"sum") for c in ["initial_stage_production_tons"]])).reset_index() 
    m_df = pd.merge(
                m_df,
                mines_df[["id",str(year)]],
                how="left",left_on=["origin_id"],
                right_on=["id"]).fillna(0)
    m_df["initial_stage_production_tons"] = m_df[str(year)] - m_df["initial_stage_production_tons"]
    m_df = m_df[m_df["initial_stage_production_tons"] > 0]
    if len(m_df.index) > 0:
        m_df["final_processing_stage"] = 1.0
        m_df["final_stage_production_tons"] = m_df["initial_stage_production_tons"]/metal_factor
        m_df["trade_type"] = "Other"
        m_df["import_country_code"] = m_df["export_country_code"]
        m_df.drop(["id",str(year)],axis=1,inplace=True)
        df = pd.concat([df,m_df],axis=0,ignore_index=True)

    return df

def add_isos_to_flows(flows_dataframe,nodes_dataframe,nodes_id_column="nid",nodes_iso_column="iso3"):
    for ft in ["from","to"]:
        flows_dataframe = pd.merge(
                                flows_dataframe,
                                nodes_dataframe[[nodes_id_column,nodes_iso_column]],
                                how="left",left_on=f"{ft}_id",right_on=nodes_id_column
                                ).fillna(0)
        flows_dataframe.rename(columns={nodes_iso_column:f"{ft}_iso_a3"},inplace=True)
        flows_dataframe.drop(nodes_id_column,axis=1,inplace=True)

    flows_dataframe = flows_dataframe[
                            ~(
                                (
                                    flows_dataframe["from_iso_a3"] == 0
                                ) & (
                                    flows_dataframe["to_iso_a3"] == 0
                                )
                            )
                            ]
    flows_dataframe["from_iso_a3"
        ] = np.where(
                    flows_dataframe["from_iso_a3"] == 0,
                    flows_dataframe["to_iso_a3"],
                    flows_dataframe["from_iso_a3"]
                    )
    flows_dataframe["to_iso_a3"
        ] = np.where(
                    flows_dataframe["to_iso_a3"] == 0,
                    flows_dataframe["from_iso_a3"],
                    flows_dataframe["to_iso_a3"]
                    )
    return flows_dataframe

def get_columns(
                    flows_dataframe,
                    location_types,
                    trade_types,
                    country_iso,
                    trade_ton_column="final_stage_production_tons"
                ):
    total_columns = defaultdict(list)
    export_columns = defaultdict(list)
    import_columns = defaultdict(list)
    for idx, (ly,ty) in enumerate(zip(location_types,trade_types)):
        tag_columns = [c for c in flows_dataframe.columns.values.tolist() if ly in c and trade_ton_column in c]
        for tg in tag_columns:
            tag_stage = tg.split(ly)[0]
            st = tag_stage.split("_")[-1]
            total_columns[st].append(tag_stage + f"_{ty}")
            if ty == "export" and f"_{country_iso}" in tg:
                export_columns[st].append(tg)
            elif ty == "import" and f"_{country_iso}" in tg:
                import_columns[st].append(tg)
            elif ty == "inter" and f"_{country_iso}" in tg:
                tag_stage = tg.split(ly)[-1]
                o_d = tag_stage.split("_")
                if o_d[0] == country_iso:
                    export_columns[st].append(tg)
                elif o_d[1] == country_iso:
                    import_columns[st].append(tg)

    return {"total_tonkm":total_columns,"export_tonkm":export_columns,"import_tonkm":import_columns}


def main(config,year,percentile,efficient_scale,country_case,constraint):
    incoming_data_path = config['paths']['incoming_data']
    processed_data_path = config['paths']['data']
    output_data_path = config['paths']['results']

    input_folder = os.path.join(output_data_path,"node_edge_flows")
    results_folder = os.path.join(output_data_path,"carbon_emissions_summaries")
    if os.path.exists(results_folder) == False:
        os.mkdir(results_folder)

    flow_results_folder = os.path.join(output_data_path,"carbon_emissions_flows")
    if os.path.exists(flow_results_folder) == False:
        os.mkdir(flow_results_folder)

    baseline_year = 2022
    if year == baseline_year:
        file_name = f"carbon_emission_totals_{year}_{percentile}"
    else:
        file_name = f"carbon_emission_totals_{year}_{percentile}_{efficient_scale}"
    """Step 1: Get the input datasets
    """
    global_epsg = 4326
    reference_minerals = ["graphite","lithium","cobalt","manganese","nickel","copper"]
    trade_ton_column = "final_stage_production_tons"
    index_columns = ["id","from_id","to_id","from_iso_a3","to_iso_a3","mode","geometry"]

    carbon_emission_df = pd.read_excel(
                                os.path.join(
                                    processed_data_path,
                                    "transport_costs",
                                    "carbon_emission_factors.xlsx")
                                )
    carbon_emission_df["CO2_pertonkm"
        ] = 1.0e-5*(
                    carbon_emission_df["spec_energ_consump"]*carbon_emission_df["Density_kg"]
                    )/(
                        carbon_emission_df["veh_wt_tons"]*carbon_emission_df["load_factor"]
                    )
    country_codes_and_projections = pd.read_excel(
                        os.path.join(
                            processed_data_path,
                            "local_projections.xlsx")
                        )
    countries = country_codes_and_projections["iso3"].values.tolist()

    global_boundaries = gpd.read_file(os.path.join(processed_data_path,
                                    "admin_boundaries",
                                    "gadm36_levels_gpkg",
                                    "gadm36_levels_continents.gpkg"))
    global_boundaries = global_boundaries[global_boundaries["ISO_A3"].isin(countries)]

    nodes_df = add_geometries_to_flows([],
                                merge_column="id",
                                modes=["rail","road"],
                                layer_type="nodes",merge=False)
    nodes_df.rename(columns={"id":"nid"},inplace=True)
    nodes_df = nodes_df[["nid","iso3"]]
    location_types = ["_origin_","_destination_","_inter_"]
    trade_types = ["export","import","inter"]
    all_flows = []
    sum_cols = []
    reassemble_flows = []
    for reference_mineral in reference_minerals:
        # Find year locations
        if year == baseline_year:
            layer_name = f"{reference_mineral}_{percentile}"
        else:
            layer_name = f"{reference_mineral}_{percentile}_{efficient_scale}"
        flows_gdf = gpd.read_parquet(
                            os.path.join(
                                input_folder,
                                f"edges_flows_{layer_name}_{year}_{country_case}_{constraint}.geoparquet"))
        flows_gdf = flows_gdf[flows_gdf["mode"].isin(["road","rail"])]
        flows_gdf = add_isos_to_flows(flows_gdf,nodes_df)
        # tag_columns = []
        # stages = list(set([float(c.split("_")[-1]) for c in stages]))
        # columns = [f"{reference_mineral}_{trade_ton_column}_{st}" for st in stages]
        # sum_cols += columns
        # flows_gdf = pd.merge(flows_gdf,carbon_emission_df[["mode","CO2_pertonkm"]],how="left",on=["mode"]).fillna(0)
        # flows_gdf[columns] = flows_gdf[columns].multiply(flows_gdf["CO2_pertonkm"],axis="index")
        # flows_gdf = flows_gdf[index_columns + columns]
        for row in country_codes_and_projections.itertuples():
            boundary_df = global_boundaries[global_boundaries["ISO_A3"] == row.iso3]
            boundary_df = boundary_df.to_crs(epsg=row.projection_epsg)
            df = flows_gdf[(flows_gdf["from_iso_a3"] == row.iso3) | (flows_gdf["to_iso_a3"] == row.iso3)]
            if len(df.index) > 0:
                df = df.to_crs(epsg=row.projection_epsg)
                df["length_km"] = 0.001*df.geometry.length
                df["reference_mineral"] = reference_mineral
                df["iso3"] = row.iso3
                all_columns = get_columns(df,location_types,trade_types,row.iso3)
                columns = []
                for k,v in all_columns.items():
                    for vk,vv in v.items():
                        df[f"{reference_mineral}_{k}_{vk}"] = df[list(set(vv))].sum(axis=1)
                        df[f"{reference_mineral}_{k}_{vk}"] = df[f"{reference_mineral}_{k}_{vk}"]*df["length_km"]
                        columns.append(f"{reference_mineral}_{k}_{vk}")
                        sum_cols.append(f"{reference_mineral}_{k}_{vk}")                        
                iso_df = df[df["from_iso_a3"] == df["to_iso_a3"]]
                reassemble_flows.append(iso_df.to_crs(epsg=global_epsg))
                df = df[df["from_iso_a3"] != df["to_iso_a3"]]
                if len(df.index) > 0:
                    df = gpd.clip(df,boundary_df)
                    if len(df.index) > 0:
                        df[columns] = df[columns].multiply(0.001*df.geometry.length/df["length_km"],axis="index")
                        df["length_km"] = 0.001*df.geometry.length
                        reassemble_flows.append(df.to_crs(epsg=global_epsg))
            print (f"Done with {row.iso3} for {reference_mineral}")
    
    sum_cols = list(set(sum_cols))
    if len(reassemble_flows) > 0:
        df = pd.concat(reassemble_flows,axis=0,ignore_index=True)
        df = pd.merge(df,carbon_emission_df[["mode","CO2_pertonkm"]],how="left",on=["mode"]).fillna(0)
        df[[f"CO2eq_{c}" for c in sum_cols]] = df[sum_cols].multiply(df["CO2_pertonkm"],axis="index")
        carbon_columns = [f"CO2eq_{c}" for c in sum_cols]
        unique_edges = df[index_columns + ["iso3","length_km"]]
        unique_edges = unique_edges.drop_duplicates(subset=["id","iso3"] ,keep="first")
        gdf = df.groupby(["id","iso3"]).agg(dict([(c,"sum") for c in sum_cols + carbon_columns])).reset_index()
        unique_edges = pd.merge(unique_edges,gdf,how="left",on=["id","iso3"])
        # unique_edges["transport_tonsCO2eq"] = unique_edges[sum_cols].sum(axis=1)
        gpd.GeoDataFrame(
                    unique_edges,
                    geometry="geometry",
                    crs=f"EPSG:{global_epsg}"
                    ).to_parquet(
                            os.path.join(
                                flow_results_folder,
                                f"{file_name}_{country_case}_{constraint}.geoparquet"
                                )
                            )
        del unique_edges,gdf
        columns = [c for c in sum_cols if "total_tonkm" in c] + carbon_columns
        rename_columns = []
        for col in columns:
            df["processing_stage"] = float(col.split("_")[-1])
            if "total_tonkm" in col and "CO2eq_" not in col:
                rename_column = "transport_total_tonkm"
            elif "total_tonkm" in col and "CO2eq_" in col:
                rename_column = "transport_total_tonsCO2eq"
            elif "export_tonkm" in col:
                rename_column = "transport_export_tonsCO2eq"
            elif "import_tonkm" in col:
                rename_column = "transport_import_tonsCO2eq"
            df.rename(columns={col:rename_column},inplace=True)
            rename_columns.append(rename_column)

            # gdf = df.groupby(
            #             [
            #                 "reference_mineral",
            #                 "iso3","mode",
            #                 "processing_stage"
            #             ]).agg(dict([(rename_column,"sum")])).reset_index()
            all_flows.append(df[[
                                "reference_mineral",
                                "iso3","mode",
                                "processing_stage",rename_column]])
            df.drop(rename_column,axis=1,inplace=True)
            # all_flows.append(gdf)


    all_flows = pd.concat(all_flows,axis=0,ignore_index=True).fillna(0)
    rename_columns = [
                        "transport_total_tonkm",
                        "transport_total_tonsCO2eq",
                        "transport_export_tonsCO2eq",
                        "transport_import_tonsCO2eq"
                    ]
    all_flows = all_flows.groupby(
                    [
                        "reference_mineral",
                        "iso3","mode",
                        "processing_stage"
                    ]).agg(dict([(rc,"sum") for rc in rename_columns])).reset_index()
    # all_flows = all_flows.reset_index()
    # if year == 2022:
    #     file_name = f"carbon_emission_totals_{year}_{percentile}"
    # else:
    #     file_name = f"carbon_emission_totals_{year}_{percentile}_{efficient_scale}"
    all_flows.to_csv(
            os.path.join(
                results_folder,
                f"{file_name}_{country_case}_{constraint}.csv"),
            index=False)


if __name__ == '__main__':
    CONFIG = load_config()
    try:
        year = int(sys.argv[1])
        percentile = str(sys.argv[2])
        efficient_scale = str(sys.argv[3])
        country_case = str(sys.argv[4])
        constraint = str(sys.argv[5])
    except IndexError:
        print("Got arguments", sys.argv)
        exit()
    main(CONFIG,year,percentile,efficient_scale,country_case,constraint)