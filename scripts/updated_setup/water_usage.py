#!/usr/bin/env python
# coding: utf-8
"""This code finds the route between mines (in Africa) and all ports (outside Africa) 
"""
import sys
import os
import pandas as pd
import fiona
pd.options.mode.copy_on_write = True
import geopandas as gpd
from collections import defaultdict
from utils import *
from transport_cost_assignment import *
from trade_functions import * 
from tqdm import tqdm
tqdm.pandas()
    
def main(
            config,
            country_case,
            constraint,
            combination = None,
            distance_from_origin=0.0,
            environmental_buffer=0.0
        ):
    incoming_data_path = config['paths']['incoming_data']
    processed_data_path = config['paths']['data']
    output_data_path = config['paths']['results']

    baseline_year = 2022
    years = [2022,2040]
    year_percentile_combinations = [
                                    (2022,"baseline","baseline"),
                                    (2040,"bau","low"),
                                    (2040,"bau","mid"),
                                    (2040,"bau","high"),
                                    (2040,"early refining","low"),
                                    (2040,"early refining","mid"),
                                    (2040,"early refining","high"),
                                    (2040,"precursor","low"),
                                    (2040,"precursor","mid"),
                                    (2040,"precursor","high")
                                    ]
    reference_minerals = ["graphite","lithium","cobalt","manganese","nickel","copper"]
    stage_types = ["initial_stage","final_stage"]
    flow_types = ["initial_stage_production_tons","final_stage_production_tons"]
    flow_factor = [-1.0,1.0]
    flow_column_types = []
    anchor_columns = ["year","scenario","reference_mineral","iso3","processing_type","processing_stage"]

    if country_case == "country":
        efficient_scale = "min_threshold_metal_tons"
    else:
        efficient_scale = "max_threshold_metal_tons"

    input_folder = os.path.join(output_data_path,"optimised_processing_locations")
    results_folder = os.path.join(output_data_path,"water_usage_summaries")
    os.makedirs(results_folder,exist_ok=True)

    location_results_folder = os.path.join(output_data_path,"water_usage_by_location")
    os.makedirs(location_results_folder ,exist_ok=True)

    stage_names_df = pd.read_excel(
                        os.path.join(
                            processed_data_path,
                            "mineral_usage_factors",
                            "stage_mapping.xlsx"),
                        sheet_name="stage_maps")[["reference_mineral","processing_stage","processing_type"]]
    water_intensity_df = pd.read_excel(
                            os.path.join(
                                processed_data_path,
                                "mineral_usage_factors",
                                "mineral_extraction_country_intensities (final units w Co edits).xlsx"
                                ), sheet_name = "Country water ratios"
                            )[["iso3","reference_mineral","processing_stage","water intensity (m3/kg)"]]
    water_intensity_df.rename(
                                columns={
                                            "water intensity (m3/kg)":"water_intensity_m3_per_kg"
                                        },
                                inplace=True)
    (
            pr_conv_factors_df, 
            _, _, _, _,_
    ) = get_common_input_dataframes("none","bau",baseline_year,baseline_year)

    if combination is None:
        input_file = f"node_locations_for_energy_conversion_{country_case}_{constraint}.gpkg"
        t_file_name = "water_totals_by_stage.xlsx"
        w_file_name = f"water_{country_case}_{constraint}"
    else:
        if distance_from_origin > 0.0 or environmental_buffer > 0.0:
            ds = str(distance_from_origin).replace('.','p')
            eb = str(environmental_buffer).replace('.','p')
            input_file = f"{combination}_node_locations_for_energy_conversion_{country_case}_{constraint}_op_{ds}km_eb_{eb}km.gpkg"
            t_file_name = f"{combination}_water_totals_by_stage_op_{ds}km_eb_{eb}km.xlsx"
            w_file_name = f"{combination}_water_{country_case}_{constraint}_op_{ds}km_eb_{eb}km"
        else:
            input_file = f"{combination}_node_locations_for_energy_conversion_{country_case}_{constraint}.gpkg"
            t_file_name = f"{combination}_water_totals_by_stage.xlsx"
            w_file_name = f"{combination}_water_{country_case}_{constraint}"

    output_file = os.path.join(
                        results_folder,
                        t_file_name)
    if os.path.isfile(output_file) is True:
        writer = pd.ExcelWriter(output_file,mode='a',engine="openpyxl",if_sheet_exists='replace')
    else:
        writer = pd.ExcelWriter(output_file)

    if country_case == "country" and constraint == "unconstrained":
        combos = year_percentile_combinations
    else:
        combos = year_percentile_combinations[1:]

    all_dfs = []
    for idx, (year,scenario,percentile) in enumerate(combos):
        scenario = scenario.replace(" ","_")
        if year == baseline_year:
            layer_name = f"{year}_{percentile}"
        else:
            layer_name = f"{scenario}_{year}_{percentile}_{efficient_scale}"

        all_flows = gpd.read_file(os.path.join(input_folder,
                            input_file),
                            layer=layer_name)

        flow_columns = all_flows.columns.values.tolist()
        flow_column_types = []
        for idx,(st,ft,ff) in enumerate(zip(stage_types,flow_types,flow_factor)):
            cols = [c for c in flow_columns if ft in c]
            flow_column_types.append((st,ft,ff,cols))

        all_flow_tuple = []
        for i, val in all_flows.iterrows():
            n_id = val.id
            iso_code = val.iso3
            location_flows = defaultdict(float)
            for idx, (st,ft,ff,fct) in enumerate(flow_column_types):
                for fc in fct:
                    fv = getattr(val,fc)
                    rf = fc.split(f"_{ft}_")[0]
                    stv = fc.split(f"_{ft}_")[1]
                    stv = float(stv.split("_")[0])
                    if stv >= 1.0:
                        location_flows[f"{rf}_{stv}"] += ff*fv
                        if st == "final_stage" and fv > 0.0:
                            country_factors = pr_conv_factors_df[
                                                    (
                                                    pr_conv_factors_df["iso3"] == iso_code
                                                    ) & (
                                                    pr_conv_factors_df["reference_mineral"] == rf
                                                    ) & (
                                                    pr_conv_factors_df["final_refined_stage"] <= stv
                                                    )
                                                ]
                            if len(country_factors.index) > 0:
                                country_factors = country_factors.sort_values(by=["final_refined_stage"],ascending=True)
                                country_factors["conversion_factors"] = country_factors["aggregate_ratio"]/country_factors["aggregate_ratio"].values[-1]
                                stages = list(zip(
                                                country_factors["final_refined_stage"].values.tolist(),
                                                country_factors["conversion_factors"].values.tolist()
                                                )
                                                )
                                for stg in stages[:-1]:
                                    location_flows[f"{rf}_{stg[0]}"] += ff*stg[1]*fv

            all_flow_tuple.append({**{"id":n_id,"iso3":iso_code}, **location_flows})


        all_flow_tuple = pd.DataFrame(all_flow_tuple).fillna(0)

        value_columns = [c for c in all_flow_tuple.columns.values.tolist() if c not in ["id","iso3"]]
        df = []
        for i,val in all_flow_tuple.iterrows():
            n_id = val.id
            iso_code = val.iso3
            for v in value_columns:
                rf = v.split("_")[0]
                st = float(v.split("_")[1])
                pr = getattr(val,v)
                df.append((n_id,iso_code,rf,st,pr))

        df = pd.DataFrame(df,columns=["id","iso3","reference_mineral","processing_stage","production_tonnes_for_water"])
        df = df[df["production_tonnes_for_water"] > 0]
        df = pd.merge(df,water_intensity_df,how="left",on=["iso3","reference_mineral","processing_stage"])
        df["water_usage_m3"] = 1.0e3*df["production_tonnes_for_water"]*df["water_intensity_m3_per_kg"]
        df.to_csv(os.path.join(location_results_folder,f"{w_file_name}_{layer_name}.csv"),index=False)
        df["year"] = year
        df["scenario"] = layer_name
        df = df.groupby(
                ["year","scenario","iso3","reference_mineral","processing_stage","water_intensity_m3_per_kg"]
                ).agg(dict([(c,"sum") for c in ["production_tonnes_for_water","water_usage_m3"]])).reset_index()

        all_dfs.append(df)

    all_dfs = pd.concat(all_dfs,axis=0,ignore_index=True)
    all_dfs = pd.merge(all_dfs,stage_names_df,how="left",on=["reference_mineral","processing_stage"])
    all_dfs = all_dfs.set_index(anchor_columns)
    all_dfs = all_dfs.sort_index()
    all_dfs.to_excel(writer,sheet_name=f"{country_case}_{constraint}")
    writer.close()

if __name__ == '__main__':
    CONFIG = load_config()
    try:
        if len(sys.argv) > 3:
            country_case = str(sys.argv[1])
            constraint = str(sys.argv[2])
            combination = str(sys.argv[3])
            distance_from_origin = float(sys.argv[4])
            environmental_buffer = float(sys.argv[5])
        else:
            country_case = str(sys.argv[1])
            constraint = str(sys.argv[2])
            combination = None
            distance_from_origin = 0.0
            environmental_buffer = 0.0
    except IndexError:
        print("Got arguments", sys.argv)
        exit()
    main(
            CONFIG,
            country_case,
            constraint,
            combination = combination,
            distance_from_origin=distance_from_origin,
            environmental_buffer=environmental_buffer)