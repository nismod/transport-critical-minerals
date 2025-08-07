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

def main(config,agg_type):
    incoming_data_path = config['paths']['incoming_data']
    processed_data_path = config['paths']['data']
    output_data_path = config['paths']['results']

    results_folder = os.path.join(output_data_path,"result_summaries")
    os.makedirs(results_folder,exist_ok=True)
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
    tonnage_thresholds = ["min_threshold_metal_tons","max_threshold_metal_tons"]
    years = list(set([y[0] for y in year_percentile_combinations]))
    price_costs_df = []
    price_df = pd.read_excel(
                        os.path.join(
                            processed_data_path,
                            "production_costs",
                            "Final_Price_and_Costs_RP.xlsx"),
                    sheet_name = "Price_final",index_col=[0])
    price_df = price_df.reset_index()
    for y in years:
        pdf = price_df[["reference_mineral","processing_stage",y]]
        pdf["year"] = y
        pdf.rename(columns={y:"price_usd_per_tonne"},inplace=True)
        price_costs_df.append(pdf)

    price_costs_df = pd.concat(price_costs_df,axis=0,ignore_index=True)
    price_costs_df.rename(columns={"processing_stage":"final_processing_stage"},inplace=True)

    if agg_type in ["BACI","OD"]:
        if agg_type == "BACI":
            fpath = os.path.join(output_data_path,"baci_trade_matrices")
        else:
            fpath = os.path.join(output_data_path,"flow_node_ods")
        fnames = []
        for idx, (year,scenario,percentile) in enumerate(year_percentile_combinations):
            scenario = scenario.replace(" ","_")
            if year == 2022:
                if agg_type == "BACI":
                    fnames.append(f"baci_ccg_country_trade_breakdown_{year}_{percentile}.csv")
                else:
                    fnames.append(f"mining_city_node_level_ods_{year}_{percentile}.csv")
            else:
                for th in tonnage_thresholds:
                    if agg_type == "BACI":
                        fnames.append(f"baci_ccg_country_trade_breakdown_{scenario}_{year}_{percentile}_{th}.csv")
                    else:
                        fnames.append(f"mining_city_node_level_ods_{scenario}_{year}_{percentile}_{th}.csv")
        dfs = []
        for fname in fnames:
            df = pd.read_csv(os.path.join(fpath,fname))
            df = df[df["trade_type"] == "Export"]
            df = df.groupby(
                        [
                            "export_country_code",
                            "reference_mineral",
                            "final_processing_stage"
                        ]
                        )["final_stage_production_tons"].sum().reset_index()
            df["scenario"] = fname
            df["year"] = year
            dfs.append(df)

        dfs = pd.concat(dfs,axis=0,ignore_index=True)

        dfs = pd.merge(
                dfs,price_costs_df,
                how="left",
                on=["reference_mineral","final_processing_stage","year"]).fillna(0)
        dfs["revenue"] = dfs["final_stage_production_tons"]*dfs["price_usd_per_tonne"]
        dfs.to_csv(
                    os.path.join(
                        results_folder,
                        f"revenue_totals_by_country_stage_mineral_scenario_{agg_type}.csv"),
                    index=False)

        dfs = dfs.groupby(
                    ["scenario","reference_mineral"]
                    ).agg(dict([(t,"sum") for t in ["final_stage_production_tons","revenue"]])).reset_index()
        dfs["total_revenue_per_tonne"] = dfs["revenue"]/dfs["final_stage_production_tons"]
        dfs.to_csv(
                    os.path.join(
                        results_folder,
                        f"revenue_totals_by_mineral_scenario_{agg_type}.csv"),
                    index=False)
    else:
        fpath = os.path.join(
                            output_data_path,"result_summaries",
                            "combined_transport_totals_by_stage.xlsx"
                            )
        if os.path.exists(fpath):
            location_cases = ["country","region"]
            optimisation_type = ["unconstrained","constrained"]
            dfs = []
            for loc in location_cases:
                for opt in optimisation_type:
                    transport_df = pd.read_excel(
                                        fpath,
                                        sheet_name=f"{loc}_{opt}",
                                        index_col=[0,1,2,3,4,5])
                    transport_df = transport_df.reset_index()
                    dfs.append(transport_df)

            dfs = pd.concat(dfs,axis=0,ignore_index=True)
            dfs = dfs.groupby(
                        ["scenario","reference_mineral"]
                        ).agg(
                        dict([(t,"sum") for t in ["production_tonnes","export_tonnes","revenue_usd"]])
                        ).reset_index()
            dfs["total_revenue_per_tonne"] = dfs["revenue"]/dfs["export_tonnes"]
            dfs.to_csv(
                        os.path.join(
                            results_folder,
                            f"revenue_totals_by_mineral_scenario_final_estimation.csv"),
                        index=False)


if __name__ == '__main__':
    CONFIG = load_config()
    try:
        agg_type = str(sys.argv[1])
    except IndexError:
        print("Got arguments", sys.argv)
        exit()
    main(CONFIG,agg_type)