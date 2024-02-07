#!/usr/bin/env python
# coding: utf-8

import os 
import pandas as pd
import sys
import geopandas as gpd
import re
from collections import defaultdict
from utils import *
from tqdm import tqdm

sa_copper_refining = [("Cupric",-26.210465466416426, 28.081329454232627),
                    ("African Pegmatite",-26.563559654313774, 28.013720446371085),
                    ("Axis House",-34.03933861128403, 18.349777281481728)]
def modify_from_string_to_float(x,column_name):
    value = str(x[column_name]).strip()
    if value != "NA":
        return float(value)
    else:
        return np.nan

def main(config):

    incoming_data_path = config['paths']['incoming_data']
    processed_data_path = config['paths']['data']

    years = [2021,2030]
    import_export_groupby_columns = ["reference_mineral",
                            "export_country_code", 
                            "import_country_code",
                            "ccg_exporter", 
                            "export_continent",
                            "import_continent"]
    trade_value_columns = ["trade_value_thousandUSD","trade_quantity_tons"]
    tons_column = "trade_quantity_tons"
    value_column = "trade_value_thousandUSD"
    conversion_factor_column = "aggregate_ratio_normalised"
    trade_df = pd.read_csv(
                    os.path.join(processed_data_path,
                        "baci","baci_full_clean_continent_trade.csv"))
    
    codes_types_df = pd.read_csv(os.path.join(processed_data_path,
                            "baci",
                            "commodity_codes_refined_unrefined.csv"))
    pr_conv_factors_df = pd.read_excel(os.path.join(processed_data_path,
                                        "mineral_usage_factors",
                                        "aggregated_stages.xlsx"))[[
                                            "reference_mineral",
                                            "initial_refined_stage",
                                            "final_refined_stage", 
                                            "aggregate_ratio_normalised"
                                            ]] 
    """Create the reference commodity level OD first
    """
    mineral_classes = list(set(codes_types_df.reference_mineral.values.tolist()))

    for year in years:
        mineral_shares_df = []
        for mineral in mineral_classes:
            code_df = codes_types_df[
                                (
                                    codes_types_df["reference_mineral"] == mineral
                                ) & (
                                    codes_types_df["year"] == year
                                    )
                                ]
            for row in code_df.itertuples():
                refined_type = row.product_type
                product_codes = [int(r) for r in str(row.product_codes).split(",")]
                initial_stage = row.initial_refined_stage   
                final_stage = row.final_refined_stage
                t_df = trade_df[
                        trade_df["product_code"
                        ].isin(product_codes)
                    ].groupby(import_export_groupby_columns)[trade_value_columns].sum().reset_index()
                t_df["cost_to_tons_ratio"] = t_df[value_column]/t_df[tons_column]
                t_df["initial_mineral_type"] = row.product_type
                t_df["initial_refined_stage"] = row.initial_refined_stage
                t_df["final_refined_stage"] = row.final_refined_stage

                # t_df["export_country_to_importers_share"] = t_df["trade_quantity_tons"]/(t_df.groupby(
                #                                                     ["export_country_code"]
                #                                                         )["trade_quantity_tons"].transform('sum'))
                t_df["importer_country_global_share"] = (t_df.groupby(
                                            ["import_country_code"]
                                                    )["trade_quantity_tons"].transform('sum'))/t_df["trade_quantity_tons"].sum()
                t_df["importer_country_ccg_share"] = (t_df[t_df["ccg_exporter"] == 1].groupby(
                                            ["import_country_code"]
                                                    )["trade_quantity_tons"].transform('sum'))/t_df[t_df["ccg_exporter"] == 1]["trade_quantity_tons"].sum()
                t_df["average_importers_cost_to_tons_ratio"] = (t_df.groupby(
                                            ["import_country_code"]
                                                    )[value_column].transform('sum'))/(t_df.groupby(
                                            ["import_country_code"]
                                                    )[tons_column].transform('sum'))
                mineral_shares_df.append(t_df)

            mineral_shares_df = pd.concat(mineral_shares_df,axis=0,ignore_index=True)
            mineral_shares_df.to_csv("test_total.csv")

            # global_importers = mineral_shares_df[mineral_shares_df["initial_mineral_type"] == "refined"]
            # global_importers[["import_country_code",
            #         "importer_country_global_share"]].drop_duplicates(subset=["import_country_code"],
            #             keep="first").to_csv("global_importers.csv")
            ccg_importers = mineral_shares_df[["import_country_code",
                                    "import_continent",
                                    "initial_refined_stage",
                                    "ccg_exporter",
                                    "importer_country_ccg_share",
                                    "average_importers_cost_to_tons_ratio"]
                            ][mineral_shares_df["ccg_exporter"] == 1].drop_duplicates(
                                subset=["import_country_code","initial_refined_stage"],
                                keep="first")
            if len(ccg_importers) == 0:
                ccg_importers = mineral_shares_df[["import_country_code",
                                    "import_continent",
                                    "initial_refined_stage",
                                    "ccg_exporter",
                                    "importer_country_global_share",
                                    "average_importers_cost_to_tons_ratiover"]
                            ].drop_duplicates(
                                subset=["import_country_code","initial_refined_stage"],
                                keep="first")
                ccg_importers.rename(
                                columns={"importer_country_global_share":"importer_country_ccg_share"},
                                inplace=True)

            ccg_importers.to_csv("ccg_global_importers.csv")
            ccg_importers.drop("ccg_exporter",axis=1,inplace=True)

            conversion_df = mineral_shares_df[
                                (mineral_shares_df[
                                    "initial_refined_stage"
                                    ] != mineral_shares_df["final_refined_stage"]
                                ) & (mineral_shares_df["ccg_exporter"] == 1)]
            if len(conversion_df.index) > 0:
                print ("Convert for year:",year)
                df = mineral_shares_df[
                                (mineral_shares_df[
                                    "initial_refined_stage"
                                    ] == mineral_shares_df["final_refined_stage"]
                                ) & (mineral_shares_df["ccg_exporter"] == 1)]
                df["trade_type"] = "unchanged trade"
                conv_df = []
                conv_df.append(df)
                del df
                conversion_df = pd.merge(conversion_df,
                                            pr_conv_factors_df,
                                            how="left",
                                            on=["reference_mineral",
                                                "initial_refined_stage",
                                                "final_refined_stage"]
                                        )
                conversion_df[tons_column] = 1.0*conversion_df[tons_column]/conversion_df[conversion_factor_column]
                export_df = conversion_df.groupby(
                                ["reference_mineral",
                                "export_country_code",
                                "export_continent",
                                "ccg_exporter",
                                "initial_mineral_type",
                                "initial_refined_stage",
                                "final_refined_stage"])[tons_column].sum().reset_index()
                # export_df.rename(columns={tons_column:f"total_{tons_column}"},inplace=True)
                i_f_stages = list(
                                set(
                                    zip(
                                        conversion_df["initial_refined_stage"].values.tolist(),
                                        conversion_df["final_refined_stage"].values.tolist()
                                        )
                                    )
                                )
                for idx,(i,f) in enumerate(i_f_stages):
                    c_df = ccg_importers[ccg_importers["initial_refined_stage"] == f]
                    c_df.drop("initial_refined_stage",axis=1,inplace=True) 
                    export_df = pd.merge(export_df,c_df,how="cross")
                    export_df[tons_column] = export_df[tons_column]*export_df["importer_country_ccg_share"]
                    m_df = mineral_shares_df[
                                        mineral_shares_df["initial_refined_stage"] == f][[
                                                                        "export_country_code", 
                                                                        "import_country_code",
                                                                        "trade_value_thousandUSD",
                                                                        "cost_to_tons_ratio"]] 
                    n_df = pd.merge(export_df,m_df,
                                        how="left",
                                        on=["import_country_code",
                                            "export_country_code"]).fillna(0)
                    n_df[value_column] = np.where(n_df["cost_to_tons_ratio"] > 0,
                                                n_df["cost_to_tons_ratio"]*n_df[tons_column],
                                                n_df["average_importers_cost_to_tons_ratio"]*n_df[tons_column])
                    n_df["trade_type"] = "processed mineral"
                    conv_df.append(n_df)

                conv_df = pd.concat(conv_df,axis=0,ignore_index=True)
                conv_df.to_csv(os.path.join(processed_data_path,
                                            "baci",
                                            f"baci_ccg_country_level_trade_{year}.csv"),index=False)

            else:
                mineral_shares_df[
                        mineral_shares_df["ccg_exporter"] == 1
                        ].to_csv(os.path.join(processed_data_path,
                                            "baci",
                                            f"baci_ccg_country_level_trade_{year}.csv"),index=False)

    

    

        



if __name__ == '__main__':
    CONFIG = load_config()
    main(CONFIG)


