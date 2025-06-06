#!/usr/bin/env python
# coding: utf-8

import os 
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
from utils import *

def main(config):

    incoming_data_path = config['paths']['incoming_data']
    processed_data_path = config['paths']['data']
    output_data_path = config['paths']['results']

    results_folder = os.path.join(output_data_path,"baci_trade_matrices")
    if os.path.exists(results_folder) == False:
        os.mkdir(results_folder)

    years = [2022,2030,2040]
    data_type = {"initial_refined_stage":"str","final_refined_stage":"str"}
    new_trade_minerals = ["cobalt","graphite"]
    import_export_groupby_columns = ["reference_mineral",
                            "export_country_code", 
                            "import_country_code",
                            "ccg_exporter", 
                            "export_continent",
                            "import_continent"]
    tons_column = "trade_quantity_tons"
    value_column = "trade_value_thousandUSD"
    conversion_factor_column = "aggregate_ratio"

    trade_value_columns = [value_column,tons_column]
    final_columns = import_export_groupby_columns + ["initial_mineral_type",
                        "initial_refined_stage",
                        "final_refined_stage",
                        "trade_type"] + trade_value_columns
    # reference_mineral  
    # export_country_code 
    # import_country_code 
    # ccg_exporter    
    # export_continent    
    # import_continent    
    # initial_mineral_type    
    # initial_refined_stage   
    # final_refined_stage 
    # trade_type
    
    trade_df = pd.read_csv(
                    os.path.join(processed_data_path,
                        "baci","baci_ccg_minerals_trade_2022_updated.csv"))
    
    codes_types_df = pd.read_csv(os.path.join(processed_data_path,
                            "baci",
                            "commodity_codes_refined_unrefined.csv"),dtype=data_type)
    # codes_types_df["initial_refined_stage"] = codes_types_df["initial_refined_stage"].astype(str)
    # codes_types_df["final_refined_stage"] = codes_types_df["final_refined_stage"].astype(str)
    pr_conv_factors_df = pd.read_excel(os.path.join(processed_data_path,
                                        "mineral_usage_factors",
                                        "aggregated_stages.xlsx"),dtype=data_type)[[
                                            "reference_mineral",
                                            "initial_refined_stage",
                                            "final_refined_stage", 
                                            "aggregate_ratio"
                                            ]] 
    # pr_conv_factors_df["initial_refined_stage"] = pr_conv_factors_df["initial_refined_stage"].astype(str)
    # pr_conv_factors_df["final_refined_stage"] = pr_conv_factors_df["final_refined_stage"].astype(str)
                                           
    """Get existing trade relationships for each processing type 
    """
    existing_code_df = codes_types_df.drop_duplicates(
                                subset=["reference_mineral","initial_refined_stage"],
                                keep="first")
    
    mineral_shares_df = []
    ccg_importers_df = []
    for row in existing_code_df.itertuples():
        product_codes = [int(r) for r in str(row.product_codes).split(",")]
        initial_stage = row.initial_refined_stage   
        t_df = trade_df[
                trade_df["product_code"
                ].isin(product_codes)
            ].groupby(import_export_groupby_columns)[trade_value_columns].sum().reset_index()
        t_df["cost_to_tons_ratio"] = t_df[value_column]/t_df[tons_column]
        t_df["initial_mineral_type"] = row.product_type
        t_df["initial_refined_stage"] = row.initial_refined_stage
        t_df["final_refined_stage"] = row.final_refined_stage

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

        c_df = t_df[["reference_mineral",
                            "import_country_code",
                            "import_continent",
                            "initial_refined_stage",
                            "importer_country_global_share",
                            "average_importers_cost_to_tons_ratio"]
                    ].drop_duplicates(
                        subset=["import_country_code","initial_refined_stage"],
                        keep="first")
        c_df.rename(
                        columns={"importer_country_global_share":"importer_country_ccg_share"},
                        inplace=True)
        # c_df = t_df[t_df["ccg_exporter"] == 1]
        # if len(c_df) == 0:
        #     c_df = t_df[["reference_mineral",
        #                     "import_country_code",
        #                     "import_continent",
        #                     "initial_refined_stage",
        #                     "importer_country_global_share",
        #                     "average_importers_cost_to_tons_ratio"]
        #             ].drop_duplicates(
        #                 subset=["import_country_code","initial_refined_stage"],
        #                 keep="first")
        #     c_df.rename(
        #                     columns={"importer_country_global_share":"importer_country_ccg_share"},
        #                     inplace=True)
        # else:
        #     c_df = c_df[["reference_mineral",
        #                 "import_country_code",
        #                 "import_continent",
        #                 "initial_refined_stage",
        #                 "importer_country_ccg_share",
        #                 "average_importers_cost_to_tons_ratio"]
        #                 ].drop_duplicates(
        #                         subset=["reference_mineral","import_country_code","initial_refined_stage"],
        #                         keep="first")
        ccg_importers_df.append(c_df)


    mineral_shares_df = pd.concat(mineral_shares_df,axis=0,ignore_index=True)
    ccg_importers = pd.concat(ccg_importers_df,axis=0,ignore_index=True)
    mineral_shares_df.to_csv("mineral_shares.csv")
    ccg_importers.to_csv("ccg_global_importers.csv")

    mineral_classes = list(set(codes_types_df.reference_mineral.values.tolist()))
    m_s_df = mineral_shares_df.copy()
    for year in years:
        trade_shares_df = []
        for mineral in mineral_classes:
            code_df = codes_types_df[
                                (
                                    codes_types_df["reference_mineral"] == mineral
                                ) & (
                                    codes_types_df["year"] == year
                                    )
                                ]

            for row in code_df.itertuples():
                initial_stage = row.initial_refined_stage   
                final_stage = row.final_refined_stage
                if initial_stage == final_stage:
                    df = mineral_shares_df[
                                (mineral_shares_df[
                                    "initial_refined_stage"
                                    ] == initial_stage
                                ) & (mineral_shares_df["ccg_exporter"] == 1
                                ) & (mineral_shares_df["reference_mineral"] == mineral)]
                    df["trade_type"] = "unchanged trade"
                    trade_shares_df.append(df)
                    del df
                else:
                    df = m_s_df[
                                (
                                    m_s_df["final_refined_stage"] == initial_stage
                                ) & (
                                    m_s_df["reference_mineral"] == mineral
                                )
                                ]
                    # print (df)
                    df = df.groupby(
                                    ["reference_mineral",
                                    "export_country_code",
                                    "export_continent",
                                    "ccg_exporter",
                                    "initial_mineral_type",
                                    "final_refined_stage"])[tons_column].sum().reset_index()
                    df.rename(columns={"final_refined_stage":"initial_refined_stage"},inplace=True)
                    df["final_refined_stage"] = final_stage
                    # if year == 2030:
                    #     print (df)

                    conv_factor_df = pr_conv_factors_df[pr_conv_factors_df["reference_mineral"] == mineral]
                    conversion_factor = conv_factor_df[
                                                conv_factor_df["final_refined_stage"] == final_stage
                                                ][conversion_factor_column].values[0]/conv_factor_df[
                                                conv_factor_df["final_refined_stage"] == initial_stage
                                                ][conversion_factor_column].values[0]
                    df[tons_column] = 1.0*df[tons_column]/conversion_factor
                    if year == 2040 and mineral in new_trade_minerals:
                        final_import_stage = initial_stage
                    else:
                        final_import_stage = final_stage

                    c_df = ccg_importers[
                                            (
                                                ccg_importers["reference_mineral"] == mineral
                                            ) & (
                                                ccg_importers["initial_refined_stage"] == final_import_stage
                                            )
                                        ]
                    c_df.drop(["reference_mineral","initial_refined_stage"],axis=1,inplace=True) 
                    # if mineral == "lithium":
                    #     print (year,final_import_stage)
                    #     print (c_df)
                    df = pd.merge(df,c_df,how="cross")
                    # if year == 2030:
                    #     print (c_df)
                    #     print (df)
                    df[tons_column] = df[tons_column]*df["importer_country_ccg_share"]
                    m_df = mineral_shares_df[
                                        (
                                            mineral_shares_df["initial_refined_stage"] == final_stage
                                        ) & (mineral_shares_df["reference_mineral"] == mineral)][[
                                                                        "export_country_code", 
                                                                        "import_country_code",
                                                                        "cost_to_tons_ratio"]] 
                    n_df = pd.merge(df,m_df,
                                        how="left",
                                        on=["import_country_code",
                                            "export_country_code"]).fillna(0)
                    n_df[value_column] = np.where(n_df["cost_to_tons_ratio"] > 0,
                                                n_df["cost_to_tons_ratio"]*n_df[tons_column],
                                                n_df["average_importers_cost_to_tons_ratio"]*n_df[tons_column])
                    n_df["trade_type"] = "processed mineral"
                    trade_shares_df.append(n_df)

        trade_shares_df = pd.concat(trade_shares_df,axis=0,ignore_index=True)
        trade_shares_df = trade_shares_df[trade_shares_df[tons_column]>0]
        trade_shares_df[final_columns].to_csv(os.path.join(results_folder,
                                    f"baci_ccg_country_level_trade_{year}.csv"),index=False)
        m_s_df = trade_shares_df.copy()


if __name__ == '__main__':
    CONFIG = load_config()
    main(CONFIG)


