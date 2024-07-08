#!/usr/bin/env python
# coding: utf-8

import os 
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
from utils import *
from trade_functions import *

def get_mine_conversion_factors(x,mcf_df,pcf_df,ini_st_column,fnl_st_column,cf_column="aggregate_ratio"):
    ref_min = x["reference_mineral"]
    exp_st = x[fnl_st_column]
    mine_st = x[ini_st_column]
    cf_df = pcf_df[pcf_df["reference_mineral"] == ref_min]
    if mine_st == 0:
        mc = mcf_df[mcf_df["reference_mineral"] == ref_min
                    ]["metal_content_factor"].values[0]
        cf_val = mc*cf_df[cf_df["final_refined_stage"] == str(exp_st).replace(".0","")
                        ][cf_column].values[0]/cf_df[
                        cf_df["final_refined_stage"] == '1'
                        ][cf_column].values[0]
    else:
        cf_val = cf_df[cf_df["final_refined_stage"] == str(exp_st).replace(".0","")
                        ][cf_column].values[0]/cf_df[
                        cf_df["final_refined_stage"] == str(mine_st).replace(".0","")
                        ][cf_column].values[0]
    return cf_val

def get_importer_shares(existing_trade_df,import_groupby_columns,value_column,tons_column,new_trade_minerals):
    global_import_df = existing_trade_df.groupby(import_groupby_columns)[[value_column,tons_column]].sum().reset_index()
    global_import_df["cost_to_tons_ratio"] = global_import_df[value_column]/global_import_df[tons_column]

    # global_import_df["import_shares"
    #     ] = global_import_df[tons_column
    #     ]/global_import_df.groupby(
    #         ["refining_stage_cam"])[tons_column
    #     ].transform("sum")
    global_import_df["average_global_cost_to_tons_ratio"] = (global_import_df.groupby(
                                ["product_code"]
                                        )[value_column].transform('sum'))/(global_import_df.groupby(
                                ["product_code"]
                                        )[tons_column].transform('sum'))
    added_import_df = [global_import_df]
    for nt in new_trade_minerals:
        product_code = nt["replicate_product"]
        aim_df = global_import_df[global_import_df["product_code"] == product_code]
        aim_df["refining_stage_cam"] = nt["future_stage"]
        added_import_df.append(aim_df)

    added_import_df = pd.concat(added_import_df,axis=0,ignore_index=True)
    added_import_df["import_shares"
        ] = added_import_df[tons_column
        ]/added_import_df.groupby(
            ["reference_mineral","refining_stage_cam"])[tons_column
        ].transform("sum")

    return added_import_df

def main(config,
        year,
        percentile,
        efficient_scale):

    incoming_data_path = config['paths']['incoming_data']
    processed_data_path = config['paths']['data']
    output_data_path = config['paths']['results']

    results_folder = os.path.join(output_data_path,"baci_trade_matrices")
    if os.path.exists(results_folder) == False:
        os.mkdir(results_folder)

    baseline_year = 2022

    new_trade_minerals = [
                            {
                                "reference_mineral":"cobalt",
                                "replicate_product":282200,
                                "future_stage":5.0
                            },
                            {
                                "reference_mineral":"graphite",
                                "replicate_product":250410,
                                "future_stage":3.0
                            },
                            {
                                "reference_mineral":"graphite",
                                "replicate_product":250410,
                                "future_stage":4.0
                            }
                            ]
    # Define a number of columns names
    (
        data_type, 
        export_country_columns, 
        import_country_columns,
        product_columns, 
        conversion_factor_column, 
        trade_balance_columns,
        final_trade_columns,
        reference_minerals
    ) = get_columns_names()

    #  Get a number of input dataframes
    (
        pr_conv_factors_df, 
        metal_content_factors_df, 
        ccg_countries, mine_city_stages, trade_df
    ) = get_common_input_dataframes(data_type,year,baseline_year)

    # Add the mine final stage to te trade dataframe and also estimate the cost-to-ton ratios
    trade_df_columns = trade_df.columns.values.tolist()
    trade_df["cost_to_tons_ratio"] = trade_df["trade_value_thousandUSD"]/trade_df["trade_quantity_tons"]
    trade_df = pd.merge(trade_df,mine_city_stages,how="left",on=["reference_mineral"])

    # Read data on production scales
    production_scales_df = pd.read_excel(
                                os.path.join(
                                    processed_data_path,
                                    "production_costs",
                                    "scales.xlsx"),
                                sheet_name="efficient_scales"
                                )
    # Read the trade proportions
    trade_proportion_df = pd.read_csv(
                            os.path.join(
                                results_folder,
                                f"baci_ccg_country_metal_content_production_{baseline_year}_baseline.csv")
                            )[["export_country_code","reference_mineral","production_to_trade_fraction"]]

    import_proportion_df = pd.read_csv(
                            os.path.join(
                                results_folder,
                                f"baci_import_shares_{baseline_year}_baseline.csv")
                            )[["import_country_code","reference_mineral",
                                "initial_processing_stage","import_location",
                            "location_fraction"]]
    import_proportion_df = import_proportion_df[import_proportion_df["import_location"] == "city_demand"]
    import_proportion_df.rename(
                        columns={
                            "import_country_code":"export_country_code",
                            "initial_processing_stage":"final_processing_stage"},
                        inplace=True)
    # Read the mine level tonnages and convert them to the final stage to be produced and traded
    mine_exports_df = []
    for reference_mineral in reference_minerals:
        mines_df = gpd.read_file(
                                os.path.join(
                                    processed_data_path,
                                    "minerals",
                                    "s_and_p_mines_current_and_future_estimates.gpkg"),
                                layer=f"{reference_mineral}_{percentile}")
        mines_df = mines_df.groupby(["ISO_A3"])[str(year)].sum().reset_index()
        mines_df = mines_df[mines_df[str(year)]>0]
        mines_df["reference_mineral"] = reference_mineral
        mines_df["initial_processing_stage"] = 0
        # mines_df["final_processing_stage"] = mine_city_stages[
        #                                       mine_city_stages["reference_mineral"
        #                                       ] == reference_mineral
        #                                       ]["mine_final_refined_stage"].values[0]
        mine_exports_df.append(mines_df)

    mine_exports_df = pd.concat(mine_exports_df,axis=0,ignore_index=True)
    mine_exports_df.rename(
                    columns={
                            "ISO_A3":"export_country_code",
                            str(year):"future_metal_content_tons"
                            },
                    inplace=True)
    mine_exports_df = pd.merge(
                        mine_exports_df,
                        trade_proportion_df,
                        how="outer",on=["export_country_code","reference_mineral"]).fillna(0)
    mine_exports_df["production_to_trade_fraction"
            ] = np.where(
                    (mine_exports_df["production_to_trade_fraction"] == 0
                    ) & (mine_exports_df["future_metal_content_tons"] > 0),
                    1, mine_exports_df["production_to_trade_fraction"])
    mine_exports_df["future_metal_content_trade_tons"
        ] = mine_exports_df["production_to_trade_fraction"]*mine_exports_df["future_metal_content_tons"]
    mine_exports_df = mine_exports_df[mine_exports_df["future_metal_content_tons"] > 0]
    mine_exports_df = pd.merge(mine_exports_df,production_scales_df,how="left",on=["reference_mineral"])
    mine_exports_df["final_processing_stage"
     ] = mine_exports_df.progress_apply(
        lambda x:mine_city_stages[
                    mine_city_stages["reference_mineral"] == x["reference_mineral"
                    ]]["mine_final_refined_stage"].values[0] if x["future_metal_content_tons"
                    ] > x[efficient_scale] else 1.0,
        axis=1)
    mine_exports_df["future_trade_tons"] = mine_exports_df.progress_apply(
                                                lambda x:x["future_metal_content_trade_tons"
                                                ]/get_mine_conversion_factors(
                                                    x,metal_content_factors_df,
                                                    pr_conv_factors_df,
                                                    "initial_processing_stage",
                                                    "final_processing_stage"),axis=1)
    # mine_exports_df.to_csv("test.csv",index=False)


    # Get the total tonnage of exports and imports of each CCG country
    trade_balance_df, export_df, import_df = get_trade_exports_imports(trade_df,ccg_countries)
    trade_balance_df.rename(columns={"refining_stage_cam":"final_processing_stage"},inplace=True)
    
    # Estimate trade balancing under the new stage conversion
    trade_balance_df = pd.merge(
                            trade_balance_df,
                            mine_exports_df,
                            how="outer",
                            on=["export_country_code",
                                "reference_mineral",
                                "final_processing_stage"]).fillna(0)
    trade_balance_df = pd.merge(
                        trade_balance_df,
                        import_proportion_df,
                        how="left",
                        on=["export_country_code",
                            "reference_mineral",
                            "final_processing_stage"]).fillna(0)
    trade_balance_df["mine_final_refined_stage"
        ] = np.where(
                trade_balance_df["future_trade_tons"] > 0,
                trade_balance_df["final_processing_stage"],
                0)
    trade_balance_df["mine_final_refined_stage"
        ] = trade_balance_df.groupby(
            ["export_country_code","reference_mineral"]
            )["mine_final_refined_stage"].transform("sum")
    trade_balance_df["mine_final_refined_stage"
        ] = np.where(
                trade_balance_df["mine_final_refined_stage"] > 0,
                trade_balance_df["mine_final_refined_stage"],
                1)

    trade_balance_df["future_export_binary"] = np.where(
                                                trade_balance_df["mine_final_refined_stage"] >= trade_balance_df["final_processing_stage"],
                                                0,1)
    trade_balance_df["future_export_tons"] = trade_balance_df[
                                                "future_export_binary"
                                                ]*trade_balance_df[
                                                "trade_quantity_tons_export"
                                                ] + trade_balance_df["future_trade_tons"]
    trade_balance_df["future_export_tons"] = np.where(
                                                trade_balance_df["mine_final_refined_stage"] == trade_balance_df["final_processing_stage"],
                                                trade_balance_df["future_export_tons"] - trade_balance_df["trade_quantity_tons_import"],
                                                trade_balance_df["future_export_tons"])
    trade_balance_df["future_export_tons"] = np.where(
                                                trade_balance_df["future_export_tons"] > 0,
                                                trade_balance_df["future_export_tons"],
                                                0)
    trade_balance_df["future_domestic_tons"] = np.where(
                                                trade_balance_df["mine_final_refined_stage"] == trade_balance_df["final_processing_stage"],
                                                trade_balance_df["future_trade_tons"] - trade_balance_df["future_export_tons"],
                                                0)
    trade_balance_df["future_import_tons"] = trade_balance_df["trade_quantity_tons_import"] - trade_balance_df["future_domestic_tons"]
    trade_balance_df.loc[trade_balance_df["future_import_tons"] < 1e-6,"future_import_tons"] = 0

    trade_balance_df = trade_balance_df[trade_balance_df["final_processing_stage"] > 0]
    
    trade_balance_df["stage_conversion_factor"] = trade_balance_df.progress_apply(
                            lambda x:get_mine_conversion_factors(
                            x,metal_content_factors_df,
                            pr_conv_factors_df,
                            "mine_final_refined_stage",
                            "final_processing_stage"),
                            axis=1)
    
    trade_balance_df["mine_production_consumed_tons"
        ] = np.where(
                trade_balance_df["mine_final_refined_stage"] < trade_balance_df["final_processing_stage"],
                trade_balance_df[
                    "future_export_tons"
                    ]*trade_balance_df["stage_conversion_factor"],
                0
            )    
    trade_balance_df["mine_production_consumed_total_tons"
        ] = trade_balance_df.groupby(
                ["export_country_code","reference_mineral"]
                )["mine_production_consumed_tons"].transform("sum")
    
    trade_balance_df["mine_production_consumed_total_tons"
        ] = np.where(
            trade_balance_df["mine_final_refined_stage"] == trade_balance_df["final_processing_stage"],
            trade_balance_df["mine_production_consumed_total_tons"],
            0
            )

    trade_balance_df["future_domestic_tons_rem"
        ] = trade_balance_df["future_domestic_tons"
        ] - trade_balance_df["mine_production_consumed_total_tons"]
    
    trade_balance_df["future_export_tons_rem"
        ] = np.where(trade_balance_df["future_domestic_tons_rem"] < 0,
            trade_balance_df["future_export_tons"] + trade_balance_df["future_domestic_tons_rem"],
            trade_balance_df["future_export_tons"])
    
    trade_balance_df["future_domestic_tons_rem"
        ] = np.where(trade_balance_df["future_domestic_tons_rem"] < 0,
            0,
            trade_balance_df["future_domestic_tons_rem"])

    trade_balance_df["mine_production_consumed_total_tons"
        ] = np.where(
            trade_balance_df["future_export_tons_rem"] < 0,
            trade_balance_df["mine_production_consumed_total_tons"] + trade_balance_df["future_export_tons_rem"],
            trade_balance_df["mine_production_consumed_total_tons"]
            )

    trade_balance_df["mine_production_actual_consumed_tons"
        ] = np.where(
                trade_balance_df.groupby(
                    ["export_country_code","reference_mineral"]
                    )["mine_production_consumed_tons"].transform("sum") > 0,
                trade_balance_df["mine_production_consumed_tons"
                ]*trade_balance_df.groupby(
                    ["export_country_code","reference_mineral"]
                    )["mine_production_consumed_total_tons"].transform("sum")/trade_balance_df.groupby(
                    ["export_country_code","reference_mineral"]
                    )["mine_production_consumed_tons"].transform("sum"),
                0
            ) 
    trade_balance_df["future_extra_import_tons"
        ] = trade_balance_df["mine_production_consumed_tons"
        ] - trade_balance_df["mine_production_actual_consumed_tons"]

    trade_balance_df["future_export_tons_rem"
        ] = np.where(
                trade_balance_df["future_export_tons_rem"] > 0,
                trade_balance_df["future_export_tons_rem"],
                0
            )

    trade_balance_df["future_extra_import_tons"
        ] = np.where(
                (
                    trade_balance_df["mine_final_refined_stage"] > 1.0
                ) & (
                    trade_balance_df["mine_final_refined_stage"] == trade_balance_df["final_processing_stage"]
                ) & (
                    trade_balance_df["future_export_tons_rem"] < trade_balance_df["trade_quantity_tons_export"]
                ),
                trade_balance_df["trade_quantity_tons_export"] - trade_balance_df["future_export_tons_rem"],
                trade_balance_df["future_extra_import_tons"]
            )

    trade_balance_df["mine_production_notenough"
        ] = np.where(trade_balance_df.groupby(
                    ["export_country_code","reference_mineral"]
                    )["future_extra_import_tons"].transform("sum") > 1e-6,
            1,
            0)
    trade_balance_df = trade_balance_df.sort_values(
        ["export_country_code","reference_mineral","final_processing_stage"],ascending=True)
    trade_balance_df["import_available"
        ] = trade_balance_df["future_import_tons"]*trade_balance_df["stage_conversion_factor"]
    trade_balance_df["import_available_cumsum"
        ] = trade_balance_df.groupby(
        ["export_country_code","reference_mineral"]
        )["import_available"].transform(lambda x: x.cumsum().shift(fill_value=0))

    trade_balance_df["future_extra_import_tons"
        ] = trade_balance_df[["future_extra_import_tons","import_available_cumsum"]].min(axis=1)

    trade_balance_df["import_available_cumsum"
        ] = trade_balance_df["import_available_cumsum"
        ] - trade_balance_df.groupby(
        ["export_country_code","reference_mineral"]
        )["future_extra_import_tons"].transform(lambda x: x.cumsum().shift(fill_value=0))

    trade_balance_df["import_export_ratio"
        ] = np.where(trade_balance_df["import_available_cumsum"] > 0,
            trade_balance_df["future_extra_import_tons"]/trade_balance_df["import_available_cumsum"],
            0)
    trade_balance_df["future_actual_export"
        ] = (trade_balance_df["mine_production_actual_consumed_tons"
        ] + trade_balance_df["future_extra_import_tons"])/trade_balance_df["stage_conversion_factor"]
    trade_balance_df["future_export_tons_rem"
        ] =  np.where(
                trade_balance_df["mine_production_notenough"] == 1,
                trade_balance_df[
                ["future_export_tons_rem","future_actual_export"]].min(axis=1),
                trade_balance_df["future_export_tons_rem"])

    trade_balance_df = trade_balance_df.sort_values(
        ["export_country_code","reference_mineral","final_processing_stage"],ascending=False)
    trade_balance_df["import_export_ratio_cumsum"
        ] = trade_balance_df.groupby(
        ["export_country_code","reference_mineral"]
        )["import_export_ratio"].transform(lambda x: x.cumsum().shift(fill_value=0))
    trade_balance_df["ratio_for_consumption"
        ] = 1 - trade_balance_df["import_export_ratio_cumsum"]
    trade_balance_df["future_import_for_consumption"
        ] = (trade_balance_df[
            ["ratio_for_consumption",
            "location_fraction"]].min(axis=1))*trade_balance_df["future_import_tons"]
    trade_balance_df["future_import_for_processing"
        ] = trade_balance_df["import_export_ratio_cumsum"]*trade_balance_df["future_import_tons"]

    trade_balance_df["future_metal_content_trade_tons"
        ] = trade_balance_df.groupby(
                ["export_country_code","reference_mineral"]
                )["future_metal_content_trade_tons"].transform("sum")
    trade_balance_df["future_trade_tons"
        ] = trade_balance_df.groupby(
                ["export_country_code","reference_mineral"]
                )["future_trade_tons"].transform("sum")

    mine_city_stages.rename(columns={"mine_final_refined_stage":"mine_highest_stage"},inplace=True)
    trade_balance_df = pd.merge(trade_balance_df,mine_city_stages,how="left",on=["reference_mineral"])
    # trade_balance_df.to_csv("export_import_tons_breakdown.csv",index=False)

    export_df = trade_balance_df[
                    trade_balance_df["mine_final_refined_stage"
                    ] < trade_balance_df["final_processing_stage"]
                    ]
    export_df["initial_processing_stage"
        ] = np.where(
                export_df["final_processing_stage"] <= export_df["mine_highest_stage"],
                0,
                export_df["mine_final_refined_stage"])
    export_df["initial_tons"
        ] = np.where(
                (export_df["final_processing_stage"] <= export_df["mine_highest_stage"]
                ) & (export_df["future_trade_tons"] > 0),
                export_df["mine_production_actual_consumed_tons"
                ]*export_df["future_metal_content_trade_tons"
                ]/export_df["future_trade_tons"
                ],
                export_df["mine_production_actual_consumed_tons"])
    export_df["final_tons"] = export_df["future_export_tons_rem"]
    export_df["trade_type"] = "Export"
    export_df["initial_processing_location"
        ] = np.where(
            export_df["final_processing_stage"] <= export_df["mine_highest_stage"],
            "mine",
            "city_process")
    export_df["final_processing_location"] = "outside"

    mine_export_df = trade_balance_df[
                    trade_balance_df["mine_final_refined_stage"
                    ] == trade_balance_df["final_processing_stage"]
                    ]
    mine_export_df["initial_processing_stage"] = 0
    mine_export_df["initial_tons"
        ] = np.where(
                mine_export_df["future_trade_tons"] > 0,
                mine_export_df["future_export_tons_rem"
                ]*mine_export_df["future_metal_content_trade_tons"
                ]/mine_export_df["future_trade_tons"
                ],
                0) 
    mine_export_df["final_tons"] = mine_export_df["future_export_tons_rem"]
    mine_export_df["trade_type"] = "Export"
    mine_export_df["initial_processing_location"] = "mine"
    mine_export_df["final_processing_location"] = "outside"


    mine_domestic_df = trade_balance_df[
                    (trade_balance_df["mine_final_refined_stage"
                    ] == trade_balance_df["final_processing_stage"]
                    ) | (trade_balance_df["final_processing_stage"
                    ] > trade_balance_df["mine_highest_stage"])
                    ]

    # mine_domestic_df["final_tons"
    #     ] = mine_domestic_df["future_domestic_tons_rem"]

    mine_domestic_df["final_tons"
        ] = np.where(
            mine_domestic_df["mine_final_refined_stage"] == mine_domestic_df["final_processing_stage"],
            mine_domestic_df["future_domestic_tons_rem"],
            mine_domestic_df["mine_production_actual_consumed_tons"]
            )
    mine_domestic_df["initial_processing_stage"] = 0
    mine_domestic_df["initial_tons"
        ] = np.where(
                mine_domestic_df["future_trade_tons"] > 0,
                mine_domestic_df["final_tons"
                ]*mine_domestic_df["future_metal_content_trade_tons"
                ]/mine_domestic_df["future_trade_tons"
                ],
                0)
    mine_domestic_df["trade_type"] = "Domestic"
    mine_domestic_df["initial_processing_location"] = "mine"
    mine_domestic_df["final_processing_location"
        ] = np.where(
                mine_domestic_df["mine_final_refined_stage"
                ] < mine_domestic_df["final_processing_stage"
                ],
                "city_process",
                "city_demand")

    mine_domestic_df["final_processing_stage"] = mine_domestic_df["mine_final_refined_stage"]

    import_processing_df = trade_balance_df[trade_balance_df["future_import_for_processing"] > 0]
    import_processing_df["initial_processing_stage"] = import_processing_df["final_processing_stage"]
    import_processing_df["initial_tons"
        ] =  import_processing_df["final_tons"
        ] = import_processing_df["future_import_for_processing"
        ]
    import_processing_df["trade_type"] = "Import"
    import_processing_df["initial_processing_location"] = "outside"
    import_processing_df["final_processing_location"] = "city_process"

    import_consumption_df = trade_balance_df[trade_balance_df["future_import_for_consumption"] > 0]
    import_consumption_df["initial_processing_stage"] = import_consumption_df["final_processing_stage"]
    import_consumption_df["initial_tons"
        ] =  import_consumption_df["final_tons"
        ] = import_consumption_df["future_import_for_consumption"
        ]
    import_consumption_df["trade_type"] = "Import"
    import_consumption_df["initial_processing_location"] = "outside"
    import_consumption_df["final_processing_location"] = "city_demand"

    trade_balance_columns = [
                                "export_country_code",
                                "reference_mineral",
                                "initial_processing_stage",
                                "final_processing_stage",
                                "initial_tons",
                                "final_tons",
                                "trade_type",
                                "initial_processing_location",
                                "final_processing_location"
                            ]
    final_trade_df = pd.concat(
                        [
                            export_df,
                            mine_export_df,
                            mine_domestic_df,
                            import_processing_df,
                            import_consumption_df
                        ],axis=0,ignore_index=True)
    final_trade_df = final_trade_df[final_trade_df["final_tons"]>0]
    final_trade_df = final_trade_df[trade_balance_columns]
    # final_trade_df.to_csv("final_trade.csv",index=False)

    final_trade_df["refining_stage_cam"] = final_trade_df["final_processing_stage"]

    export_df = final_trade_df[final_trade_df["trade_type"] == "Export"]
    export_df.rename(
                    columns={
                        "final_tons":"final_tons_export",
                        # "final_processing_stage":"refining_stage_cam"
                        },
                    inplace=True)
    t_df = pd.merge(
                    trade_df,
                    export_df,
                    how="outer",
                    on=["export_country_code",
                        "reference_mineral",
                        "refining_stage_cam"]).fillna(0)
    t_df["future_trade_quantity_tons"
                    ] = np.where(
                            (t_df["trade_quantity_tons"] > 0) & (t_df["export_country_code"].isin(ccg_countries)),
                            t_df["trade_quantity_tons"
                            ]*t_df["final_tons_export"
                            ]/t_df.groupby(
                                ["export_country_code","reference_mineral",
                                "refining_stage_cam"])["trade_quantity_tons"].transform("sum"),
                            t_df["trade_quantity_tons"])
    t_df["adjusted_exports"
        ] = t_df.groupby(
            [
                "export_country_code",
                "reference_mineral",
                "refining_stage_cam"]
            )["future_trade_quantity_tons"].transform("sum")
    # Adjust the imports of CCG and non CCG next
    import_df = final_trade_df[final_trade_df["trade_type"] == "Import"
                        ][["export_country_code","reference_mineral","refining_stage_cam","final_tons"]]
    import_df = import_df.groupby(
                    ["export_country_code",
                    "reference_mineral",
                    "refining_stage_cam"])["final_tons"].sum().reset_index()
    import_df.rename(
        columns={
            "export_country_code":"import_country_code",
            # "final_processing_stage":"refining_stage_cam",
            "final_tons":"final_tons_import"
            },
        inplace=True)
    t_df = pd.merge(
                    t_df,import_df,how="outer",
                    on=["import_country_code",
                        "reference_mineral",
                        "refining_stage_cam"]).fillna(0)
    t_df["future_trade_quantity_tons"
        ] = np.where(
                (t_df["future_trade_quantity_tons"] > 0) & (t_df["import_country_code"].isin(ccg_countries)),
                t_df["future_trade_quantity_tons"
                ]*t_df["final_tons_import"
                ]/t_df.groupby(["import_country_code",
                "reference_mineral",
                "refining_stage_cam"])["future_trade_quantity_tons"].transform("sum"),
                t_df["future_trade_quantity_tons"])
    t_df["readjusted_exports"
        ] = t_df.groupby(
            [
                "export_country_code",
                "reference_mineral",
                "refining_stage_cam"]
            )["future_trade_quantity_tons"].transform("sum")
    t_df["export_diff"] = t_df["adjusted_exports"] - t_df["readjusted_exports"]
    t_df["future_trade_temp"
        ] = np.where(
                ~t_df["import_country_code"].isin(ccg_countries),
                t_df["trade_quantity_tons"],
                0
            )
    t_df["future_trade_adjust"
            ] = np.where(
                    (t_df["future_trade_temp"] > 0) & (~t_df["import_country_code"].isin(ccg_countries)),
                    t_df["future_trade_temp"
                    ]*t_df["export_diff"
                    ]/t_df.groupby(["export_country_code",
                        "reference_mineral","refining_stage_cam"])["future_trade_temp"].transform("sum"),
                    0)
    t_df["future_trade_quantity_tons"] += t_df["future_trade_adjust"]
    updated_trade_df = t_df.copy()
    updated_trade_df["historic_exports"
        ] = updated_trade_df.groupby(
                    [
                        "export_country_code",
                        "reference_mineral",
                        "refining_stage_cam"]
                    )["trade_quantity_tons"].transform("sum")
    updated_trade_df["historic_imports"
        ] = updated_trade_df.groupby(
                    [
                        "import_country_code",
                        "reference_mineral",
                        "refining_stage_cam"]
                    )["trade_quantity_tons"].transform("sum")
    updated_trade_df["future_exports"
        ] = updated_trade_df.groupby(
                    [
                        "export_country_code",
                        "reference_mineral",
                        "refining_stage_cam"]
                    )["future_trade_quantity_tons"].transform("sum")
    updated_trade_df["future_imports"
        ] = updated_trade_df.groupby(
                    [
                        "import_country_code",
                        "reference_mineral",
                        "refining_stage_cam"]
                    )["future_trade_quantity_tons"].transform("sum")
    updated_trade_df["ccg_export_diff"] = updated_trade_df["final_tons_export"] - updated_trade_df["future_exports"]
    # updated_trade_df.to_csv("test0.csv")
    new_trade_df = updated_trade_df[
                        (updated_trade_df["ccg_export_diff"] > 1e-2
                        ) & (updated_trade_df["export_country_code"].isin(ccg_countries))]
    updated_trade_df = updated_trade_df[
                        (updated_trade_df["ccg_export_diff"] <= 1e-2
                        )]
    updated_trade_df["trade_quantity_tons"] = updated_trade_df["future_trade_quantity_tons"]
    updated_trade_df["trade_value_thousandUSD"] = updated_trade_df["cost_to_tons_ratio"]*updated_trade_df["trade_quantity_tons"]

    
    new_trade_df = new_trade_df[
                    ["export_country_code","reference_mineral",
                    "refining_stage_cam","ccg_export_diff"]
                    ].drop_duplicates(
                    subset=[
                            "export_country_code",
                            "reference_mineral",
                            "refining_stage_cam"],
                    keep="first")
    export_countries = trade_df[export_country_columns]
    import_countries = trade_df[import_country_columns]
    import_countries.columns = export_country_columns
    all_countries = pd.concat([export_countries,import_countries],axis=0,ignore_index=True)
    all_countries = all_countries.drop_duplicates(subset=["export_country_code"],keep="first")
    new_trade_df = pd.merge(new_trade_df,all_countries,how="left",on=["export_country_code"])
    value_column = "trade_value_thousandUSD" 
    tons_column = "trade_quantity_tons"
    existing_trade_df = trade_df[
                            (~trade_df["import_country_code"].isin(ccg_countries)
                            ) & (trade_df[tons_column] > 0)]
    # import_groupby_columns = [
    #     c for c in trade_df.columns.values.tolist() if c not in export_country_columns + [value_column,tons_column]
    #     ]
    import_groupby_columns = import_country_columns + product_columns + ["ccg_mineral","refining_stage_cam"]
    # print (import_groupby_columns)
    global_import_df = get_importer_shares(
                            existing_trade_df,
                            import_groupby_columns,
                            value_column,tons_column,
                            new_trade_minerals)
    # global_import_df.to_csv("global_imports.csv",index=False)

    added_trade_df = []
    minerals_stages = list(
                        set(
                            zip(
                                new_trade_df["reference_mineral"].values.tolist(),
                                new_trade_df["refining_stage_cam"].values.tolist()
                                )
                            )
                        )
    for idx,(rm,st) in enumerate(minerals_stages):
        c_df = global_import_df[
                            (
                                global_import_df["reference_mineral"] == rm
                            ) & (
                                global_import_df["refining_stage_cam"] == st
                            )
                        ]
        df = new_trade_df[
                            (
                                new_trade_df["reference_mineral"] == rm
                            ) & (
                                new_trade_df["refining_stage_cam"] == st
                            )
                        ]
        df.drop(["reference_mineral","refining_stage_cam"],axis=1,inplace=True)
        df = pd.merge(df,c_df,how="cross")
        df["trade_quantity_tons"] = df["ccg_export_diff"]*df["import_shares"]
        df["trade_value_thousandUSD"] = df["trade_quantity_tons"]*df["cost_to_tons_ratio"]
        df["ccg_exporter"] = 1
        added_trade_df.append(df[trade_df_columns])
    added_trade_df = pd.concat(added_trade_df,axis=0,ignore_index=True)
    # added_trade_df["ccg_exporter"] = 1
    # added_trade_df.to_csv("new_trade.csv",index=False)

    updated_trade_df = pd.concat([updated_trade_df[trade_df_columns],added_trade_df],axis=0,ignore_index=True)
    # updated_trade_df.to_csv("test3.csv",index=False)


    # Now we need to incorporate the Export and Import Breakdown and location into the trade matrix
    # First get all the exports balances in terms of the production out of exporting countries 
    export_df = final_trade_df[final_trade_df["trade_type"] == "Export"]
    t_df = updated_trade_df[updated_trade_df["export_country_code"].isin(ccg_countries)]
    t_df = pd.merge(t_df,export_df,how="left",
                        on=[
                            "export_country_code",
                            "reference_mineral",
                            "refining_stage_cam"
                            ]).fillna(0)
    t_df["initial_stage_production_tons"] = np.where(
                                        t_df["final_tons"] > 0,
                                        t_df["initial_tons"]*t_df["trade_quantity_tons"]/t_df["final_tons"],
                                        0)
    t_df["final_stage_production_tons"] = t_df["trade_quantity_tons"]
    # Get the balances for the products going out of Africa completely
    port_bound_df = t_df[(t_df["import_continent"] != "Africa") & (~t_df["import_country_code"].isin(ccg_countries))]
    port_bound_df["final_processing_location"] = "port"
    # port_bound_df.to_csv("port_bound.csv",index=False)
    
    # Get the balances for the products in Africa to non-CCG countries
    non_ccg_africa_bound_df = t_df[(t_df["import_continent"] == "Africa") & (~t_df["import_country_code"].isin(ccg_countries))]
    non_ccg_africa_bound_df["final_processing_location"] = "city_demand"
    # non_ccg_africa_bound_df.to_csv("non_ccg_africa_bound.csv",index=False)

    import_df = final_trade_df[final_trade_df["trade_type"] == "Import"]
    import_df["add_tons"] = import_df.groupby(
                                ["export_country_code","reference_mineral",
                                "final_processing_stage"])["final_tons"].transform("sum")
    import_df["location_fraction"
        ] = import_df["final_tons"]/import_df["add_tons"] 
    # import_df.to_csv("test.csv")
    import_df.rename(
        columns={
            "export_country_code":"import_country_code",
            # "final_processing_stage":"refining_stage_cam",
            # "final_tons":"final_tons_import"
            },
        inplace=True)
    # Get the balances for the products in Africa to CCG countries
    ccg_africa_bound_df = t_df[t_df["import_country_code"].isin(ccg_countries)]
    ccg_africa_bound_df.drop("final_processing_location",axis=1,inplace=True)
    ccg_africa_bound_df = pd.merge(ccg_africa_bound_df,
                                import_df[["import_country_code",
                                "reference_mineral","refining_stage_cam",
                                "final_processing_location","location_fraction"]],how="left",
                                on=["import_country_code",
                                "reference_mineral","refining_stage_cam"]).fillna(0)
    ccg_africa_bound_df["initial_stage_production_tons"
        ] = ccg_africa_bound_df["location_fraction"]*ccg_africa_bound_df["initial_stage_production_tons"
        ] 
    ccg_africa_bound_df["final_stage_production_tons"
        ] = ccg_africa_bound_df["location_fraction"]*ccg_africa_bound_df["final_stage_production_tons"
        ] 
    # ccg_africa_bound_df.to_csv("test.csv")
    ccg_africa_bound_df = pd.merge(ccg_africa_bound_df,mine_city_stages,how="left",on=["reference_mineral"])
    ccg_africa_bound_df["initial_processing_location"
        ] = np.where(
                ccg_africa_bound_df["final_processing_stage"] <= ccg_africa_bound_df["mine_highest_stage"],
                "mine","city_process")
    ccg_africa_bound_df["final_processing_location"
        ] = np.where(
                (ccg_africa_bound_df["final_processing_stage"
                ] < ccg_africa_bound_df["mine_highest_stage"]
                ) & (ccg_africa_bound_df["final_processing_location"] == "city_process"),
                "mine",ccg_africa_bound_df["final_processing_location"
        ])
    ccg_imports_df = updated_trade_df[
                        (
                            ~updated_trade_df["export_country_code"].isin(ccg_countries)
                        ) & (
                            updated_trade_df["import_country_code"].isin(ccg_countries)
                        )]
    ccg_imports_df = pd.merge(ccg_imports_df,
                                import_df[["import_country_code",
                                "reference_mineral","refining_stage_cam",
                                "final_processing_location","location_fraction"]],how="left",
                                on=["import_country_code",
                                "reference_mineral","refining_stage_cam"]).fillna(0)
    ccg_imports_df[
        "initial_processing_stage"
        ] = ccg_imports_df[
        "final_processing_stage"
        ] = ccg_imports_df["refining_stage_cam"]
    ccg_imports_df["trade_type"] = "Import"  
    ccg_imports_df[
            "initial_stage_production_tons"
            ] = ccg_imports_df[
            "final_stage_production_tons"
            ] = ccg_imports_df["trade_quantity_tons"]*ccg_imports_df["location_fraction"]

    ccg_imports_df["initial_processing_location"] = np.where(
                                                        ccg_imports_df["export_continent"] == "Africa",
                                                        "city_process","port") 

    ccg_africa_bound_df = pd.concat([ccg_africa_bound_df,ccg_imports_df],axis=0,ignore_index=True).fillna(0)
    # ccg_africa_bound_df.to_csv("ccg_africa_bound.csv",index=False)

    # Get the domestic trade now
    st_df = updated_trade_df[product_columns + ["refining_stage_cam"]]
    st_df = st_df.drop_duplicates(
                            subset=[
                                    "reference_mineral","refining_stage_cam"],
                            keep="first")
    st_df.rename(columns={"refining_stage_cam":"final_processing_stage"},inplace=True)
    domestic_df = final_trade_df[final_trade_df["trade_type"]=="Domestic"]
    domestic_df.rename(
                        columns={
                                "initial_tons":"initial_stage_production_tons",
                                "final_tons":"final_stage_production_tons"},
                                inplace=True)
    
    domestic_df = pd.merge(domestic_df,st_df,how="left",on=["reference_mineral","final_processing_stage"])
    domestic_df = pd.merge(
                        domestic_df,
                        all_countries,
                        how="left",on=["export_country_code"])
    domestic_df[import_country_columns] = domestic_df[export_country_columns]
    # domestic_df.to_csv("test5.csv")

    final_trade_matrix_df = pd.concat(
                                    [
                                        port_bound_df[final_trade_columns],
                                        non_ccg_africa_bound_df[final_trade_columns],
                                        ccg_africa_bound_df[final_trade_columns],
                                        domestic_df[final_trade_columns]
                                    ],
                                axis=0,ignore_index=True)
    final_trade_matrix_df = final_trade_matrix_df[
                            final_trade_matrix_df["final_stage_production_tons"]>0
                            ]
    # print (final_trade_matrix_df)
    groupby_cols = [c for c in final_trade_matrix_df.columns.values.tolist() if c not in ["initial_stage_production_tons","final_stage_production_tons"]]
    final_trade_matrix_df = final_trade_matrix_df.groupby(
                                groupby_cols
                                )[["initial_stage_production_tons",
                                "final_stage_production_tons"]].sum().reset_index()
    # print (final_trade_matrix_df)
    final_trade_matrix_df.to_csv(
                            os.path.join(
                                results_folder,
                                f"baci_ccg_country_trade_breakdown_{year}_{percentile}_{efficient_scale}.csv"),
                            index=False)

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

