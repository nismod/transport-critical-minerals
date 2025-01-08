#!/usr/bin/env python
# coding: utf-8

import os 
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
from utils import *

config = load_config()
incoming_data_path = config['paths']['incoming_data']
processed_data_path = config['paths']['data']
output_data_path = config['paths']['results']

def get_columns_names():
    data_type = {"initial_refined_stage":"str","final_refined_stage":"str"}
    export_country_columns = [
                                "export_country_name",
                                "export_country_code",
                                "export_continent",
                                "export_landlocked"
                            ]
    import_country_columns = [
                                "import_country_name",
                                "import_country_code",
                                "import_continent",
                                "import_landlocked"
                            ]
    product_columns = [
                        "product_code",
                        "product_description",
                        "refining_stage",
                        "reference_mineral",   
                        "processing_level"
                        ]
    conversion_factor_column = "aggregate_ratio"

    trade_balance_columns = [
                                "export_country_code",
                                "reference_mineral",
                                "refining_stage_cam",
                                "initial_processing_stage",
                                "final_processing_stage",
                                "initial_processing_location",
                                "initial_processed_tons",
                                "final_processed_tons",
                                "trade_type"
                            ]
    final_trade_columns = export_country_columns + import_country_columns + product_columns + [ 
                            "initial_processing_stage",
                            "final_processing_stage",
                            "initial_processing_location",
                            "final_processing_location",
                            "trade_type",
                            "initial_stage_production_tons",
                            "final_stage_production_tons"
                        ]
    reference_minerals = ["copper","cobalt","manganese","lithium","graphite","nickel"]

    return (data_type, export_country_columns, 
            import_country_columns,product_columns, 
            conversion_factor_column, trade_balance_columns,
            final_trade_columns,reference_minerals)

def get_common_input_dataframes(data_type,refining_year,trade_year):    
    # Read the data on the conversion factors to go from one stage to another
    # This will help in understanding material requirements for production of a stage output
    # from the inputs of another stage                        
    pr_conv_factors_df = pd.read_excel(os.path.join(processed_data_path,
                                        "mineral_usage_factors",
                                        "aggregated_stages.xlsx"),dtype=data_type)[[
                                            "reference_mineral",
                                            "initial_refined_stage",
                                            "final_refined_stage",
                                            "aggregate_ratio"
                                            ]]
    # Read the data on the usage of stage 1 (or metal content converted to higher stage)
    mineral_usage_factor_df = pd.read_excel(os.path.join(processed_data_path,
                                        "mineral_usage_factors",
                                        "mineral_usage_factors.xlsx"))[[
                                            "reference_mineral",
                                            "final_refined_stage",
                                            "usage_factor"
                                            ]]
    mineral_usage_factor_df = mineral_usage_factor_df.drop_duplicates(
                                    subset=["reference_mineral","final_refined_stage"],
                                    keep="first")
    # Read the data on how much metal content goes into ores and concentrates
    metal_content_factors_df = pd.read_csv(os.path.join(processed_data_path,
                                            "mineral_usage_factors",
                                            "metal_content.csv"))
    metal_content_factors_df.rename(
                        columns={
                            "Reference mineral":"reference_mineral",
                            "Input metal content":"metal_content_factor"},
                        inplace=True)
    
    # Read the finalised version of the BACI trade data
    ccg_countries = pd.read_csv(
                        os.path.join(processed_data_path,
                                "baci","ccg_country_codes.csv"))
    ccg_countries = ccg_countries[ccg_countries["ccg_country"] == 1]["iso_3digit_alpha"].values.tolist()

    # Read the data on the highest stages at the mines
    # This will help identify which stage goes to mine and which outside
    mine_city_stages = pd.read_csv(os.path.join(processed_data_path,"baci","mine_city_stages.csv"))
    mine_city_stages = mine_city_stages[
                            mine_city_stages["year"] == refining_year
                            ][["reference_mineral","mine_final_refined_stage"]]
    
    trade_df = pd.read_csv(
                    os.path.join(processed_data_path,
                        "baci",f"baci_ccg_minerals_trade_{trade_year}_bgs_corrected.csv"))
    trade_df = trade_df[trade_df["trade_quantity_tons"]>0]

    return (pr_conv_factors_df, 
            metal_content_factors_df, ccg_countries, 
            mine_city_stages, trade_df, mineral_usage_factor_df)

def modify_mineral_usage_factors(future_year=2030,baseline_year=2022):
    (data_type, _, _,_, _, _,_,_) = get_columns_names()
    (_, _, _,mcs_df,_, muf_df) = get_common_input_dataframes(
                                        data_type,future_year,baseline_year)

    muf_df["mod_usage_factor"
        ] = muf_df.groupby(["reference_mineral"])["usage_factor"].cumprod()
    muf_df = pd.merge(muf_df,mcs_df,how="left",on=["reference_mineral"])
    muf_df["mod_usage_factor"
        ] = np.where(
                    muf_df["final_refined_stage"] > muf_df["mine_final_refined_stage"],
                    0,
                    muf_df["mod_usage_factor"])
    muf_df = muf_df.sort_values(by=["reference_mineral","final_refined_stage"],ascending=False)
    muf_df["final_usage_factor"
        ] = muf_df.groupby(["reference_mineral"])["mod_usage_factor"].diff()
    muf_df["final_usage_factor"] = muf_df["final_usage_factor"].fillna(muf_df["mod_usage_factor"])
    muf_df["usage_factor"] = muf_df["final_usage_factor"]
    muf_df["cum_usage_factor"
        ] = muf_df[muf_df["final_refined_stage"] > 1.0
            ].groupby(["reference_mineral"])["usage_factor"].transform("sum")
    muf_df["cum_usage_factor"] = muf_df["cum_usage_factor"].fillna(0)
    muf_df.drop(
                [
                    "mod_usage_factor",
                    "final_usage_factor",
                    "mine_final_refined_stage"
                ],
                axis=1,inplace=True)
    return muf_df[(muf_df["usage_factor"] > 0) & (muf_df["cum_usage_factor"] > 0)]


def get_trade_exports_imports(trade_df,ccg_countries):
    export_df = trade_df[trade_df["export_country_code"].isin(ccg_countries)]
    import_df = trade_df[trade_df["import_country_code"].isin(ccg_countries)]

    sum_columns = ["reference_mineral","refining_stage_cam"]
    export_df = export_df.groupby(["export_country_code"]+sum_columns)["trade_quantity_tons"].sum().reset_index()
    export_df.rename(
            columns={
                    "trade_quantity_tons":"trade_quantity_tons_export"},
            inplace=True)

    import_df = import_df.groupby(["import_country_code"]+sum_columns)["trade_quantity_tons"].sum().reset_index()
    import_df.rename(
                columns={
                        "import_country_code":"export_country_code",
                        "trade_quantity_tons":"trade_quantity_tons_import"},
                inplace=True)
    trade_balance_df = pd.merge(
                                export_df,
                                import_df,
                                how="outer",
                                on=["export_country_code"] + sum_columns,
                                ).fillna(0)
    return trade_balance_df, export_df, import_df

def get_mine_layer(reference_mineral,year,percentile,mine_id_col="id",return_columns=None):
    mines_df = gpd.read_file(
                        os.path.join(
                            processed_data_path,
                            "minerals",
                            "s_and_p_mines_current_and_future_estimates.gpkg"),
                    layer=f"{reference_mineral}_{percentile}")
    mines_df.rename(columns={"ISO_A3":"iso3","mine_id":mine_id_col},inplace=True)
    mines_df["weight"] = mines_df[str(year)]
    if return_columns is None:
        return mines_df
    else:
        return mines_df[return_columns]

def bgs_tonnage_estimates():
    bgs_totals = pd.read_excel(
                        os.path.join(
                            processed_data_path,
                            "baci","BGS_SnP_comparison.xlsx"),
                        index_col=[0],header=[0,1]).fillna(0)
    bgs_totals = bgs_totals.reset_index()
    original_columns = bgs_totals.columns.values.tolist()
    columns = [original_columns[0]] + [c for c in original_columns[1:] if c[0] == 'Max SP BGS']
    bgs_totals = bgs_totals[columns]
    reference_minerals = [c[1].lower() for c in columns[1:]]
    bgs_totals.columns = ["export_country_code"] + reference_minerals
    
    bgs_totals_by_mineral = []
    for reference_mineral in reference_minerals:
        df = bgs_totals[["export_country_code",reference_mineral]]
        df["reference_mineral"] = reference_mineral
        df.rename(columns={reference_mineral:"SP_BGS_max"},inplace=True)
        bgs_totals_by_mineral.append(df)

    bgs_totals_by_mineral = pd.concat(bgs_totals_by_mineral,axis=0,ignore_index=True)
    bgs_totals_by_mineral[
        "SP_BGS_max"
        ] = np.where(bgs_totals_by_mineral["reference_mineral"] == "lithium",
            5.323*bgs_totals_by_mineral["SP_BGS_max"],
            bgs_totals_by_mineral["SP_BGS_max"]
        )
    return bgs_totals_by_mineral, "SP_BGS_max"

if __name__ == '__main__':
    modify_mineral_usage_factors(future_year=2030)
    