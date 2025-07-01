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

def get_common_input_dataframes(data_type,planned_scenario,refining_year,trade_year):    
    # Read the data on the conversion factors to go from one stage to another
    # This will help in understanding material requirements for production of a stage output
    # from the inputs of another stage                        
    pr_conv_factors_df = pd.read_excel(os.path.join(processed_data_path,
                                        "mineral_usage_factors",
                                        "aggregated_stages_modified.xlsx"))
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
    mine_city_stages = pd.read_csv(os.path.join(processed_data_path,"baci","mine_city_stages_new.csv"))
    mine_city_stages = mine_city_stages[
                            (
                                mine_city_stages["year"] == refining_year
                            ) & (
                                mine_city_stages["planned_scenario"] == planned_scenario
                            )
                            ][["reference_mineral","mine_final_refined_stage"]]
    
    trade_df = pd.read_csv(
                    os.path.join(processed_data_path,
                        "baci",f"baci_ccg_minerals_trade_{trade_year}_bgs_corrected.csv"))
    trade_df = trade_df[trade_df["trade_quantity_tons"]>0]

    return (pr_conv_factors_df, 
            metal_content_factors_df, ccg_countries, 
            mine_city_stages, trade_df, mineral_usage_factor_df)

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

def modify_mineral_usage_factors(scenario,future_year=2030,baseline_year=2022):
    (data_type, _, _,_, _, _,_,_) = get_columns_names()
    (_, _, ccg_cnts,mcs_df,_, muf_df) = get_common_input_dataframes(
                                        data_type,scenario,future_year,baseline_year)
    cnt_df = pd.DataFrame(ccg_cnts,columns=["export_country_code"])
    mineral_df = pd.DataFrame(
                        list(
                                set(muf_df["reference_mineral"].values.tolist()
                                    )
                            ),columns=["reference_mineral"]
                        )
    cnt_df = pd.merge(cnt_df,mineral_df,how="cross")
    del mineral_df

    if scenario == "bau":
        tr_df = pd.read_csv(
                        os.path.join(
                            output_data_path,
                            "baci_trade_matrices",
                            f"baci_ccg_country_trade_breakdown_{baseline_year}_baseline.csv")
                        )
        tr_df = tr_df[tr_df["initial_stage_production_tons"] > 0]
        exp_df = tr_df[tr_df["export_country_code"].isin(ccg_cnts)]
        exp_df = exp_df[exp_df["initial_processing_stage"] == 0.0]
        muf_df = exp_df.groupby(
                    [
                        "export_country_code",
                        "reference_mineral",
                        "final_processing_stage"
                    ]
                    ).agg({"initial_stage_production_tons":"sum"}).reset_index()
        muf_df["usage_factor"
            ] = muf_df["initial_stage_production_tons"]/muf_df.groupby(
                                                    [
                                                        "export_country_code",
                                                        "reference_mineral"]
                                                    )["initial_stage_production_tons"
                                                    ].transform("sum")
        muf_df.rename(columns={"final_processing_stage":"final_refined_stage"},inplace=True)
        muf_df.drop(
                    [
                        "initial_stage_production_tons"
                    ],
                    axis=1,inplace=True)
        cnt_df["final_refined_stage"] = 1.0
        muf_df = pd.merge(
                        muf_df,
                        cnt_df,
                        how="outer",
                        on=["export_country_code","reference_mineral","final_refined_stage"]
                        )
        muf_df["usage_factor"
            ] = muf_df["usage_factor"].fillna(1.0)
        muf_df["cum_usage_factor"
                ] = muf_df.groupby(
                        ["export_country_code",
                        "reference_mineral"]
                        )["usage_factor"].transform("sum")
    else:
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
        muf_df.drop(
                    [
                        "mod_usage_factor",
                        "final_usage_factor",
                        "mine_final_refined_stage"
                    ],
                    axis=1,inplace=True)
        muf_df = pd.merge(cnt_df,muf_df,how="left",on=["reference_mineral"])

        muf_df["cum_usage_factor"
                ] = muf_df[muf_df["final_refined_stage"] > 1.0
                    ].groupby(
                        ["export_country_code",
                        "reference_mineral"]
                        )["usage_factor"].transform("sum")
    muf_df["cum_usage_factor"] = muf_df["cum_usage_factor"].fillna(0)
    return muf_df[(muf_df["usage_factor"] > 0) & (muf_df["cum_usage_factor"] > 0)]

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
    reference_minerals = list(set([c[1] for c in original_columns[1:]]))
    for rf in reference_minerals:
        bgs_totals[('Max SP BGS',rf)] = bgs_totals[[("SP",rf),("BGS",rf)]].max(axis=1)
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

    # Read the data on how much metal content goes into ores and concentrates
    mine_metal_conversion_df = pd.read_csv(os.path.join(processed_data_path,
                                            "mineral_usage_factors",
                                            "mine_metal_content_conversion.csv"))
    ccg_countries = list(set(mine_metal_conversion_df["ISO_A3"].values.tolist()))
    bgs_ccg = bgs_totals_by_mineral[bgs_totals_by_mineral["export_country_code"].isin(ccg_countries)]
    bgs_ccg = pd.merge(
                        bgs_ccg,
                        mine_metal_conversion_df,
                        how="left",
                        left_on=["export_country_code","reference_mineral"],
                        right_on=["ISO_A3","reference_mineral"])
    bgs_ccg.drop("ISO_A3",axis=1,inplace=True)
    mine_metal_conversion_df = mine_metal_conversion_df.groupby(
                                        ["reference_mineral"]
                                        )[["mine_conversion_factor"]].agg(pd.Series.mode).reset_index()

    bgs_non_ccg = bgs_totals_by_mineral[~bgs_totals_by_mineral["export_country_code"].isin(ccg_countries)]
    bgs_non_ccg = pd.merge(bgs_non_ccg,mine_metal_conversion_df,how="left",on=["reference_mineral"])

    bgs_totals_by_mineral = pd.concat([bgs_ccg,bgs_non_ccg],axis=0,ignore_index=True)
    bgs_totals_by_mineral["SP_BGS_metal_content"
        ] = bgs_totals_by_mineral["SP_BGS_max"
        ]*bgs_totals_by_mineral["mine_conversion_factor"]
    # bgs_totals_by_mineral[
    #     "SP_BGS_max"
    #     ] = np.where(bgs_totals_by_mineral["reference_mineral"] == "lithium",
    #         5.323*bgs_totals_by_mineral["SP_BGS_max"],
    #         bgs_totals_by_mineral["SP_BGS_max"]
    #     )
    return bgs_totals_by_mineral, "SP_BGS_metal_content"

def get_modified_aggregate_ratios():
    df = pd.read_excel(
                    os.path.join(
                        processed_data_path,
                        "mineral_usage_factors",
                        "mineral_extraction_country_intensities (final units w Co edits).xlsx"),
                    sheet_name = "Country material ratios")
    columns = [
                "iso3","reference_mineral","processing_stage",
                "aggregate_ratio (metal content for stage 1 and mass of relevant output for the other stages)"
                ]
    rename_columns = [
                "iso3","reference_mineral","final_refined_stage",
                "aggregate_ratio"
                ]
    df = df[columns]
    df.columns = rename_columns
    df["initial_refined_stage"] = 1.0
    df_stage1 = df[df["final_refined_stage"] == 1.0]
    df_stage1.rename(columns={"aggregate_ratio":"metal_content_factor"},inplace=True)
    df = pd.merge(
                    df,
                    df_stage1[["iso3","reference_mineral","metal_content_factor"]],
                    how="left",
                    on=["iso3","reference_mineral"]
                )
    df["aggregate_ratio_normalised"] = df["aggregate_ratio"]/df["metal_content_factor"]
    # df.drop("stage1_factor",axis=1,inplace=True)

    writer = pd.ExcelWriter(os.path.join(
                        processed_data_path,
                        "mineral_usage_factors",
                        "aggregated_stages_modified.xlsx"))
    df.to_excel(writer,index=False)
    writer.close()

def add_conversion_factors(df_trade,df_factors,ccg_countries,trade_iso_column="iso_code"):
    df_ccg = df_trade[df_trade[trade_iso_column].isin(ccg_countries)]
    df_non_ccg = df_trade[~df_trade[trade_iso_column].isin(ccg_countries)]
    df_ccg = pd.merge(
                    df_ccg,
                    df_factors[[
                                "iso3","reference_mineral",
                                "final_refined_stage",
                                "aggregate_ratio",
                                "metal_content_factor"
                            ]],
                    how="left",
                    left_on=[trade_iso_column,"reference_mineral","refining_stage_cam"],
                    right_on=["iso3","reference_mineral","final_refined_stage"]
                )
    df_ccg.drop("iso3",axis=1,inplace=True)
    df_factors_non_ccg = df_factors.groupby(
                                        ["reference_mineral","final_refined_stage"]
                                        )[["aggregate_ratio","metal_content_factor"]].agg(pd.Series.mode).reset_index()
    df_factors_non_ccg["refining_stage_cam"] = df_factors_non_ccg["final_refined_stage"]
    df_non_ccg = pd.merge(df_non_ccg,df_factors_non_ccg,how="left")

    df = pd.concat([df_ccg,df_non_ccg],axis=0,ignore_index=True)
    return df 

if __name__ == '__main__':
    # modify_mineral_usage_factors("early refining",future_year=2040)
    # modify_mineral_usage_factors("precursor",future_year=2040)
    # modify_mineral_usage_factors("bau",future_year=2040)
    get_modified_aggregate_ratios()
    # bgs_tonnage_estimates()
    