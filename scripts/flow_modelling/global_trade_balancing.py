#!/usr/bin/env python
# coding: utf-8

import os 
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
from utils import *

def get_conversion_factors(x,mc_df,pcf_df,cf_column="aggregate_ratio"):
    ref_min = x["reference_mineral"]
    mc_exp = mc_df[mc_df["reference_mineral"] == ref_min]["metal_content_factor"].values[0]

    exp_st = x["refining_stage_cam"]
    imp_st = x["import_stage_consumed"]
    cf_df = pcf_df[pcf_df["reference_mineral"] == ref_min]
    cf_exp = cf_df[cf_df["final_refined_stage"] == str(exp_st).replace(".0","")
                    ][cf_column].values[0]/cf_df[
                    cf_df["final_refined_stage"] == '1'
                    ][cf_column].values[0]
    cf_imp = cf_df[cf_df["final_refined_stage"] == str(exp_st).replace(".0","")
                    ][cf_column].values[0]/cf_df[
                    cf_df["final_refined_stage"] == str(imp_st).replace(".0","")
                    ][cf_column].values[0]

    mc_imp = mc_exp*cf_df[cf_df["final_refined_stage"] == str(imp_st).replace(".0","")
                    ][cf_column].values[0]/cf_df[
                    cf_df["final_refined_stage"] == '1'
                    ][cf_column].values[0]

    return mc_exp, mc_imp, cf_exp, cf_imp


def main(config):

    incoming_data_path = config['paths']['incoming_data']
    processed_data_path = config['paths']['data']
    output_data_path = config['paths']['results']

    results_folder = os.path.join(output_data_path,"baci_trade_matrices")
    if os.path.exists(results_folder) == False:
        os.mkdir(results_folder)

    baseline_year = 2022
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
                                "import_stage_consumed",     
                                "total_metal_content_production_for_export_tons",
                                "total_metal_content_import_for_export_tons", 
                                "import_consumed_fraction",    
                                "production_metal_content_factor",
                                "import_metal_content_factor", 
                                "production_stage_factor",   
                                "import_stage_factor", 
                                "total_production_for_export_tons",    
                                "total_import_for_export_tons",
                                "total_export_tons",
                            ]
    reference_minerals = ["copper","cobalt","manganese","lithium","graphite","nickel"]
    
    # Read the data on the conversion factors to go from one stage ot another
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
    # Read the data on how much metal content goes into ores and concentrates
    metal_content_factors_df = pd.read_csv(os.path.join(processed_data_path,
                                            "mineral_usage_factors",
                                            "metal_content.csv"))
    metal_content_factors_df.rename(
                        columns={
                            "Reference mineral":"reference_mineral",
                            "Input metal content":"metal_content_factor"},
                        inplace=True)
    
    # Read the BGS total of metal production
    bgs_totals = pd.read_excel(
                        os.path.join(
                            processed_data_path,
                            "baci","BGS_SnP_comparison.xlsx"),
                        index_col=[0,1])
    bgs_totals = bgs_totals.reset_index()
    bgs_totals.rename(columns={"level_0":"reference_mineral","level_1":"export_country_code"},inplace=True)
    bgs_totals["reference_mineral"] = bgs_totals["reference_mineral"].str.lower()

    # Read the finalised version of the BACI trade data
    ccg_countries = pd.read_csv(
                        os.path.join(processed_data_path,
                                "baci","ccg_country_codes.csv"))
    ccg_countries = ccg_countries[ccg_countries["ccg_country"] == 1]["iso_3digit_alpha"].values.tolist()

    # Read the global trade data 
    trade_df = pd.read_csv(
                    os.path.join(processed_data_path,
                        "baci",f"baci_ccg_minerals_trade_{baseline_year}_updated.csv"))
    final_trade_columns = trade_df.columns.values.tolist()
    trade_df = trade_df[trade_df["trade_quantity_tons"]>0]
    trade_df = pd.merge(
                    trade_df,
                    metal_content_factors_df[
                        ["reference_mineral",
                        "metal_content_factor"]],
                    how="left",on=["reference_mineral"])
    trade_df["trade_quantity_tons"] = np.where(trade_df["refining_stage_cam"] == 1,
                                        trade_df["trade_quantity_tons"]/trade_df["metal_content_factor"],
                                        trade_df["trade_quantity_tons"])
    trade_df.drop("metal_content_factor",axis=1,inplace=True)

    # Get the total tonnage of exports and imports of each country
    sum_columns = ["reference_mineral","refining_stage_cam"]
    export_df = trade_df.groupby(["export_country_code"]+sum_columns)["trade_quantity_tons"].sum().reset_index()
    export_df.rename(
            columns={
                    "export_country_code":"iso_code",
                    "trade_quantity_tons":"trade_quantity_tons_export"},
            inplace=True)

    import_df = trade_df.groupby(["import_country_code"]+sum_columns)["trade_quantity_tons"].sum().reset_index()
    import_df.rename(
                columns={
                        "import_country_code":"iso_code",
                        "trade_quantity_tons":"trade_quantity_tons_import"},
                inplace=True)
    trade_balance_df = pd.merge(
                                export_df,
                                import_df,
                                how="outer",
                                on=["iso_code"] + sum_columns,
                                ).fillna(0)
    trade_balance_df["metal_content_factor"] = 0
    trade_balance_df["stage_1_factor"] = 0
    for reference_mineral in reference_minerals:
        trade_balance_df.loc[trade_balance_df["reference_mineral"] == reference_mineral,
                "metal_content_factor"] = metal_content_factors_df[
                                            metal_content_factors_df["reference_mineral"] == reference_mineral
                                                ]["metal_content_factor"].values[0]
        conv_factor_df = pr_conv_factors_df[pr_conv_factors_df["reference_mineral"] == reference_mineral]
        stages = sorted([float(s) for s in list(
                        set(
                            trade_df[trade_df["reference_mineral"] == reference_mineral]["refining_stage_cam"].values.tolist()
                            )
                        )])
        for st in stages:
            conversion_factor = conv_factor_df[
                                    conv_factor_df["final_refined_stage"] == str(st).replace(".0","")
                                    ][conversion_factor_column].values[0]/conv_factor_df[
                                    conv_factor_df["final_refined_stage"] == '1'
                                    ][conversion_factor_column].values[0]
            trade_balance_df.loc[
                (trade_balance_df["reference_mineral"] == reference_mineral
                ) & (trade_balance_df["refining_stage_cam"] == st),
                "stage_1_factor"] = conversion_factor


    trade_balance_df.to_csv("export_import_tons_global.csv",index=False)

    # Get the tonnage of extra metal content production  + stage 1 export 
    # required to satisfy the exports of each mineral stage tonnage
    # And also the most efficient tonnage and stage of import that is 
    # consumed to produce a higher stage export tonnage of each mineral stage 
    mineral_balance_df = []
    for reference_mineral in reference_minerals:
        conv_factor_df = pr_conv_factors_df[pr_conv_factors_df["reference_mineral"] == reference_mineral]
        metal_content = metal_content_factors_df[
                                metal_content_factors_df["reference_mineral"] == reference_mineral
                                ]["metal_content_factor"].values[0]
        mb_df = trade_balance_df[
                                trade_balance_df["reference_mineral"] == reference_mineral
                                ]
        countries = list(set(mb_df["iso_code"].values.tolist()))
        for cnt in countries:
            stages = sorted([float(s) for s in list(
                        set(
                            mb_df[mb_df["iso_code"] == cnt]["refining_stage_cam"].values.tolist()
                            )
                        )])
            for st in stages:
                domestic_prod = 0.0
                stage_1_prod = 0.0
                import_consumed = 0.0
                import_st = st
                ex_st = mb_df[
                                    (
                                        mb_df["iso_code"] == cnt
                                    ) & (
                                        mb_df["refining_stage_cam"] == st
                                    )]["trade_quantity_tons_export"].values[0]
                lower_stages = [lst for lst in stages if lst < st]
                if len(lower_stages) > 0:
                    domestic_prods = []
                    for l_st in lower_stages:
                        im_l_st = mb_df[
                                        (
                                            mb_df["iso_code"] == cnt
                                        ) & (
                                            mb_df["refining_stage_cam"] == l_st
                                        )]["trade_quantity_tons_import"].values[0]
                        conversion_factor = conv_factor_df[
                                                conv_factor_df["final_refined_stage"] == str(st).replace(".0","")
                                                ][conversion_factor_column].values[0]/conv_factor_df[
                                                conv_factor_df["final_refined_stage"] == str(l_st).replace(".0","")
                                                ][conversion_factor_column].values[0] 
                        domestic_prods.append(
                                        (
                                            l_st,
                                            im_l_st,
                                            ex_st*conversion_factor,
                                            ex_st*conversion_factor - im_l_st
                                        )
                                        )

                    domestic_prods = sorted(domestic_prods,key=lambda x:x[-1])
                    import_st = domestic_prods[0][0]
                    min_import = domestic_prods[0][1]
                    needed_import = domestic_prods[0][2]
                    import_consumed = min(min_import,needed_import)
                    if domestic_prods[0][-1] > 0:
                        min_prod = domestic_prods[0][-1]
                        conversion_factor = conv_factor_df[
                                                conv_factor_df["final_refined_stage"] == str(import_st).replace(".0","")
                                                ][conversion_factor_column].values[0]/conv_factor_df[
                                                conv_factor_df["final_refined_stage"] == '1'
                                                ][conversion_factor_column].values[0]
                        domestic_prod = metal_content*min_prod*conversion_factor
                        stage_1_prod = min_prod*conversion_factor
                    
                    mb_df.loc[
                            (
                                mb_df["iso_code"] == cnt
                                ) & (
                                    mb_df["refining_stage_cam"] == import_st
                        ),"trade_quantity_tons_import"] = max(0,min_import - import_consumed)
                elif st != 1:
                    conversion_factor = conv_factor_df[
                                                conv_factor_df["final_refined_stage"] == str(st).replace(".0","")
                                                ][conversion_factor_column].values[0]/conv_factor_df[
                                                conv_factor_df["final_refined_stage"] == '1'
                                                ][conversion_factor_column].values[0]

                    domestic_prod = metal_content*ex_st*conversion_factor
                    stage_1_prod = ex_st*conversion_factor
                else:
                    domestic_prod = ex_st*metal_content
                    stage_1_prod = ex_st

                mineral_balance_df.append(
                                            (
                                                cnt,reference_mineral,
                                                st,ex_st,domestic_prod,
                                                stage_1_prod,import_consumed,import_st
                                            )
                                        )
    mineral_balance_df = pd.DataFrame(
                                    mineral_balance_df,
                                    columns=[
                                                "export_country_code",
                                                "reference_mineral",
                                                "refining_stage_cam",
                                                "total_export_tons",
                                                "total_metal_content_production_for_export_tons",
                                                "total_production_stage_1_tons",
                                                "import_consumed_tons",
                                                "import_stage_consumed"]
                                    )
    import_df.rename(columns={"refining_stage_cam":"import_stage_consumed"},inplace=True)
    mineral_balance_df = pd.merge(
                            mineral_balance_df,
                            import_df[["iso_code","reference_mineral","import_stage_consumed","trade_quantity_tons_import"]],
                            how="left",
                            left_on=["export_country_code","reference_mineral","import_stage_consumed"],
                            right_on=["iso_code","reference_mineral","import_stage_consumed"])
    mineral_balance_df["import_consumed_fraction"] = np.where(mineral_balance_df["trade_quantity_tons_import"] >0,
                                mineral_balance_df["import_consumed_tons"]/mineral_balance_df["trade_quantity_tons_import"],
                                0)

    mineral_balance_df["conversion_factors"] = mineral_balance_df.progress_apply(
                                                lambda x:get_conversion_factors(
                                                    x,metal_content_factors_df,pr_conv_factors_df),
                                                axis=1)
    mineral_balance_df[
                [
                    "production_metal_content_factor",
                    "import_metal_content_factor",
                    "production_stage_factor",
                    "import_stage_factor"
                ]] = mineral_balance_df["conversion_factors"].apply(pd.Series)
    mineral_balance_df.drop("conversion_factors",axis=1,inplace=True)
    mineral_balance_df[
        "total_metal_content_import_for_export_tons"
        ] = mineral_balance_df[
            "import_consumed_tons"]*mineral_balance_df["import_metal_content_factor"]
    mineral_balance_df[
        "total_production_for_export_tons"
        ] = mineral_balance_df[
            "total_production_stage_1_tons"]/mineral_balance_df["production_stage_factor"]
    mineral_balance_df[
        "total_import_for_export_tons"
        ] = mineral_balance_df[
            "import_consumed_tons"]/mineral_balance_df["import_stage_factor"]


    mb_df = mineral_balance_df[trade_balance_columns]
    mb_df["baci_tons"] = mb_df.groupby(
                            [
                                "export_country_code",
                                "reference_mineral"
                            ]
                            )["total_metal_content_production_for_export_tons"].transform("sum")
    mb_df.to_csv("metal_production_global.csv",index=False)
    
    print (trade_df)
    t_df = pd.merge(
                trade_df,
                mb_df,
                how="left",
                on=["export_country_code","reference_mineral","refining_stage_cam"]).fillna(0)
    t_df = pd.merge(t_df,bgs_totals,how="left",on=["export_country_code","reference_mineral"]).fillna(0)
    t_df[
        "actual_export_tons"
        ] = t_df[["baci_tons","BGS"]].min(axis=1)
    print (t_df)

    t_df[
      "trade_production_metal_content_export_tons"
      ] = np.where(t_df["baci_tons"] > 0, 
            (
                t_df["trade_quantity_tons"
                ]/t_df["total_export_tons"]
            )*(
                t_df["total_metal_content_production_for_export_tons"
                ]*t_df["actual_export_tons"
                ]/t_df["baci_tons"]
            ),0
            )
    t_df["trade_metal_content_export_tons"] = t_df.groupby(
                                    [
                                        "export_country_code",
                                        "reference_mineral",
                                        "refining_stage_cam"
                                    ])["trade_production_metal_content_export_tons"].transform("sum")

    import_metal_df = t_df.groupby(["import_country_code","reference_mineral","refining_stage_cam"
                      ])["trade_production_metal_content_export_tons"].sum().reset_index()
    import_metal_df.rename(
          columns={
              "import_country_code":"export_country_code",
              "trade_production_metal_content_export_tons":"trade_import_tons",
              "refining_stage_cam":"import_stage_consumed"},inplace=True)
    t_df = pd.merge(
          t_df,
          import_metal_df,
          how="left",
          on=["export_country_code","reference_mineral","import_stage_consumed"]).fillna(0)
    t_df["trade_metal_content_import_tons"] = t_df["trade_import_tons"]*t_df["import_consumed_fraction"]

    t_df[
        "trade_export_stage_tons"
        ] = t_df[
        "trade_metal_content_export_tons"
        ]/(t_df["production_metal_content_factor"]*t_df["production_stage_factor"])

    t_df[
        "trade_import_stage_tons"
        ] = t_df[
        "trade_metal_content_import_tons"
        ]/(t_df["import_metal_content_factor"]*t_df["import_stage_factor"])
    t_df["updated_total_export_tons"] = t_df["trade_export_stage_tons"] + t_df["trade_import_stage_tons"]
    t_df["usd_per_tons"] = t_df["trade_value_thousandUSD"]/t_df["trade_quantity_tons"]
    t_df[
        "trade_quantity_tons"
        ] = t_df["trade_quantity_tons"
        ]*t_df["updated_total_export_tons"]/t_df["total_export_tons"]
    t_df["trade_value_thousandUSD"] = t_df["usd_per_tons"]*t_df["trade_quantity_tons"]
    t_df[final_trade_columns].to_csv(
        os.path.join(processed_data_path,
                    "baci",
                    "baci_ccg_minerals_trade_2022_bgs_corrected.csv"),
                    index=False)
    # mb_df[
    #     "total_metal_content_production_for_domestic_tons"
    #     ] = mb_df["BGS"] - mb_df["actual_export_tons"]
    # mb_df = mb_df.groupby(
    #             ["export_country_code","reference_mineral"]
    #             )["total_metal_content_production_for_export_tons"].sum().reset_index()
    # mb_df.rename(columns={"total_metal_content_production_for_export_tons":"baci_tons"},inplace=True)
    # mb_df = pd.merge(mb_df,bgs_totals,how="outer",on=["export_country_code","reference_mineral"]).fillna(0)
    # mb_df[
    #     "actual_export_tons"
    #     ] = mb_df[["baci_tons","BGS"]].min(axis=1)
    # mb_df[
    #     "total_metal_content_production_for_domestic_tons"
    #     ] = mb_df["BGS"] - mb_df["actual_export_tons"]
    # mb_df.to_csv("baci_bgs_comparison_global.csv",index=False)

    # mb_df = pd.merge(
    #             mineral_balance_df[trade_balance_columns],
    #             md_df[
    #                 [
    #                     "export_country_code",
    #                     "reference_mineral",
    #                     "baci_tons",
    #                     "actual_export_tons"]],
    #             how="left",on=["export_country_code","reference_mineral"])

    
    # print (trade_df)
    # t_df = pd.merge(trade_df,mb_df,how="left",on=["export_country_code","reference_mineral","refining_stage_cam"])
    # print (t_df)
    # t_df.to_csv("test.csv",index=False)



if __name__ == '__main__':
    CONFIG = load_config()
    main(CONFIG)


