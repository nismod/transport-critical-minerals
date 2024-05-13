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
                                "initial_processting_stage",
                                "final_processing_stage",
                                "initial_processing_location",
                                "initial_processed_tons",
                                "final_processed_tons",
                                "trade_type"
                            ]
    final_trade_columns = export_country_columns + import_country_columns + product_columns + [ 
                            "initial_processting_stage",
                            "final_processing_stage",
                            "initial_processing_location",
                            "final_processing_location",
                            "trade_type",
                            "initial_stage_production_tons",
                            "final_stage_production_tons"
                        ]
    reference_minerals = ["copper","cobalt","manganese","lithium","graphite","nickel"]

    # Read the data on the highest stages at the mines
    # This will help identify which stage goes to mine and which outside
    mine_city_stages = pd.read_csv(os.path.join(processed_data_path,"baci","mine_city_stages.csv"))
    mine_city_stages = mine_city_stages[
                            mine_city_stages["year"] == baseline_year
                            ][["reference_mineral","mine_final_refined_stage"]]
    
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
    # Read the data on how much metal concent goes into ores and concentrates
    metal_content_factors_df = pd.read_csv(os.path.join(processed_data_path,
                                            "mineral_usage_factors",
                                            "metal_content.csv"))
    
    # Read the finalised version of the BACI trade data
    ccg_countries = pd.read_csv(
                        os.path.join(processed_data_path,
                                "baci","ccg_country_codes.csv"))
    ccg_countries = ccg_countries[ccg_countries["ccg_country"] == 1]["iso_3digit_alpha"].values.tolist()
    trade_df = pd.read_csv(
                    os.path.join(processed_data_path,
                        "baci",f"baci_ccg_minerals_trade_{baseline_year}_updated.csv"))
    trade_df = trade_df[trade_df["trade_quantity_tons"]>0]

    # Get the total tonnage of exports and imports of each CCG country
    # ccg_countries = list(set(trade_df[trade_df["ccg_exporter"] == 1]["export_country_code"].values.tolist()))
    
    export_df = trade_df[trade_df["export_country_code"].isin(ccg_countries)]
    import_df = trade_df[trade_df["import_country_code"].isin(ccg_countries)]

    sum_columns = ["reference_mineral","refining_stage_cam"]
    export_df = export_df.groupby(["export_country_code"]+sum_columns)["trade_quantity_tons"].sum().reset_index()
    export_df.rename(
            columns={
                    "export_country_code":"iso_code",
                    "trade_quantity_tons":"trade_quantity_tons_export"},
            inplace=True)

    import_df = import_df.groupby(["import_country_code"]+sum_columns)["trade_quantity_tons"].sum().reset_index()
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
    trade_balance_df.to_csv("export_import_tons.csv",index=False)

    # Get the tonnage of extra metal content production  + stage 1 export 
    # required to satisfy the exports of each mineral stage tonnage
    # And also the most efficient tonnage and stage of import that is 
    # consumed to produce a higher stage export tonnage of each mineral stage 
    mineral_balance_df = []
    for reference_mineral in reference_minerals:
        conv_factor_df = pr_conv_factors_df[pr_conv_factors_df["reference_mineral"] == reference_mineral]
        metal_content = metal_content_factors_df[
                                metal_content_factors_df["Reference mineral"] == reference_mineral
                                ]["Input metal content"].values[0]
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
                                            l_st,im_l_st,
                                            conversion_factor,
                                            metal_content,
                                            ex_st*conversion_factor,
                                            ex_st*conversion_factor - im_l_st
                                        )
                                        )

                    domestic_prods = sorted(domestic_prods,key=lambda x:x[-1])
                    import_st = domestic_prods[0][0]
                    import_consumed = min(domestic_prods[0][1],domestic_prods[0][4])
                    if domestic_prods[0][-1] > 0:
                        min_im = domestic_prods[0][1]
                        min_f = domestic_prods[0][2]
                        min_me = domestic_prods[0][3]
                        min_prod = domestic_prods[0][5]

                        conversion_factor = conv_factor_df[
                                                conv_factor_df["final_refined_stage"] == str(import_st).replace(".0","")
                                                ][conversion_factor_column].values[0]/conv_factor_df[
                                                conv_factor_df["final_refined_stage"] == '1'
                                                ][conversion_factor_column].values[0]
                        domestic_prod = min_me*min_prod*conversion_factor
                        stage_1_prod = min_prod*conversion_factor
                        mb_df.loc[
                                (
                                    mb_df["iso_code"] == cnt
                                    ) & (
                                        mb_df["refining_stage_cam"] == import_st
                            ),"trade_quantity_tons_import"] = max(0,min_im - import_consumed)
                elif st != 1:
                    conversion_factor = conv_factor_df[
                                                conv_factor_df["final_refined_stage"] == str(st).replace(".0","")
                                                ][conversion_factor_column].values[0]/conv_factor_df[
                                                conv_factor_df["final_refined_stage"] == '1'
                                                ][conversion_factor_column].values[0]

                    domestic_prod = metal_content*ex_st*conversion_factor
                    stage_1_prod = ex_st*conversion_factor
                else:
                	domestic_prod = ex_st
                	ex_st = ex_st/metal_content
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
                                                "total_export_metal_content_tons",
                                                "total_export_stage_1_tons",
                                                "import_consumed_tons",
                                                "import_stage_consumed"]
                                    )
    mineral_balance_df = pd.merge(mineral_balance_df,mine_city_stages,how="left",on=["reference_mineral"])
    # To identify if the produced metal content will be exported directly from mine or sent to city in-country
    # For example: 
    #     metal to produce copper stage 1 will be exported from mine
    #     metal to produce copper stage 5 will be produced at mine and then sent to city first before export
    mineral_balance_df["trade_type"] = np.where(
                            mineral_balance_df["refining_stage_cam"] <= mineral_balance_df["mine_final_refined_stage"],
                            "Export","Domestic")
    # Location where the import will be headed to within country
    mineral_balance_df["import_location"] = np.where(
                            mineral_balance_df["refining_stage_cam"] < mineral_balance_df["mine_final_refined_stage"],
                            "mine","city_process")
    mineral_balance_df.to_csv("metal_production.csv",index=False)

    # Breakdown the production into in-country trade and final export 
    export_df = mineral_balance_df[mineral_balance_df["trade_type"] == "Export"]
    domestic_df = mineral_balance_df[mineral_balance_df["trade_type"] == "Domestic"]
    domestic_df_copy = domestic_df.copy()

    export_df["initial_processing_location"] = "mine"
    # export_df["final_processing_location"] = "port"

    domestic_df["initial_processing_location"] = "city_process"
    # domestic_df["final_processing_location"] = "port"
    domestic_df["trade_type"] = "Export"

    domestic_df_copy["initial_processing_location"] = "mine"
    # domestic_df_copy["final_processing_location"] = "city"

    export_df["initial_processting_stage"] = 0
    export_df["final_processing_stage"] = export_df["refining_stage_cam"]
    domestic_df["initial_processting_stage"] = 1
    domestic_df["final_processing_stage"] = domestic_df["refining_stage_cam"]
    domestic_df_copy["initial_processting_stage"] = 0
    domestic_df_copy["final_processing_stage"] = 1


    export_df.rename(
                columns={
                    "total_export_tons":"final_processed_tons",
                    "total_export_metal_content_tons":"initial_processed_tons"
                    },inplace=True)
    domestic_df.rename(
                columns={
                    "total_export_tons":"final_processed_tons",
                    "total_export_stage_1_tons":"initial_processed_tons"
                    },inplace=True)
    domestic_df_copy.rename(
                columns={
                    "total_export_stage_1_tons":"final_processed_tons",
                    "total_export_metal_content_tons":"initial_processed_tons"
                    },inplace=True)
    
    export_balance_df = pd.concat(
            [
                export_df[trade_balance_columns],
                domestic_df[trade_balance_columns],
                domestic_df_copy[trade_balance_columns]
            ],axis=0,ignore_index=True)

    export_balance_df.to_csv("export_trade_breakdown.csv",index=False)

    # Get the fraction of the import that goes to a country for
    #     Mine level processing
    #     City level processing
    #     City level final demand  
    country_imports_df = mineral_balance_df.groupby(
                                                [
                                                "export_country_code",
                                                "reference_mineral",
                                                "refining_stage_cam",
                                                "import_stage_consumed",
                                                "import_location"
                                                ])["import_consumed_tons"].sum().reset_index()
    country_imports_df.rename(
                        columns={
                                    "export_country_code":"import_country_code",
                                    "refining_stage_cam":"final_processing_stage",
                                    "import_stage_consumed":"initial_processting_stage"
                                },
                        inplace=True)
    import_df.rename(
            columns={
                        "iso_code":"import_country_code",
                        "refining_stage_cam":"initial_processting_stage"
                    },
            inplace=True)
    country_imports_df = pd.merge(
                            import_df[
                                        [
                                            "import_country_code",
                                            "initial_processting_stage",
                                            "reference_mineral",
                                            "trade_quantity_tons_import"
                                    ]],
                            country_imports_df,
                            how="left",
                            on=["import_country_code","initial_processting_stage","reference_mineral"]).fillna(0)
    country_imports_df = country_imports_df[country_imports_df["trade_quantity_tons_import"] > 0]
    country_imports_df["location_fraction"] = country_imports_df["import_consumed_tons"]/country_imports_df["trade_quantity_tons_import"]
    country_imports_df = country_imports_df[
                                    [
                                        "import_country_code",
                                        "reference_mineral",
                                        "initial_processting_stage",
                                        "final_processing_stage",
                                        "import_location",
                                        "location_fraction"
                                    ]]
    remaining_imports = country_imports_df.groupby(
                                    ["import_country_code",
                                    "reference_mineral",
                                    "initial_processting_stage"])["location_fraction"].sum().reset_index()
    remaining_imports["location_fraction"] = 1 - remaining_imports["location_fraction"]
    remaining_imports["final_processing_stage"] = 100
    remaining_imports["import_location"] = "city_demand"

    country_imports_df = pd.concat([country_imports_df,remaining_imports],axis=0,ignore_index=True)
    country_imports_df = country_imports_df[country_imports_df["location_fraction"] > 0]
    country_imports_df.to_csv("import_shares.csv",index=False)

    # Now we need to incorporate the Export and Import Breakdown and location into the trade matrix
    # First get all the exports balances in terms of the production out of exporting countries 
    export_df = export_balance_df[export_balance_df["trade_type"]=="Export"]
    t_df = trade_df[trade_df["export_country_code"].isin(ccg_countries)]
    t_df = pd.merge(t_df,export_df,how="left",
                        on=[
                            "export_country_code",
                            "reference_mineral",
                            "refining_stage_cam"
                            ])
    # print (t_df)

    t_df["initial_stage_production_tons"] = np.where(
                                        t_df["final_processing_stage"] == 1,
                                        t_df["trade_quantity_tons"],
                                        t_df["initial_processed_tons"]*t_df["trade_quantity_tons"]/t_df["final_processed_tons"])
    t_df["final_stage_production_tons"] = np.where(
                                        t_df["final_processing_stage"] == 1,
                                        t_df["final_processed_tons"]*t_df["trade_quantity_tons"]/t_df["initial_processed_tons"],
                                        t_df["trade_quantity_tons"])
    # t_df["final_stage_value_thousandUSD"] = t_df["trade_value_thousandUSD"]
    print (t_df)
    t_df.to_csv("test.csv",index=False)

    # Get the balances for the products going out of Africa completely
    port_bound_df = t_df[(t_df["import_continent"] != "Africa") & (~t_df["import_country_code"].isin(ccg_countries))]
    port_bound_df["final_processing_location"] = "port"
    port_bound_df.to_csv("port_bound.csv",index=False)
    
    # Get the balances for the products in Africa to non-CCG countries
    non_ccg_africa_bound_df = t_df[(t_df["import_continent"] == "Africa") & (~t_df["import_country_code"].isin(ccg_countries))]
    non_ccg_africa_bound_df["final_processing_location"] = "city_demand"
    non_ccg_africa_bound_df.to_csv("non_ccg_africa_bound.csv",index=False)

    # Get the balances for the products in Africa to non-CCG countries
    ccg_africa_bound_df = t_df[t_df["import_country_code"].isin(ccg_countries)]
    ccg_imports_df = trade_df[
                        (
                            ~trade_df["export_country_code"].isin(ccg_countries)
                        ) & (
                            trade_df["import_country_code"].isin(ccg_countries)
                        )]
    ccg_imports_df[
        "initial_processting_stage"
        ] = ccg_imports_df[
        "final_processing_stage"
        ] = ccg_imports_df["refining_stage_cam"]
    ccg_imports_df["trade_type"] = "Import"  
    ccg_imports_df[
            "initial_stage_production_tons"
            ] = ccg_imports_df[
            "final_stage_production_tons"
            ] = ccg_imports_df["trade_quantity_tons"]

    ccg_imports_df["initial_processing_location"] = np.where(
                                                        ccg_imports_df["export_continent"] == "Africa",
                                                        "city_process","port") 

    ccg_africa_bound_df = pd.concat([ccg_africa_bound_df,ccg_imports_df],axis=0,ignore_index=True).fillna(0)
    ccg_africa_bound_df.to_csv("ccg_africa_bound.csv",index=False)

    country_imports_df.drop("final_processing_stage",axis=1,inplace=True)
    country_imports_df.rename(
            columns={
                "initial_processting_stage":"final_processing_stage",
                "import_location":"final_processing_location"},
            inplace=True)
    ccg_africa_bound_df = pd.merge(
                                ccg_africa_bound_df,
                                country_imports_df,
                                how="left",
                                on=["import_country_code","reference_mineral","final_processing_stage"])
    ccg_africa_bound_df[
            "initial_stage_production_tons"
            ] = ccg_africa_bound_df["location_fraction"]*ccg_africa_bound_df["initial_stage_production_tons"]
    ccg_africa_bound_df[
            "final_stage_production_tons"
            ] = ccg_africa_bound_df["location_fraction"]*ccg_africa_bound_df["final_stage_production_tons"]
    ccg_africa_bound_df.to_csv("ccg_africa_bound_import_shares.csv",index=False)

    # Get the domestic trade now
    st_one_df = trade_df[(trade_df["refining_stage_cam"] == 1)]
    st_one_df = st_one_df[product_columns + ["refining_stage_cam"]]
    st_one_df = st_one_df.drop_duplicates(
                            subset=[
                                    "reference_mineral"],
                            keep="first")
    st_one_df.rename(columns={"refining_stage_cam":"final_processing_stage"},inplace=True)
    domestic_df = export_balance_df[export_balance_df["trade_type"]=="Domestic"]
    domestic_df["final_processing_location"] = "city_process"
    domestic_df = domestic_df.groupby(
                            [
                                "export_country_code",
                                "reference_mineral",
                                "initial_processting_stage",
                                "final_processing_stage",
                                "initial_processing_location",
                                "final_processing_location",
                                "trade_type"
                            ])[["initial_processed_tons","final_processed_tons"]].sum().reset_index()
    domestic_df = domestic_df[domestic_df["final_processed_tons"] > 0]
    domestic_df.rename(
                        columns={
                                "initial_processed_tons":"initial_stage_production_tons",
                                "final_processed_tons":"final_stage_production_tons"},
                                inplace=True)
    
    domestic_df = pd.merge(domestic_df,st_one_df,how="left",on=["reference_mineral","final_processing_stage"])
    domestic_df = pd.merge(
                        domestic_df,
                        trade_df[
                            [
                                "export_country_name",
                                "export_country_code",
                                "export_continent",
                                "export_landlocked",
                                "ccg_exporter",
                                "ccg_mineral"]].drop_duplicates(subset=["export_country_code"],keep="first"),
                        how="left",on=["export_country_code"])
    domestic_df[import_country_columns] = domestic_df[export_country_columns]

    final_trade_matrix_df = pd.concat(
                                    [
                                        port_bound_df[final_trade_columns],
                                        non_ccg_africa_bound_df[final_trade_columns],
                                        ccg_africa_bound_df[final_trade_columns],
                                        domestic_df[final_trade_columns]
                                    ],
                                axis=0,ignore_index=True)
    final_trade_matrix_df.to_csv(
                            os.path.join(
                                results_folder,
                                f"baci_ccg_country_trade_breakdown_{baseline_year}.csv"),
                            index=False)

    metal_content_df = final_trade_matrix_df[
                            (
                                final_trade_matrix_df["export_country_code"].isin(ccg_countries)
                            ) & (final_trade_matrix_df["initial_processting_stage"] == 0)
                        ].groupby(
                            ["export_country_code",
                            "reference_mineral",
                            "initial_processting_stage"
                        ]
                        )["initial_stage_production_tons"].sum().reset_index()
    metal_content_df.to_csv(
                            os.path.join(
                                results_folder,
                                f"baci_ccg_country_metal_content_production_{baseline_year}.csv"),
                            index=False)













    






    
    


if __name__ == '__main__':
    CONFIG = load_config()
    main(CONFIG)


