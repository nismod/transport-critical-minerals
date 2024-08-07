#!/usr/bin/env python
# coding: utf-8

import os 
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
from utils import *
import subprocess 

"""Notes of BACI updates
    - Correct the codes for Singapore manually
    - Run the script baci_trade_data.py
    - Run the script global_trade_balancing.py
"""
def main(config):

    incoming_data_path = config['paths']['incoming_data']
    processed_data_path = config['paths']['data']
    year_percentile_combinations = [
                                    (2022,"baseline"),
                                    (2030,"low"),
                                    (2030,"mid"),
                                    (2030,"high"),
                                    (2040,"low"),
                                    (2040,"mid"),
                                    (2040,"high")
                                    ]
    tonnage_thresholds = ["min_threshold_metal_tons","max_threshold_metal_tons"]
    # percentiles = [25,50,75]
    # # percentiles = [25]
    reference_minerals = ["graphite","lithium","cobalt","manganese","nickel","copper"]
    # reference_minerals = ["cobalt"]

    args = [
            "python",
            "existing_trade_balancing.py"
            ]
    print ("* Start the creation of the high-level OD matrices in the present")
    print (args)
    subprocess.run(args)

    for th in tonnage_thresholds:
        for idx, (year,percentile) in enumerate(year_percentile_combinations):
            if year > 2022:
                args = [
                    "python",
                    "future_trade_balancing.py",
                    f"{year}",
                    f"{percentile}",
                    f"{th}",
                    ]
                print (f"* Start the creation of the {year} {percentile} percentile high-level OD matrices under {th} limits")
                print (args)
                subprocess.run(args)

    for th in tonnage_thresholds:
        for idx, (year,percentile) in enumerate(year_percentile_combinations):
            args = [
                "python",
                "mineral_node_ods.py",
                f"{year}",
                f"{percentile}",
                f"{th}",
                ]
            print (f"* Start the creation of the {year} {percentile} percentile node OD matrices under {th} limits")
            print (args)
            subprocess.run(args)

    # # for percentile in percentiles:
    # #     args = [
    # #         "python",
    # #         "mineral_node_ods.py",
    # #         f"{percentile}"
    # #         ]
    # #     print ("* Start the creation of the mine-level outputs")
    # #     print (args)
    # #     subprocess.run(args)

    
    # # for reference_mineral in reference_minerals:
    # #     for percentile in percentiles:
    # #         args = [
    # #             "python",
    # #             "city_mine_scenarios.py",
    # #             f"{reference_mineral}",
    # #             f"{percentile}"
    # #             ]
    # #         print ("* Start the creation of the mine-level outputs")
    # #         print (args)
    # #         subprocess.run(args)


    # # reference_minerals = ["graphite"]
    # for reference_mineral in reference_minerals:
    #     for idx, (year,percentile) in enumerate(year_percentile_combinations):
    #         if year == 2022:
    #             args = [
    #                 "python",
    #                 "flow_allocation.py",
    #                 f"{reference_mineral}",
    #                 f"{year}",
    #                 f"{percentile}",
    #                 f"0"
    #                 ]
    #             print ("* Start the creation of the flow allocation outputs")
    #             print (args)
    #             subprocess.run(args)
    #         else:
    #             for th in tonnage_thresholds:
    #                 args = [
    #                     "python",
    #                     "flow_allocation.py",
    #                     f"{reference_mineral}",
    #                     f"{year}",
    #                     f"{percentile}",
    #                     f"{th}"
    #                     ]

    #                 print ("* Start the creation of the flow allocation outputs")
    #                 print (args)
    #                 subprocess.run(args)

    # # reference_minerals = ["copper"]
    # # reference_minerals = ["graphite","lithium","cobalt","manganese","nickel","copper"]
    # for reference_mineral in reference_minerals:
    #     for idx, (year,percentile) in enumerate(year_percentile_combinations):
    #         if year == 2022:
    #             args = [
    #                 "python",
    #                 "node_edge_flows.py",
    #                 f"{reference_mineral}",
    #                 f"{year}",
    #                 f"{percentile}",
    #                 f"0"
    #                 ]
    #             print ("* Start the creation of the flow flow allocation outputs")
    #             print (args)
    #             subprocess.run(args)
    #         else:
    #             for th in tonnage_thresholds:
    #                 args = [
    #                     "python",
    #                     "node_edge_flows.py",
    #                     f"{reference_mineral}",
    #                     f"{year}",
    #                     f"{percentile}",
    #                     f"{th}"
    #                     ]
    #                 print ("* Start the creation of the flow flow allocation outputs")
    #                 print (args)
    #                 subprocess.run(args)


    # # for reference_mineral in reference_minerals:
    # #     for idx, (year,percentile) in enumerate(year_percentile_combinations):
    # #         if year != 2022:
    # #             args = [
    # #                 "python",
    # #                 "location_identification.py",
    # #                 f"{reference_mineral}",
    # #                 f"{year}",
    # #                 f"{percentile}"
    # #                 ]
    # #             print ("* Start the creation of the flow flow allocation outputs")
    # #             print (args)
    # #             subprocess.run(args)
    # for idx, (year,percentile) in enumerate(year_percentile_combinations):
    #     if year == 2022:
    #         args = [
    #             "python",
    #             "flow_locations_for_processing.py",
    #             f"{year}",
    #             f"{percentile}",
    #             f"0",
    #             ]
    #         print (f"* Start the creation of processing locations")
    #         print (args)
    #         subprocess.run(args)

    #         args = [
    #             "python",
    #             "flow_location_identification.py",
    #             f"{year}",
    #             f"{percentile}",
    #             f"0",
    #             ]
    #         print (f"* Start the creation of energy locations")
    #         print (args)
    #         subprocess.run(args)
    #     else:
    #         for th in tonnage_thresholds:
    #             args = [
    #             "python",
    #             "flow_locations_for_processing.py",
    #             f"{year}",
    #             f"{percentile}",
    #             f"{th}",
    #             ]
    #             print (f"* Start the creation of processing locations")
    #             print (args)
    #             subprocess.run(args)
    #             args = [
    #                 "python",
    #                 "flow_location_identification.py",
    #                 f"{year}",
    #                 f"{percentile}",
    #                 f"{th}",
    #                 ]
    #             print (f"* Start the creation of energy locations")
    #             print (args)
    #             subprocess.run(args)

    # for idx, (year,percentile) in enumerate(year_percentile_combinations):
    #     if year > 2022:
    #         for th in tonnage_thresholds:
    #             args = [
    #             "python",
    #             "flow_location_optimisation.py",
    #             f"{year}",
    #             f"{percentile}",
    #             f"{th}",
    #             ]
    #             print (f"* Start the creation of processing locations")
    #             print (args)
    #             subprocess.run(args)
    #     else:
    #         args = [
    #         "python",
    #         "flow_location_optimisation.py",
    #         f"{year}",
    #         f"{percentile}",
    #         "0",
    #         ]
    #         print (f"* Start the creation of processing locations")
    #         print (args)
    #         subprocess.run(args)



if __name__ == '__main__':
    CONFIG = load_config()
    main(CONFIG)


