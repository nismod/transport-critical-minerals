#!/usr/bin/env python
# coding: utf-8

import os 
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
from utils import *
import subprocess 

def main(config):

    incoming_data_path = config['paths']['incoming_data']
    processed_data_path = config['paths']['data']

    year_percentile_combinations = [
                                    (2022,0),
                                    # (2030,25),
                                    # (2030,50),
                                    # (2030,75),
                                    # (2040,25),
                                    # (2040,50),
                                    # (2040,75)
                                    ]
    percentiles = [25,50,75]
    # percentiles = [25]
    reference_minerals = ["graphite","lithium","cobalt","manganese","nickel","copper"]

    # args = [
    #         "python",
    #         "trade_balancing.py"
    #         ]
    # print ("* Start the creation of the high-level OD matrices")
    # print (args)
    # subprocess.run(args)
    
    # for reference_mineral in reference_minerals:
    #     for percentile in percentiles:
    #         args = [
    #             "python",
    #             "city_mine_scenarios.py",
    #             f"{reference_mineral}",
    #             f"{percentile}"
    #             ]
    #         print ("* Start the creation of the mine-level outputs")
    #         print (args)
    #         subprocess.run(args)

    for percentile in percentiles:
        args = [
            "python",
            "mineral_node_ods.py",
            f"{percentile}"
            ]
        print ("* Start the creation of the mine-level outputs")
        print (args)
        subprocess.run(args)

    # reference_minerals = ["copper"]
    # for reference_mineral in reference_minerals:
    #     for idx, (year,percentile) in enumerate(year_percentile_combinations):
    #         args = [
    #             "python",
    #             "flow_allocation.py",
    #             f"{reference_mineral}",
    #             f"{year}",
    #             f"{percentile}"
    #             ]
    #         print ("* Start the creation of the flow flow allocation outputs")
    #         print (args)
    #         subprocess.run(args)

    # for reference_mineral in reference_minerals:
    #     for idx, (year,percentile) in enumerate(year_percentile_combinations):
    #         args = [
    #             "python",
    #             "node_edge_flows.py",
    #             f"{reference_mineral}",
    #             f"{year}",
    #             f"{percentile}"
    #             ]
    #         print ("* Start the creation of the flow flow allocation outputs")
    #         print (args)
    #         subprocess.run(args)

    # for reference_mineral in reference_minerals:
    #     for idx, (year,percentile) in enumerate(year_percentile_combinations):
    #         if year != 2022:
    #             args = [
    #                 "python",
    #                 "location_identification.py",
    #                 f"{reference_mineral}",
    #                 f"{year}",
    #                 f"{percentile}"
    #                 ]
    #             print ("* Start the creation of the flow flow allocation outputs")
    #             print (args)
    #             subprocess.run(args)



if __name__ == '__main__':
    CONFIG = load_config()
    main(CONFIG)


