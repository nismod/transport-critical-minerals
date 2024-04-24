#!/usr/bin/env python
# coding: utf-8

import os 
import pandas as pd
from utils import *
import subprocess 

def main(config):

    incoming_data_path = config['paths']['incoming_data']
    processed_data_path = config['paths']['data']

    args = [
            "python",
            "trade_balancing.py"
            ]
    print ("* Start the creation of the high-level OD matrices")
    print (args)
    subprocess.run(args)

    # reference_mineral = "copper"
    # mine_initial_refined_stage = 1.0
    # mine_final_refined_stage = 3.0
    # city_initial_refined_stage = 5.0
    # city_final_refined_stage = 5.0 
    # percentiles = [25,50,75]
    
    # for percentile in percentiles:
    #     args = [
    #         "python",
    #         "city_mine_scenarios.py",
    #         f"{reference_mineral}",
    #         f"{mine_initial_refined_stage}",
    #         f"{mine_final_refined_stage}",
    #         f"{city_initial_refined_stage}",
    #         f"{city_final_refined_stage}",
    #         f"{percentile}"
    #         ]
    #     print ("* Start the creation of the mine-level outputs")
    #     print (args)
    #     subprocess.run(args)


if __name__ == '__main__':
    CONFIG = load_config()
    main(CONFIG)


