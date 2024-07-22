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
    reference_minerals = ["graphite","lithium","cobalt","manganese","nickel","copper"]
    location_cases = ["country","region"]
    optimisation_type = ["unconstrained","constrained"]
    baseline_year = 2022

    run_script = False
    if run_script is True:
        args = [
                "python",
                "s_and_p_mines.py"
                ]
        print ("* Clean the S&P mine data and store new mines")
        print (args)
        subprocess.run(args)

        args = [
                "python",
                "baci_cleaning.py"
                ]
        print ("* Clean the BACI matrices in the baseline")
        print (args)
        subprocess.run(args)

        args = [
                "python",
                "global_trade_balancing.py"
                ]
        print ("* Balance global trade matrices to match BGS values")
        print (args)
        subprocess.run(args)    

        args = [
                "python",
                "existing_trade_balancing.py"
                ]
        print ("* Start the creation of the high-level OD matrices in the baseline")
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

        num_blocks = 0
        with open("parameter_set.txt","w+") as f:
            for rf in reference_minerals:
                num_blocks += 1
                for idx, (year,percentile) in enumerate(year_percentile_combinations):
                    if year == baseline_year:
                        th = "none"
                        f.write(f"{rf},{year},{percentile},{th}\n")
                    else:
                        for th in tonnage_thresholds:
                            f.write(f"{rf},{year},{percentile},{th}\n")                    
        f.close()

        """Next we call the flow analysis script and loop through the scenarios
        """
        args = [
                "parallel",
                "-j", str(num_blocks),
                "--colsep", ",",
                "-a",
                "parameter_set.txt",
                "python",
                "flow_allocation.py",
                "{}"
                ]
        print ("* Start the processing of flow allocation")
        print (args)
        subprocess.run(args)


    run_script = False
    if run_script is True:
        num_blocks = 0
        with open("optimisation_set.txt","w+") as f:
            for idx, (year,percentile) in enumerate(year_percentile_combinations):
                num_blocks += 1
                if year == baseline_year:
                    th = "none"
                    loc = "country"
                    opt = "unconstrained"
                    f.write(f"{year},{percentile},{th},{loc},{opt}\n")
                else:
                    for th in tonnage_thresholds:
                        for loc in location_cases:
                            for opt in optimisation_type:
                                f.write(f"{year},{percentile},{th},{loc},{opt}\n")                    
        f.close()

        """Next we call the flow analysis script and loop through the scenarios
        """
        args = [
                "parallel",
                "-j", str(num_blocks),
                "--colsep", ",",
                "-a",
                "optimisation_set.txt",
                "python",
                "flow_location_optimisation_v2.py",
                "{}"
                ]
        print ("* Start the processing of flow location optimisation")
        print (args)
        subprocess.run(args)

    run_script = False
    if run_script is True:
        with open("optimisation_set.txt","r") as r:
            for p in r:
                pv = p.split(",")
                opt = pv[4].strip('\n')
                args = [
                        "python",
                        "processing_locations_for_energy.py",
                        f"{pv[0]}",
                        f"{pv[1]}",
                        f"{pv[2]}",
                        f"{pv[3]}",
                        f"{opt}"
                        ]
                print ("* Start the processing of assembling locations for energy calculations")
                print (args)
                subprocess.run(args)  

    run_script = False
    if run_script is True:
        with open("optimisation_set.txt","r") as r:
            for p in r:
                pv = p.split(",")
                opt = pv[4].strip('\n')
                args = [
                        "python",
                        "country_totals_tons_and_costs.py",
                        f"{pv[0]}",
                        f"{pv[1]}",
                        f"{pv[2]}",
                        f"{pv[3]}",
                        f"{opt}"
                        ]
                print ("* Start the processing of tonnage summaries")
                print (args)
                subprocess.run(args)

    run_script = False
    if run_script is True:
        for lc in location_cases:
            for opt in optimisation_type:
                args = [
                        "python",
                        "combined_tonnages.py",
                        f"{lc}",
                        f"{opt}"
                        ]
                print ("* Start the processing of tonnage summaries into excel")
                print (args)
                subprocess.run(args)   

    run_script = False
    if run_script is True:
        num_blocks = 0
        with open("flow_set.txt","w+") as f:
            for rf in reference_minerals:
                num_blocks += 2
                for idx, (year,percentile) in enumerate(year_percentile_combinations):
                    if year == baseline_year:
                        th = "none"
                        loc = "country"
                        opt = "unconstrained"
                        f.write(f"{rf},{year},{percentile},{th},{loc},{opt}\n")
                    else:
                        for th in tonnage_thresholds:
                            for loc in location_cases:
                                for opt in optimisation_type:
                                    f.write(f"{rf},{year},{percentile},{th},{loc},{opt}\n")                    
        f.close()

        """Next we aggregate the flows through the scenarios
        """
        args = [
                "parallel",
                "-j", str(num_blocks),
                "--colsep", ",",
                "-a",
                "flow_set.txt",
                "python",
                "node_edge_flows.py",
                "{}"
                ]
        print ("* Start the processing of node edge flow allocation")
        print (args)
        subprocess.run(args)                 

    run_script = True
    if run_script is True:
        """Next we call the flow analysis script and loop through the scenarios
        """
        num_blocks = 12
        args = [
                "parallel",
                "-j", str(num_blocks),
                "--colsep", ",",
                "-a",
                "optimisation_set.txt",
                "python",
                "carbon_emissions.py",
                "{}"
                ]
        print ("* Start the processing of flow location optimisation")
        print (args)
        subprocess.run(args)
    
if __name__ == '__main__':
    CONFIG = load_config()
    main(CONFIG)


