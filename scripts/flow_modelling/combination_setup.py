#!/usr/bin/env python
# coding: utf-8

import os 
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
from utils import *
import subprocess 

"""Notes of BACI updates
    - Correct the codes for Singapore manually
    - Run the script baci_cleaning.py
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

    run_script = False
    if run_script is True:
        args = [
                "python",
                "road_proximity.py"
                ]
        print ("* Find the proximity of nodes to roads")
        print (args)
        subprocess.run(args)

    run_script = False
    if run_script is True:
        args = [
                "python",
                "location_filters.py"
                ]
        print ("* Put filters on the nodes and edges")
        print (args)
        subprocess.run(args)

    run_script = False
    if run_script is True:
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
    
    run_script = False
    if run_script is True:
        args = [
                "python",
                "existing_trade_balancing.py"
                ]
        print ("* Start the creation of the high-level OD matrices in the baseline")
        print (args)
        subprocess.run(args)

    run_script = False
    if run_script is True:
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

    run_script = False
    if run_script is True:
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

    run_script = False
    if run_script is True:
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
        num_blocks = 3
        all_scenarios = []
        distance_filters = [(x,y) for x in [0,500,1000] for y in [0,10,20]]  # for a list
        print (distance_filters)
        ref_mins = [["cobalt"],["copper"],["nickel"],["graphite"],["manganese"],["lithium"]]
        baseline_scenario = [[2022],"baseline","none","country","unconstrained"]
        for rf in ref_mins:    
            all_scenarios.append([rf] + baseline_scenario)
        ref_mins = [["cobalt","copper","nickel"],["graphite"],["manganese"],["lithium"]]
        p = "min_threshold_metal_tons"
        c = "country"
        yrs = [2030,2040]
        for rf in ref_mins:
            for s in ["low","mid","high"]:
                for o in ["unconstrained","constrained"]:
                    if o == "constrained":
                        for idx,(op,ef) in enumerate(distance_filters):
                            all_scenarios.append([rf] + [yrs] + [s,p,c,o,baseline_year,op,ef])
                    else:
                        all_scenarios.append([rf] + [yrs] + [s,p,c,o])
        p = "max_threshold_metal_tons"
        c = "region"
        yrs = [2030,2040]
        for rf in ref_mins:
            for s in ["low","mid","high"]:
                for o in ["unconstrained","constrained"]:
                    if o == "constrained":
                        for idx,(op,ef) in enumerate(distance_filters):
                            all_scenarios.append([rf] + [yrs] + [s,p,c,o,baseline_year,op,ef])
                    else:
                        all_scenarios.append([rf] + [yrs] + [s,p,c,o])

        with open("combination_set.txt","w+") as f:
            for row in all_scenarios:
                st = ""
                for r in row[:-1]:
                    st += f"{r};"
                st += f"{row[-1]}\n"
                f.write(st)               
        f.close()

        """Next we run the optimsation script
        """
        args = [
                "parallel",
                "-j", str(num_blocks),
                "--colsep", ";",
                "-a",
                "combination_set.txt",
                "python",
                "optimisation_combined.py",
                "{}"
                ]
        print ("* Start the processing of plotting flows")
        print (args)
        subprocess.run(args)

    run_script = False
    if run_script is True:
        distance_filters = [(x,y) for x in [0,500,1000] for y in [0,10,20]]  # for a list
        c = "combined"
        with open("combined_optimisation_set.txt","w+") as f:
            for idx, (year,percentile) in enumerate(year_percentile_combinations):
                if year == baseline_year:
                    th = "none"
                    loc = "country"
                    opt = "unconstrained"
                    f.write(f"{year},{percentile},{th},{loc},{opt},{c},0.0,0.0\n")
                else:
                    for loc in location_cases:
                        if loc == "country":
                            th = "min_threshold_metal_tons"
                        else:
                            th = "max_threshold_metal_tons"
                        for opt in optimisation_type:
                            if opt == "constrained":
                                for idx,(op,ef) in enumerate(distance_filters):
                                    f.write(f"{year},{percentile},{th},{loc},{opt},{c},{op},{ef}\n")
                            else:
                                f.write(f"{year},{percentile},{th},{loc},{opt},{c},0.0,0.0\n")

        f.close()

    run_script = False
    if run_script is True:
        with open("combined_optimisation_set.txt","r") as r:
            for p in r:
                pv = p.split(",")
                ls = pv[-1].strip('\n')
                args = [
                        "python",
                        "processing_locations_for_energy.py"
                        ]
                for v in pv[:-1]:
                    args.append(v)
                args.append(ls)
                print ("* Start the processing of assembling locations for energy calculations")
                print (args)
                subprocess.run(args)  

    run_script = False
    if run_script is True:
        num_blocks = 16
        args = [
                "parallel",
                "-j", str(num_blocks),
                "--colsep", ",",
                "-a",
                "combined_optimisation_set.txt",
                "python",
                "production_cost_estimation.py",
                "{}"
                ]
        print ("* Start the processing of production cost estimations")
        print (args)
        subprocess.run(args)

    run_script = False
    if run_script is True:
        num_blocks = 16
        args = [
                "parallel",
                "-j", str(num_blocks),
                "--colsep", ",",
                "-a",
                "combined_optimisation_set.txt",
                "python",
                "country_totals_tons_and_costs.py",
                "{}"
                ]
        print ("* Start the processing of tonnage summaries")
        print (args)
        subprocess.run(args)
        
        # with open("combined_optimisation_set.txt","r") as r:
        #     for p in r:
        #         pv = p.split(",")
        #         ls = pv[-1].strip('\n')
        #         args = [
        #                 "python",
        #                 "country_totals_tons_and_costs.py"
        #                 ]
        #         for v in pv[:-1]:
        #             args.append(v)
        #         args.append(ls)
        #         print ("* Start the processing of tonnage summaries")
        #         print (args)
        #         subprocess.run(args)

    run_script = False
    if run_script is True:
        num_blocks = 16
        distance_filters = [(x,y) for x in [0,500,1000] for y in [0,10,20]]  # for a list
        c = "combined"
        with open("combined_flow_set.txt","w+") as f:
            for rf in reference_minerals:
                for idx, (year,percentile) in enumerate(year_percentile_combinations):
                    if year == baseline_year:
                        th = "none"
                        loc = "country"
                        opt = "unconstrained"
                        f.write(f"{rf},{year},{percentile},{th},{loc},{opt},{c},0.0,0.0\n")
                    else:
                        for loc in location_cases:
                            if loc == "country":
                                th = "min_threshold_metal_tons"
                            else:
                                th = "max_threshold_metal_tons"
                            for opt in optimisation_type:
                                if opt == "constrained":
                                    for idx,(op,ef) in enumerate(distance_filters):
                                        f.write(f"{rf},{year},{percentile},{th},{loc},{opt},{c},{op},{ef}\n")
                                else:
                                    f.write(f"{rf},{year},{percentile},{th},{loc},{opt},{c},0.0,0.0\n")

        f.close()

        """Next we aggregate the flows through the scenarios
        """
        args = [
                "parallel",
                "-j", str(num_blocks),
                "--colsep", ",",
                "-a",
                "combined_flow_set.txt",
                "python",
                "node_edge_flows.py",
                "{}"
                ]
        print ("* Start the processing of node edge flow allocation")
        print (args)
        subprocess.run(args)                 

    run_script = False
    if run_script is True:
        """Next we call the flow analysis script and loop through the scenarios
        """
        num_blocks = 16
        args = [
                "parallel",
                "-j", str(num_blocks),
                "--colsep", ",",
                "-a",
                "combined_optimisation_set.txt",
                "python",
                "emissions_estimations.py",
                "{}"
                ]
        print ("* Start the processing of carbon emissions estimations")
        print (args)
        subprocess.run(args)

    run_script = True
    if run_script is True:
        distance_filters = [(x,y) for x in [0,500,1000] for y in [0,10,20]]  # for a list
        cx = "combined"
        for lcx in location_cases:
            for optx in optimisation_type:
                if optx == "constrained":
                    for ix,(opx,efx) in enumerate(distance_filters):
                        args = [
                                "python",
                                "combined_tonnages_v2.py",
                                f"{lcx}",
                                f"{optx}",
                                f"{cx}",
                                f"{opx}",
                                f"{efx}"
                                ]
                        print ("* Start the processing of tonnage summaries into excel")
                        print (args)
                        subprocess.run(args)
                else:
                    args = [
                            "python",
                            "combined_tonnages_v2.py",
                            f"{lcx}",
                            f"{optx}",
                            f"{cx}",
                            "0.0",
                            "0.0"
                            ]
                    print ("* Start the processing of tonnage summaries into excel")
                    print (args)
                    subprocess.run(args)
    
    run_script = False
    if run_script is True:
        # with open("optimisation_set.txt","r") as r:
        #     for p in r:
        #         pv = p.split(",")
        #         opt = pv[4].strip('\n')
        #         args = [
        #                 "python",
        #                 "aggregated_node_edge_flows.py",
        #                 f"{pv[0]}",
        #                 f"{pv[1]}",
        #                 f"{pv[2]}",
        #                 f"{pv[3]}",
        #                 f"{opt}"
        #                 ]
        #         print ("* Start the processing of aggregating node edge flows")
        #         print (args)
        #         subprocess.run(args)  
        num_blocks = 16
        args = [
                "parallel",
                "-j", str(num_blocks),
                "--colsep", ",",
                "-a",
                "combined_optimisation_set.txt",
                "python",
                "aggregated_node_edge_flows.py",
                "{}"
                ]
        print ("* Start the processing of aggregating node edge flows")
        print (args)
        subprocess.run(args) 
    
if __name__ == '__main__':
    CONFIG = load_config()
    main(CONFIG)


