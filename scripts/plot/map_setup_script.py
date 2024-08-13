#!/usr/bin/env python
# coding: utf-8

import os 
from map_plotting_utils import *
import subprocess 

"""Notes of BACI updates
    - Correct the codes for Singapore manually
    - Run the script baci_trade_data.py
    - Run the script global_trade_balancing.py
"""
def main(config):
    incoming_data_path = config['paths']['incoming_data']
    processed_data_path = config['paths']['data']
    
    reference_minerals = ["graphite","lithium","cobalt","manganese","nickel","copper"]
    baseline_scenario = [2022,"baseline","none","country","unconstrained"]
    future_scenarios = []
    p = "min_threshold_metal_tons"
    c = "country"
    for s in ["low","mid","high"]:
        for o in ["unconstrained","constrained"]:
            fsc = [tuple(baseline_scenario)]
            for y in [2030,2040]:
                fsc.append((y,s,p,c,o))

            future_scenarios.append(list(map(list,zip(*fsc))))

    p = "max_threshold_metal_tons"
    c = "region"
    for s in ["low","mid","high"]:
        for o in ["unconstrained","constrained"]:
            fsc = [tuple(baseline_scenario)]
            for y in [2030,2040]:
                fsc.append((y,s,p,c,o))

            future_scenarios.append(list(map(list,zip(*fsc))))

    p = "min_threshold_metal_tons"
    c = "country"
    for y in [2030,2040]:
        for o in ["unconstrained","constrained"]:
            fsc = []
            for s in ["low","mid","high"]:
                fsc.append((y,s,p,c,o))
            future_scenarios.append(list(map(list,zip(*fsc))))

    p = "max_threshold_metal_tons"
    c = "region"
    for y in [2030,2040]:
        for o in ["unconstrained","constrained"]:
            fsc = []
            for s in ["low","mid","high"]:
                fsc.append((y,s,p,c,o))
            future_scenarios.append(list(map(list,zip(*fsc))))

    run_script = True
    if run_script is True:
        num_blocks = 0
        with open("flow_set.txt","w+") as f:
            for rf in reference_minerals:
                num_blocks += 2
                for row in future_scenarios:
                    st = f"{rf};"
                    for r in row[:-1]:
                        st += f"{r};"
                    st += f"{row[-1]}\n"
                    f.write(st)                
        f.close()

        """Next we aggregate the flows through the scenarios
        """
        args = [
                "parallel",
                "-j", str(num_blocks),
                "--colsep", ";",
                "-a",
                "flow_set.txt",
                "python",
                "mineral_flows.py",
                "{}"
                ]
        print ("* Start the processing of plotting flows")
        print (args)
        subprocess.run(args)

    run_script = False
    if run_script is True:
        num_blocks = 0
        with open("flow_set.txt","w+") as f:
            for rf in reference_minerals:
                num_blocks += 2
                for row in future_scenarios:
                    st = f"{rf};"
                    for r in row[:-1]:
                        st += f"{r};"
                    st += f"{row[-1]}\n"
                    f.write(st)                
        f.close()

        """Next we aggregate the flows through the scenarios
        """
        args = [
                "parallel",
                "-j", str(num_blocks),
                "--colsep", ";",
                "-a",
                "flow_set.txt",
                "python",
                "zambia_mineral_flows.py",
                "{}"
                ]
        print ("* Start the processing of plotting flows")
        print (args)
        subprocess.run(args)                 

    
if __name__ == '__main__':
    CONFIG = load_config()
    main(CONFIG)


