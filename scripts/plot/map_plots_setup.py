#!/usr/bin/env python
# coding: utf-8

import os 
from map_plotting_utils import *
import subprocess 

def main(config):
    incoming_data_path = config['paths']['incoming_data']
    processed_data_path = config['paths']['data']
    
    reference_minerals = ["graphite","lithium","cobalt","manganese","nickel","copper"]
    scenarios = ["bau","early refining","precursor"]
    baseline_scenario = [["baseline"],[2022],["baseline"],["none"],["country"],["unconstrained"]]
    future_scenarios = []
    future_scenarios.append(baseline_scenario)
    p = "min_threshold_metal_tons"
    c = "country"
    for scn in scenarios:
        for y in [2040]:
            fsc = []
            for o in ["unconstrained","constrained"]:
                for s in ["mid"]:
                    fsc.append((scn,y,s,p,c,o))

        future_scenarios.append(list(map(list,zip(*fsc))))

    p = "max_threshold_metal_tons"
    c = "region"
    for scn in scenarios:
        for y in [2040]:
            fsc = []
            for o in ["unconstrained","constrained"]:
                for s in ["mid"]:
                    fsc.append((scn,y,s,p,c,o))

        future_scenarios.append(list(map(list,zip(*fsc))))

    with open("map_plots_set.txt","w+") as f:
        for rf in reference_minerals:
            # for case in ["noncombined","combined"]:
            for case in ["combined"]:
                for row in future_scenarios:
                    st = f"{rf};"
                    if case == "noncombined":
                        for r in row[:-1]:
                            st += f"{r};"
                        st += f"{row[-1]}\n"
                        f.write(st)
                    elif case == "combined":
                        if row[0][0] != 2022:
                            for r in row:
                                st += f"{r};"
                            st += "combined;0;0\n"
                            f.write(st)                
    f.close()

    run_script = False
    if run_script is True:
        args = [
                "python",
                "transport_networks_map.py"
                ]
        print ("* Plot CCG networks")
        print (args)
        subprocess.run(args)

    run_script = True
    if run_script is True:
        args = [
                "python",
                "location_maps_scenarios.py"
                ]
        print ("* Plot regional location maps")
        print (args)
        subprocess.run(args)

    run_script = True
    if run_script is True:
        args = [
                "python",
                "country_location_maps.py"
                ]
        print ("* Plot Country location maps")
        print (args)
        subprocess.run(args)

    run_script = True
    if run_script is True:
        args = [
                "python",
                "env_layers_with_mines.py"
                ]
        print ("* Plot regional location maps with environmental filters")
        print (args)
        subprocess.run(args)

    run_script = True
    if run_script is True:
        args = [
                "python",
                "country_env_maps.py"
                ]
        print ("* Plot Country location maps with environmental filters")
        print (args)
        subprocess.run(args)

    run_script = False
    if run_script is True:
        args = [
                "parallel",
                "-j", str(num_blocks),
                "--colsep", ";",
                "-a",
                "map_plots_set.txt",
                "python",
                "zambia_flow_maps.py",
                "{}"
                ]
        print ("* Start the processing of plotting flows")
        print (args)
        subprocess.run(args) 

    with open("aggregated_map_plots_set.txt","w+") as f:
        for case in ["noncombined","combined"]:
            for row in future_scenarios:
                st = ""
                if case == "noncombined":
                    for r in row[:-1]:
                        st += f"{r};"
                    st += f"{row[-1]}\n"
                    f.write(st)
                elif case == "combined":
                    if row[0][0] != 2022:
                        for r in row:
                            st += f"{r};"
                        st += "combined;0;0\n"
                        f.write(st)                
    f.close()

    num_blocks = 8
    
    run_script = True
    if run_script is True:
        args = [
                "parallel",
                "-j", str(num_blocks),
                "--colsep", ";",
                "-a",
                "aggregated_map_plots_set.txt",
                "python",
                "agg_flow_maps.py",
                "{}"
                ]
        print ("* Start the processing of plotting flows")
        print (args)
        subprocess.run(args)

    num_blocks = 12
    run_script = True
    if run_script is True:
        args = [
                "parallel",
                "-j", str(num_blocks),
                "--colsep", ";",
                "-a",
                "map_plots_set.txt",
                "python",
                "flow_maps.py",
                "{}"
                ]
        print ("* Start the processing of plotting flows")
        print (args)
        subprocess.run(args)

    run_script = False
    if run_script is True:
        args = [
                "parallel",
                "-j", str(num_blocks),
                "--colsep", ";",
                "-a",
                "aggregated_map_plots_set.txt",
                "python",
                "country_agg_flow_maps.py",
                "{}"
                ]
        print ("* Start the processing of plotting flows")
        print (args)
        subprocess.run(args)
    
if __name__ == '__main__':
    CONFIG = load_config()
    main(CONFIG)


