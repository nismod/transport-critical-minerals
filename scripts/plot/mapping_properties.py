"""Road network risks and adaptation maps
"""
import os
import sys

def mineral_properties():
    reference_minerals = ["copper","cobalt","manganese","lithium","graphite","nickel"]
    reference_mineral_colors = ["#cc4c02","#3690c0","#88419d","#d7301f","#252525","#737373"]
    
    mineral_properties = {
                            "copper":{
                                        "mine_color":"#662506",
                                        "node_colors":["#662506","#cc4c02","#fe9929"],
                                        "edge_color":"#525252",
                                        "stages":[1.0,2.0,3.0,4.3,5.0],
                                        "stage_colors":["#662506","#993404","#cc4c02","#fe9929","#fec44f"]
                                    },
                            "cobalt":{
                                        "mine_color":"#023858",
                                        "node_colors":["#023858","#0570b0","#74a9cf"],
                                        "edge_color":"#525252",
                                        "stages":[1.0,3.0,5.0],
                                        "stage_colors":["#023858","#0570b0","#74a9cf"]
                                    },
                            "manganese":{
                                        "mine_color":"#4d004b",
                                        "node_colors":["#4d004b","#8c96c6","#88419d"],
                                        "edge_color":"#525252",
                                        "stages":[1.0,3.1,4.1],
                                        "stage_colors":["#4d004b","#8c96c6","#88419d"]
                                    },
                            "lithium":{
                                        "mine_color":"#004529",
                                        "node_colors":["#004529","#238443","#78c679"],
                                        "edge_color":"#525252",
                                        "stages":[1.0,3.0,4.2],
                                        "stage_colors":["#004529","#238443","#78c679"]
                                    },
                            "graphite":{
                                        "mine_color":"#000000",
                                        "node_colors":["#000000","#737373","#bdbdbd"],
                                        "edge_color":"#525252",
                                        "stages":[1.0,3.0,4.0],
                                        "stage_colors":["#737373","#969696","#bdbdbd"]
                                    },
                            "nickel":{
                                        "mine_color":"#67000d",
                                        "node_colors":["#67000d","#cb181d","#fb6a4a"],
                                        "edge_color":"#525252",
                                        "stages":[1.0,2.0,3.0,4.3,5.0],
                                        "stage_colors":["#67000d","#a50f15","#cb181d","#fb6a4a","#fc9272"]
                                    },
                        }
    return mineral_properties
