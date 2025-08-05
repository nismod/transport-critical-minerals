"""Functions for network modelling
"""
import sys
import os
import json
import numpy as np
import pandas as pd
import igraph as ig
import geopandas as gpd
from shapely.geometry import LineString
from scipy.spatial import cKDTree
from tqdm import tqdm
tqdm.pandas()

def load_config():
    """Read config.json"""
    config_path = os.path.join(os.path.dirname(__file__), "..", "..", "config.json")
    with open(config_path, "r") as config_fh:
        config = json.load(config_fh)
    return config

def ckdnearest(gdA, gdB):
    """Function to calculate nearest points between two dataframes
        Taken from https://gis.stackexchange.com/questions/222315/finding-nearest-point-in-other-geodataframe-using-geopandas
    """
    nA = np.array(list(gdA.geometry.apply(lambda x: (x.x, x.y))))
    nB = np.array(list(gdB.geometry.apply(lambda x: (x.x, x.y))))
    btree = cKDTree(nB)
    dist, idx = btree.query(nA, k=1)
    gdB_nearest = gdB.iloc[idx].drop(columns="geometry").reset_index(drop=True)
    gdf = pd.concat(
        [
            gdA.reset_index(drop=True),
            gdB_nearest,
            pd.Series(dist, name='dist')
        ], 
        axis=1)

    return gdf

def add_lines(x,from_nodes_df,to_nodes_df,from_nodes_id,to_nodes_id):
    """Function to create a Line joining two nodes

    """
    from_point = from_nodes_df[from_nodes_df[from_nodes_id] == x[from_nodes_id]]
    to_point = to_nodes_df[to_nodes_df[to_nodes_id].isin([x[to_nodes_id]])] 
    return LineString([from_point.geometry.values[0],to_point.geometry.values[0]])

def get_flow_on_path_elements(save_paths_df,path_id,path_column,
    flow_column):
    """Function to get the flows along path elements

    """
    path_flows = defaultdict(float)
    for row in save_paths_df.itertuples():
        for item in getattr(row,path_column):
            path_flows[item] += getattr(row,flow_column)

    return pd.DataFrame([(k,v) for k,v in path_flows.items()],columns=[path_id,flow_column])


def get_flow_paths_indexes_of_edges(flow_dataframe,path_criteria):
    edge_path_index = defaultdict(list)
    for v in flow_dataframe.itertuples():
        for k in getattr(v,path_criteria):
            edge_path_index[k].append(v.Index)

    del flow_dataframe
    return edge_path_index

def network_od_node_edge_path_estimations(graph,
    source, target, cost_criteria,path_id_column):
    """Estimate the paths and costs for given OD pair

    """
    edge_paths = graph.get_shortest_paths(source, target, weights=cost_criteria, output="epath")
    node_paths = graph.get_shortest_paths(source, target, weights=cost_criteria, output="vpath")

    edge_path_list = []
    node_path_list = []
    total_path_gcost_list = []
    # for p in range(len(paths)):
    for path in edge_paths:
        edge_path = []
        total_gcost = 0
        if path:
            for n in path:
                edge_path.append(graph.es[n][path_id_column])
                path_gcost.append(graph.es[n][cost_criteria])
            total_gcost = sum(path_gcost)

        edge_path_list.append(edge_path)
        total_path_gcost_list.append(total_gcost)
    for path in node_paths:
        node_path = []
        if path:
            for n in path:
                node_path.append(graph.vs[n]["name"])
        node_path_list.append(node_path)
    
    return edge_path_list,node_path_list,total_path_gcost_list    


def network_od_node_edge_paths_assembly(points_dataframe, graph,
                                cost_criteria,
                                path_id_column,
                                origin_id_column,destination_id_column,
                                store_paths=True):

    """Assemble estimates of OD paths, distances, times, costs and tonnages on networks

    """
    save_paths = []
    points_dataframe = points_dataframe.set_index(origin_id_column)
    origins = list(set(points_dataframe.index.values.tolist()))
    for origin in origins:
        destinations = list(set(points_dataframe.loc[[origin], destination_id_column].values.tolist()))

        get_epath,get_npath,get_gcost = network_od_node_edge_path_estimations(
                graph, origin, destinations, cost_criteria,path_id_column)

        save_paths += list(zip([origin]*len(destinations),
                            destinations, get_epath,get_npath,
                            get_gcost))

    cols = [
        origin_id_column, destination_id_column, 'edge_path','node_path',
        cost_criteria
    ]
    save_paths_df = pd.DataFrame(save_paths, columns=cols)
    if store_paths is False:
        save_paths_df.drop(["edge_path","node_path"],axis=1,inplace=True)

    points_dataframe = points_dataframe.reset_index()
    save_paths_df = pd.merge(points_dataframe,save_paths_df,how='left', on=[
                             origin_id_column, destination_id_column]).fillna(0)

    save_paths_df = save_paths_df[save_paths_df[origin_id_column] != 0]

    return save_paths_df
