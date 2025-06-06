"""Functions for utlis for transport flow modelling
"""
import sys
import os
import json
import snkit
import numpy as np
import pandas as pd
import igraph as ig
import networkx
import geopandas as gpd
from collections import defaultdict
from itertools import chain
import fiona
from shapely.geometry import shape, mapping, LineString
from scipy.spatial import cKDTree
from tqdm import tqdm
tqdm.pandas()

def link_nodes_to_nearest_edge(network, condition=None, tolerance=1e-9):
    """Link nodes to all edges within some distance"""
    new_node_geoms = []
    new_edge_geoms = []
    for node in tqdm(
        network.nodes.itertuples(index=False), desc="link", total=len(network.nodes)
    ):
        # for each node, find edges within
        edge = snkit.network.nearest_edge(node.geometry, network.edges)
        if condition is not None and not condition(node, edge):
            continue
        # add nodes at points-nearest
        point = snkit.network.nearest_point_on_line(node.geometry, edge.geometry)
        if point != node.geometry:
            new_node_geoms.append(point)
            # add edges linking
            line = LineString([node.geometry, point])
            new_edge_geoms.append(line)

    new_nodes = snkit.network.matching_gdf_from_geoms(network.nodes, new_node_geoms)
    all_nodes = snkit.network.concat_dedup([network.nodes, new_nodes])

    new_edges = snkit.network.matching_gdf_from_geoms(network.edges, new_edge_geoms)
    all_edges = snkit.network.concat_dedup([network.edges, new_edges])

    # split edges as necessary after new node creation
    unsplit = snkit.network.Network(nodes=all_nodes, edges=all_edges)

    # this step is typically the majority of processing time
    split = snkit.network.split_edges_at_nodes(unsplit,tolerance=tolerance)

    return split
def convert_json_geopandas(df,epsg=4326):
    layer_dict = []    
    for key, value in df.items():
        if key == "features":
            for feature in value:
                if any(feature["geometry"]["coordinates"]):
                    d1 = {"geometry":shape(feature["geometry"])}
                    d1.update(feature["properties"])
                    layer_dict.append(d1)

    return gpd.GeoDataFrame(pd.DataFrame(layer_dict),geometry="geometry", crs=f"EPSG:{epsg}")

def components(edges,nodes,
                node_id_column="id",edge_id_column="id",
                from_node_column="from_id",to_node_column="to_id"):
    G = networkx.Graph()
    G.add_nodes_from(
        (getattr(n, node_id_column), {"geometry": n.geometry}) for n in nodes.itertuples()
    )
    G.add_edges_from(
        (getattr(e,from_node_column), getattr(e,to_node_column), 
            {edge_id_column: getattr(e,edge_id_column), "geometry": e.geometry})
        for e in edges.itertuples()
    )
    components = networkx.connected_components(G)
    for num, c in enumerate(components):
        print(f"Component {num} has {len(c)} nodes")
        edges.loc[(edges[from_node_column].isin(c) | edges[to_node_column].isin(c)), "component"] = num
        nodes.loc[nodes[node_id_column].isin(c), "component"] = num

    return edges, nodes

def add_lines(x,from_nodes_df,to_nodes_df,from_nodes_id,to_nodes_id):
    from_point = from_nodes_df[from_nodes_df[from_nodes_id] == x[from_nodes_id]]
    to_point = to_nodes_df[to_nodes_df[to_nodes_id].isin([x[to_nodes_id]])] 
    return LineString([from_point.geometry.values[0],to_point.geometry.values[0]])

def ckdnearest(gdA, gdB):
    """Taken from https://gis.stackexchange.com/questions/222315/finding-nearest-point-in-other-geodataframe-using-geopandas
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

def gdf_geom_clip(gdf_in, clip_geom):
    """Filter a dataframe to contain only features within a clipping geometry

    Parameters
    ---------
    gdf_in
        geopandas dataframe to be clipped in
    province_geom
        shapely geometry of province for what we do the calculation

    Returns
    -------
    filtered dataframe
    """
    return gdf_in.loc[gdf_in['geometry'].apply(lambda x: x.within(clip_geom))].reset_index(drop=True)

def get_nearest_values(x,input_gdf,column_name):
    polygon_index = input_gdf.distance(x.geometry).sort_values().index[0]
    return input_gdf.loc[polygon_index,column_name]

def extract_gdf_values_containing_nodes(x, input_gdf, column_name):
    a = input_gdf.loc[list(input_gdf.geometry.contains(x.geometry))]
    if len(a.index) > 0:
        return a[column_name].values[0]
    else:
        polygon_index = input_gdf.distance(x.geometry).sort_values().index[0]
        return input_gdf.loc[polygon_index,column_name]

def load_config():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    config_path = os.path.join(script_dir,'..','..', 'config.json')

    with open(config_path, 'r') as config_fh:
        config = json.load(config_fh)
    return config

def create_network_from_nodes_and_edges(nodes,edges,node_edge_prefix,
                        snap_distance=None,geometry_precision=False,by=None):
    edges.columns = map(str.lower, edges.columns)
    if "id" in edges.columns.values.tolist():
        edges.rename(columns={"id": "e_id"}, inplace=True)

    # Deal with empty edges (drop)
    empty_idx = edges.geometry.apply(lambda e: e is None or e.is_empty)
    if empty_idx.sum():
        empty_edges = edges[empty_idx]
        print(f"Found {len(empty_edges)} empty edges.")
        print(empty_edges)
        edges = edges[~empty_idx].copy()

    network = snkit.Network(nodes, edges)
    print("* Done with network creation")

    network = snkit.network.split_multilinestrings(network)
    print("* Done with splitting multilines")
    if geometry_precision is True:
        network = snkit.network.round_geometries(network, precision=5)
        print("* Done with rounding off geometries")

    if nodes is not None:
        if snap_distance is not None:
            # network = snkit.network.link_nodes_to_edges_within(network, snap_distance, tolerance=1e-10)
            network = link_nodes_to_nearest_edge(network, tolerance=1e-9)
            print ('* Done with joining nodes to edges')
        else:
            network = snkit.network.snap_nodes(network)
            print ('* Done with snapping nodes to edges')
        # network.nodes = snkit.network.drop_duplicate_geometries(network.nodes)
        # print ('* Done with dropping same geometries')

        # network = snkit.network.split_edges_at_nodes(network,tolerance=9e-10)
        # print ('* Done with splitting edges at nodes')

    network = snkit.network.add_endpoints(network)   
    print ('* Done with adding endpoints')

    network.nodes = snkit.network.drop_duplicate_geometries(network.nodes)
    print ('* Done with dropping same geometries')

    network = snkit.network.split_edges_at_nodes(network,tolerance=1e-20)
    print ('* Done with splitting edges at nodes')
    
    network = snkit.network.add_ids(network, 
                            edge_prefix=f"{node_edge_prefix}e", 
                            node_prefix=f"{node_edge_prefix}n")
    network = snkit.network.add_topology(network, id_col='id')
    print ('* Done with network topology')

    if by is not None:
        network = snkit.network.merge_edges(network,by=by)
        print ('* Done with merging network')

    network.edges.rename(columns={'from_id':'from_node',
                                'to_id':'to_node',
                                'id':'edge_id'},
                                inplace=True)
    network.nodes.rename(columns={'id':'node_id'},inplace=True)
    
    return network

def network_od_path_estimations(graph,
    source, target, cost_criteria,path_id_column):
    """Estimate the paths, distances, times, and costs for given OD pair

    Parameters
    ---------
    graph
        igraph network structure
    source
        String/Float/Integer name of Origin node ID
    source
        String/Float/Integer name of Destination node ID
    tonnage : float
        value of tonnage
    vehicle_weight : float
        unit weight of vehicle
    cost_criteria : str
        name of generalised cost criteria to be used: min_gcost or max_gcost
    time_criteria : str
        name of time criteria to be used: min_time or max_time
    fixed_cost : bool

    Returns
    -------
    edge_path_list : list[list]
        nested lists of Strings/Floats/Integers of edge ID's in routes
    path_dist_list : list[float]
        estimated distances of routes
    path_time_list : list[float]
        estimated times of routes
    path_gcost_list : list[float]
        estimated generalised costs of routes

    """
    paths = graph.get_shortest_paths(source, target, weights=cost_criteria, output="epath")


    edge_path_list = []
    path_gcost_list = []
    # for p in range(len(paths)):
    for path in paths:
        edge_path = []
        path_gcost = 0
        if path:
            for n in path:
                edge_path.append(graph.es[n][path_id_column])
                path_gcost += graph.es[n][cost_criteria]

        edge_path_list.append(edge_path)
        path_gcost_list.append(path_gcost)

    
    return edge_path_list, path_gcost_list

def network_od_path_estimations_multiattribute(graph,
    source, target, cost_criteria,path_id_column,attribute_list=None):
    """Estimate the paths, distances, times, and costs for given OD pair

    Parameters
    ---------
    graph
        igraph network structure
    source
        String/Float/Integer name of Origin node ID
    source
        String/Float/Integer name of Destination node ID
    tonnage : float
        value of tonnage
    vehicle_weight : float
        unit weight of vehicle
    cost_criteria : str
        name of generalised cost criteria to be used: min_gcost or max_gcost
    time_criteria : str
        name of time criteria to be used: min_time or max_time
    fixed_cost : bool

    Returns
    -------
    edge_path_list : list[list]
        nested lists of Strings/Floats/Integers of edge ID's in routes
    path_dist_list : list[float]
        estimated distances of routes
    path_time_list : list[float]
        estimated times of routes
    path_gcost_list : list[float]
        estimated generalised costs of routes

    """
    paths = graph.get_shortest_paths(source, target, weights=cost_criteria, output="epath")

    paths_list = []
    if attribute_list is None:
        for path in paths:
            path_dict = {'edge_path':[],cost_criteria:0}
            if path:
                for n in path:
                    path_dict['edge_path'].append(graph.es[n][path_id_column])
                    path_dict[cost_criteria] += graph.es[n][cost_criteria]

            paths_list.append(path_dict)
    else:
        for path in paths:
            path_dict = dict([('edge_path',[]),(cost_criteria,0)] + [(a,0) for a in attribute_list])
            if path:
                for n in path:
                    path_dict['edge_path'].append(graph.es[n][path_id_column])
                    path_dict[cost_criteria] += graph.es[n][cost_criteria]
                    for a in attribute_list:
                        path_dict[a] += graph.es[n][a]

            paths_list.append(path_dict)

    return pd.DataFrame(paths_list)

def network_od_node_edge_path_estimations(graph,
    source, target, cost_criteria,distance_criteria,time_criteria,border_cost_criteria,path_id_column):
    """Estimate the paths, distances, times, and costs for given OD pair

    Parameters
    ---------
    graph
        igraph network structure
    source
        String/Float/Integer name of Origin node ID
    source
        String/Float/Integer name of Destination node ID
    tonnage : float
        value of tonnage
    vehicle_weight : float
        unit weight of vehicle
    cost_criteria : str
        name of generalised cost criteria to be used: min_gcost or max_gcost
    time_criteria : str
        name of time criteria to be used: min_time or max_time
    fixed_cost : bool

    Returns
    -------
    edge_path_list : list[list]
        nested lists of Strings/Floats/Integers of edge ID's in routes
    path_dist_list : list[float]
        estimated distances of routes
    path_time_list : list[float]
        estimated times of routes
    path_gcost_list : list[float]
        estimated generalised costs of routes

    """
    edge_paths = graph.get_shortest_paths(source, target, weights=cost_criteria, output="epath")
    node_paths = graph.get_shortest_paths(source, target, weights=cost_criteria, output="vpath")

    edge_path_list = []
    node_path_list = []
    path_gcost_list = []
    path_dcost_list = []
    path_tcost_list = []
    path_bcost_list = []
    total_path_gcost_list = []
    # for p in range(len(paths)):
    for path in edge_paths:
        edge_path = []
        path_gcost = []
        path_dcost = []
        path_tcost = []
        path_bcost = []
        total_gcost = 0
        if path:
            for n in path:
                edge_path.append(graph.es[n][path_id_column])
                path_gcost.append(graph.es[n][cost_criteria])
                path_dcost.append(graph.es[n][distance_criteria])
                path_tcost.append(graph.es[n][time_criteria])
                path_bcost.append(graph.es[n][border_cost_criteria])
            total_gcost = sum(path_gcost)

        edge_path_list.append(edge_path)
        path_gcost_list.append(path_gcost)
        path_dcost_list.append(path_dcost)
        path_tcost_list.append(path_tcost)
        path_bcost_list.append(path_bcost)
        total_path_gcost_list.append(total_gcost)
    for path in node_paths:
        node_path = []
        if path:
            for n in path:
                node_path.append(graph.vs[n]["name"])
        node_path_list.append(node_path)
    
    return edge_path_list,node_path_list,path_gcost_list,path_dcost_list,path_tcost_list,path_bcost_list,total_path_gcost_list    

def network_od_node_edge_paths_assembly(points_dataframe, graph,
                                cost_criteria,distance_criteria,time_criteria,
                                border_cost_criteria,
                                path_id_column,
                                origin_id_column,destination_id_column,
                                store_paths=True):

    """Assemble estimates of OD paths, distances, times, costs and tonnages on networks

    Parameters
    ----------
    points_dataframe : pandas.DataFrame
        OD nodes and their tonnages
    graph
        igraph network structure
    region_name : str
        name of Province
    excel_writer
        Name of the excel writer to save Pandas dataframe to Excel file

    Returns
    -------
    save_paths_df : pandas.DataFrame
        - origin - String node ID of Origin
        - destination - String node ID of Destination
        - edge_path - List of string of edge ID's for paths with minimum generalised cost flows
        - gcost - Float values of estimated generalised cost for paths with minimum generalised cost flows

    """
    save_paths = []
    points_dataframe = points_dataframe.set_index(origin_id_column)
    origins = list(set(points_dataframe.index.values.tolist()))
    for origin in origins:
        destinations = list(set(points_dataframe.loc[[origin], destination_id_column].values.tolist()))

        get_epath,get_npath,get_cpath,get_dpath,get_tpath,get_bpath,get_gcost = network_od_node_edge_path_estimations(
                graph, origin, destinations, cost_criteria,distance_criteria,time_criteria,border_cost_criteria,path_id_column)

        # tons = points_dataframe.loc[[origin], tonnage_column].values
        save_paths += list(zip([origin]*len(destinations),
                            destinations, get_epath,get_npath,
                            get_cpath,get_dpath,get_tpath,get_bpath,get_gcost))

        # print(f"done with {origin}")
        # except:
        #     print(f"* no path between {origin}-{destinations}")
    
    cols = [
        origin_id_column, destination_id_column, 'edge_path','node_path',
        f'{cost_criteria}_path',f'{distance_criteria}_path',f'{time_criteria}_path',
        f'{border_cost_criteria}_path',cost_criteria
    ]
    save_paths_df = pd.DataFrame(save_paths, columns=cols)
    if store_paths is False:
        save_paths_df.drop(["edge_path","node_path"],axis=1,inplace=True)

    points_dataframe = points_dataframe.reset_index()
    # save_paths_df = pd.merge(save_paths_df, points_dataframe, how='left', on=[
    #                          'origin_id', 'destination_id']).fillna(0)

    save_paths_df = pd.merge(points_dataframe,save_paths_df,how='left', on=[
                             origin_id_column, destination_id_column]).fillna(0)

    # save_paths_df = save_paths_df[(save_paths_df[tonnage_column] > 0)
    #                               & (save_paths_df['origin_id'] != 0)]
    save_paths_df = save_paths_df[save_paths_df[origin_id_column] != 0]

    return save_paths_df

def network_od_paths_assembly(points_dataframe, graph,
                                cost_criteria,path_id_column,store_edge_path=True):
    """Assemble estimates of OD paths, distances, times, costs and tonnages on networks

    Parameters
    ----------
    points_dataframe : pandas.DataFrame
        OD nodes and their tonnages
    graph
        igraph network structure
    region_name : str
        name of Province
    excel_writer
        Name of the excel writer to save Pandas dataframe to Excel file

    Returns
    -------
    save_paths_df : pandas.DataFrame
        - origin - String node ID of Origin
        - destination - String node ID of Destination
        - edge_path - List of string of edge ID's for paths with minimum generalised cost flows
        - gcost - Float values of estimated generalised cost for paths with minimum generalised cost flows

    """
    save_paths = []
    points_dataframe = points_dataframe.set_index('origin_id')
    origins = list(set(points_dataframe.index.values.tolist()))
    for origin in origins:
        try:
            destinations = list(set(points_dataframe.loc[[origin], 'destination_id'].values.tolist()))

            get_path, get_gcost = network_od_path_estimations(
                    graph, origin, destinations, cost_criteria,path_id_column)

            # tons = points_dataframe.loc[[origin], tonnage_column].values
            save_paths += list(zip([origin]*len(destinations),
                                destinations, get_path,
                                get_gcost))

            # print(f"done with {origin}")
        except:
            print(f"* no path between {origin}-{destinations}")
    
    cols = [
        'origin_id', 'destination_id', 'edge_path',cost_criteria
    ]
    save_paths_df = pd.DataFrame(save_paths, columns=cols)
    if store_edge_path is False:
        save_paths_df.drop("edge_path",axis=1,inplace=True)

    points_dataframe = points_dataframe.reset_index()
    # save_paths_df = pd.merge(save_paths_df, points_dataframe, how='left', on=[
    #                          'origin_id', 'destination_id']).fillna(0)

    save_paths_df = pd.merge(points_dataframe,save_paths_df,how='left', on=[
                             'origin_id', 'destination_id']).fillna(0)

    # save_paths_df = save_paths_df[(save_paths_df[tonnage_column] > 0)
    #                               & (save_paths_df['origin_id'] != 0)]
    save_paths_df = save_paths_df[save_paths_df['origin_id'] != 0]

    return save_paths_df

def find_nodes_in_path(x,network_dataframe,edge_id_column,edge_path_column):
    if len(x[edge_path_column]) > 0:
        # df = network_dataframe[network_dataframe[edge_id_column].isin(x[edge_path_column])]
        df = network_dataframe.set_index(edge_id_column).loc[x[edge_path_column]].reset_index(inplace=False)
        if len(df.index) > 0:
            from_to_nodes = df[["from_id","to_id"]].values.tolist()
            from_to_nodes = [item for sublist in from_to_nodes for item in sublist]
            return list(dict.fromkeys(from_to_nodes))
        else:
            return []
    else:
        return []

def add_node_paths(flow_dataframe,network_dataframe,edge_id_column,edge_path_column):
    flow_dataframe["node_path"] = flow_dataframe.progress_apply(
                                    lambda x:find_nodes_in_path(
                                        x,network_dataframe,
                                        edge_id_column,edge_path_column),
                                    axis=1)
    return flow_dataframe



def network_od_paths_assembly_multiattributes(points_dataframe, graph,
                                cost_criteria,path_id_column,
                                origin_id_column,destination_id_column,
                                attribute_list=None,store_edge_path=True):
    """Assemble estimates of OD paths, distances, times, costs and tonnages on networks

    Parameters
    ----------
    points_dataframe : pandas.DataFrame
        OD nodes and their tonnages
    graph
        igraph network structure
    region_name : str
        name of Province
    excel_writer
        Name of the excel writer to save Pandas dataframe to Excel file

    Returns
    -------
    save_paths_df : pandas.DataFrame
        - origin - String node ID of Origin
        - destination - String node ID of Destination
        - edge_path - List of string of edge ID's for paths with minimum generalised cost flows
        - gcost - Float values of estimated generalised cost for paths with minimum generalised cost flows

    """
    save_paths_df = []
    points_dataframe = points_dataframe.set_index(origin_id_column)
    origins = list(set(points_dataframe.index.values.tolist()))
    # for origin in origins:
    #     try:
    #         destinations = list(set(points_dataframe.loc[[origin], destination_id_column].values.tolist()))

    #         get_path_df = network_od_path_estimations(
    #                 graph, origin, destinations, cost_criteria,path_id_column,
    #                 attribute_list=attribute_list)
    #         get_path_df[origin_id_column] = origin
    #         get_path_df[destination_id_column] = destinations
    #         # tons = points_dataframe.loc[[origin], tonnage_column].values
    #         # save_paths += list(zip([origin]*len(destinations),
    #         #                     destinations, get_path,
    #         #                     get_gcost))
    #         save_paths_df.append(get_path_df)
    #         # print(f"done with {origin}")
    #     except:
    #         print(f"* no path between {origin}-{destinations}")

    for origin in origins:
        destinations = list(set(points_dataframe.loc[[origin], destination_id_column].values.tolist()))

        get_path_df = network_od_path_estimations_multiattribute(
                graph, origin, destinations, cost_criteria,path_id_column,
                attribute_list=attribute_list)
        get_path_df[origin_id_column] = origin
        get_path_df[destination_id_column] = destinations
        # tons = points_dataframe.loc[[origin], tonnage_column].values
        # save_paths += list(zip([origin]*len(destinations),
        #                     destinations, get_path,
        #                     get_gcost))
        save_paths_df.append(get_path_df)
        # print(f"done with {origin}")
    
    # cols = [
    #     'origin_id', 'destination_id', 'edge_path',cost_criteria
    # ]
    # save_paths_df = pd.DataFrame(save_paths, columns=cols)
    save_paths_df = pd.concat(save_paths_df,axis=0,ignore_index=True)
    if store_edge_path is False:
        save_paths_df.drop("edge_path",axis=1,inplace=True)

    points_dataframe = points_dataframe.reset_index()
    # save_paths_df = pd.merge(save_paths_df, points_dataframe, how='left', on=[
    #                          'origin_id', 'destination_id']).fillna(0)

    save_paths_df = pd.merge(points_dataframe,save_paths_df,how='left', on=[
                             origin_id_column, destination_id_column]).fillna(0)

    # save_paths_df = save_paths_df[(save_paths_df[tonnage_column] > 0)
    #                               & (save_paths_df['origin_id'] != 0)]
    save_paths_df = save_paths_df[save_paths_df[origin_id_column] != 0]

    return save_paths_df

def get_flow_on_edges(save_paths_df,edge_id_column,edge_path_column,
    flow_column):
    """Write results to Shapefiles

    Outputs ``gdf_edges`` - a shapefile with minimum and maximum tonnage flows of all
    commodities/industries for each edge of network.

    Parameters
    ---------
    save_paths_df
        Pandas DataFrame of OD flow paths and their tonnages
    industry_columns
        List of string names of all OD commodities/industries indentified
    min_max_exist
        List of string names of commodity/industry columns for which min-max tonnage column names already exist
    gdf_edges
        GeoDataFrame of network edge set
    save_csv
        Boolean condition to tell code to save created edge csv file
    save_shapes
        Boolean condition to tell code to save created edge shapefile
    shape_output_path
        Path where the output shapefile will be stored
    csv_output_path
        Path where the output csv file will be stored

    """
    edge_flows = defaultdict(float)
    for row in save_paths_df.itertuples():
        for item in getattr(row,edge_path_column):
            edge_flows[item] += getattr(row,flow_column)

    return pd.DataFrame([(k,v) for k,v in edge_flows.items()],columns=[edge_id_column,flow_column])

def create_igraph_from_dataframe(graph_dataframe, directed=False, simple=False):
    graph = ig.Graph.TupleList(
        graph_dataframe.itertuples(index=False),
        edge_attrs=list(graph_dataframe.columns)[2:],
        directed=directed
    )
    if simple:
        graph.simplify()

    es, vs, simple = graph.es, graph.vs, graph.is_simple()
    d = "directed" if directed else "undirected"
    s = "simple" if simple else "multi"
    print(
        "Created {}, {} {}: {} edges, {} nodes.".format(
            s, d, "igraph", len(es), len(vs)))

    return graph

def get_path_indexes_for_edges(edge_ids_with_paths,selected_edge_list):
    return list(
            set(
                list(
                    chain.from_iterable([
                        path_idx for path_key,path_idx in edge_ids_with_paths.items() if path_key in selected_edge_list
                                        ]
                                        )
                    )
                )
            )

def get_flow_paths_indexes_of_edges(flow_dataframe,path_criteria):
    edge_path_index = defaultdict(list)
    for v in flow_dataframe.itertuples():
        for k in getattr(v,path_criteria):
            edge_path_index[k].append(v.Index)

    del flow_dataframe
    return edge_path_index

def get_flow_paths_indexes_and_edges_dataframe(flow_dataframe,path_criteria,id_column="id"):
    edge_path_index = []
    for v in flow_dataframe.itertuples():
        path = getattr(v,path_criteria)
        edge_path_index += list(zip(path,[v.Index]*len(path)))
    del flow_dataframe
    return pd.DataFrame(edge_path_index,columns=[id_column,"path_index"])


def update_flow_and_overcapacity(od_dataframe,network_dataframe,flow_column,edge_id_column="edge_id",subtract=False):
    edge_flows = get_flow_on_edges(od_dataframe,edge_id_column,"edge_path",flow_column)
    edge_flows.rename(columns={flow_column:"added_flow"},inplace=True)
    network_dataframe = pd.merge(network_dataframe,edge_flows,how="left",on=[edge_id_column]).fillna(0)
    del edge_flows
    if subtract is True:
        network_dataframe[flow_column] = network_dataframe[flow_column] - network_dataframe["added_flow"]
    else:
        network_dataframe[flow_column] += network_dataframe["added_flow"]
    network_dataframe["over_capacity"] = network_dataframe["capacity"] - network_dataframe[flow_column]

    return network_dataframe

def find_minimal_flows_along_overcapacity_paths(over_capacity_ods,network_dataframe,
                                        over_capacity_edges,edge_id_paths,edge_id_column,flow_column):
    over_capacity_edges_df = pd.DataFrame([
                                (
                                    path_key,path_idx
                                ) for path_key,path_idx in edge_id_paths.items() if path_key in over_capacity_edges
                            ],columns = [edge_id_column,"path_indexes"]
                                )
    over_capacity_edges_df = pd.merge(over_capacity_edges_df,
                                network_dataframe[[edge_id_column,"residual_capacity","added_flow"]],
                                how="left",
                                on=[edge_id_column])
    # print (over_capacity_edges_df)
    # print (over_capacity_ods)
    over_capacity_edges_df["edge_path_flow"] = over_capacity_edges_df.progress_apply(
                                        lambda x:over_capacity_ods[
                                            over_capacity_ods.path_indexes.isin(x.path_indexes)
                                            ][flow_column].values,
                                        axis=1
                                        )
    over_capacity_edges_df["edge_path_flow_cor"] = over_capacity_edges_df.progress_apply(
                                        lambda x:list(
                                            1.0*x.residual_capacity*x.edge_path_flow/x.added_flow),
                                        axis=1
                                        )
    over_capacity_edges_df["path_flow_tuples"] = over_capacity_edges_df.progress_apply(
                                        lambda x:list(zip(x.path_indexes,x.edge_path_flow_cor)),axis=1)

    min_flows = []
    for r in over_capacity_edges_df.itertuples():
        min_flows += r.path_flow_tuples

    min_flows = pd.DataFrame(min_flows,columns=["path_indexes","min_flows"])
    min_flows = min_flows.sort_values(by=["min_flows"],ascending=True)
    min_flows = min_flows.drop_duplicates(subset=["path_indexes"],keep="first")

    over_capacity_ods = pd.merge(over_capacity_ods,min_flows,how="left",on=["path_indexes"])
    del min_flows, over_capacity_edges_df
    over_capacity_ods["residual_flows"] = over_capacity_ods[flow_column] - over_capacity_ods["min_flows"]

    return over_capacity_ods

def od_flow_allocation_capacity_constrained(flow_ods,network_dataframe,
                                            flow_column,cost_column,
                                            distance_column,time_column,
                                            border_column,
                                            path_id_column,origin_id_column,
                                            destination_id_column,
                                            store_edge_path=True):
    network_dataframe["over_capacity"] = network_dataframe["capacity"] - network_dataframe[flow_column]
    capacity_ods = []
    unassigned_paths = []
    while len(flow_ods.index) > 0:
        # print (flow_ods)
        graph = create_igraph_from_dataframe(
                    network_dataframe[network_dataframe["over_capacity"] > 1e-3],
                    directed=True)
        graph_nodes = [x['name'] for x in graph.vs]
        unassigned_paths.append(flow_ods[~((flow_ods[origin_id_column].isin(graph_nodes)) & (flow_ods[destination_id_column].isin(graph_nodes)))])
        flow_ods = flow_ods[(flow_ods[origin_id_column].isin(graph_nodes)) & (flow_ods[destination_id_column].isin(graph_nodes))]
        if len(flow_ods.index) > 0:
            # flow_ods = network_od_paths_assembly(flow_ods,graph,cost_column)
            # flow_ods = network_od_paths_assembly_multiattributes(
            #                     flow_ods,graph,cost_column,
            #                     path_id_column,origin_id_column,
            #                     destination_id_column)
            flow_ods = network_od_node_edge_paths_assembly(
                                    flow_ods,graph,
                                    cost_column,
                                    distance_column,
                                    time_column,
                                    border_column,
                                    path_id_column,origin_id_column,
                                    destination_id_column)
            unassigned_paths.append(flow_ods[flow_ods[cost_column] == 0])
            flow_ods = flow_ods[flow_ods[cost_column] > 0]
            if len(flow_ods.index) > 0:
                # print (flow_ods)
                network_dataframe["residual_capacity"] = network_dataframe["over_capacity"]
                network_dataframe = update_flow_and_overcapacity(flow_ods,
                                        network_dataframe,flow_column,edge_id_column=path_id_column)
                over_capacity_edges = network_dataframe[network_dataframe["over_capacity"] < -1.0e-3][path_id_column].values.tolist()
                if len(over_capacity_edges) > 0:
                    edge_id_paths = get_flow_paths_indexes_of_edges(flow_ods,"edge_path")
                    edge_paths_overcapacity = get_path_indexes_for_edges(edge_id_paths,over_capacity_edges)
                    if store_edge_path is False:
                        cap_ods = flow_ods[~flow_ods.index.isin(edge_paths_overcapacity)]
                        cap_ods.drop(["edge_path","node_path"],axis=1,inplace=True)
                        capacity_ods.append(cap_ods)
                        del cap_ods
                    else:
                        capacity_ods.append(flow_ods[~flow_ods.index.isin(edge_paths_overcapacity)])

                    over_capacity_ods = flow_ods[flow_ods.index.isin(edge_paths_overcapacity)]
                    over_capacity_ods["path_indexes"] = over_capacity_ods.index.values.tolist()
                    over_capacity_ods = find_minimal_flows_along_overcapacity_paths(over_capacity_ods,network_dataframe,
                                                                over_capacity_edges,
                                                                edge_id_paths,path_id_column,flow_column)
                    cap_ods = over_capacity_ods.copy() 
                    cap_ods.drop(["path_indexes",flow_column,"residual_flows"],axis=1,inplace=True)
                    cap_ods.rename(columns={"min_flows":flow_column},inplace=True)
                    if store_edge_path is False:
                        cap_ods.drop(["edge_path","node_path"],axis=1,inplace=True)
                    
                    capacity_ods.append(cap_ods)
                    del cap_ods

                    over_capacity_ods["residual_ratio"] = over_capacity_ods["residual_flows"]/over_capacity_ods[flow_column]
                    over_capacity_ods.drop(["path_indexes",flow_column,"min_flows"],axis=1,inplace=True)
                    over_capacity_ods.rename(columns={"residual_flows":flow_column},inplace=True)

                    network_dataframe.drop("added_flow",axis=1,inplace=True)
                    network_dataframe = update_flow_and_overcapacity(over_capacity_ods,
                                                        network_dataframe,flow_column,path_id_column,subtract=True)
                    network_dataframe.drop("added_flow",axis=1,inplace=True)
                    flow_ods = over_capacity_ods[over_capacity_ods["residual_ratio"] > 0.01]
                    flow_ods.drop(["edge_path","node_path",f"{cost_column}_path",
                                    f"{distance_column}_path",f"{time_column}_path",
                                    f"{border_column}_path",cost_column,
                                    "residual_ratio"],axis=1,inplace=True)
                    del over_capacity_ods
                else:
                    if store_edge_path is False:
                        flow_ods.drop(["edge_path","node_path",f"{cost_column}_path",
                                    f"{distance_column}_path",f"{time_column}_path",
                                    f"{border_column}_path"],axis=1,inplace=True)
                    capacity_ods.append(flow_ods)
                    network_dataframe.drop(["residual_capacity","added_flow"],axis=1,inplace=True)
                    flow_ods = pd.DataFrame()

    return capacity_ods, unassigned_paths, network_dataframe

def truncate_by_threshold(flow_dataframe, flow_column='flux', threshold=.99):
    print(f"Truncating paths with threshold {threshold * 100:.0f}%.")
    flows_sorted = flow_dataframe.reset_index(drop=True).sort_values(by=flow_column, ascending=False)
    fluxes_sorted = flows_sorted[flow_column]
    total_flux = fluxes_sorted.sum()
    flux_percentiles = fluxes_sorted.cumsum() / total_flux
    excess = flux_percentiles[flux_percentiles >= threshold]
    cutoff = excess.idxmin()
    keep = flux_percentiles[flux_percentiles <= threshold].index
    flows_truncated = flows_sorted.loc[keep, :]
    print(f"Number of paths before: {len(flows_sorted):,.0f}.")
    print(f"Number of paths after: {len(flows_truncated):,.0f}.")
    return flows_truncated