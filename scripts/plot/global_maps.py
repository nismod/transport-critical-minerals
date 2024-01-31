"""Road network risks and adaptation maps
"""
import os
import sys
from collections import OrderedDict
import pandas as pd
import geopandas as gpd
import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from map_plotting_utils import *
from tqdm import tqdm
tqdm.pandas()

def select_nodes(x,flow_column,threshold_ton_flows):
    if x["infra"] == "port":
        return 1
    elif (x["infra"] in ["station","stop","road"]) and (x["degree"] > 2) and (x[flow_column] > threshold_ton_flows):
        return 1
    else:
        return 0

def main(config):
    data_path = config['paths']['data']
    output_path = config['paths']['results']
    figure_path = config['paths']['figures']

    figures = os.path.join(figure_path)
    ccg_countries = pd.read_csv(os.path.join(data_path,"admin_boundaries","ccg_country_codes.csv"))
    ccg_isos = ccg_countries[ccg_countries["ccg_country"] == 1]["iso_3digit_alpha"].values.tolist()
    # mine_sites_df = gpd.read_file(os.path.join(data_path,
    #                                 "Minerals",
    #                                 "copper_mines_tons_refined_unrefined.gpkg"))
    # mine_sites_df["geometry"] = mine_sites_df.geometry.centroid
    # mine_sites_df["refining_type"] = np.where(mine_sites_df["process_binary"] == 0,"Unrefined","Refined")
    # output_column = "mine_output_approx_copper"
    # values_range = mine_sites_df[output_column].values.tolist()
    # ax_proj = get_projection(epsg=4326)
    # fig, ax_plots = plt.subplots(1,1,
    #                 subplot_kw={'projection': ax_proj},
    #                 figsize=(12,12),
    #                 dpi=500)
    # # ax_plots = ax_plots.flatten()
    # ax = plot_africa_basemap(ax_plots)
    # ax = point_map_plotting_colors_width(ax,mine_sites_df,
    #                             output_column,
    #                             values_range,
    #                             point_classify_column="refining_type",
    #                             point_categories=["Unrefined","Refined"],
    #                             point_colors=["#e31a1c","#41ae76"],
    #                             point_labels=[s.upper() for s in ["Unrefined","Refined"]],
    #                             point_zorder=[6,7,8,9],
    #                             point_steps=8,
    #                             width_step = 40.0,
    #                             interpolation = 'fisher-jenks',
    #                             legend_label="Annual output (tons)",
    #                             legend_size=16,
    #                             legend_weight=2.0,
    #                             no_value_label="No output",
    #                             )
    # plt.tight_layout()
    # save_fig(os.path.join(figures,"ccg_copper_outputs.png"))
    # plt.close()

    # s_and_p_mines = gpd.read_file(os.path.join(
    #                 data_path,
    #                 "Minerals",
    #                 "s_and_p_mines.gpkg"))
    # mineral = "Copper"
    # s_and_p_mines["refining_type"] = np.where(s_and_p_mines["PROCESSING_METHODS"].isna(),"Unrefined","Refined")
    # s_and_p_mines["copper_mine_binary"] = s_and_p_mines.progress_apply(
    #                                 lambda x: 1 if mineral.lower() in x["COMMODITIES_LIST"].lower() else 0,axis=1)
    # # s_and_p_mines[["COMMODITIES_LIST","copper_mine_binary"]].to_csv("test.csv")
    # commodity_column = "COMMODITY_PRODUCTION_TONNE_BY_PERIOD"
    # years = [2022,2030,2040]

    # for year in years:
    #     output_column = f"{commodity_column}_{year}_{mineral}"
    #     # mine_sites_df = s_and_p_mines[s_and_p_mines[output_column] > 0] 
    #     mine_sites_df = s_and_p_mines[s_and_p_mines["copper_mine_binary"] == 1] 
    #     mine_sites_df[output_column] = mine_sites_df[output_column].fillna(0)
    #     values_range = mine_sites_df[output_column].values.tolist()
    #     ax_proj = get_projection(epsg=4326)
    #     fig, ax_plots = plt.subplots(1,1,
    #                     subplot_kw={'projection': ax_proj},
    #                     figsize=(12,12),
    #                     dpi=500)
    #     ax = plot_africa_basemap(ax_plots)
    #     ax = point_map_plotting_colors_width(ax,mine_sites_df,
    #                                 output_column,
    #                                 values_range,
    #                                 point_classify_column="refining_type",
    #                                 point_categories=["Unrefined","Refined"],
    #                                 point_colors=["#e31a1c","#41ae76"],
    #                                 point_labels=[s.upper() for s in ["Unrefined","Refined"]],
    #                                 point_zorder=[6,7,8,9],
    #                                 point_steps=8,
    #                                 width_step = 40.0,
    #                                 interpolation = 'fisher-jenks',
    #                                 legend_label="Annual output (tons)",
    #                                 legend_size=16,
    #                                 legend_weight=2.0,
    #                                 no_value_label="No output",
    #                                 )
    #     plt.tight_layout()
    #     save_fig(os.path.join(figures,f"s_and_p_{mineral}_{year}_outputs.png"))
    #     plt.close()

    # flow_df = gpd.read_file(os.path.join(output_path,
    #                                 "flow_mapping",
    #                                 "copper_flows.gpkg"),
    #                                 layer="edges")
    # flow_df = flow_df[~flow_df.geometry.isna()]
    # # print (flow_df.columns.values.tolist())
    # output_columns = ["copper_mine_output_tons",
    #                     "copper_refined_mine_output_tons",
    #                     "copper_unrefined_mine_output_tons"]
    # output_types = ["total","refined","unrefined"]
    # output_colors = ['#7f0000',"#41ae76","#e31a1c"]
    # for idx,(oc,ot,ocl) in enumerate(zip(output_columns,output_types,output_colors)): 
    #     values_range = flow_df[oc].values.tolist()
    #     ax_proj = get_projection(epsg=4326)
    #     fig, ax_plots = plt.subplots(1,1,
    #                     subplot_kw={'projection': ax_proj},
    #                     figsize=(12,6),
    #                     dpi=500)
    #     # ax_plots = ax_plots.flatten()
    #     ax = plot_global_basemap(ax_plots)
    #     ax = line_map_plotting_colors_width(ax,flow_df,oc,
    #                         1.0,f"{ot.title()} Annual output (tons)","flows",
    #                         line_colors = 8*[ocl],
    #                         no_value_color = '#969696',
    #                         line_steps = 8,
    #                         width_step = 0.08,
    #                         interpolation='fisher-jenks')
    #     plt.tight_layout()
    #     save_fig(os.path.join(figures,f"ccg_copper_{ot}_flows.png"))
    #     plt.close()

    flow_df = gpd.read_file(os.path.join(output_path,
                                    "flow_mapping",
                                    "copper_flows.gpkg"),
                                    layer="edges")
    flow_df = flow_df[~flow_df.geometry.isna()]
    nodes_df = gpd.read_file(os.path.join(output_path,
                                    "flow_mapping",
                                    "copper_flows.gpkg"),
                                    layer="nodes")
    nodes_df = nodes_df[nodes_df["iso3"].isin(ccg_isos)]
    # print (flow_df.columns.values.tolist())
    output_columns = ["copper_mine_output_tons",
                        "copper_refined_mine_output_tons",
                        "copper_unrefined_mine_output_tons"]
    output_types = ["total","refined","unrefined"]
    output_colors = ['#7f0000',"#41ae76","#e31a1c"]
    for idx,(oc,ot,ocl) in enumerate(zip(output_columns,output_types,output_colors)): 
        values_range = flow_df[oc].values.tolist()
        threshold_ton_flows = nodes_df[oc].quantile(0.95)

        nodes_df["select_nodes_binary"] = nodes_df.progress_apply(
                                lambda x:select_nodes(x,oc,threshold_ton_flows),
                                axis=1)
        print (threshold_ton_flows)
        ax_proj = get_projection(epsg=4326)
        fig, ax_plots = plt.subplots(1,1,
                        subplot_kw={'projection': ax_proj},
                        figsize=(12,12),
                        dpi=500)
        # ax_plots = ax_plots.flatten()
        # ax = plot_global_basemap(ax_plots)
        ax = plot_africa_basemap(ax_plots)
        ax = line_map_plotting_colors_width(ax,flow_df,oc,
                            1.0,f"{ot.title()} Annual output (tons)","flows",
                            line_colors = 8*[ocl],
                            no_value_color = '#969696',
                            line_steps = 8,
                            width_step = 0.08,
                            interpolation='fisher-jenks')
        n_df = nodes_df[nodes_df["select_nodes_binary"] == 1]
        n_cls = ["sea","rail","road"]
        print (n_df)
        ax = point_map_plotting_colors_width(ax,n_df,
                                oc,
                                n_df[oc].values.tolist(),
                                point_classify_column="mode",
                                point_categories=n_cls,
                                point_colors=["#1f78b4","#00441b","#993404"],
                                point_labels=[s.upper() for s in n_cls],
                                point_zorder=[20,21,22,23],
                                point_steps=8,
                                width_step = 40.0,
                                interpolation = 'fisher-jenks',
                                legend_label="Annual output (tons)",
                                legend_size=16,
                                legend_weight=2.0,
                                no_value_label="No output",
                                )
        plt.tight_layout()
        save_fig(os.path.join(figures,f"ccg_copper_{ot}_node_edge_flows.png"))
        plt.close()

if __name__ == '__main__':
    CONFIG = load_config()
    main(CONFIG)
