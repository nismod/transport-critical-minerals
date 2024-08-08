#!/usr/bin/env python
# coding: utf-8

import sys
import os
import ast
import pandas as pd
import numpy as np
import xlsxwriter
from xlsxwriter import Workbook
pd.options.mode.copy_on_write = True
import geopandas as gpd
from map_plotting_utils import *
from tqdm import tqdm
tqdm.pandas()

def find_cell(x,column_one,columns,color_dataframe):
    text_color = color_dataframe[color_dataframe.index == x[column_one]]["text_color"].values[0]
    background_color = color_dataframe[color_dataframe.index == x[column_one]]["background_color"].values[0]
    s = pd.Series(dict([(c,f'color:{text_color};background-color:{background_color}') for c in columns]))
    return s

def find_index(s,column_one,color_dataframe):
    l = []
    print (s)
    for r in s[column_one].values.tolist():
        text_color = color_dataframe[color_dataframe.index == r]["text_color"].values[0]
        background_color = color_dataframe[color_dataframe.index == r]["background_color"].values[0]
        l.append(f'color:{text_color};background-color:{background_color}')
    return pd.Series(l)


def main(config):
    processed_data_path = config['paths']['data']
    output_data_path = config['paths']['results']

    input_folder = os.path.join(output_data_path,"result_summaries")
    results_folder = os.path.join(output_data_path,"result_summaries")
    if os.path.exists(results_folder) == False:
        os.mkdir(results_folder)

    df = pd.read_excel(
                os.path.join(
                    input_folder,
                    "Critical Minerals Results Indicators Mock-Up Final.xlsx"),
                sheet_name="Per Scenario Per Mineral",header=[0,1])
    cols = df.columns.values.tolist()
    col_one  = cols[0]
    len_max = df[col_one].str.len().max()
    cdf = pd.read_excel(
                os.path.join(
                    input_folder,
                    "Critical Minerals Results Indicators Mock-Up Final.xlsx"),
                sheet_name="colors",index_col=[0])
    writer = pd.ExcelWriter(os.path.join(
                                results_folder,
                                "Test.xlsx")) 
    df.style.apply(
        lambda x:find_cell(x,col_one,cols,cdf),axis=1
        ).to_excel(writer,sheet_name="Test")

    # print (df)
    writer.sheets['Test'].set_column(1,1,len_max)
    writer.sheets['Test'].set_row(2, None, None, {'hidden': True})
    # writer.sheets['Test'].set_column(0,0,0,0,{'hidden': True})
    writer.close()




if __name__ == '__main__':
    CONFIG = load_config()
    main(CONFIG)