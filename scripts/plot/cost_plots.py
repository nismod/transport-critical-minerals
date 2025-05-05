"""Generate bar plots
"""
import os
import sys
import pandas as pd
pd.options.mode.copy_on_write = True
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from matplotlib.ticker import (MaxNLocator,LinearLocator, MultipleLocator)
from itertools import cycle, islice
import matplotlib.pyplot as plt
from matplotlib import cm
from map_plotting_utils import *
from mapping_properties import *
from tqdm import tqdm
import re
tqdm.pandas()


# Define the directory
output_dir = os.path.expanduser('~/critical_minerals_Africa/transport-outputs/figures/regional_figures/')


def main(config):
    output_data_path = config['paths']['results']
    figure_path = config['paths']['figures']

    # Unit cost plots
    make_plot = True
    if make_plot is True:
        
        df = pd.read_excel(
            output_data_path + '/all_data_pivots.xlsx',
            sheet_name="unit costs",
            skiprows=176,
            header=[0, 1]
        )

        # Subset to the first 20 columns (if that’s what you need)
        df = df.iloc[:, :20]

        # Filter rows where key header cells are missing.
        df_all = df[~df[('Country unconstrained: aggregated shares', 'Scenario')].isnull()]
        df_all = df_all[~df_all[('Constraint', 'share_type')].isnull()]

        # Clean the MultiIndex: remove trailing .1, .2, etc. from the first level (mineral names)
        df_all.columns = pd.MultiIndex.from_tuples(
            [(re.sub(r'\.\d+$', '', str(col[0])), col[1]) for col in df_all.columns]
        )

        # Function to remove trailing .1, .2, etc. from mineral names
        def clean_mineral(name):
            return re.sub(r'\.\d+$', '', str(name))

        # Create new multi-index tuples where level 0 is cleaned for minerals
        new_columns = []
        for col in df.columns:
            level0, level1 = col
            new_level0 = clean_mineral(level0)
            new_columns.append((new_level0, level1))
        df.columns = pd.MultiIndex.from_tuples(new_columns)

        # Rename the first and last columns to standard names.
        # The first column originally is something like ("Country unconstrained: aggregated shares", "Scenario")
        # and the last column is ("Constraint", "share_type")
        df = df.rename(columns={
            (df.columns[0][0], df.columns[0][1]): ("Scenario", "Scenario"),
            (df.columns[-1][0], df.columns[-1][1]): ("Constraint", "Constraint")
        })

        # Flatten the multi-index columns for easier processing
        df.columns = ['_'.join(col).strip() for col in df.columns]
        df = df.rename(columns={
            "Scenario_Scenario": "Scenario",
            "Constraint_Constraint": "Constraint"
        })

        # Drop any rows that might be duplicate header rows (e.g. where Scenario is "Scenario")
        df = df[df["Scenario"] != "Scenario"]

        # --- Step 2. Reshape the Data into Long Format ---

        # All columns except "Scenario" and "Constraint" hold mineral cost data.
        value_vars = [col for col in df.columns if col not in ["Scenario", "Constraint"]]

        # Melt the DataFrame so each row is one observation for a mineral-cost combination.
        df_long = df.melt(id_vars=["Scenario", "Constraint"], 
                        value_vars=value_vars, 
                        var_name="mineral_cost", 
                        value_name="share")

        # The column names are now like "cobalt_production", "copper_transport", etc.
        # Split this into separate 'mineral' and 'cost' columns.
        df_long[['mineral', 'cost']] = df_long['mineral_cost'].str.split("_", expand=True)
        df_long.drop(columns=["mineral_cost"], inplace=True)

        # Convert the share values to numeric (non-numeric entries become NaN)
        df_long["share"] = pd.to_numeric(df_long["share"], errors="coerce")

        # --- Step 3. Extract Year and Create a Combined Label ---

        # Extract a four-digit year from the Scenario (e.g., "2022_baseline")
        df_long["year"] = df_long["Scenario"].str.extract(r"(\d{4})")
        # Forward-fill any missing years (for sub-scenarios, if needed)
        df_long["year"] = df_long["year"].fillna(method="ffill")

        # Create a label for the x-axis combining year and mineral (e.g., "2022 - cobalt")
        df_long["year_mineral"] = df_long["year"] + " - " + df_long["mineral"]

        # --- Step 4. Aggregate and Pivot the Data for Plotting ---

        # Aggregate the share values (here we take the mean in case there are duplicates)
        agg = df_long.groupby(["Constraint", "year", "mineral", "cost", "year_mineral"])["share"] \
                    .mean().reset_index()

        # Pivot the data so that each cost type becomes its own column
        pivot_table = agg.pivot_table(index=["Constraint", "year_mineral"], 
                                    columns="cost", 
                                    values="share").fillna(0)
        pivot_table = pivot_table.reset_index()

        # --- Step 5. Plotting ---

        # Plot one subplot per Constraint type (e.g., "Country unconstrained" and "Region unconstrained")
        constraints = pivot_table["Constraint"].unique()
        n = len(constraints)
        fig, axs = plt.subplots(nrows=n, figsize=(12, 6 * n), squeeze=False)

        for ax, constraint in zip(axs.flat, constraints):
            data = pivot_table[pivot_table["Constraint"] == constraint].copy()
            data = data.sort_values("year_mineral")
            data.set_index("year_mineral", inplace=True)
            data.drop(columns=["Constraint"], inplace=True)
            
            data.plot(kind="bar", stacked=True, ax=ax)
            ax.set_title(f"Cost Shares by Mineral and Year ({constraint})")
            ax.set_xlabel("Year - Mineral")
            ax.set_ylabel("Share")
            ax.legend(title="Cost Type", bbox_to_anchor=(1.05, 1), loc="upper left")
            ax.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.show()

# I can't use average upper and lower costs for efficient scale because they don't disaggregate by stage...

if __name__ == '__main__':
    CONFIG = load_config()
    main(CONFIG)
    

