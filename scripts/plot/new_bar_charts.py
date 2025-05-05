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
from scipy.stats import zscore
import seaborn as sns
tqdm.pandas()

# Set a clean, modern global style
# Base seaborn setup
sns.set_theme(style="whitegrid", context="talk")

# Apply custom settings for even finer control
custom_params = {
    "axes.titlesize": 24,        # Bigger titles
    "axes.labelsize": 21,        # Bigger x/y labels
    "xtick.labelsize": 18,       # Bigger x-axis tick labels
    "ytick.labelsize": 18,       # Bigger y-axis tick labels
    "legend.fontsize": 18,       # Bigger legend text
    "legend.title_fontsize": 20, # Bigger legend title
    "figure.titlesize": 26,      # Bigger figure suptitles
    "axes.grid": True,           # Keep subtle gridlines
    "grid.linestyle": "--",      # Make grid dashed
    "grid.alpha": 0.6,           # Soft grid, not too dark
    "axes.edgecolor": "black",   # Stronger frame around plots
    "axes.linewidth": 1.2,       # Thicker axis lines
}

sns.set_context("talk", rc=custom_params)

mpl.rcParams["font.family"] = "DejaVu Sans"
mpl.rcParams["font.size"] = 12
mpl.rcParams["axes.spines.top"] = False
mpl.rcParams["axes.spines.right"] = False
mpl.rcParams["axes.facecolor"] = "#FFFFFF"
mpl.rcParams["grid.alpha"] = 0.3
mpl.rcParams["grid.linestyle"] = "--"
# Increase figure DPI globally
mpl.rcParams["figure.dpi"] = 120

# mpl.style.use('ggplot')
# mpl.rcParams['font.size'] = 10.
# mpl.rcParams['font.family'] = 'tahoma'
# mpl.rcParams['axes.labelsize'] = 12.
# mpl.rcParams['xtick.labelsize'] = 10.
# mpl.rcParams['ytick.labelsize'] = 10.

# Define the directory
output_dir = os.path.expanduser('~/critical_minerals_Africa/transport-outputs/figures/regional_figures')

# Create the folder if it doesn't exist
os.makedirs(output_dir, exist_ok=True)
# Main logic here
print(f"Output directory created or already exists: {output_dir}")


def plot_clustered_stacked(fig,axe,
                            dfall,
                            bar_colors,
                            labels=None,
                            scenario_labels = True,
                            shift_bars = True,
                            ylabel="Y-values",
                            title="multiple stacked bar plot",
                            H="/",
                            stacked=True, 
                            df_style = 'multiple',
                            orientation="vertical",
                            width=0.7, 
                            **kwargs):

    """ 
        Source: https://stackoverflow.com/questions/22787209/how-to-have-clusters-of-stacked-bars
        Given a list of dataframes, with identical columns and index, create a clustered stacked bar plot. 
        labels is a list of the names of the dataframe, used for the legend
        title is a string for the title of the plot
        H is the hatch used for identification of the different dataframe
    """
    
    n_df = len(dfall)

    if df_style == 'multiple':
        n_col = len(dfall[0].columns) 
        n_ind = len(dfall[0].index)

        for df in dfall : # for each data frame
            # bar_colors = list(islice(cycle(bar_colors), None, len(df)))
            color_cycle = list(islice(cycle(bar_colors), None, len(df.columns)))
            if orientation == "horizontal":
                print("got multiple and horizontal")
                # Plot horizontally
                axe = df.plot(
                    kind="barh",
                    ax=axe,
                    legend=False,
                    grid=False,
                    color=color_cycle,
                    edgecolor='white',
                    stacked=False,
                    width=width,
                    **kwargs
                )
            else:
                print("got multiple and vertical")
                # Original vertical plotting
                axe = df.plot(
                    kind="bar",
                    # linewidth=1.0,
                    ax=axe,
                    legend=False,
                    grid=False,
                    color=color_cycle,
                    edgecolor='white',
                    stacked=False,
                    width=width,
                    **kwargs
                )

    # For cases when only one year is used
    elif df_style == 'single_yr':
        n_col = len(dfall.columns) 
        n_ind = len(dfall.index)
        df = dfall
    # axe = plt.subplot(111)

        # bar_colors = list(islice(cycle(bar_colors), None, len(dfall)))
        color_cycle = list(islice(cycle(bar_colors), None, n_col))
        if orientation == "horizontal":
            axe = df.plot(
                kind="barh",
                # linewidth=1.0,
                stacked=stacked,
                ax=axe,
                legend=False,
                grid=False,
                color=color_cycle,
                edgecolor='white',
                width=width,
                **kwargs
            )
        else:
            axe = df.plot(
                kind="bar",
                # linewidth=1.0,
                stacked=stacked,
                ax=axe,
                legend=False,
                grid=False,
                color=color_cycle,
                edgecolor='white',
                width=width,
                **kwargs
            )
    else:
        raise ValueError(f"Unknown df_style: {df_style}")

 # -------------------------------------------
    # SHIFT THE BARS IF MULTIPLE DATAFRAMES
    # -------------------------------------------
    if shift_bars:
        h, l = axe.get_legend_handles_labels()
        num_groups = len(dfall)  # Number of datasets (e.g., 2030, 2040)
        num_categories = len(dfall[0].columns)  # Number of processing stages

        for i in range(num_groups):
            for j, patch in enumerate(h[i * num_categories: (i + 1) * num_categories]):
                for rect in patch.patches:
                    if orientation == "horizontal":
                        rect.set_y(rect.get_y() - (i / float(num_groups)) + (j / float(num_categories)) * 0.4)
                        rect.set_height(0.6 / float(num_categories))  # Reduce overlap by decreasing height
                    else:
                        rect.set_x(rect.get_x() - (i / float(num_groups)) + (j / float(num_categories)) * 0.4)
                        rect.set_width(0.6 / float(num_categories))  # Reduce overlap by decreasing width
    # -------------------------------------------
    # SET TICKS, LABELS, ETC.
    # -------------------------------------------
    if orientation == "horizontal":
        # The numeric axis is x, the categorical axis is y
        # We'll set the y-ticks to the midpoint of each cluster
        # (similar logic as before, but for y-axis).
        if df_style == "multiple":
            df_first = dfall[0]
            # number of categories
            n_ind = len(df_first.index)
        # else:
        if df_style == "single_yr":
            n_ind = len(dfall.index)
        
        # For each category, we have 2.0 steps, plus offset
        # This is the vertical version of your original set_xticks(...) code
        new_positions = (np.arange(0, 2 * n_ind, 2) + 1 / float(n_df + 1)) / 2.0

        axe.set_yticks(new_positions)
        # label with the index
        if df_style == "multiple":
            # axe.set_yticks(np.arange(len(df_first.index)) + width/2) # new to fix width
            axe.set_yticklabels(df_first.index, rotation=0, fontsize=12, fontweight="bold")
        else:
            # axe.set_yticks(np.arange(len(dfall.index)) + width/2) # new to fix width
            axe.set_yticklabels(dfall.index, rotation=0, fontsize=12, fontweight="bold")

        # x-axis label is now your numeric axis (production)
        axe.set_xlabel(ylabel, fontweight='bold', fontsize=15)
        # We'll call this something like 'Countries' or 'ISO' or similar
        axe.set_ylabel('')
        axe.tick_params(axis='x', labelsize=12)

        # Optionally you can flip the grid lines so it draws major lines along x
        axe.grid(which='major', axis='y', linestyle='-', zorder=0)
        axe.set_axisbelow(True)
    else:
        # the original vertical code
        # The numeric axis is y, the categorical axis is x
        if df_style == 'multiple':
            df_first = dfall[0]
            # new_positions = np.arange(len(df_first.index)) + width/2 # new to fix width
            n_ind = len(df_first.index)
        else:
            # new_positions = np.arange(len(dfall.index)) + width/2 # new to fix width
            n_ind = len(dfall.index)
        
        new_positions = (np.arange(0, 2 * n_ind, 2) + 1 / float(n_df + 1)) / 2.0 # removed to fix width
        axe.set_xticks(new_positions)

        if df_style == "multiple":
            axe.set_xticklabels(df_first.index, rotation=0, fontsize=12, fontweight="bold")
        else:
            axe.set_xticklabels(dfall.index, rotation=0, fontsize=12, fontweight="bold")

        axe.set_xlabel('')
        axe.set_ylabel(ylabel, fontweight='bold', fontsize=15)
        axe.tick_params(axis='y', labelsize=12)
        axe.grid(which='major', axis='x', linestyle='-', zorder=0)
        axe.set_axisbelow(True)

    # Title
    axe.set_title(title, fontweight='bold', fontsize=16)

    # -------------------------------------------
    # BUILD LEGEND
    # -------------------------------------------
    legend_handles = []
    # titles = ["$\\bf{Mineral \\, processing \\, stages}$", "$\\bf Scenarios$"]
    # legend_handles.append(axe.plot([],[], color="none", 
    #             label="$\\bf{Mineral \\, processing \\, stages}$")[0])

    # For the columns in the first DataFrame (the "stacked" part):
    if df_style == 'multiple':
        col_labels = dfall[0].columns
    else:
        col_labels = dfall.columns

    used_colors = list(islice(cycle(bar_colors), None, len(col_labels)))
    for bc, bl in zip(used_colors, col_labels):
        legend_handles.append(mpatches.Patch(color=bc, label=bl))

    # If we have multiple scenarios, add them with option to not have scenarios
    if scenario_labels == True:
        if df_style == 'multiple' and len(labels) == n_df:
            legend_handles.append(axe.plot([], [], color="none", label="$\\bf Scenario$")[0])
            scenario_colors = list(islice(cycle(bar_colors), None, n_df))
            for idx, lbl in enumerate(labels):
                legend_handles.append(
                    mpatches.Patch(facecolor=scenario_colors[idx], edgecolor='white', label=lbl)
                )
        elif df_style == 'single_yr' and len(labels) > 0:
            legend_handles.append(axe.plot([], [], color="none", label="$\\bf Scenario$")[0])
            for idx, lbl in enumerate(labels):
                scenario_color = list(islice(cycle(bar_colors), idx, idx+1))[0]
                legend_handles.append(
                    mpatches.Patch(facecolor=scenario_color, edgecolor='white', label=lbl)
                )
    # else:
        # titles = ["$\\bf{Mineral \\, processing \\, stages}$"]
    if orientation == "horizontal":
        # leg = axe.legend(handles=legend_handles, fontsize=12, loc='upper left', frameon=False)
        leg = axe.legend( handles=legend_handles,  fontsize=12, loc="upper center",
                        bbox_to_anchor=(0.5, -0.1),  ncol=3,  frameon=False )
        leg.set_title("Mineral processing stages\n", prop={"size":14, "weight":"bold"})

    if orientation == "vertical":
        leg = axe.legend(handles=legend_handles, fontsize=12, loc='lower left', frameon=False)

    if shift_bars == False:
        leg = axe.legend(handles=legend_handles, fontsize=12, loc='upper left', frameon=False)

    # Optionally move titles in legend
    # for item, label_ in zip(leg.legend_handles, leg.texts):
    #     if label_._text in titles:
    #         label_.set_ha('left')

    return axe

def plot_mineral_differences_all_minerals_env(df, region_level, mineral_colors, mineral_order):
    fig, axs = plt.subplots(2, 1, figsize=(14, 14), dpi=300, sharex=True)

    for i, year in enumerate(["2030", "2040"]):
        ax = axs[i]
        stage = "Metal content"
        
        df_year = df[(df["year"] == year) & (df["region_level"] == region_level)]
        if df_year.empty:
            continue

        # Aggregate across processing types to get total delta per mineral per country
        df_agg = df_year.groupby(["iso3", "reference_mineral"])["value"].sum().unstack(fill_value=0)

        # Ensure consistent mineral order and fill any missing ones with 0s
        df_agg = df_agg.reindex(columns=mineral_order, fill_value=0)

        # Plot horizontal stacked bar
        bottom = np.zeros(len(df_agg))
        for j, mineral in enumerate(mineral_order):
            ax.barh(df_agg.index, df_agg[mineral], left=bottom, color=mineral_colors[j], label=mineral)
            bottom += df_agg[mineral].values

        ax.set_title(f"{year} - {stage} - Constrained minus Unconstrained ({region_level.title()})", fontsize=24)
        ax.set_xlabel("Difference in annual metal content production (kt)", fontsize=24) #18
        ax.axvline(0, color="black", linewidth=1)
        ax.tick_params(axis='both', which='major', labelsize=24)
        ax.grid(True, axis="x", linestyle="--", alpha=0.6)

        if i == 0:
            ax.legend(title="Mineral", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=24, title_fontsize=24)

    plt.tight_layout()
    return fig

def plot_mineral_differences_all_minerals_env_rel(df, region_level, mineral_colors, mineral_order):
    fig, axs = plt.subplots(2, 1, figsize=(16, 14), dpi=300, sharex=True)

    for i, year in enumerate(["2030", "2040"]):
        ax = axs[i]
        stage = "Metal content"
        
        df_year = df[(df["year"] == year) & (df["region_level"] == region_level)]
        if df_year.empty:
            continue

        # Aggregate across processing types to get total delta per mineral per country
        df_agg = df_year.groupby(["iso3", "reference_mineral"])["value"].sum().unstack(fill_value=0)

        # Ensure consistent mineral order and fill any missing ones with 0s
        df_agg = df_agg.reindex(columns=mineral_order, fill_value=0)

        # Plot horizontal stacked bar
        bottom = np.zeros(len(df_agg))
        for j, mineral in enumerate(mineral_order):
            ax.barh(df_agg.index, df_agg[mineral], left=bottom, color=mineral_colors[j], label=mineral)
            bottom += df_agg[mineral].values

        ax.set_title(f"{year} - {stage} - Constrained minus Unconstrained ({region_level.title()})", fontsize=24)
        # ax.set_xlabel("Difference in annual metal content production (%)", fontsize=24) #18
        ax.axvline(0, color="black", linewidth=1)
        # ax.tick_params(axis='both', which='major', labelsize=24)
        ax.grid(True, axis="x", linestyle="--", alpha=0.6)

        # Ensure both subplots have x-axis labels and ticks
        ax.set_xlabel("Relative change in annual metal content production (%)", fontsize=24)
        ax.tick_params(axis='both', which='major', labelsize=24)
        ax.xaxis.set_tick_params(labelbottom=True)

        if i == 0:
            ax.legend(title="Mineral", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=24, title_fontsize=24)

    plt.tight_layout()
    return fig

def plot_mineral_heatmap_all_minerals_env_rel(df, region_level, mineral_order, cmap="RdBu_r"):
    """
    Plots side-by-side heatmaps (2030 and 2040) of percentage production differences
    across minerals and countries for a given region level.

    Parameters:
    - df: DataFrame with 'year', 'region_level', 'iso3', 'reference_mineral', 'value'
    - region_level: 'country' or 'region'
    - mineral_order: list of minerals to order columns
    - cmap: color map to use, default "RdBu_r"
    """

    fig, axes = plt.subplots(1, 2, figsize=(24, 12), dpi=300, sharey=True)

    years = ["2030", "2040"]
    for idx, year in enumerate(years):
        ax = axes[idx]

        # Filter data
        df_year = df[(df["year"] == year) & (df["region_level"] == region_level)]
        if df_year.empty:
            continue

        # Create pivot table: rows = countries, columns = minerals
        pivot_df = df_year.pivot_table(
            index="iso3",
            columns="reference_mineral",
            values="value",
            aggfunc="sum",
            fill_value=0
        )

        # Reorder minerals consistently
        pivot_df = pivot_df.reindex(columns=mineral_order, fill_value=0)

        # Sort countries alphabetically
        pivot_df = pivot_df.sort_index()

        # Plot heatmap
        sns.heatmap(
            pivot_df,
            annot=True,
            fmt=".1f",
            cmap=cmap,
            center=0,
            linewidths=0.5,
            cbar_kws={'label': 'Change in Production (%)'},
            ax=ax
        )

        ax.set_title(f"{year} - Metal Content Production Change (%) ({region_level.title()})", fontsize=20)
        ax.set_xlabel("Mineral", fontsize=16)
        if idx == 0:
            ax.set_ylabel("Country", fontsize=16)
        else:
            ax.set_ylabel("")
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

    plt.tight_layout()
    return fig

def plot_mineral_differences_all_minerals(df, constraint_label, mineral_colors, mineral_order):
    fig, axs = plt.subplots(2, 1, figsize=(14, 13), dpi=300, sharex=True)

    for i, year in enumerate(["2030", "2040"]):
        ax = axs[i]
        if "Beneficiation" in df['processing_type'].unique():
            stage = "All stages"
        elif year == "2030":
            stage = "Early refining"
        elif year == "2040":
            stage = "Precursor related product"
        

        df_year = df[(df["year"] == year) & (df["constraint_type"] == constraint_label)]
        if df_year.empty:
            continue

        # Aggregate across processing types to get total delta per mineral per country
        df_agg = df_year.groupby(["iso3", "reference_mineral"])["value"].sum().unstack(fill_value=0)

        # Ensure consistent mineral order and fill any missing ones with 0s
        df_agg = df_agg.reindex(columns=mineral_order, fill_value=0)

        # Plot horizontal stacked bar
        bottom = np.zeros(len(df_agg))
        for j, mineral in enumerate(mineral_order):
            ax.barh(df_agg.index, df_agg[mineral], left=bottom, color=mineral_colors[j], label=mineral)
            bottom += df_agg[mineral].values

        ax.set_title(f"{year} - {stage} - Regional minus National ({constraint_label.title()})", fontsize=24)
        ax.set_xlabel("Difference in annual production (kt)", fontsize=24)
        ax.axvline(0, color="black", linewidth=1)
        ax.grid(True, axis="x", linestyle="--", alpha=0.6)
        ax.tick_params(axis='y', labelsize=24)
        ax.tick_params(axis='x', labelsize=24)

        if i == 0:
            ax.legend(title="Mineral", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=24, title_fontsize=24)

    plt.tight_layout()
    return fig


def plot_stacked_bar(df, group_by, stack_col, value_col, orientation="vertical", 
                     ax=None, colors=None, short_val_label=None, units=None, 
                     annotate_totals=False, percentage=False, grouped_categ="", 
                     short_labels=None, **kwargs):
    """
    Creates a stacked bar plot from a DataFrame.
    
    Data is aggregated by the 'group_by' column and 'stack_col' so that each group 
    is represented as one bar, and each bar is split into segments according to the unique 
    values in stack_col. The numeric values are summed from value_col for each combination.
    
    If annotate_totals is True, the function calculates the total for each group and places
    the total value as text at the top (for vertical plots) or to the right (for horizontal plots)
    of each bar.
    
    Additionally, this function annotates each stacked segment with its category label. If a 
    segment is sufficiently large, the label is drawn inside the bar. If the segment is small,
    no external annotation is added.
    
    Parameters:
      - df: The input DataFrame.
      - group_by: The column name to group by (defines the categories on the primary axis).
      - stack_col: The column name that defines the stacking (each unique value becomes a segment).
      - value_col: The column name containing the numeric values.
      - orientation: "vertical" (default) for vertical bars or "horizontal" for horizontal bars.
      - ax: Optional matplotlib Axes to plot on. If None, a new figure and Axes are created.
      - colors: Optional. Either a dictionary mapping each unique value in stack_col to a color,
                or a list/tuple of colors to use in order.
      - short_val_label: A short label for the value column. Defaults to value_col if not provided.
      - units: The units to be displayed alongside the value label (e.g. "USD per tonne").
               Defaults to an empty string if not provided.
      - annotate_totals: Boolean. If True, annotate each bar with the total for that group.
      - percentage: Boolean. If True, the values in the pivot are converted to percentages.
      - grouped_categ: Title for the legend.
      - short_labels: Optional list of short names to use for the legend labels.
      - **kwargs: Additional keyword arguments to pass to the plotting function.
    
    Returns:
      The matplotlib Axes object containing the stacked bar plot.
    """
    # Set defaults for short_val_label and units.
    if short_val_label is None:
        short_val_label = value_col
    if units is None:
        units = ""
    custom_val_label = short_val_label
    if units:
        custom_val_label += " (" + units + ")"
    
    # Create the pivot table.
    df_pivot = df.pivot_table(index=group_by, columns=stack_col, values=value_col, 
                              aggfunc='sum', fill_value=0)
    
    if percentage:
        # Convert absolute values to percentages.
        df_shares = df_pivot.div(df_pivot.sum(axis=1), axis=0)
        df_plot = df_shares * 100
    else:
        df_plot = df_pivot.copy()
    
    # Process the colors.
    if isinstance(colors, dict):
        color_list = [colors.get(col, None) for col in df_plot.columns]
    elif isinstance(colors, (list, tuple)):
        color_list = colors
    else:
        color_list = None

    # Create the axes if not provided.
    if ax is None:
        fig, ax = plt.subplots()
    
    # Plot the stacked bar chart.
    if orientation == "horizontal":
        df_plot.plot(kind='barh', stacked=True, ax=ax, color=color_list, **kwargs)
        ax.set_xlabel(custom_val_label)
        ax.set_ylabel(group_by.replace("_", " "))
    elif orientation == "vertical":
        df_plot.plot(kind='bar', stacked=True, ax=ax, color=color_list, **kwargs)
        ax.set_xlabel(group_by.replace("_", " "))
        ax.set_ylabel(custom_val_label)
    else:
        raise ValueError("Orientation must be either 'vertical' or 'horizontal'.")
    
    ax.set_title(f"{value_col} by {group_by} and {stack_col}")
    
    # Annotate totals if requested.
    if annotate_totals:
        group_totals = df_pivot.sum(axis=1)

        if percentage:
            offset = 100 * 0.01  # Ensures text appears just after 100%
        else:
            offset = group_totals.max() * 0.05  # Dynamic offset for absolute values

        if orientation == "vertical":
            xticks = ax.get_xticks()
            for i, group in enumerate(df_plot.index):
                total = group_totals.loc[group]
                x = xticks[i] if i < len(xticks) else i
                
                # Ensure correct placement for percentage-based bars
                y_pos = 100 + offset if percentage else df_pivot.loc[group].sum() + offset
                
                ax.text(x, y_pos, f"{total:,.0f}kt CO2e", ha='center', va='bottom', fontsize=10, fontweight='bold', color="black")
        
        else:  # Horizontal orientation
            yticks = ax.get_yticks()
            for i, group in enumerate(df_plot.index):
                total = group_totals.loc[group]
                y = yticks[i] if i < len(yticks) else i

                # Ensure correct placement for percentage-based bars
                x_pos = 100 + offset if percentage else df_pivot.loc[group].sum() + offset
                
                ax.text(x_pos, y, f"{total:,.0f}kt CO2e", ha='left', va='center', fontsize=10, fontweight='bold', color="black")
    
    # Add the legend outside the plot area.
    if short_labels is not None:
        handles, _ = ax.get_legend_handles_labels()
        ax.legend(handles, short_labels, title=grouped_categ, loc='center left', bbox_to_anchor=(1, 0.5))
    else:
        ax.legend(title=grouped_categ, loc='center left', bbox_to_anchor=(1, 0.5))
    
    # Compute threshold per group to ensure consistent annotation
    threshold = df_pivot.abs().sum(axis=1).max() * 0.05  # Use absolute values for correct thresholding

    # Annotate each stacked segment with its category label.
    if orientation == "vertical":
        x_positions = np.arange(len(df_plot.index))
        
        for i, group in enumerate(df_plot.index):
            row = df_plot.loc[group]
            bottom_positive = 0  # Tracks stacking position for positive values
            bottom_negative = 0  # Tracks stacking position for negative values
            has_negative = any(row[col] < 0 for col in df_plot.columns)  # Check for negative values

            for col in df_plot.columns:
                val = row[col]
                abs_val = abs(val)  # Absolute value for threshold comparison
                
                # Only annotate if the absolute value is above the threshold
                if abs_val >= threshold:
                    x = x_positions[i]
                    label_text = str(col)

                    # Use short labels if available
                    if short_labels is not None and col in df_plot.columns:
                        label_text = short_labels[df_plot.columns.get_loc(col)]

                    # Adjust positioning for both cases:
                    if has_negative:  
                        # Mixed positive & negative case
                        if val > 0:
                            y = bottom_positive + val / 2  # Center inside positive stack
                            bottom_positive += val  # Move stacking up for positive side
                        else:
                            y = bottom_negative + val / 2  # Center inside negative stack
                            bottom_negative += val  # Move stacking down for negative side
                    else:  
                        # Fully positive case
                        y = bottom_positive + val / 2  # Center inside segment
                        bottom_positive += val  # Move stacking up
                    
                    # Ensure text is only added if there's enough space
                    if abs_val >= threshold * 1.5:  
                        ax.text(x, y, label_text, ha='center', va='center', fontsize=10, color='white', rotation=90)

    else:  # Horizontal orientation
        y_positions = np.arange(len(df_plot.index))
        
        for i, group in enumerate(df_plot.index):
            row = df_plot.loc[group]
            left_positive = 0  # Tracks stacking position for positive values
            left_negative = 0  # Tracks stacking position for negative values
            has_negative = any(row[col] < 0 for col in df_plot.columns)  # Check for negative values
            
            for col in df_plot.columns:
                val = row[col]
                abs_val = abs(val)  # Absolute value for threshold comparison
                
                # Only annotate if the absolute value is above the threshold
                if abs_val >= threshold:
                    y = y_positions[i]
                    label_text = str(col)

                    # Use short labels if available
                    if short_labels is not None and col in df_plot.columns:
                        label_text = short_labels[df_plot.columns.get_loc(col)]

                    # Adjust positioning for both cases:
                    if has_negative:  
                        # Mixed positive & negative case
                        if val > 0:
                            x = left_positive + val / 2  # Center inside positive stack
                            left_positive += val  # Move stacking right for positive side
                        else:
                            x = left_negative + val / 2  # Center inside negative stack
                            left_negative += val  # Move stacking left for negative side
                    else:  
                        # Fully positive case
                        x = left_positive + val / 2  # Center inside segment
                        left_positive += val  # Move stacking right
                    
                    # Ensure text is only added if there's enough space
                    if abs_val >= threshold * 1.5:  
                        ax.text(x, y, label_text, ha='center', va='center', fontsize=10, color='white')

    if percentage:
        ax.figure.subplots_adjust(right=0.8)  # Expands the right side so the legend is fully visible
    return ax

# Group by mineral
# def calculate_value_added(group):
#     # Find max stage (Nmax) for the mineral
#     Nmax = group["processing_stage"].max()
    
#     # Get rows for Nmax and N=1
#     stage_Nmax = group[group["processing_stage"] == Nmax]
#     stage_1 = group[group["processing_stage"] == 1]
    
#     if stage_Nmax.empty or stage_1.empty:
#         return None  # Skip if missing necessary data
    
#     # Extract required values
#     price_Nmax = stage_Nmax["price_usd_per_tonne"].values[0]
#     mass_Nmax = stage_Nmax["production_tonnes"].values[0]
#     cost_stage_1 = stage_1["production_cost_usd_per_tonne"].values[0]
#     mass_stage_1 = stage_1["production_tonnes"].values[0]

#     # Compute value added
#      # Calculate value added for each processing stage
#     group["value_added"] = (group["price_usd_per_tonne"] * group["production_tonnes"]) - (cost_stage_1 * mass_stage_1)
    
#     return group

def calculate_value_added(group):
    """
    Calculate value added at each processing stage based on earlier stages.
    """
    # Sort by processing stage in ascending order
    group = group.sort_values(by="processing_stage", ascending=True)

    # Ensure processing_stage is numeric
    group["processing_stage"] = group["processing_stage"].astype(float)

    # Initialize value_added as NaN to fill later
    group["value_added"] = 0.0

    # Iterate over each processing stage to compute cumulative value addition
    for i in range(1, len(group)):  
        # Previous stage
        prev_stage = group.iloc[i - 1]
        # Current stage
        current_stage = group.iloc[i]

        # Ensure valid calculations
        if prev_stage["production_tonnes"] > 0:
            group.at[current_stage.name, "value_added"] = (
                (current_stage["price_usd_per_tonne"] * current_stage["production_tonnes"]) -
                (prev_stage["production_cost_usd_per_tonne"] * prev_stage["production_tonnes"])
            )

    return group

def plot_supply_curve_bars(df, tons_column, unit_cost_column, ax=None, sort=True, 
                           colors=None, color_map=None, hide_small_labels=True, min_label_threshold=0.03, **kwargs):
    """
    Create a supply curve plot using bars instead of a step-line.
    
    Each bar represents the production for one row. The bar's width is the production 
    (from `tons_column`), and its height is the unit cost (from `unit_cost_column`).
    
    The function calculates the cumulative production and computes a left boundary 
    for each bar. The x‑axis shows cumulative production (in production units) while 
    the bars are annotated in their center with the label from the DataFrame’s index.
    
    Parameters:
      - df: Input DataFrame containing production and unit cost data.
      - tons_column: Name of the column representing production.
      - unit_cost_column: Name of the column representing unit cost.
      - ax: Optional matplotlib Axes to plot on. If None, a new figure and axes are created.
      - sort: If True, sort the DataFrame by the unit cost column in ascending order.
      - colors: Optional list of colors to use as fallback if an index is not found in `color_map`.
      - color_map: Optional dictionary mapping index values (e.g. country codes) to color strings.
      - hide_small_labels: If True, hide labels for very small values.
      - min_label_threshold: Minimum fraction of max height for a label to be shown.
      - **kwargs: Additional keyword arguments to pass to ax.bar().
    
    Returns:
      The matplotlib Axes object with the supply curve plotted.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    # Work on a copy to preserve the original DataFrame.
    df = df.copy()

    # Optionally sort the DataFrame by unit cost (so that cheaper production comes first)
    if sort:
        df = df.sort_values(by=unit_cost_column)

    # Compute cumulative production.
    df['cum_production'] = df[tons_column].cumsum()
    # The left boundary for each bar: cumulative production minus the current production.
    df['left'] = df['cum_production'] - df[tons_column]
    # Compute the center of each bar (used for annotation).
    df['center'] = df['left'] + df[tons_column] / 2
    # If no fallback colors are provided, use Matplotlib's default cycle.
    if colors is None:
        colors = plt.rcParams['axes.prop_cycle'].by_key().get('color', None)

    # Define a threshold for when text should be moved above the bar
    threshold = df[unit_cost_column].max() * 0.08  # 8% of max height

    # Store y-label positions to avoid overlap
    used_y_positions = []

    # Plot each bar using a loop.
    for i, (idx, row) in enumerate(df.iterrows()):
        # Use the color_map if provided.
        if color_map is not None and idx in color_map:
            bar_color = color_map[idx]
        else:
            # Fallback: if a list of colors is provided, cycle through it.
            if colors is not None:
                bar_color = colors[i % len(colors)]
            else:
                bar_color = None

        ax.bar(row['left'], row[unit_cost_column],
               width=row[tons_column],
               align='edge',
               color=bar_color,
               **kwargs)

        # Determine label position
        x_center = row['center']
        y_center = row[unit_cost_column] / 2
        label_color = "black"  # Ensure all labels are black

        # If the unit cost is too small, consider hiding the label
        if hide_small_labels and row[unit_cost_column] < min_label_threshold * df[unit_cost_column].max():
            continue  # Skip this label

        if row[unit_cost_column] < threshold:
            # Move label above the bar if too small
            y_text = row[unit_cost_column] + threshold * 0.6  # A bit above the bar
            line_y = row[unit_cost_column]  # Line from the top of the bar

            # Adjust y position dynamically to avoid overlap
            while any(abs(y_text - used_y) < threshold * 0.8 for used_y in used_y_positions):
                y_text += threshold * 0.6  # Move further up

            # Store this y position to prevent overlapping
            used_y_positions.append(y_text)

            # Draw a guiding line
            ax.plot([x_center, x_center], [line_y, y_text - threshold * 0.2], color="black", linestyle="dotted", linewidth=1)

        else:
            y_text = y_center  # Normal placement inside the bar

        # Slightly shift names randomly left/right to prevent squishing
        x_shift = np.random.uniform(-0.3, 0.3)  # Small random shift
        ax.text(x_center + x_shift, y_text, str(idx),
                ha='center', va='center',
                color=label_color, fontsize=10, fontweight='bold')

    # Set the x-axis label to indicate cumulative production.
    tons_column_clean = tons_column.replace("_", " ") 
    unit_cost_column_clean = "Energy, transport and production unit cost (USD/tonne)"
    ax.set_xlabel("Cumulative " + tons_column_clean, fontsize=14)
    ax.set_ylabel(unit_cost_column_clean, fontsize=14)
    ax.set_title("Supply Curve", fontsize=16, fontweight='bold')

    # Add a grid on the x-axis for better readability.
    ax.grid(True, axis='x', linestyle='--', alpha=0.5)
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)

    return ax

def plot_donut_chart(df, group_col, value_col='transport_total_tonsCO2eq',
                     ax=None, colors=None, title=None,
                     total_fmt="{:,.0f}", total_fontsize=16,
                     leader_offset=1.05, leader_line=True,
                     **kwargs):
    """
    Create a donut chart (pie chart with a hole) from a DataFrame by grouping the data 
    by `group_col` and summing the values in `value_col`. This function annotates each slice
    with its label. For slices with percentages 5% or higher, the label is placed at the wedge
    centroid with no leader line. For smaller slices, the label is shifted further away and 
    (if leader_line is True) an arrow is drawn from the wedge to the label.
    The total value (sum over groups) is shown in the center.
    
    Parameters:
      - df: The input DataFrame.
      - group_col: The column by which to group the data (e.g. "region").
      - value_col: The column containing the values to sum (default: "transport_total_tonsCO2eq").
      - ax: Optional matplotlib Axes. If None, a new figure and Axes are created.
      - colors: Optional list of colors for the slices.
      - title: Optional title for the chart.
      - total_fmt: Format string for the total value in the center.
      - total_fontsize: Font size for the central total text.
      - leader_offset: Base factor to offset the annotation from the wedge centroid.
      - leader_line: If True, draw an arrow (leader line) from the wedge to the label for small slices.
      - **kwargs: Additional keyword arguments passed to ax.pie().
      
    Returns:
      The matplotlib Axes with the donut chart.
    """
    if ax is None:
        fig, ax = plt.subplots()

    # Group the data by group_col and sum the value_col.
    grouped = df.groupby(group_col)[value_col].sum()
    total_value = grouped.sum()

    # Create the pie chart WITHOUT built-in labels; we'll annotate manually.
    wedges, texts, autotexts = ax.pie(
        grouped,
        labels=None,  # omit built-in labels
        autopct='%1.1f%%',  # display percentages inside
        startangle=90,
        colors=colors,
        wedgeprops={'width': 0.3, 'edgecolor': 'white'},
        **kwargs
    )
    ax.axis('equal')  # Ensure the donut is circular.

    # Add the total value at the center.
    ax.text(0, 0, total_fmt.format(total_value),
            horizontalalignment='center', verticalalignment='center',
            fontsize=total_fontsize, fontweight='bold')

    # Annotation: For each wedge, decide where to place the label.
    # We set a threshold of 5%: if the slice is >=5%, label at centroid with no arrow.
    threshold_pct = 5.0
    for i, w in enumerate(wedges):
        # Get the midpoint angle in degrees.
        theta1, theta2 = w.theta1, w.theta2
        ang = (theta1 + theta2) / 2.0  
        ang_rad = np.deg2rad(ang)

        # Compute the wedge's angular span.
        wedge_span = theta2 - theta1
        
        # Compute approximate centroid (using average of inner and outer radii).
        r_inner = w.r - w.width
        r_outer = w.r
        r = (r_inner + r_outer) / 2.0
        x = r * np.cos(ang_rad)
        y = r * np.sin(ang_rad)

        # Get the percentage from the autotext.
        pct_str = autotexts[i].get_text().replace('%','')
        try:
            pct = float(pct_str)
        except:
            pct = 0.0

        label = str(grouped.index[i])

        # If the percentage is >= threshold, annotate at the centroid.
        if pct >= threshold_pct:
            ax.text(x, y, label,
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontsize=12, fontweight='bold')
        else:
            # For small slices, add extra offset.
            extra_offset = 0.0
            # If wedge span is very small (say, below 20°), add extra offset.
            if wedge_span < 20.0:
                extra_offset = (20.0 - wedge_span) / 40.0  # Adjust this factor as needed

            local_offset = leader_offset + extra_offset
            x_text = local_offset * np.cos(ang_rad)
            y_text = local_offset * np.sin(ang_rad)
            ha = "left" if x_text > 0 else "right"
            if leader_line:
                ax.annotate(label,
                            xy=(x, y), xycoords='data',
                            xytext=(x_text, y_text), textcoords='data',
                            horizontalalignment=ha,
                            verticalalignment='center',
                            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.2"),
                            fontsize=12, fontweight='bold')
            else:
                ax.text(x_text, y_text, label,
                        horizontalalignment=ha,
                        verticalalignment='center',
                        fontsize=12, fontweight='bold')

    if title:
        ax.set_title(title, fontsize=16, fontweight='bold')

    return ax



def main(config):
    processed_data_path = config['paths']['data']
    output_data_path = config['paths']['results']
    figure_path = config['paths']['figures']

    figures = os.path.join(figure_path,"regional_figures")
    if os.path.exists(figures) is False:
        os.mkdir(figures)

    figures = os.path.join(figure_path,"regional_figures","bar_plots")
    if os.path.exists(figures) is False:
        os.mkdir(figures)

    multiply_factor = 1.0
    columns = [
                "production_tonnes", 
                "export_tonnes",
                "export_transport_cost_usd", 
                "tonsCO2eq",
                "revenue_usd",
                "production_cost_usd",
                "water_usage_m3"
            ]
    column_titles = [
                        "production volume ('000 tonnes)",
                        "export volume ('000 tonnes)",
                        "transport costs (million USD)",
                        "transport carbon emissions (000' tonsCO2eq)",
                        "revenue (million USD)",
                        "production costs (million USD)",
                        "water usage ('000 m3)"
                    ]
    multiply_factors = [1.0e-3,1.0e-3,1.0e-6,1e-3,1.0e-6,1.0e-6,1.0e-3]
    column_titles = [f"Annual {c}" for c in column_titles]
    scenarios_descriptions = [
                                {
                                    "scenario_type":"country_unconstrained",
                                    "scenario_name":"MN",
                                    "scenarios":[
                                                    "2022_baseline",
                                                    "2030_mid_min_threshold_metal_tons",
                                                    "2040_mid_min_threshold_metal_tons"
                                                ],
                                    "scenario_labels":[
                                                    "2022 SQ",
                                                    "2030 MN",
                                                    "2040 MN"
                                                ]
                                },
                                {
                                    "scenario_type":"region_unconstrained",
                                    "scenario_name":"MR",
                                    "scenarios":[
                                                    "2022_baseline",
                                                    "2030_mid_max_threshold_metal_tons",
                                                    "2040_mid_max_threshold_metal_tons"
                                                ],
                                    "scenario_labels":[
                                                    "2022 SQ",
                                                    "2030 MR",
                                                    "2040 MR"
                                                ]
                                },
                    ]
    """
    Aggregated data in Excel
    """
    make_data = False
    if make_data is True:
        # Define the input file and sheets to read
        input_file = os.path.join(output_data_path,
                                "result_summaries",
                                "combined_energy_transport_totals_by_stage.xlsx")
        sheets_to_read = ["country_unconstrained", "country_constrained", 
                        "region_unconstrained", "region_constrained"]  # Modify with the required sheet names

        # Read and process each sheet
        df_list = pd.DataFrame()
        for sheet in sheets_to_read:
            df = pd.read_excel(input_file, sheet_name=sheet, index_col=[0,1,2,3,4]).reset_index()
            df['constraint'] = sheet
            df_list = pd.concat([df_list, df], ignore_index=True)
        
        unit_costs = [
                        "export_transport_cost_usd_per_tonne",
                        "import_transport_cost_usd_per_tonne",
                        "production_cost_usd_per_tonne",
                        "energy_opex_per_tonne",
                        "energy_investment_usd_per_tonne"
                    ]
        
        # Add energy when ready
        df_list['production_transport_energy_unit_cost_usd_per_tonne'] = [x+y+z+zz+zy for x,y,z,zz,zy in zip(df_list[unit_costs[0]],df_list[unit_costs[1]],
                                                                                                            df_list[unit_costs[2]], df_list[unit_costs[3]],
                                                                                                            df_list[unit_costs[4]])]
        df_list['production_transport_energy_unit_cost_usd_per_tonne'] = df_list['production_transport_energy_unit_cost_usd_per_tonne'].fillna(0)

        costs = [
                    "export_transport_cost_usd",
                    "import_transport_cost_usd",
                    "production_cost_usd",
                    "energy_opex",
                    "energy_investment_usd"
                ]

        # Compute total cost
        df_list["all_cost_usd"] = [x+y+z+zz+zy for x,y,z,zz,zy in zip(df_list[costs[0]],df_list[costs[1]],
                                                                                                            df_list[costs[2]], df_list[costs[3]],
                                                                                                            df_list[costs[4]])]

        # Save to a new Excel file
        output_file = os.path.join(output_data_path, "all_data.xlsx")  
        df_list.to_excel(output_file, index=False)

        uc_list = df_list[["scenario", "reference_mineral", "iso3", 'processing_type',
                             'processing_stage', "constraint"] + unit_costs + ["production_transport_energy_unit_cost_usd_per_tonne"]]
        uc_list = uc_list.drop_duplicates(subset=[ "scenario", "reference_mineral", "iso3", "processing_type",
                                        "processing_stage", "constraint", "production_transport_energy_unit_cost_usd_per_tonne"
                                            ])

        # Define the grouping columns
        group_cols = ["constraint", "scenario", "reference_mineral", "iso3", "processing_type", "processing_stage"]

        # Define aggregation rules: 
        # - Use 'first' if a unit cost is expected to be constant within a group
        # - Use 'sum' for transport costs (since they can vary by country)

        agg_dict = {col: 'sum' for col in ["export_transport_cost_usd_per_tonne","import_transport_cost_usd_per_tonne",
                                           "energy_opex_per_tonne", "energy_investment_usd_per_tonne"]}  # If cost is different per country
        agg_dict["production_cost_usd_per_tonne"] = 'first'  

        # Aggregate data
        aggregated_df = uc_list.groupby(group_cols, as_index=False).agg(agg_dict)

        # Compute the total unit cost
        aggregated_df["production_transport_energy_unit_cost_usd_per_tonne"] = aggregated_df[unit_costs].sum(axis=1)

        uc_output_file = os.path.join(output_data_path, "unit_costs.xlsx")  
        aggregated_df.to_excel(uc_output_file, index=False)

            
    make_plot = False
    if make_plot is True:
        results_file = os.path.join(output_data_path,
                                "result_summaries",
                                "transport_totals_by_stage.xlsx")
        for sd in scenarios_descriptions:
            sc_t = sd["scenario_type"]
            sc_n = sd["scenario_name"]
            scenarios = sd["scenarios"]
            sc_l = sd["scenario_labels"]
            results_folder = os.path.join(figures,f"{sc_n}_{sc_t}")
            if os.path.exists(results_folder) == False:
                os.mkdir(results_folder)
            data_df = pd.read_excel(
                            results_file,
                            sheet_name=sc_t,
                            index_col=[0,1,2,3,4])
            data_df = data_df.reset_index()
            if sc_t != "country_unconstrained":
                baseline_df = pd.read_excel(
                                results_file,
                                sheet_name="country_unconstrained",
                                index_col=[0,1,2,3,4])
                baseline_df = baseline_df.reset_index()
                data_df = pd.concat(
                                [
                                    baseline_df[baseline_df["year"] == 2022],
                                    data_df
                                ],axis=0,ignore_index=True
                                )

            reference_minerals = list(set(data_df["reference_mineral"].values.tolist()))
            all_properties = mineral_properties()
            for cdx,(col,col_t,m_t) in enumerate(zip(columns,column_titles,multiply_factors)):
                for rf in reference_minerals:
                    df = data_df[(data_df["reference_mineral"] == rf) & (data_df["processing_stage"] > 0)]
                    stages = sorted(list(set(df["processing_stage"].values.tolist())))
                    stage_colors = all_properties[rf]["stage_colors"][:len(stages)]
                    countries = sorted(list(set(df["iso3"].values.tolist())))
                    fig, ax = plt.subplots(1,1,figsize=(18,9),dpi=500)
                    f = 0
                    dfall = []
                    for sc in scenarios:
                        m_df = pd.DataFrame(countries,columns=["iso3"])
                        # xvals = 2*(np.arange(0,len(countries)) + f)
                        for st in stages:
                            s_df = df[(df["processing_stage"] == st) & (df["scenario"] == sc)]
                            s_df[col] = m_t*s_df[col]
                            s_df.rename(columns={col:f"Stage {st}"},inplace=True)
                            m_df = pd.merge(m_df,s_df[["iso3",f"Stage {st}"]],how="left",on=["iso3"]).fillna(0)
                        dfall.append(m_df.set_index(["iso3"]))
                    ax = plot_clustered_stacked(
                                                fig,ax,dfall,stage_colors,
                                                labels=sc_l,
                                                ylabel=col_t, 
                                                title=f"{rf.title()} {sc_n} scenario")
                    plt.grid()
                    plt.tight_layout()
                    save_fig(os.path.join(results_folder,
                                f"{rf}_{col}_{sc_t}.png"))
                    plt.close()


    """CO2 emissions and Value Added 
    """
    make_plot = False
    if make_plot is True:
        results_file = os.path.join(output_data_path,
                                "result_summaries",
                                # "totals_by_country.xlsx"
                                "combined_transport_totals_by_stage.xlsx"
                                )
        df = pd.read_excel(
                            results_file,
                            index_col=[0,1,2])
        df = df.reset_index() 
        countries = sorted(list(set(df["iso3"].values.tolist())))
        scenarios_descriptions = [s for s in scenarios_descriptions if s["scenario_type"] == "country_unconstrained"]
        plot_type = [
                        {
                            "type":"carbon_emissions",
                            "columns": ["transport_total_tonsCO2eq",
                                        # "energy_tonsCO2eq"
                                        ],
                            "columns_labels":["Annual transport emissions",
                                                # "Annual energy emissions"
                                                ],
                            "columns_colors":["#bdbdbd",
                                            # "#969696"
                                            ],
                            "ylabel":"Annual carbon emissions ('000 tonsCO2eq)",
                            "factor":1.0e-3
                        },
                        # {
                        #     "type":"value_added",
                        #     "columns": ["export_value_added_usd"],
                        #     "columns_labels":["Annual value added"],
                        #     "columns_colors":["#66c2a4"],
                        #     "ylabel":"Annual value added (USD millions)",
                        #     "factor":1.0e-6
                        # }
                        ] 
        for sc in scenarios_descriptions:
            sc_t = sc["scenario_type"]
            sc_n = sc["scenario_name"]
            for pt in plot_type:
                cols = pt["columns"]
                ptype = pt["type"]
                fig, ax = plt.subplots(1,1,figsize=(18,9),dpi=500)
                dfall = []
                for st in sc["scenarios"]:
                    m_df = pd.DataFrame(countries,columns=["iso3"])
                    s_df = df[df["scenario"] == st]
                    m_df = pd.merge(m_df,s_df[["iso3"] + cols],how="left",on=["iso3"]).fillna(0)
                    m_df[cols] = pt["factor"]*m_df[cols]
                    m_df.rename(columns=dict(list(zip(cols,pt["columns_labels"]))),inplace=True)
                    dfall.append(m_df.set_index(["iso3"]))
                
                ax = plot_clustered_stacked(
                                            fig,ax,dfall,pt["columns_colors"],
                                            labels=sc["scenario_labels"],
                                            ylabel=pt["ylabel"], 
                                            title=f"{sc_n} scenario")
                plt.grid()
                plt.tight_layout()
                save_fig(os.path.join(figures,
                            f"{ptype}_{sc_t}.png"))
                plt.close()
    
    """CO2 emissions - NEW - transport and energy
    """
    make_plot = False
    if make_plot is True:
        multiply_factor = 1.0e-3
        constraints = ["country_unconstrained", "region_unconstrained",
                       "country_constrained", "region_constrained"]
        scenarios = [
            "2030_mid_min_threshold_metal_tons",
            "2030_mid_max_threshold_metal_tons",
            "2040_mid_min_threshold_metal_tons",
            "2040_mid_max_threshold_metal_tons"
        ]
        reference_minerals = ["cobalt","copper","graphite","lithium","manganese","nickel"]
        reference_minerals_short = ["Co","Cu", "Gr", "Li","Mn","Ni"]
        reference_mineral_colors = ["#fdae61","#f46d43","#66c2a5","#c2a5cf","#fee08b","#3288bd"]
        reference_mineral_colormap = dict(zip(reference_minerals, reference_mineral_colors))

        reference_mineral_namemap = dict(zip(reference_minerals, reference_minerals_short))
        reference_mineral_colormapshort = dict(zip(reference_minerals_short, reference_mineral_colors))

        results_file = os.path.join(output_data_path,
                                "result_summaries",
                                "combined_energy_transport_totals_by_stage.xlsx"
                                )
        
        for cs in constraints:
            print(f"Constraint: {cs}")
            # Read the full data for this constraint.
            df = pd.read_excel(results_file, sheet_name=cs, index_col=[0,1,2,3,4]).reset_index()
            # unit conversion
            df['transport_total_tonsCO2eq'] = multiply_factor*df['transport_total_tonsCO2eq']
            df['energy_tonsCO2eq'] = multiply_factor*df["energy_tonsCO2eq"]
            df['energy_transport_tonsCO2eq'] = [x+y for x,y in zip(df['transport_total_tonsCO2eq'],df['energy_tonsCO2eq'])]

            # Get unique countries
            countries = sorted(df['iso3'].unique())
            num_countries = len(countries)
            # Get the colormap and sample distinct colors
            colormap = plt.colormaps['tab20']  # Get the 'tab20' colormap
            colors = colormap(np.linspace(0, 1, num_countries))  # Sample 'num_countries' colors

            # Convert colors to hex format
            country_colors = [plt.matplotlib.colors.rgb2hex(color) for color in colors]
            country_color_map = dict(zip(countries, country_colors))

            data_2030 = pd.DataFrame()
            data_2040 = pd.DataFrame()
            
            for sc in scenarios:
                if "2030" in sc:
                    year = '2030'
                if "2040" in sc:
                    year = '2040'
                df_sc = df[df['scenario'] == sc]
                if df_sc.empty:
                    print(f"No data for {cs} and {sc}. Skipping...")
                    continue

                fig, axs = plt.subplots(1, 2, figsize=(14, 7))
                plot_donut_chart(df_sc, group_col='iso3', 
                                value_col='energy_transport_tonsCO2eq', 
                                ax=axs[0],
                                colors = country_colors,
                                title="Emissions by country (kilotonne CO2eq)",
                                leader_offset=1.05,
                                leader_line = True
                                )

                plot_donut_chart(df_sc, group_col='reference_mineral', 
                                value_col='energy_transport_tonsCO2eq', 
                                ax=axs[1],
                                colors= reference_mineral_colors,
                                title="Emissions by mineral (kilotonne CO2eq)",
                                leader_offset=1.05,
                                leader_line = True
                                )

                # Save the figure with an appropriate filename.
                filename = f"{cs}_{year}_CO2_donut.png"
                # Ensure you have a defined 'figures' folder or path.
                plt.tight_layout()
                plt.savefig(os.path.join(figures, filename))
                plt.close()

                if "2030" in sc:
                    year = '2030'
                    df_slice2030 = df[df['scenario'] == sc]
                    data_2030 = pd.concat([data_2030, df_slice2030], ignore_index = True)
                if "2040" in sc:
                    year = '2040'
                    df_slice2040 = df[df['scenario'] == sc]
                    data_2040 = pd.concat([data_2040, df_slice2040], ignore_index = True)

            # Stacked bars
            cs_clean = cs.replace("_", " ").title()
            data_2030['reference_mineral_short'] = data_2030['reference_mineral'].map(reference_mineral_namemap)
            data_2040['reference_mineral_short'] = data_2040['reference_mineral'].map(reference_mineral_namemap)

            data_2030.rename(columns = {'reference_mineral_short':'Mineral'}, inplace = True)
            data_2040.rename(columns = {'reference_mineral_short':'Mineral'}, inplace = True)

            # **Sort bars by total emissions (lowest first)**
            data_2030_sorted = data_2030.groupby('Mineral')['energy_transport_tonsCO2eq'].sum().sort_values(ascending=True).index
            data_2040_sorted = data_2040.groupby('Mineral')['energy_transport_tonsCO2eq'].sum().sort_values(ascending=True).index

            # Reorder data based on sorted order
            data_2030 = data_2030.set_index('Mineral').loc[data_2030_sorted].reset_index()
            data_2040 = data_2040.set_index('Mineral').loc[data_2040_sorted].reset_index()

            all_data = pd.concat([data_2030, data_2040], ignore_index=True)
            if all_data.empty:
                print(f"No data for {cs}. Skipping...")
                continue
            all_data = all_data[['scenario','Mineral', 'iso3', 'energy_transport_tonsCO2eq']]
            all_data.rename(columns = {'iso3':'Country', 'scenario':'Scenario'}, inplace = True)

            fig, axs = plt.subplots(2,1, figsize=(7,9), dpi=500, sharex=True) #horizontal
            axs[0] = plot_stacked_bar(data_2030, group_by='Mineral', stack_col='iso3', value_col='energy_transport_tonsCO2eq',
                                orientation="horizontal", ax=axs[0], colors=country_color_map, short_val_label = 'Emissions by mineral', 
                                units = 'kilotonne CO2eq', grouped_categ = 'Country')
            axs[0].set_title(f"2030 - {cs_clean}")

            axs[1] =plot_stacked_bar(data_2040, group_by='Mineral', stack_col='iso3', value_col='energy_transport_tonsCO2eq',
                                orientation="horizontal", ax=axs[1], colors=country_color_map, short_val_label = 'Emissions by mineral', 
                                units = 'kilotonne CO2eq', grouped_categ = 'Country')
            axs[1].set_title(f"2040 - {cs_clean}")
            
            axs[0].set_xlabel(f"Emissions by mineral (kilotonne CO2eq)")
            axs[0].xaxis.label.set_visible(True)
            axs[0].tick_params(axis="x", which="both", labelsize=12, labelbottom=True)

            for ax in axs:
                ax.grid(which="major", axis="x", linestyle="-", zorder=0)
                ax.grid(False, axis="y")

            plt.tight_layout()
            # Save figure
            filename = f"{cs}_CO2_mineral_comparisons_2030_2040H.png"
            save_fig(os.path.join(figures, filename))
            plt.close()
            # Saving data
            all_data.to_csv(os.path.join(output_data_path,
                                "result_summaries", f"{cs}_CO2_mineral_comparisons_2030_2040H.csv"), index=False)

            # Stacked percentage bars
            fig, axs = plt.subplots(2,1, figsize=(8.2,9), dpi=500, sharex=True) #horizontal
            axs[0] = plot_stacked_bar(data_2030, group_by='Mineral', stack_col='iso3', value_col='energy_transport_tonsCO2eq',
                                orientation="horizontal", ax=axs[0], colors=country_color_map, short_val_label = 'Share of CO2eq emissions', 
                                units = "%", grouped_categ = 'Country', annotate_totals=True, percentage = True)
            axs[0].set_title(f"2030 - {cs_clean}")

            axs[1] =plot_stacked_bar(data_2040, group_by='Mineral', stack_col='iso3', value_col='energy_transport_tonsCO2eq',
                                orientation="horizontal", ax=axs[1], colors=country_color_map, short_val_label = 'Share of CO2eq emissions', 
                                units = "%", grouped_categ = 'Country', annotate_totals=True, percentage = True)
            axs[1].set_title(f"2040 - {cs_clean}")

            # axs[0].set_xlabel(f"Share of CO2eq emissions (%)")
            # axs[0].xaxis.label.set_visible(True)
            axs[0].tick_params(axis="x", which="both", labelsize=12, labelbottom=True)
                
            # Adjust grid and axis settings
            legend_objects = []  # Store legend objects for better positioning
            for ax in axs:
                ax.grid(which="major", axis="x", linestyle="-", zorder=0)
                ax.grid(False, axis="y")

                # Move legend further to the right
                legend = ax.legend(loc='upper left', bbox_to_anchor=(1.2, 1))  # Increased x-offset
                legend.set_frame_on(False)  # Optional: Remove box around legend
                legend_objects.append(legend)  # Store legend for saving fix
            # Remove legend from axs[1]
            axs[1].get_legend().remove()
            # Increase right margin to ensure legend doesn't get cut off
            plt.subplots_adjust(right=0.75)  # Adjust figure to accommodate legend

            # Save figure
            filename = f"{cs}_CO2_mineral_comparisons_shares_2030_2040H.png"
            # save_fig(os.path.join(figures, filename), bbox_inches="tight", bbox_extra_artists=legend_objects)
            plt.savefig(os.path.join(figures, filename), bbox_inches="tight", dpi=500)
            plt.close()

    """Delta tonnage unconstrained and constrained: metal content
    """
    make_plot = False # this no longer works with the plot function definition
    if make_plot is True:
        multiply_factor = 1.0e-3
        results_file = os.path.join(output_data_path,
                                "result_summaries",
                                "transport_totals_by_stage.xlsx")
        constraints = ["country_unconstrained","country_constrained"]
        reference_minerals = ["cobalt","copper","graphite","lithium","manganese","nickel"]
        reference_minerals_short = ["Co","Cu", "Gr", "Li","Mn","Ni"]
        reference_mineral_colors = ["#fdae61","#f46d43","#66c2a5","#c2a5cf","#fee08b","#3288bd"]
        scenarios = ["2030_mid_min_threshold_metal_tons","2040_mid_min_threshold_metal_tons"]
        tons_column = "production_tonnes"
        index_cols = ["reference_mineral","iso3"]
        dfs = []
        for scenario in scenarios:
            for cs in constraints:
                df = pd.read_excel(
                                    results_file,
                                    sheet_name=cs,
                                    index_col=[0,1,2,3])
                df = df.reset_index() 
                m_df = pd.DataFrame(sorted(list(set(df["iso3"].values.tolist()))),columns=["iso3"])
                df = df[(df["scenario"] == scenario) & (df["processing_stage"] == 0)]
                df = df.groupby(index_cols)[tons_column].sum().reset_index()
                df[tons_column] = multiply_factor*df[tons_column]
                df = (df.set_index(["iso3"]).pivot(
                                        columns="reference_mineral"
                                        )[tons_column].reset_index().rename_axis(None, axis=1)).fillna(0)
                df_cols = df.columns.values.tolist()
                df.columns = ["iso3"] + [c.title() for c in df_cols[1:]] 
                m_df = pd.merge(m_df,df,how="left",on=["iso3"]).fillna(0)
                print (m_df)
                dfs.append(m_df.set_index("iso3"))
        delta_df = []
        delta_df.append(dfs[1].subtract(dfs[0], fill_value=0))
        delta_df.append(dfs[3].subtract(dfs[2], fill_value=0))
        fig, ax = plt.subplots(1,1,figsize=(18,9),dpi=500)
        ax = plot_clustered_stacked(
                                fig,ax,dfs,reference_mineral_colors,
                                labels=["National 2030 unconstrained",
                                        "National 2030 constrained",
                                        "National 2040 unconstrained",
                                        "National 2040 constrained"],
                                ylabel="Annual metal content produced (000' tonnes)", 
                                title=f"Mid National scenarios - metal content production")
        plt.grid()
        plt.tight_layout()
        save_fig(os.path.join(figures,
                    f"MN_scenarios_unconstrained_constrained_side_by_side.png"))
        plt.close()

        fig, ax = plt.subplots(1,1,figsize=(18,9),dpi=500)
        ax = plot_clustered_stacked(
                                fig,ax,delta_df,reference_mineral_colors,
                                labels=["2030 difference", "2040 difference"],
                                ylabel="Difference in annual metal content produced (000' tonnes)", 
                                title=f"Mid National 2030 and 2040 scenarios - Constrained minus Unconstrained production of metal content")
        plt.grid()
        plt.tight_layout()
        save_fig(os.path.join(figures,
                    f"MN_scenarios_unconstrained_constrained_delta_production.png"))
        plt.close()

    """COUNTRY Delta tonnage unconstrained and constrained for 2030 and 2040 separately: metal content
    """ # doesn't work w plotting function
    make_plot = False # these aren't very useful because the differences between minerals are too big and Mn dominates
    if make_plot is True:
        multiply_factor = 1.0e-3
        results_file = os.path.join(output_data_path,
                                "result_summaries",
                                "transport_totals_by_stage.xlsx")
        constraints = ["country_unconstrained","country_constrained"]
        reference_minerals = ["cobalt","copper","graphite","lithium","manganese","nickel"]
        reference_minerals_short = ["Co","Cu", "Gr", "Li","Mn","Ni"]
        reference_mineral_colors = ["#fdae61","#f46d43","#66c2a5","#c2a5cf","#fee08b","#3288bd"]
        scenarios = ["2030_mid_min_threshold_metal_tons","2040_mid_min_threshold_metal_tons"]
        tons_column = "production_tonnes"
        index_cols = ["reference_mineral","iso3"]
        dfs = []
        for scenario in scenarios:
            for cs in constraints:
                df = pd.read_excel(
                                    results_file,
                                    sheet_name=cs,
                                    index_col=[0,1,2,3])
                df = df.reset_index() 
                m_df = pd.DataFrame(sorted(list(set(df["iso3"].values.tolist()))),columns=["iso3"])
                df = df[(df["scenario"] == scenario) & (df["processing_stage"] == 0)]
                df = df.groupby(index_cols)[tons_column].sum().reset_index()
                df[tons_column] = multiply_factor*df[tons_column]
                df = (df.set_index(["iso3"]).pivot(
                                        columns="reference_mineral"
                                        )[tons_column].reset_index().rename_axis(None, axis=1)).fillna(0)
                df_cols = df.columns.values.tolist()
                df.columns = ["iso3"] + [c.title() for c in df_cols[1:]] 
                m_df = pd.merge(m_df,df,how="left",on=["iso3"]).fillna(0)
                print (m_df)
                dfs.append(m_df.set_index("iso3"))
        delta_df = []
        delta_df.append(dfs[1].subtract(dfs[0], fill_value=0))
        delta_df.append(dfs[3].subtract(dfs[2], fill_value=0))
        # fig, ax = plt.subplots(1,1,figsize=(18,9),dpi=500)
        # ax = plot_clustered_stacked(
        #                         fig,ax,dfs,reference_mineral_colors,
        #                         labels=["National 2030 unconstrained",
        #                                 "National 2030 constrained",
        #                                 "National 2040 unconstrained",
        #                                 "National 2040 constrained"],
        #                         ylabel="Annual metal content produced (000' tonnes)", 
        #                         title=f"Mid National scenarios - metal content production")

        # plt.grid()
        # plt.tight_layout()
        # save_fig(os.path.join(figures,
        #             f"MN_scenarios_unconstrained_constrained_side_by_side.png"))
        # plt.close()

        fig, axs = plt.subplots(1, 2, figsize=(19, 9), dpi=500)  # Create a 1-row, 2-column layout
        delta_df4r = pd.DataFrame(delta_df[0]) # otherwise they are a list
        delta_df4r2 = pd.DataFrame(delta_df[1])
        

        # Plot the 2030 difference in the first subplot (axs[0])
        axs[0] = plot_clustered_stacked(
            fig, axs[0], delta_df4r, reference_mineral_colors,  
            labels=["2030 difference"],
            ylabel="Difference in annual metal content produced (000' tonnes)", 
            title="Mid National 2030 - Constrained \n minus unconstrained (metal content) (Country)",
            df_style = 'single_yr',
            stacked = False,
            scenario_labels = False,
        )

        # Plot the 2040 difference in the second subplot (axs[1])
        axs[1] = plot_clustered_stacked(
            fig, axs[1], delta_df4r2, reference_mineral_colors,  
            labels=["2040 difference"],
            ylabel="Difference in annual metal content produced (000' tonnes)", 
            title="Mid National 2040 - Constrained \n minus unconstrained (metal content) (Country)",
            df_style = 'single_yr',
            stacked = False,
            scenario_labels = False,
        )
    
        # Disable gridlines on the x-axis
        axs[0].grid(False, axis='x')
        axs[1].grid(False, axis='x')
        # Enable gridlines on the y-axis
        axs[0].grid(which='major', axis='y', linestyle='-', zorder=0)
        axs[1].grid(which='major', axis='y', linestyle='-', zorder=0)
        
        # for ax in axs:
        #     ax.axvline(x=0, color="black", linewidth=1, zorder=3)

        # plt.grid()
        plt.tight_layout()

        # Save the figure
        save_fig(os.path.join(figures, "MN_scenarios_unconstrained_constrained_delta_production_year_separated.png"))
        plt.close()

    """
    REGION Delta tonnage unconstrained and constrained for 2030 and 2040 separately: metal content
    """
    make_plot = False # these aren't very useful because the differences between minerals are too big and Mn dominates
    if make_plot is True:
        multiply_factor = 1.0e-3
        results_file = os.path.join(output_data_path,
                                "result_summaries",
                                "transport_totals_by_stage.xlsx")
        constraints = ["region_unconstrained","region_constrained"]

        reference_minerals = ["cobalt","copper","graphite","lithium","manganese","nickel"]
        reference_minerals_short = ["Co","Cu", "Gr", "Li","Mn","Ni"]
        reference_mineral_colors = ["#fdae61","#f46d43","#66c2a5","#c2a5cf","#fee08b","#3288bd"]

        scenarios = ["2030_mid_max_threshold_metal_tons","2040_mid_max_threshold_metal_tons"]
        tons_column = "production_tonnes"
        index_cols = ["reference_mineral","iso3"]
        dfs = []
        for scenario in scenarios:
            for cs in constraints:
                df = pd.read_excel(
                                    results_file,
                                    sheet_name=cs,
                                    index_col=[0,1,2,3])
                df = df.reset_index() 
                m_df = pd.DataFrame(sorted(list(set(df["iso3"].values.tolist()))),columns=["iso3"])
                df = df[(df["scenario"] == scenario) & (df["processing_stage"] == 0)] 
                df = df.groupby(index_cols)[tons_column].sum().reset_index()
                df[tons_column] = multiply_factor*df[tons_column]
                df = (df.set_index(["iso3"]).pivot(
                                        columns="reference_mineral"
                                        )[tons_column].reset_index().rename_axis(None, axis=1)).fillna(0)
                df_cols = df.columns.values.tolist()
                df.columns = ["iso3"] + [c.title() for c in df_cols[1:]] 
                m_df = pd.merge(m_df,df,how="left",on=["iso3"]).fillna(0)
                print (m_df)
                dfs.append(m_df.set_index("iso3"))
        delta_df = []
        delta_df.append(dfs[1].subtract(dfs[0], fill_value=0))
        delta_df.append(dfs[3].subtract(dfs[2], fill_value=0))
        # fig, ax = plt.subplots(1,1,figsize=(18,9),dpi=500)
        # ax = plot_clustered_stacked(
        #                         fig,ax,dfs,reference_mineral_colors,
        #                         labels=["Regional 2030 unconstrained",
        #                                 "Regional 2030 constrained",
        #                                 "Regional 2040 unconstrained",
        #                                 "Regional 2040 constrained"],
        #                         ylabel="Annual metal content produced (000' tonnes)", 
        #                         title=f"Mid Regional scenarios - metal content production",
        #                         df_style = "multiple",
        #                         orientation = "verical")

        # plt.grid()
        # plt.tight_layout()
        # save_fig(os.path.join(figures,
        #             f"MR_scenarios_unconstrained_constrained_side_by_side.png"))
        # plt.close()

        fig, axs = plt.subplots(1, 2, figsize=(19, 9), dpi=500)  
        delta_df_2030 = pd.DataFrame(delta_df[0]) # otherwise they are a list
        delta_df_2040 = pd.DataFrame(delta_df[1])
        

        # Plot the 2030 difference in the first subplot (axs[0])
        axs[0] = plot_clustered_stacked(
            fig, axs[0], [delta_df_2030], reference_mineral_colors,  
            labels=["2030 difference"],
            ylabel="Difference in annual metal content produced (000' tonnes)", 
            title="Mid Regional 2030 - Constrained \n minus unconstrained (metal content) (Region)",
            width=0.7,   
            df_style = 'multiple',
            orientation = "vertical",
            scenario_labels = False,
            stacked=False,
        )

        # Plot the 2040 difference in the second subplot (axs[1])
        axs[1] = plot_clustered_stacked(
            fig, axs[1], [delta_df_2040], reference_mineral_colors,  
            labels=["2040 difference"],
            ylabel="Difference in annual metal content produced (000' tonnes) (Region)", 
            title="Mid Regional 2040 - Constrained \n minus unconstrained (metal content) (Region)",
            width=0.7,   
            df_style = 'multiple',
            orientation = "vertical",
            scenario_labels = False,
            stacked=False,
        )
    
        # Disable gridlines on the x-axis
        axs[0].grid(False, axis='x')
        axs[1].grid(False, axis='x')
        # Enable gridlines on the y-axis
        axs[0].grid(which='major', axis='y', linestyle='-', zorder=0)
        axs[1].grid(which='major', axis='y', linestyle='-', zorder=0)

        # for ax in axs:
        #     ax.axvline(x=0, color="black", linewidth=1, zorder=3)

        # plt.grid()
        plt.tight_layout()

        # Save the figure
        save_fig(os.path.join(figures, "MR_scenarios_unconstrained_constrained_delta_production_metalcont_year_separated.png"))
        plt.close()
    """
    REGION Delta tonnage unconstrained and constrained for 2030 and 2040 separately in subplots: metal content 
    """
    make_plot = False
    if make_plot:
        multiply_factor = 1.0e-3
        results_file = os.path.join(output_data_path, "result_summaries", "transport_totals_by_stage.xlsx")
        constraints = ["region_unconstrained", "region_constrained"]
        reference_minerals = ["cobalt","copper","graphite","lithium","manganese","nickel"]
        reference_minerals_short = ["Co","Cu", "Gr", "Li","Mn","Ni"]
        reference_mineral_colors = ["#fdae61","#f46d43","#66c2a5","#c2a5cf","#fee08b","#3288bd"]
        scenarios = [
            "2030_mid_max_threshold_metal_tons",
            "2040_mid_max_threshold_metal_tons"
        ]
        tons_column = "production_tonnes"
        index_cols = ["reference_mineral", "iso3"]
        all_properties = mineral_properties()

        all_results = []
        for rf in reference_minerals:
            dfs = []  # To store DataFrames for each scenario and constraint
            stages_all = all_properties[rf]["stage_labels"]
            stage_colors_all = all_properties[rf]["stage_label_colors"]
            
            for scenario in scenarios:
                # Keeping only the relevant stages for each year
                if '2030' in scenario:
                    # stages = [item for item in stages_all if 'Precursor related product' not in item]
                    # stage_colors_2030 = [item for item in stage_colors_all if '#7f0000' not in item]
                    stages = stages_all
                    stage_colors_2030 = stage_colors_all
                if '2040' in scenario:
                    # stages = [item for item in stages_all if 'Early refining' not in item]
                    # stage_colors_2040 = [item for item in stage_colors_all if '#fb6a4a' not in item]
                    stages = stages_all
                    stage_colors_2040 = stage_colors_all

                for cs in constraints:
                    df = pd.read_excel(results_file, sheet_name=cs, index_col=[0, 1, 2, 3, 4])
                    df = df.reset_index()

                    # Filter by scenario, processing_stage, and mineral
                    filtered_df = df[(df["scenario"] == scenario) & 
                                    (df["processing_stage"] == 0) & 
                                    (df["reference_mineral"] == rf)]

                    if filtered_df.empty:
                        print(f"No data available for scenario '{scenario}' and constraint '{cs}'. Skipping...")
                        continue

                    m_df = pd.DataFrame(sorted(filtered_df["iso3"].unique()), columns=["iso3"])

                    # Aggregate and process stage-wise data
                    for st in stages:
                        s_df = filtered_df[filtered_df["processing_type"] == st]
                        if not s_df.empty:
                            s_df[tons_column] = multiply_factor * s_df[tons_column]
                            s_df = s_df.groupby(["iso3", "processing_type"])[tons_column].sum().reset_index()
                            s_df.rename(columns={tons_column: f"{st}"}, inplace=True)
                            m_df = pd.merge(m_df, s_df[["iso3", f"{st}"]], how="left", on="iso3").fillna(0)
                        else:
                            m_df[f"{st}"] = 0

                    dfs.append(m_df.set_index("iso3"))
            # Add metadata columns
                    m_df["Constraint"] = cs
                    m_df["Scenario"] = scenario
                    m_df["Mineral"] = rf
                
                    all_results.append(m_df.copy())

            # Compute the differences for 2030 and 2040
            delta_df = []
            if len(dfs) >= 4:  # Ensure we have all the required DataFrames
                delta_df.append(dfs[1].subtract(dfs[0], fill_value=0))  # 2030 constrained minus unconstrained
                delta_df.append(dfs[3].subtract(dfs[2], fill_value=0))  # 2040 constrained minus unconstrained

            # Plot results for 2030 and 2040
            fig, axs = plt.subplots(1, 2, figsize=(19, 9), dpi=500, sharey=True)

            if not delta_df:
                print(f"No data differences to plot for {rf}. Skipping...")
                continue
            delta_df_2030 = pd.DataFrame(delta_df[0]) if delta_df else pd.DataFrame()
            delta_df_2040 = pd.DataFrame(delta_df[1]) if delta_df else pd.DataFrame()

            if delta_df_2030 is None or delta_df_2040 is None:
                print(f"Missing data for 2030 or 2040 for {rf}. Skipping...")
                continue
            
        # Plot results for 2030 and 2040
        fig, axs = plt.subplots(1, 2, figsize=(19, 9), dpi=500, sharey=True)
        
        
        if not delta_df_2030.empty:
            print(delta_df_2030)
            axs[0] = plot_clustered_stacked(
                fig, axs[0], [delta_df_2030], stage_colors_2030,
                labels=["2030 difference"],
                ylabel="Difference in annual metal content production (000' tonnes)",
                title=f"{rf.title()} 2030 - Constrained minus unconstrained (Region) (metal content)",
                df_style = 'multiple',
                stacked=False,
                scenario_labels = False,
                width = 0.7
            )

        if not delta_df_2040.empty:
            axs[1] = plot_clustered_stacked(
                fig, axs[1], [delta_df_2040], stage_colors_2040,
                labels=["2040 difference"],
                ylabel="Difference in annual metal content production (000' tonnes)",
                title=f"{rf.title()} 2040 - Constrained minus unconstrained (Region) (metal content)",
                df_style = 'multiple',
                stacked=False,
                scenario_labels = False,
                width = 0.7
            )

        # Configure gridlines
        for ax in axs:
            ax.grid(False, axis="x")
            ax.grid(which="major", axis="y", linestyle="-", zorder=0)
            ax.axhline(y=0, color="black", linewidth=1, zorder=3)

        plt.tight_layout()
        save_fig(os.path.join(figures, f"{rf}_MR_constrained_unconstrained_production_metalcont_comparisons_2030_2040.png"))
        plt.close()

        # HORIZONTAL plot

        # Plot results for 2030 and 2040
        fig, axs = plt.subplots(2, 1, figsize=(9, 12), dpi=500, sharex=True)

        if not delta_df_2030.empty:
            axs[0] = plot_clustered_stacked(
                fig, axs[0], [delta_df_2030], stage_colors_2030,
                labels=["2030 difference"],
                ylabel="Difference in annual metal content production (000' tonnes)",
                title=f"{rf.title()} 2030 - Constrained minus unconstrained (Regional) (metal content)",
                df_style = 'multiple',
                orientation = "horizontal",
                stacked=False,
                scenario_labels = False,
                width = 0.7
            )

        if not delta_df_2040.empty:
            axs[1] = plot_clustered_stacked(
                fig, axs[1], [delta_df_2040], stage_colors_2040,
                labels=["2040 difference"],
                ylabel="Difference in annual metal content production (000' tonnes)",
                title=f"{rf.title()} 2040 - Constrained minus unconstrained (Regional) (metal content)",
                df_style = 'multiple',
                orientation = "horizontal",
                stacked=False,
                scenario_labels = False,
                width = 0.7
            )

        # Configure gridlines
        for ax in axs:
            ax.grid(False, axis="y")
            ax.grid(which="major", axis="x", linestyle="-", zorder=0)
            ax.axvline(x=0, color="black", linewidth=1, zorder=3)

        plt.tight_layout()
        save_fig(os.path.join(figures, f"{rf}_MR_constrained_unconstrained_production_metalcont_comparisons_2030_2040_H.png"))
        plt.close()
        # Concatenate all results after the loop ends
        final_df = pd.concat(all_results, ignore_index=True)
        final_df = final_df[['Constraint','Scenario', 'Mineral',  'iso3'] + stages]
        final_df.rename(columns={"iso3": "Country", 'Beneficiation': 'Beneficiation (kt)',
                                'Precursor related product': 'Precursor related product (kt)'}, inplace=True)
        # Define output file path for Excel
        excel_filename = os.path.join(output_data_path, "result_summaries", 
                                    "MR_constrained_unconstrained_production_mineral_comparisons_2030_2040.xlsx")

        # Save to Excel with formatting
        with pd.ExcelWriter(excel_filename, engine="xlsxwriter") as writer:
            final_df.to_excel(writer, sheet_name="Results", index=False)

            # Formatting improvements
            workbook = writer.book
            worksheet = writer.sheets["Results"]

            # Adjust column widths dynamically for better readability
            for col_num, value in enumerate(final_df.columns.values):
                worksheet.set_column(col_num, col_num, min(len(value) + 2, 30))  

    """
    COUNTRY Delta tonnage unconstrained and constrained for 2030 and 2040 separately in subplots: metal content
    """
    make_plot = False
    if make_plot:
        multiply_factor = 1.0e-3
        results_file = os.path.join(output_data_path, "result_summaries", "transport_totals_by_stage.xlsx")
        constraints = ["country_unconstrained", "country_constrained"]
        reference_minerals = ["cobalt","copper","graphite","lithium","manganese","nickel"]
        reference_minerals_short = ["Co","Cu", "Gr", "Li","Mn","Ni"]
        reference_mineral_colors = ["#fdae61","#f46d43","#66c2a5","#c2a5cf","#fee08b","#3288bd"]
        scenarios = [
            "2030_mid_min_threshold_metal_tons",
            "2040_mid_min_threshold_metal_tons"
        ]
        tons_column = "production_tonnes"
        index_cols = ["reference_mineral", "iso3"]
        all_properties = mineral_properties()

        for rf in reference_minerals:
            dfs = []  # To store DataFrames for each scenario and constraint
            stages_all = all_properties[rf]["stage_labels"]
            stage_colors_all = all_properties[rf]["stage_label_colors"]

            for scenario in scenarios:
                # Keeping only the relevant stages for each year
                if '2030' in scenario:
                    stages = [item for item in stages_all if 'Precursor related product' not in item]
                    stage_colors_2030 = [item for item in stage_colors_all if '#7f0000' not in item]
                if '2040' in scenario:
                    stages = [item for item in stages_all if 'Early refining' not in item]
                    stage_colors_2040 = [item for item in stage_colors_all if '#fb6a4a' not in item]

                for cs in constraints:
                    df = pd.read_excel(results_file, sheet_name=cs, index_col=[0, 1, 2, 3, 4])
                    df = df.reset_index()

                    # Filter by scenario, processing_stage, and mineral
                    filtered_df = df[(df["scenario"] == scenario) & 
                                    (df["processing_stage"] == 0) & 
                                    (df["reference_mineral"] == rf)]

                    if filtered_df.empty:
                        print(f"No data available for scenario '{scenario}' and constraint '{cs}'. Skipping...")
                        continue

                    m_df = pd.DataFrame(sorted(filtered_df["iso3"].unique()), columns=["iso3"])

                    # Aggregate and process stage-wise data
                    for st in stages:
                        s_df = filtered_df[filtered_df["processing_type"] == st]
                        if not s_df.empty:
                            s_df[tons_column] = multiply_factor * s_df[tons_column]
                            s_df = s_df.groupby(["iso3", "processing_type"])[tons_column].sum().reset_index()
                            s_df.rename(columns={tons_column: f"{st}"}, inplace=True)
                            m_df = pd.merge(m_df, s_df[["iso3", f"{st}"]], how="left", on="iso3").fillna(0)
                        else:
                            m_df[f"{st}"] = 0

                    dfs.append(m_df.set_index("iso3"))

            # Compute the differences for 2030 and 2040
            delta_df = []
            if len(dfs) >= 4:  # Ensure we have all the required DataFrames
                delta_df.append(dfs[1].subtract(dfs[0], fill_value=0))  # 2030 constrained minus unconstrained
                delta_df.append(dfs[3].subtract(dfs[2], fill_value=0))  # 2040 constrained minus unconstrained

            # Plot results for 2030 and 2040
            fig, axs = plt.subplots(1, 2, figsize=(19, 9), dpi=500, sharey=True)

            if not delta_df:
                print(f"No data differences to plot for {rf}. Skipping...")
                continue
            delta_df_2030 = pd.DataFrame(delta_df[0]) if delta_df else pd.DataFrame()
            delta_df_2040 = pd.DataFrame(delta_df[1]) if delta_df else pd.DataFrame()

            # Remove rows where all columns (except index 'iso3') are zero
            delta_df_2030 = delta_df_2030.loc[(delta_df_2030 != 0).any(axis=1)]
            delta_df_2040 = delta_df_2040.loc[(delta_df_2040 != 0).any(axis=1)]

            if delta_df_2030 is None or delta_df_2040 is None:
                print(f"Missing data for 2030 or 2040 for {rf}. Skipping...")
                continue
            
            # VERTICAL PLOT
            if not delta_df_2030.empty:
                axs[0] = plot_clustered_stacked(
                    fig, axs[0], [delta_df_2030], stage_colors_2030,
                    labels=["2030 difference"],
                    ylabel="Difference in annual metal content production (000' tonnes)",
                    title=f"{rf.title()} 2030 - Constrained minus unconstrained (Country) (metal content)",
                    df_style = 'multiple',#'single_yr', #'multiple',
                    stacked=False,
                    scenario_labels = False,
                    width = 0.8
                )

            if not delta_df_2040.empty:
                axs[1] = plot_clustered_stacked(
                    fig, axs[1], [delta_df_2040], stage_colors_2040,
                    labels=["2040 difference"],
                    ylabel="Difference in annual metal content production (000' tonnes)",
                    title=f"{rf.title()} 2040 - Constrained minus unconstrained (Country) (metal content)",
                    df_style = 'multiple',#'single_yr',
                    stacked=False,
                    scenario_labels = False,
                    width = 0.8
                )

            # Configure gridlines
            for ax in axs:
                ax.grid(False, axis="x")
                ax.grid(which="major", axis="y", linestyle="-", zorder=0)
                ax.axhline(y=0, color="black", linewidth=1, zorder=3)

            plt.tight_layout()
            save_fig(os.path.join(figures, f"{rf}_MN_constrained_unconstrained_production_metalcont_comparisons_2030_2040.png"))
            plt.close()

            # HORIZONTAL PLOT
            # Plot results for 2030 and 2040
            fig, axs = plt.subplots(2, 1, figsize=(9, 19), dpi=500, sharex=True)
            
            if not delta_df_2030.empty:
                axs[0] = plot_clustered_stacked(
                    fig, axs[0], [delta_df_2030], stage_colors_2030,
                    labels=["2030 difference"],
                    ylabel="Difference in annual metal content production (000' tonnes)",
                    title=f"{rf.title()} 2030 - Constrained minus unconstrained (Country) (metal content)",
                    df_style = 'multiple',#'single_yr',
                    stacked=False,
                    orientation="horizontal",
                    scenario_labels = False,
                    width = 0.5
                )

            if not delta_df_2040.empty:
                axs[1] = plot_clustered_stacked(
                    fig, axs[1], [delta_df_2040], stage_colors_2040,
                    labels=["2040 difference"],
                    ylabel="Difference in annual metal content production (000' tonnes)",
                    title=f"{rf.title()} 2040 - Constrained minus unconstrained (Country) (metal content)",
                    df_style = 'multiple',#'single_yr',
                    stacked=False,
                    orientation="horizontal",
                    scenario_labels = False,
                    width = 0.5
                )

            # Configure gridlines
            for ax in axs:
                ax.grid(False, axis="y")
                ax.grid(which="major", axis="x", linestyle="-", zorder=0)
                ax.axvline(x=0, color="black", linewidth=1, zorder=3)

            plt.tight_layout()
            save_fig(os.path.join(figures, f"{rf}_MN_constrained_unconstrained_production_metalcont_comparisons_2030_2040_H.png"))
            plt.close()
    
    
    """
    REGION Delta tonnage unconstrained and constrained for 2030 and 2040 separately in subplots: production tonnes
    """
    make_plot = False
    if make_plot:
        multiply_factor = 1.0e-3
        results_file = os.path.join(output_data_path, "result_summaries", "transport_totals_by_stage.xlsx")
        constraints = ["region_unconstrained", "region_constrained"]
        reference_minerals = ["cobalt","copper","graphite","lithium","manganese","nickel"]
        reference_minerals_short = ["Co","Cu", "Gr", "Li","Mn","Ni"]
        reference_mineral_colors = ["#fdae61","#f46d43","#66c2a5","#c2a5cf","#fee08b","#3288bd"]
        scenarios = [
            "2030_mid_max_threshold_metal_tons",
            "2040_mid_max_threshold_metal_tons"
        ]
        tons_column = "production_tonnes"
        index_cols = ["reference_mineral", "iso3"]
        all_properties = mineral_properties()

        all_results = []
        for rf in reference_minerals:
            dfs = []  # To store DataFrames for each scenario and constraint
            stages_all = all_properties[rf]["stage_labels"]
            stage_colors_all = all_properties[rf]["stage_label_colors"]
            
            for scenario in scenarios:
                # Keeping only the relevant stages for each year
                if '2030' in scenario:
                    # stages = [item for item in stages_all if 'Precursor related product' not in item]
                    # stage_colors_2030 = [item for item in stage_colors_all if '#7f0000' not in item]
                    stages = stages_all
                    stage_colors_2030 = stage_colors_all
                if '2040' in scenario:
                    # stages = [item for item in stages_all if 'Early refining' not in item]
                    # stage_colors_2040 = [item for item in stage_colors_all if '#fb6a4a' not in item]
                    stages = stages_all
                    stage_colors_2040 = stage_colors_all

                for cs in constraints:
                    df = pd.read_excel(results_file, sheet_name=cs, index_col=[0, 1, 2, 3, 4])
                    df = df.reset_index()

                    # Filter by scenario, processing_stage, and mineral
                    filtered_df = df[(df["scenario"] == scenario) & 
                                    (df["processing_stage"] > 0) & 
                                    (df["reference_mineral"] == rf)]

                    if filtered_df.empty:
                        print(f"No data available for scenario '{scenario}' and constraint '{cs}'. Skipping...")
                        continue

                    m_df = pd.DataFrame(sorted(filtered_df["iso3"].unique()), columns=["iso3"])

                    # Aggregate and process stage-wise data
                    for st in stages:
                        s_df = filtered_df[filtered_df["processing_type"] == st]
                        if not s_df.empty:
                            s_df[tons_column] = multiply_factor * s_df[tons_column]
                            s_df = s_df.groupby(["iso3", "processing_type"])[tons_column].sum().reset_index()
                            s_df.rename(columns={tons_column: f"{st}"}, inplace=True)
                            m_df = pd.merge(m_df, s_df[["iso3", f"{st}"]], how="left", on="iso3").fillna(0)
                        else:
                            m_df[f"{st}"] = 0

                    dfs.append(m_df.set_index("iso3"))
            # Add metadata columns
                    m_df["Constraint"] = cs
                    m_df["Scenario"] = scenario
                    m_df["Mineral"] = rf
                
                    all_results.append(m_df.copy())

            # Compute the differences for 2030 and 2040
            delta_df = []
            if len(dfs) >= 4:  # Ensure we have all the required DataFrames
                delta_df.append(dfs[1].subtract(dfs[0], fill_value=0))  # 2030 constrained minus unconstrained
                delta_df.append(dfs[3].subtract(dfs[2], fill_value=0))  # 2040 constrained minus unconstrained

            # Plot results for 2030 and 2040
            fig, axs = plt.subplots(1, 2, figsize=(19, 9), dpi=500, sharey=True)

            if not delta_df:
                print(f"No data differences to plot for {rf}. Skipping...")
                continue
            delta_df_2030 = pd.DataFrame(delta_df[0]) if delta_df else pd.DataFrame()
            delta_df_2040 = pd.DataFrame(delta_df[1]) if delta_df else pd.DataFrame()

            if delta_df_2030 is None or delta_df_2040 is None:
                print(f"Missing data for 2030 or 2040 for {rf}. Skipping...")
                continue
            
            if not delta_df_2030.empty:
                print(delta_df_2030)
                axs[0] = plot_clustered_stacked(
                    fig, axs[0], [delta_df_2030], stage_colors_2030,
                    labels=["2030 difference"],
                    ylabel="Difference in annual production (000' tonnes)",
                    title=f"{rf.title()} 2030 - Constrained minus unconstrained (Regional)",
                    df_style = 'multiple',
                    stacked=False,
                    scenario_labels = False,
                    width = 0.7
                )

            if not delta_df_2040.empty:
                axs[1] = plot_clustered_stacked(
                    fig, axs[1], [delta_df_2040], stage_colors_2040,
                    labels=["2040 difference"],
                    ylabel="Difference in annual production (000' tonnes)",
                    title=f"{rf.title()} 2040 - Constrained minus unconstrained (Regional)",
                    df_style = 'multiple',
                    stacked=False,
                    scenario_labels = False,
                    width = 0.7
                )

            # Configure gridlines
            for ax in axs:
                ax.grid(False, axis="x")
                ax.grid(which="major", axis="y", linestyle="-", zorder=0)
                ax.axhline(y=0, color="black", linewidth=1, zorder=3)

            plt.tight_layout()
            save_fig(os.path.join(figures, f"{rf}_MR_constrained_unconstrained_production_comparisons_2030_2040.png"))
            plt.close()

            # HORIZONTAL plot

            # Plot results for 2030 and 2040
            fig, axs = plt.subplots(2, 1, figsize=(9, 12), dpi=500, sharex=True)

            if not delta_df_2030.empty:
                axs[0] = plot_clustered_stacked(
                    fig, axs[0], [delta_df_2030], stage_colors_2030,
                    labels=["2030 difference"],
                    ylabel="Difference in annual production (000' tonnes)",
                    title=f"{rf.title()} 2030 - Constrained minus unconstrained (Regional)",
                    df_style = 'multiple',
                    orientation = "horizontal",
                    stacked=False,
                    scenario_labels = False,
                    width = 0.7
                )

            if not delta_df_2040.empty:
                axs[1] = plot_clustered_stacked(
                    fig, axs[1], [delta_df_2040], stage_colors_2040,
                    labels=["2040 difference"],
                    ylabel="Difference in annual production (000' tonnes)",
                    title=f"{rf.title()} 2040 - Constrained minus unconstrained (Regional)",
                    df_style = 'multiple',
                    orientation = "horizontal",
                    stacked=False,
                    scenario_labels = False,
                    width = 0.7
                )

            # Configure gridlines
            for ax in axs:
                ax.grid(False, axis="y")
                ax.grid(which="major", axis="x", linestyle="-", zorder=0)
                ax.axvline(x=0, color="black", linewidth=1, zorder=3)

            plt.tight_layout()
            save_fig(os.path.join(figures, f"{rf}_MR_constrained_unconstrained_production_comparisons_2030_2040_H.png"))
            plt.close()
        # Concatenate all results after the loop ends
        final_df = pd.concat(all_results, ignore_index=True)
        final_df = final_df[['Constraint','Scenario', 'Mineral',  'iso3'] + stages]
        final_df.rename(columns={"iso3": "Country", 'Beneficiation': 'Beneficiation (kt)',
                                'Precursor related product': 'Precursor related product (kt)'}, inplace=True)
        # Define output file path for Excel
        excel_filename = os.path.join(output_data_path, "result_summaries", 
                                    "MR_constrained_unconstrained_production_mineral_comparisons_2030_2040.xlsx")

        # Save to Excel with formatting
        with pd.ExcelWriter(excel_filename, engine="xlsxwriter") as writer:
            final_df.to_excel(writer, sheet_name="Results", index=False)

            # Formatting improvements
            workbook = writer.book
            worksheet = writer.sheets["Results"]

            # Adjust column widths dynamically for better readability
            for col_num, value in enumerate(final_df.columns.values):
                worksheet.set_column(col_num, col_num, min(len(value) + 2, 30))  

    """
    COUNTRY Delta tonnage unconstrained and constrained for 2030 and 2040 separately in subplots: production tonnes
    """
    make_plot = False
    if make_plot:
        multiply_factor = 1.0e-3
        results_file = os.path.join(output_data_path, "result_summaries", "transport_totals_by_stage.xlsx")
        constraints = ["country_unconstrained", "country_constrained"]
        reference_minerals = ["cobalt","copper","graphite","lithium","manganese","nickel"]
        reference_minerals_short = ["Co","Cu", "Gr", "Li","Mn","Ni"]
        reference_mineral_colors = ["#fdae61","#f46d43","#66c2a5","#c2a5cf","#fee08b","#3288bd"]
        scenarios = [
            "2030_mid_min_threshold_metal_tons",
            "2040_mid_min_threshold_metal_tons"
        ]
        tons_column = "production_tonnes"
        index_cols = ["reference_mineral", "iso3"]
        all_properties = mineral_properties()

        for rf in reference_minerals:
            dfs = []  # To store DataFrames for each scenario and constraint
            stages_all = all_properties[rf]["stage_labels"]
            stage_colors_all = all_properties[rf]["stage_label_colors"]

            for scenario in scenarios:
                # Keeping only the relevant stages for each year
                if '2030' in scenario:
                    stages = [item for item in stages_all if 'Precursor related product' not in item]
                    stage_colors_2030 = [item for item in stage_colors_all if '#7f0000' not in item]
                if '2040' in scenario:
                    stages = [item for item in stages_all if 'Early refining' not in item]
                    stage_colors_2040 = [item for item in stage_colors_all if '#fb6a4a' not in item]

                for cs in constraints:
                    df = pd.read_excel(results_file, sheet_name=cs, index_col=[0, 1, 2, 3, 4])
                    df = df.reset_index()

                    # Filter by scenario, processing_stage, and mineral
                    filtered_df = df[(df["scenario"] == scenario) & 
                                    (df["processing_stage"] > 0) & 
                                    (df["reference_mineral"] == rf)]

                    if filtered_df.empty:
                        print(f"No data available for scenario '{scenario}' and constraint '{cs}'. Skipping...")
                        continue

                    m_df = pd.DataFrame(sorted(filtered_df["iso3"].unique()), columns=["iso3"])

                    # Aggregate and process stage-wise data
                    for st in stages:
                        s_df = filtered_df[filtered_df["processing_type"] == st]
                        if not s_df.empty:
                            s_df[tons_column] = multiply_factor * s_df[tons_column]
                            s_df = s_df.groupby(["iso3", "processing_type"])[tons_column].sum().reset_index()
                            s_df.rename(columns={tons_column: f"{st}"}, inplace=True)
                            m_df = pd.merge(m_df, s_df[["iso3", f"{st}"]], how="left", on="iso3").fillna(0)
                        else:
                            m_df[f"{st}"] = 0

                    dfs.append(m_df.set_index("iso3"))

            # Compute the differences for 2030 and 2040
            delta_df = []
            if len(dfs) >= 4:  # Ensure we have all the required DataFrames
                delta_df.append(dfs[1].subtract(dfs[0], fill_value=0))  # 2030 constrained minus unconstrained
                delta_df.append(dfs[3].subtract(dfs[2], fill_value=0))  # 2040 constrained minus unconstrained

            # Plot results for 2030 and 2040
            fig, axs = plt.subplots(1, 2, figsize=(19, 9), dpi=500, sharey=True)

            if not delta_df:
                print(f"No data differences to plot for {rf}. Skipping...")
                continue
            delta_df_2030 = pd.DataFrame(delta_df[0]) if delta_df else pd.DataFrame()
            delta_df_2040 = pd.DataFrame(delta_df[1]) if delta_df else pd.DataFrame()

            # Remove rows where all columns (except index 'iso3') are zero
            delta_df_2030 = delta_df_2030.loc[(delta_df_2030 != 0).any(axis=1)]
            delta_df_2040 = delta_df_2040.loc[(delta_df_2040 != 0).any(axis=1)]

            if delta_df_2030 is None or delta_df_2040 is None:
                print(f"Missing data for 2030 or 2040 for {rf}. Skipping...")
                continue
            
            # VERTICAL PLOT
            if not delta_df_2030.empty:
                axs[0] = plot_clustered_stacked(
                    fig, axs[0], [delta_df_2030], stage_colors_2030,
                    labels=["2030 difference"],
                    ylabel="Difference in annual production (000' tonnes)",
                    title=f"{rf.title()} 2030 - Constrained minus unconstrained (Country)",
                    df_style = 'multiple',#'single_yr', #'multiple',
                    stacked=False,
                    scenario_labels = False,
                    width = 0.8
                )

            if not delta_df_2040.empty:
                axs[1] = plot_clustered_stacked(
                    fig, axs[1], [delta_df_2040], stage_colors_2040,
                    labels=["2040 difference"],
                    ylabel="Difference in annual production (000' tonnes)",
                    title=f"{rf.title()} 2040 - Constrained minus unconstrained (Country)",
                    df_style = 'multiple',#'single_yr',
                    stacked=False,
                    scenario_labels = False,
                    width = 0.8
                )

            # Configure gridlines
            for ax in axs:
                ax.grid(False, axis="x")
                ax.grid(which="major", axis="y", linestyle="-", zorder=0)
                ax.axhline(y=0, color="black", linewidth=1, zorder=3)

            plt.tight_layout()
            save_fig(os.path.join(figures, f"{rf}_MN_constrained_unconstrained_production_comparisons_2030_2040.png"))
            plt.close()

            # HORIZONTAL PLOT
            # Plot results for 2030 and 2040
            fig, axs = plt.subplots(2, 1, figsize=(9, 19), dpi=500, sharex=True)
            
            if not delta_df_2030.empty:
                axs[0] = plot_clustered_stacked(
                    fig, axs[0], [delta_df_2030], stage_colors_2030,
                    labels=["2030 difference"],
                    ylabel="Difference in annual production (000' tonnes)",
                    title=f"{rf.title()} 2030 - Constrained minus unconstrained (Country)",
                    df_style = 'multiple',#'single_yr',
                    stacked=False,
                    orientation="horizontal",
                    scenario_labels = False,
                    width = 0.5
                )

            if not delta_df_2040.empty:
                axs[1] = plot_clustered_stacked(
                    fig, axs[1], [delta_df_2040], stage_colors_2040,
                    labels=["2040 difference"],
                    ylabel="Difference in annual production (000' tonnes)",
                    title=f"{rf.title()} 2040 - Constrained minus unconstrained (Country)",
                    df_style = 'multiple',#'single_yr',
                    stacked=False,
                    orientation="horizontal",
                    scenario_labels = False,
                    width = 0.5
                )

            # Configure gridlines
            for ax in axs:
                ax.grid(False, axis="y")
                ax.grid(which="major", axis="x", linestyle="-", zorder=0)
                ax.axvline(x=0, color="black", linewidth=1, zorder=3)

            plt.tight_layout()
            save_fig(os.path.join(figures, f"{rf}_MN_constrained_unconstrained_production_comparisons_2030_2040_H.png"))
            plt.close()
    
    
    """
    DELTA Regional constrained unconstrained production costs' comparisons for 2030 and 2040 separately in subplots: production costs
    """
    make_plot = False
    if make_plot:
        multiply_factor = 1.0e-6
        results_file = os.path.join(output_data_path, "result_summaries", "energy_transport_totals_by_stage.xlsx")
        constraints = ["region_unconstrained", "region_constrained"]
        reference_minerals = ["cobalt","copper","graphite","lithium","manganese","nickel"]
        reference_minerals_short = ["Co","Cu", "Gr", "Li","Mn","Ni"]
        reference_mineral_colors = ["#fdae61","#f46d43","#66c2a5","#c2a5cf","#fee08b","#3288bd"]
        scenarios = [
            "2030_mid_max_threshold_metal_tons",
            "2040_mid_max_threshold_metal_tons"
        ]
        tons_column = "production_cost_usd"
        index_cols = ["reference_mineral", "iso3"]
        all_properties = mineral_properties()

        for rf in reference_minerals:
            dfs = []  # To store DataFrames for each scenario and constraint
            stages_all = all_properties[rf]["stage_labels"]
            stage_colors_all = all_properties[rf]["stage_label_colors"]
            print(stage_colors_all)

            # for scenario in scenarios:
            #     # Keeping only the relevant stages for each year
            #     if '2030' in scenario:
            #         stages = [item for item in stages_all if 'Precursor related product' not in item]
            #         stage_colors = [item for item in stage_colors_all if '#7f0000' not in item]
            #     if '2040' in scenario:
            #         stages = [item for item in stages_all if 'Early refining' not in item]
            #         stage_colors = [item for item in stage_colors_all if '#fb6a4a' not in item]


            #     for cs in constraints:
            #         df = pd.read_excel(results_file, sheet_name=cs, index_col=[0, 1, 2, 3, 4])
            #         df = df.reset_index()

            #         # Filter by scenario, processing_stage, and mineral
            #         filtered_df = df[(df["scenario"] == scenario) & 
            #                         (df["processing_stage"] > 0) & 
            #                         (df["reference_mineral"] == rf)]

            #         if filtered_df.empty:
            #             print(f"No data available for scenario '{scenario}' and constraint '{cs}'. Skipping...")
            #             continue

            #         m_df = pd.DataFrame(sorted(filtered_df["iso3"].unique()), columns=["iso3"])

            #         # Aggregate and process stage-wise data
            #         for st in stages:
            #             s_df = filtered_df[filtered_df["processing_type"] == st]
            #             if not s_df.empty:
            #                 s_df[tons_column] = multiply_factor * s_df[tons_column]
            #                 s_df = s_df.groupby(["iso3", "processing_type"])[tons_column].sum().reset_index()
            #                 s_df.rename(columns={tons_column: f"{st}"}, inplace=True)
            #                 m_df = pd.merge(m_df, s_df[["iso3", f"{st}"]], how="left", on="iso3").fillna(0)
            #             else:
            #                 m_df[f"{st}"] = 0

            #         dfs.append(m_df.set_index("iso3"))

            # # Compute the differences for 2030 and 2040
            # delta_df = []
            # if len(dfs) >= 4:  # Ensure we have all the required DataFrames
            #     delta_df.append(dfs[1].subtract(dfs[0], fill_value=0))  # 2030 constrained minus unconstrained
            #     delta_df.append(dfs[3].subtract(dfs[2], fill_value=0))  # 2040 constrained minus unconstrained

            # # VERTICAL
            # # Plot results for 2030 and 2040
            # fig, axs = plt.subplots(1, 2, figsize=(19, 9), dpi=500, sharey=True)

            # if not delta_df:
            #     print(f"No data differences to plot for {rf}. Skipping...")
            #     continue
            # delta_df_2030 = pd.DataFrame(delta_df[0]) if delta_df else pd.DataFrame()
            # delta_df_2040 = pd.DataFrame(delta_df[1]) if delta_df else pd.DataFrame()

            # if delta_df_2030 is None or delta_df_2040 is None:
            #     print(f"Missing data for 2030 or 2040 for {rf}. Skipping...")
            #     continue
            
            
            # if not delta_df_2030.empty:
            #     axs[0] = plot_clustered_stacked(
            #         fig, axs[0], [delta_df_2030], stage_colors,
            #         labels=["2030 difference"],
            #         ylabel="Difference in annual production costs (million USD)",
            #         title=f"{rf.title()} 2030 - Constrained minus unconstrained (Region)",
            #         df_style = 'multiple',
            #         orientation = "horizontal",
            #         stacked=False,
            #         scenario_labels = False,
            #         width = 0.7
            #     )

            # if not delta_df_2040.empty:
            #     axs[1] = plot_clustered_stacked(
            #         fig, axs[1], [delta_df_2040], stage_colors,
            #         labels=["2040 difference"],
            #         ylabel="Difference in annual production costs (million USD)",
            #         title=f"{rf.title()} 2040 - Constrained minus unconstrained (Region)",
            #         df_style = 'multiple',
            #         orientation = "horizontal",
            #         stacked=False,
            #         scenario_labels = False,
            #         width = 0.7
            #     )

            # # Configure gridlines
            # for ax in axs:
            #     ax.grid(False, axis="x")
            #     ax.grid(which="major", axis="y", linestyle="-", zorder=0)
            #     ax.axhline(y=0, color="black", linewidth=1, zorder=3)

            # plt.tight_layout()
            # save_fig(os.path.join(figures, f"{rf}_MR_constrained_unconstrained_costs_comparisons_2030_2040.png"))
            # plt.close()

            # # HORIZONTAL
            # # Plot results for 2030 and 2040
            # fig, axs = plt.subplots(2, 1, figsize=(9, 19), dpi=500, sharex=True)

            # if not delta_df_2030.empty:
            #     axs[0] = plot_clustered_stacked(
            #         fig, axs[0], [delta_df_2030], stage_colors,
            #         labels=["2030 difference"],
            #         ylabel="Difference in annual production costs (million USD)",
            #         title=f"{rf.title()} 2030 - Constrained minus unconstrained (Region)",
            #         df_style = 'multiple',
            #         scenario_labels = False,
            #         stacked = False,
            #         orientation = 'horizontal'
            #     )

            # if not delta_df_2040.empty:
            #     axs[1] = plot_clustered_stacked(
            #         fig, axs[1], [delta_df_2040], stage_colors,
            #         labels=["2040 difference"],
            #         ylabel="Difference in annual production costs (million USD)",
            #         title=f"{rf.title()} 2040 - Constrained minus unconstrained (Region)",
            #         df_style = 'multiple',
            #         scenario_labels = False,
            #         stacked = False,
            #         orientation = 'horizontal'
            #     )

            # # Configure gridlines
            # for ax in axs:
            #     ax.grid(False, axis="y")
            #     ax.grid(which="major", axis="x", linestyle="-", zorder=0)
            #     ax.axvline(x=0, color="black", linewidth=1, zorder=3)

            # plt.tight_layout()
            # save_fig(os.path.join(figures, f"{rf}_MR_constrained_unconstrained_costs_comparisons_2030_2040_H.png"))
            # plt.close()
    """
    DELTA Country constrained unconstrained production costs' comparisons for 2030 and 2040 separately in subplots: production costs
    """
    make_plot = False
    if make_plot:
        multiply_factor = 1.0e-6
        results_file = os.path.join(output_data_path, "result_summaries", "combined_transport_totals_by_stage.xlsx")
        constraints = ["country_unconstrained", "country_constrained"]
        reference_minerals = ["cobalt","copper","graphite","lithium","manganese","nickel"]
        reference_minerals_short = ["Co","Cu", "Gr", "Li","Mn","Ni"]
        reference_mineral_colors = ["#fdae61","#f46d43","#66c2a5","#c2a5cf","#fee08b","#3288bd"]
        scenarios = [
            "2030_mid_min_threshold_metal_tons",
            "2040_mid_min_threshold_metal_tons"
        ]
        tons_column = "production_cost_usd"
        index_cols = ["reference_mineral", "iso3"]
        all_properties = mineral_properties()

        for rf in reference_minerals:
            dfs = []  # To store DataFrames for each scenario and constraint
            stages_all = all_properties[rf]["stage_labels"]
            stage_colors_all = all_properties[rf]["stage_label_colors"]

            for scenario in scenarios:
                # Keeping only the relevant stages for each year
                if '2030' in scenario:
                    stages = [item for item in stages_all if 'Precursor related product' not in item]
                    stage_colors = [item for item in stage_colors_all if '#7f0000' not in item]
                if '2040' in scenario:
                    stages = [item for item in stages_all if 'Early refining' not in item]
                    stage_colors = [item for item in stage_colors_all if '#fb6a4a' not in item]

                for cs in constraints:
                    df = pd.read_excel(results_file, sheet_name=cs, index_col=[0, 1, 2, 3, 4])
                    df = df.reset_index()

                    # Filter by scenario, processing_stage, and mineral
                    filtered_df = df[(df["scenario"] == scenario) & 
                                    (df["processing_stage"] > 0) & 
                                    (df["reference_mineral"] == rf)]

                    if filtered_df.empty:
                        print(f"No data available for scenario '{scenario}' and constraint '{cs}'. Skipping...")
                        continue

                    m_df = pd.DataFrame(sorted(filtered_df["iso3"].unique()), columns=["iso3"])

                    # Aggregate and process stage-wise data
                    for st in stages:
                        s_df = filtered_df[filtered_df["processing_type"] == st]
                        if not s_df.empty:
                            s_df[tons_column] = multiply_factor * s_df[tons_column]
                            s_df = s_df.groupby(["iso3", "processing_type"])[tons_column].sum().reset_index()
                            s_df.rename(columns={tons_column: f"{st}"}, inplace=True)
                            m_df = pd.merge(m_df, s_df[["iso3", f"{st}"]], how="left", on="iso3").fillna(0)
                        else:
                            m_df[f"{st}"] = 0

                    dfs.append(m_df.set_index("iso3"))

            # Compute the differences for 2030 and 2040
            delta_df = []
            if len(dfs) >= 4:  # Ensure we have all the required DataFrames
                delta_df.append(dfs[1].subtract(dfs[0], fill_value=0))  # 2030 constrained minus unconstrained
                delta_df.append(dfs[3].subtract(dfs[2], fill_value=0))  # 2040 constrained minus unconstrained

            # VERTICAL
            # Plot results for 2030 and 2040
            fig, axs = plt.subplots(1, 2, figsize=(19, 9), dpi=500, sharey=True)

            if not delta_df:
                print(f"No data differences to plot for {rf}. Skipping...")
                continue
            delta_df_2030 = pd.DataFrame(delta_df[0]) if delta_df else pd.DataFrame()
            delta_df_2040 = pd.DataFrame(delta_df[1]) if delta_df else pd.DataFrame()

            if delta_df_2030 is None or delta_df_2040 is None:
                print(f"Missing data for 2030 or 2040 for {rf}. Skipping...")
                continue
            
            if not delta_df_2030.empty:
                axs[0] = plot_clustered_stacked(
                    fig, axs[0], [delta_df_2030], stage_colors,
                    labels=["2030 difference"],
                    ylabel="Difference in annual production costs (million USD)",
                    title=f"{rf.title()} 2030 - Constrained minus unconstrained (Country)",
                    df_style = 'multiple',
                    orientation = "vertical",
                    stacked=False,
                    scenario_labels = False,
                    width = 0.7
                )


            if not delta_df_2040.empty:
                axs[1] = plot_clustered_stacked(
                    fig, axs[1], [delta_df_2040], stage_colors,
                    labels=["2040 difference"],
                    ylabel="Difference in annual production costs (million USD)",
                    title=f"{rf.title()} 2040 - Constrained minus unconstrained (Country)",
                    df_style = 'multiple',
                    orientation = "vertical",
                    stacked=False,
                    scenario_labels = False,
                    width = 0.7
                )

            # Configure gridlines
            for ax in axs:
                ax.grid(False, axis="x")
                ax.grid(which="major", axis="y", linestyle="-", zorder=0)
                ax.axhline(y=0, color="black", linewidth=1, zorder=3)

            plt.tight_layout()
            save_fig(os.path.join(figures, f"{rf}_MN_constrained_unconstrained_costs_comparisons_2030_2040.png"))
            plt.close()

            # HORIZONTAL
            # Plot results for 2030 and 2040
            fig, axs = plt.subplots(2, 1, figsize=(9, 19), dpi=500, sharex=True)

            if not delta_df_2030.empty:
                axs[0] = plot_clustered_stacked(
                    fig, axs[0], [delta_df_2030], stage_colors,
                    labels=["2030 difference"],
                    ylabel="Difference in annual production costs (million USD)",
                    title=f"{rf.title()} 2030 - Constrained minus unconstrained (Country)",
                    df_style = 'multiple',
                    orientation = "horizontal",
                    stacked=False,
                    scenario_labels = False,
                    width = 0.7
                )

            if not delta_df_2040.empty:
                axs[1] = plot_clustered_stacked(
                    fig, axs[1], [delta_df_2040], stage_colors,
                    labels=["2040 difference"],
                    ylabel="Difference in annual production costs (million USD)",
                    title=f"{rf.title()} 2040 - Constrained minus unconstrained (Country)",
                    df_style = 'multiple',
                    orientation = "horizontal",
                    stacked=False,
                    scenario_labels = False,
                    width = 0.7
                )

            # Configure gridlines
            for ax in axs:
                ax.grid(False, axis="y")
                ax.grid(which="major", axis="x", linestyle="-", zorder=0)
                ax.axvline(x=0, color="black", linewidth=1, zorder=3)

            plt.tight_layout()
            save_fig(os.path.join(figures, f"{rf}_MN_constrained_unconstrained_costs_comparisons_2030_2040_H.png"))
            plt.close()


    """Unconstrained country and regional comparisons: annual production
    """
    make_plot = False
    if make_plot is True:
        multiply_factor = 1.0e-3
        results_file = os.path.join(output_data_path,
                                "result_summaries",
                                "combined_transport_totals_by_stage.xlsx")
        constraints = ["country_unconstrained","region_unconstrained"]
        reference_minerals = ["cobalt","copper","graphite","lithium","manganese","nickel"]
        reference_minerals_short = ["Co","Cu", "Gr", "Li","Mn","Ni"]
        reference_mineral_colors = ["#fdae61","#f46d43","#66c2a5","#c2a5cf","#fee08b","#3288bd"]
        scenarios =  [
                        "2030_mid_min_threshold_metal_tons",
                        "2030_mid_max_threshold_metal_tons", 
                        "2040_mid_min_threshold_metal_tons",
                        "2040_mid_max_threshold_metal_tons"
                    ]
        col = "export_tonnes"
        index_cols = ["reference_mineral","iso3"]
        all_properties = mineral_properties()
        for rf in reference_minerals:
            dfall = []
            stages = all_properties[rf]["stage_labels"]
            stage_colors = all_properties[rf]["stage_label_colors"]
            for scenario in scenarios:
                for cs in constraints:
                    df = pd.read_excel(
                                        results_file,
                                        sheet_name=cs,
                                        index_col=[0,1,2,3,4])
                    df = df.reset_index()
                    countries = sorted(list(set(df["iso3"].values.tolist())))
                    df = df[(df["scenario"] == scenario) & (df["processing_stage"] > 0) & (df["reference_mineral"] == rf)]  
                    if len(df.index) > 0:
                         # stages = sorted(list(set(df["processing_stage"].values.tolist())))
                        # stage_colors = all_properties[rf]["stage_colors"][:len(stages)]
                        m_df = pd.DataFrame(countries,columns=["iso3"])
                        # xvals = 2*(np.arange(0,len(countries)) + f)
                        for st in stages:
                            s_df = df[df["processing_type"] == st]
                            if len(s_df.index) > 0:
                                s_df[col] = multiply_factor*s_df[col]
                                s_df = s_df.groupby(["iso3","processing_type"])[col].sum().reset_index()
                                s_df.rename(columns={col:f"{st}"},inplace=True)
                                m_df = pd.merge(m_df,s_df[["iso3",f"{st}"]],how="left",on=["iso3"]).fillna(0)
                            else:
                                m_df[f"{st}"] = 0
                        dfall.append(m_df.set_index(["iso3"]))
                    
            fig, ax = plt.subplots(1,1,figsize=(18,9),dpi=500)   
            ax = plot_clustered_stacked(
                                        fig,ax,dfall,stage_colors,
                                        labels=["National 2030","Regional 2030","National 2040","Regional 2040"],
                                        ylabel="Annual export volumes (000' tonnes)", 
                                        title=f"{rf.title()} Mid National and Mid Regional scenario comparisons")
            plt.grid()
            plt.tight_layout()
            save_fig(os.path.join(figures,
                        f"{rf}_MN_MR_production_comparisions_side_by_side.png"))
            plt.close()
    
    """Delta Constrained and unconstrained: Country and regional comparisons for 2030 and 2040 separately in subplots: annual production
    """
    make_plot = False
    if make_plot:
        multiply_factor = 1.0e-3
        results_file = os.path.join(output_data_path, "result_summaries", "transport_totals_by_stage.xlsx")
        constraints = ["country_unconstrained", "region_unconstrained",
                        "country_constrained", "region_constrained"]
        reference_minerals = ["cobalt","copper","graphite","lithium","manganese","nickel"]
        reference_minerals_short = ["Co","Cu", "Gr", "Li","Mn","Ni"]
        reference_mineral_colors = ["#fdae61","#f46d43","#66c2a5","#c2a5cf","#fee08b","#3288bd"]
        scenarios = [
                    "2030_mid_min_threshold_metal_tons",
                    "2030_mid_max_threshold_metal_tons",
                    "2040_mid_min_threshold_metal_tons",
                    "2040_mid_max_threshold_metal_tons"
                    ]
        tons_column = "production_tonnes"
        index_cols = ["reference_mineral", "iso3"]
        all_properties = mineral_properties()

        for rf in reference_minerals:
            dfs = []  # To store DataFrames for each scenario and constraint
            stages = all_properties[rf]["stage_labels"]
            stage_colors = all_properties[rf]["stage_label_colors"]
            # print(stage_colors)

            for scenario in scenarios:
                for cs in constraints[:2]:
                    print(f"{cs}")
                    df = pd.read_excel(results_file, sheet_name=cs, index_col=[0, 1, 2, 3, 4])
                    df = df.reset_index()

                    # Filter by scenario, processing_stage, and mineral
                    filtered_df = df[(df["scenario"] == scenario) & 
                                    (df["processing_stage"] > 0) & 
                                    (df["reference_mineral"] == rf)]

                    if filtered_df.empty:
                        print(f"No data available for scenario '{scenario}' and constraint '{cs}'. Skipping...")
                        continue

                    m_df = pd.DataFrame(sorted(filtered_df["iso3"].unique()), columns=["iso3"])

                    # Aggregate and process stage-wise data
                    for st in stages:
                        s_df = filtered_df[filtered_df["processing_type"] == st]
                        if not s_df.empty:
                            s_df[tons_column] = multiply_factor * s_df[tons_column]
                            s_df = s_df.groupby(["iso3", "processing_type"])[tons_column].sum().reset_index()
                            s_df.rename(columns={tons_column: f"{st}"}, inplace=True)
                            m_df = pd.merge(m_df, s_df[["iso3", f"{st}"]], how="left", on="iso3").fillna(0)
                        else:
                            m_df[f"{st}"] = 0

                    dfs.append(m_df.set_index("iso3"))
                    # print("dfs", dfs)

            # Compute the differences for 2030 and 2040
            delta_df = []
            if len(dfs) >= 4:  # Ensure we have all the required DataFrames
                delta_df.append(dfs[1].subtract(dfs[0], fill_value=0))  # 2030 Regional - National 
                delta_df.append(dfs[3].subtract(dfs[2], fill_value=0))  # 2040 Regional - National
                # print("delta_df", delta_df)

            # Plot results for 2030 and 2040
            fig, axs = plt.subplots(2, 1, figsize=(9, 18), dpi=500, sharex=True)

            if not delta_df:
                print(f"No data differences to plot for {rf}. Skipping...")
                continue
            delta_df_2030 = pd.DataFrame(delta_df[0]) if delta_df else pd.DataFrame()
            delta_df_2040 = pd.DataFrame(delta_df[1]) if delta_df else pd.DataFrame()
            
            # First, drop rows that are completely NaN.
            delta_df_2030 = delta_df_2030.dropna(how='all')
            mask = ~(delta_df_2030.fillna(0) == 0).all(axis=1)
            delta_df_2030 = delta_df_2030[mask]

            delta_df_2040 = delta_df_2040.dropna(how='all')
            mask = ~(delta_df_2040.fillna(0) == 0).all(axis=1)
            delta_df_2040 = delta_df_2040[mask]

            # print('2030 data: Regional - National')
            # print([delta_df_2030]) # The fifferences in stages not relevant to 2030 and 2040 are so small that they don't show in the plot

            if delta_df_2030 is None or delta_df_2040 is None:
                print(f"Missing data for 2030 or 2040 for {rf}. Skipping...")
                continue

            if not delta_df_2030.empty:    
                axs[0] = plot_clustered_stacked(
                    fig, axs[0], [delta_df_2030], 
                    bar_colors = stage_colors,
                    labels=list(delta_df_2030.columns), 
                    ylabel="Difference in annual production (000' tonnes)",
                    width=0.7,   
                    title=f"{rf.title()} 2030 - Regional minus National (Unconstrained)",
                    df_style = 'multiple', 
                    orientation="horizontal",
                    scenario_labels = False,
                    shift_bars = True
                )
                axs[0].get_legend()  # If a legend was automatically generated, you can remove it
                axs[0].legend().remove()  # Removes the legend if it exists

            if not delta_df_2040.empty:
                axs[1] = plot_clustered_stacked(
                    fig, axs[1], [delta_df_2040], 
                    bar_colors = stage_colors,
                    labels=list(delta_df_2040.columns), 
                    ylabel="Difference in annual production (000' tonnes)",
                    width=0.7,   
                    title=f"{rf.title()} 2040 - Regional minus National (Unconstrained)",
                    df_style = 'multiple', #single_yr
                    orientation="horizontal",
                    scenario_labels = False,
                    shift_bars = True
                )

            # Configure gridlines
            for ax in axs:
                ax.grid(False, axis="y")
                ax.grid(which="major", axis="x", linestyle="-", zorder=0)
                ax.axvline(x=0, color="black", linewidth=1, zorder=3)

            plt.tight_layout()
            save_fig(os.path.join(figures, f"{rf}_unconstrained_MN_MR_production_comparisons_2030_2040.png"))
            plt.close()

            for scenario in scenarios:
                for cs in constraints[3:]:
                    df = pd.read_excel(results_file, sheet_name=cs, index_col=[0, 1, 2, 3, 4])
                    df = df.reset_index()

                    # Filter by scenario, processing_stage, and mineral
                    filtered_df = df[(df["scenario"] == scenario) & 
                                    (df["processing_stage"] > 0) & 
                                    (df["reference_mineral"] == rf)]

                    if filtered_df.empty:
                        print(f"No data available for scenario '{scenario}' and constraint '{cs}'. Skipping...")
                        continue

                    m_df = pd.DataFrame(sorted(filtered_df["iso3"].unique()), columns=["iso3"])

                    # Aggregate and process stage-wise data
                    for st in stages:
                        s_df = filtered_df[filtered_df["processing_type"] == st]
                        if not s_df.empty:
                            s_df[tons_column] = multiply_factor * s_df[tons_column]
                            s_df = s_df.groupby(["iso3", "processing_type"])[tons_column].sum().reset_index()
                            s_df.rename(columns={tons_column: f"{st}"}, inplace=True)
                            m_df = pd.merge(m_df, s_df[["iso3", f"{st}"]], how="left", on="iso3").fillna(0)
                        else:
                            m_df[f"{st}"] = 0

                    dfs.append(m_df.set_index("iso3"))

            # Compute the differences for 2030 and 2040
            delta_df = []
            if len(dfs) >= 4:  # Ensure we have all the required DataFrames
                delta_df.append(dfs[1].subtract(dfs[0], fill_value=0))  # 2030 Regional - National 
                delta_df.append(dfs[3].subtract(dfs[2], fill_value=0))  # 2040 Regional - National

            # Plot results for 2030 and 2040
            fig, axs = plt.subplots(2, 1, figsize=(9, 18), dpi=500, sharex=True)

            if not delta_df:
                print(f"No data differences to plot for {rf}. Skipping...")
                continue
            delta_df_2030 = pd.DataFrame(delta_df[0]) if delta_df else pd.DataFrame()
            delta_df_2040 = pd.DataFrame(delta_df[1]) if delta_df else pd.DataFrame()
            
            # First, drop rows that are completely NaN.
            delta_df_2030 = delta_df_2030.dropna(how='all')
            mask = ~(delta_df_2030.fillna(0) == 0).all(axis=1)
            delta_df_2030 = delta_df_2030[mask]

            delta_df_2040 = delta_df_2040.dropna(how='all')
            mask = ~(delta_df_2040.fillna(0) == 0).all(axis=1)
            delta_df_2040 = delta_df_2040[mask]

            # print('2030 data: Regional - National')
            # print(delta_df_2030)

            if delta_df_2030 is None or delta_df_2040 is None:
                print(f"Missing data for 2030 or 2040 for {rf}. Skipping...")
                continue

            if not delta_df_2030.empty:
                axs[0] = plot_clustered_stacked(
                    fig, axs[0], [delta_df_2030], 
                    bar_colors = stage_colors,
                    labels=list(delta_df_2030.columns),
                    ylabel="Difference in annual production (000' tonnes)",
                    width=0.7,   
                    title=f"{rf.title()} 2030 - Regional minus National (Constrained)",
                    df_style = 'multiple',
                    orientation="horizontal",
                    scenario_labels = False,
                    shift_bars = True
                )
                axs[0].get_legend()  # If a legend was automatically generated, you can remove it
                axs[0].legend().remove()  # Removes the legend if it exists

            if not delta_df_2040.empty:
                axs[1] = plot_clustered_stacked(
                    fig, axs[1], [delta_df_2040], 
                    bar_colors = stage_colors,
                    labels=list(delta_df_2040.columns),
                    ylabel="Difference in annual production (000' tonnes)",
                    width=0.7,   
                    title=f"{rf.title()} 2040 - Regional minus National (Constrained)",
                    df_style = 'multiple',
                    orientation="horizontal",
                    scenario_labels = False,
                    shift_bars = True
                )

            # Configure gridlines
            for ax in axs:
                ax.grid(False, axis="y")
                ax.grid(which="major", axis="x", linestyle="-", zorder=0)
                ax.axvline(x=0, color="black", linewidth=1, zorder=3)
            fig.subplots_adjust(bottom=0.2)
            plt.tight_layout()
            save_fig(os.path.join(figures, f"{rf}_constrained_MN_MR_production_comparisons_2030_2040.png"))
            plt.close()
   
    """Delta Constrained and unconstrained: Country and regional comparisons for 2030 and 2040 separately in subplots: annual production
    FOR PAPER - all differences in two subplots (one per year)
    """
    make_plot = False
    if make_plot:
        multiply_factor = 1.0e-3
        results_file = os.path.join(output_data_path, "result_summaries", "combined_energy_transport_totals_by_stage.xlsx")
        
        constraints_map = {
                            "unconstrained": ["country_unconstrained", "region_unconstrained"],
                            "constrained": ["country_constrained", "region_constrained"]
                        }

        scenarios_map = {
                        "2030": ("2030_mid_min_threshold_metal_tons", "2030_mid_max_threshold_metal_tons"),
                        "2040": ("2040_mid_min_threshold_metal_tons", "2040_mid_max_threshold_metal_tons")
                         }

        reference_minerals = ["cobalt", "copper", "graphite", "lithium", "manganese", "nickel"]
        reference_minerals_short = ["Co","Cu", "Gr", "Li","Mn","Ni"]
        reference_mineral_colors = ["#fdae61","#f46d43","#66c2a5","#c2a5cf","#fee08b","#3288bd"]
        tons_column = "production_tonnes"

        all_properties = mineral_properties()

        # This will store all the deltas in long format
        differences_long = []

        for rf in reference_minerals:
            stages = all_properties[rf]["stage_labels"]

            for constraint_type, (cs_national, cs_regional) in constraints_map.items():
                for year, (sc_national, sc_regional) in scenarios_map.items():

                    # Load national and regional data
                    df_nat = pd.read_excel(results_file, sheet_name=cs_national, index_col=[0, 1, 2, 3, 4]).reset_index()
                    df_reg = pd.read_excel(results_file, sheet_name=cs_regional, index_col=[0, 1, 2, 3, 4]).reset_index()

                    # Filter both
                    df_nat = df_nat[
                        (df_nat["scenario"] == sc_national) & 
                        (df_nat["processing_stage"] > 0) & 
                        (df_nat["reference_mineral"] == rf)
                    ]

                    df_reg = df_reg[
                        (df_reg["scenario"] == sc_regional) & 
                        (df_reg["processing_stage"] > 0) & 
                        (df_reg["reference_mineral"] == rf)
                    ]

                    if df_nat.empty or df_reg.empty:
                        print(f"Missing data for {rf}, {year}, {constraint_type}. Skipping...")
                        continue

                    # Aggregate national
                    nat_agg = df_nat.copy()
                    nat_agg[tons_column] *= multiply_factor
                    nat_agg = nat_agg.groupby(["iso3", "processing_type"])[tons_column].sum().unstack(fill_value=0)

                    # Aggregate regional
                    reg_agg = df_reg.copy()
                    reg_agg[tons_column] *= multiply_factor
                    reg_agg = reg_agg.groupby(["iso3", "processing_type"])[tons_column].sum().unstack(fill_value=0)

                    # Align both dataframes on iso3 and processing types
                    reg_agg, nat_agg = reg_agg.align(nat_agg, fill_value=0)

                    # Compute delta
                    delta = reg_agg - nat_agg

                    # Stack to long format
                    delta_long = delta.stack().reset_index()
                    delta_long.columns = ["iso3", "processing_type", "value"]
                    delta_long["reference_mineral"] = rf
                    delta_long["year"] = year
                    delta_long["constraint_type"] = constraint_type

                    differences_long.append(delta_long)

        # Combine all minerals and save or use directly
        differences_df = pd.concat(differences_long, ignore_index=True)
        differences_df.to_csv(f"{output_data_path}/country_mineral_year_differences.csv", index=False)

        mineral_order = ["cobalt", "copper", "graphite", "lithium", "manganese", "nickel"]
        mineral_short = ["Co", "Cu", "Gr", "Li", "Mn", "Ni"]
        mineral_colors = ["#fdae61", "#f46d43", "#66c2a5", "#c2a5cf", "#fee08b", "#3288bd"]

        for constraint in ["unconstrained", "constrained"]:
            fig = plot_mineral_differences_all_minerals(
                                                        differences_df, 
                                                        constraint_label=constraint, 
                                                        mineral_colors=mineral_colors,
                                                        mineral_order=mineral_order
                                                    )
            save_fig(os.path.join(figures, f"ALL_minerals_{constraint}_production_differences_2030_2040.png"))
            plt.close(fig)
        for constraint in ["unconstrained", "constrained"]:
            differences_df_selected = differences_df[
                                                    ((differences_df['year'] == "2030") & (differences_df['processing_type'] == "Early refining")) |
                                                    ((differences_df['year'] == "2040") & (differences_df['processing_type'] == "Precursor related product"))
                                                    ]
            fig = plot_mineral_differences_all_minerals(
                                                        differences_df_selected, 
                                                        constraint_label=constraint, 
                                                        mineral_colors=mineral_colors,
                                                        mineral_order=mineral_order
                                                    )
            save_fig(os.path.join(figures, f"ALL_minerals_{constraint}_production_differences_2030_EARLY_2040_PRECURSOR.png"))
            plt.close(fig)

    """Delta country and region constrained minus unconstrained: Country and regional comparisons for 2030 and 2040 separately in subplots: annual production
    FOR PAPER - all differences METAL CONTENT in two subplots (one per year). For slides: just edit fontsize in function.
    """
    make_plot = False
    if make_plot:
        multiply_factor = 1.0e-3
        results_file = os.path.join(output_data_path, "result_summaries", "combined_energy_transport_totals_by_stage.xlsx")
        
        region_levels = ["country", "region"]
        # unconstrained, constrained
        scenario_suffix_map = {
                            "country": "mid_min",
                            "region": "mid_max"
                        }

        years = ["2030", "2040"]
        reference_minerals = ["cobalt", "copper", "graphite", "lithium", "manganese", "nickel"]
        tons_column = "production_tonnes"
        all_properties = mineral_properties()
        results_by_key = {}
        differences_long = []

        for rf in reference_minerals:
            stages = all_properties[rf]["stage_labels"]

            for region_level in region_levels:
                scenario_suffix = scenario_suffix_map[region_level]

                for constraint_type in ["constrained", "unconstrained"]:
                    sheet_name = f"{region_level}_{constraint_type}"
                    print(region_level, constraint_type)

                    for year in years:
                        scenario_name = f"{year}_{scenario_suffix}_threshold_metal_tons"
                        print(f"→ {rf} | {year} | {sheet_name} | Scenario: {scenario_name}")

                        try:
                            df = pd.read_excel(results_file, sheet_name=sheet_name, index_col=[0, 1, 2, 3, 4]).reset_index()
                        except Exception as e:
                            print(f"Failed to read sheet {sheet_name}: {e}")
                            continue

                        df = df[
                                (df["scenario"] == scenario_name) &
                                (df["processing_stage"] == 0) &
                                (df["reference_mineral"] == rf)
                                ]

                        if df.empty:
                            print(f" No data for {rf}, {year}, {region_level}, {constraint_type}")
                            continue

                        df[tons_column] *= multiply_factor
                        df_agg = df.groupby(["iso3", "processing_type"])[tons_column].sum().unstack(fill_value=0)

                        key = (rf, year, region_level, constraint_type)
                        results_by_key[key] = df_agg

        # Compute constrained - unconstrained deltas
        for rf in reference_minerals:
            for year in years:
                for region_level in region_levels:
                    key_con = (rf, year, region_level, "constrained")
                    key_unc = (rf, year, region_level, "unconstrained")

                    if key_con in results_by_key and key_unc in results_by_key:
                        con_df = results_by_key[key_con]
                        unc_df = results_by_key[key_unc]

                        con_df, unc_df = con_df.align(unc_df, fill_value=0)
                        delta = con_df - unc_df

                        delta_long = delta.stack().reset_index()
                        delta_long.columns = ["iso3", "processing_type", "value"]
                        delta_long["reference_mineral"] = rf
                        delta_long["year"] = year
                        delta_long["region_level"] = region_level

                        differences_long.append(delta_long)
                    else:
                        print(f" Missing keys for diff: {key_con} or {key_unc}")

        # Final result
        if differences_long:
            differences_df = pd.concat(differences_long, ignore_index=True)
            differences_df.to_csv(f"{output_data_path}/country_mineral_year_metal_cont_differences.csv", index=False)
            print(" Saved differences_df.")
        else:
            print(" No differences computed. Check input filters and scenario names.")
    if make_plot:
        # Plotting
        mineral_order = reference_minerals
        mineral_colors = ["#fdae61", "#f46d43", "#66c2a5", "#c2a5cf", "#fee08b", "#3288bd"]

        for region in region_levels:
            fig = plot_mineral_differences_all_minerals_env(
                                                        differences_df, 
                                                        region_level=region, 
                                                        mineral_colors=mineral_colors,
                                                        mineral_order=mineral_order
                                                    )
            save_fig(os.path.join(
                    figures, 
                    f"ALL_minerals_{region}_constrained_minus_unconstrained_metal_content_production_differences_2030_2040.png"
                ))
            plt.close(fig)

        # mineral_order = ["cobalt", "copper", "graphite", "lithium", "manganese", "nickel"]
        # mineral_short = ["Co", "Cu", "Gr", "Li", "Mn", "Ni"]
        # mineral_colors = ["#fdae61", "#f46d43", "#66c2a5", "#c2a5cf", "#fee08b", "#3288bd"]

        # for region in ["country", "region"]:
        #     fig = plot_mineral_differences_all_minerals(
        #                                                 differences_df, 
        #                                                 region_level=region, 
        #                                                 mineral_colors=mineral_colors,
        #                                                 mineral_order=mineral_order
        #                                             )
        #     save_fig(os.path.join(figures, f"ALL_minerals_{region}_constrained_minus_unconstrained_metal_content_production_differences_2030_2040.png"))
        #     plt.close(fig)

        # Plot selected stage: 2030 with Early refining, 2040 with Precursor related product
        # for region in ["country", "region"]:
        #     differences_df_selected = differences_df[
        #         ((differences_df['year'] == "2030") & (differences_df['processing_type'] == "Early refining")) |
        #         ((differences_df['year'] == "2040") & (differences_df['processing_type'] == "Precursor related product"))
        #     ]

        #     fig = plot_mineral_differences_all_minerals(
        #         differences_df_selected, 
        #         region_level=region, 
        #         mineral_colors=mineral_colors,
        #         mineral_order=mineral_order
        #     )
        #     save_fig(os.path.join(figures, f"ALL_minerals_{region}_metal_content_production_differences_2030_EARLY_2040_PRECURSOR.png"))
        #     plt.close(fig)
    
    """Delta country and region constrained minus unconstrained: Country and regional comparisons for 2030 and 2040 separately in subplots: annual production
    NEW: normalises differences by the unconstrained value
    FOR PAPER - all differences METAL CONTENT in two subplots (one per year). For slides: just edit fontsize in function.
    """
    make_plot = True
    if make_plot:
        multiply_factor = 1.0e-3
        results_file = os.path.join(output_data_path, "result_summaries", "combined_energy_transport_totals_by_stage.xlsx")
        
        region_levels = ["country", "region"]
        # unconstrained, constrained
        scenario_suffix_map = {
                            "country": "mid_min",
                            "region": "mid_max"
                        }

        years = ["2030", "2040"]
        reference_minerals = ["cobalt", "copper", "graphite", "lithium", "manganese", "nickel"]
        tons_column = "production_tonnes"
        all_properties = mineral_properties()
        results_by_key = {}
        differences_long = []

        for rf in reference_minerals:
            stages = all_properties[rf]["stage_labels"]

            for region_level in region_levels:
                scenario_suffix = scenario_suffix_map[region_level]

                for constraint_type in ["constrained", "unconstrained"]:
                    sheet_name = f"{region_level}_{constraint_type}"
                    print(region_level, constraint_type)

                    for year in years:
                        scenario_name = f"{year}_{scenario_suffix}_threshold_metal_tons"
                        print(f"→ {rf} | {year} | {sheet_name} | Scenario: {scenario_name}")

                        try:
                            df = pd.read_excel(results_file, sheet_name=sheet_name, index_col=[0, 1, 2, 3, 4]).reset_index()
                        except Exception as e:
                            print(f"Failed to read sheet {sheet_name}: {e}")
                            continue

                        df = df[
                                (df["scenario"] == scenario_name) &
                                (df["processing_stage"] == 0) &
                                (df["reference_mineral"] == rf)
                                ]

                        if df.empty:
                            print(f" No data for {rf}, {year}, {region_level}, {constraint_type}")
                            continue

                        df[tons_column] *= multiply_factor
                        df_agg = df.groupby(["iso3", "processing_type"])[tons_column].sum().unstack(fill_value=0)

                        key = (rf, year, region_level, constraint_type)
                        results_by_key[key] = df_agg

        # Compute constrained - unconstrained deltas
        for rf in reference_minerals:
            for year in years:
                for region_level in region_levels:
                    key_con = (rf, year, region_level, "constrained")
                    key_unc = (rf, year, region_level, "unconstrained")

                    if key_con in results_by_key and key_unc in results_by_key:
                        con_df = results_by_key[key_con]
                        unc_df = results_by_key[key_unc]

                        con_df, unc_df = con_df.align(unc_df, fill_value=0)
                        # relative difference
                        relative_diff = (con_df - unc_df) / unc_df * 100
                        relative_diff = relative_diff.replace([np.inf, -np.inf], np.nan).fillna(0)

                        relative_diff_long = relative_diff.stack().reset_index()
                        relative_diff_long.columns = ["iso3", "processing_type", "value"]
                        relative_diff_long["reference_mineral"] = rf
                        relative_diff_long["year"] = year
                        relative_diff_long["region_level"] = region_level

                        differences_long.append(relative_diff_long)

                    else:
                        print(f" Missing keys for diff: {key_con} or {key_unc}")

        # Final result
        if differences_long:
            differences_df = pd.concat(differences_long, ignore_index=True)
            differences_df.to_csv(f"{output_data_path}/country_mineral_year_metal_cont_relative_differences.csv", index=False)
            print(" Saved differences_df.")
        else:
            print(" No differences computed. Check input filters and scenario names.")
    if make_plot:
        # Plotting
        mineral_order = reference_minerals
        mineral_colors = ["#fdae61", "#f46d43", "#66c2a5", "#c2a5cf", "#fee08b", "#3288bd"]

        # for region in region_levels:
        #     fig = plot_mineral_differences_all_minerals_env_rel(
        #                                                 differences_df, 
        #                                                 region_level=region, 
        #                                                 mineral_colors=mineral_colors,
        #                                                 mineral_order=mineral_order
        #                                             )
        #     save_fig(os.path.join(
        #             figures, 
        #             f"ALL_minerals_{region}_constrained_minus_unconstrained_metal_content_relative_production_differences_2030_2040.png"
        #         ))
        #     plt.close(fig)
        fig = plot_mineral_heatmap_all_minerals_env_rel(
                                                    differences_df, 
                                                    region_level="country", 
                                                    mineral_order=["cobalt", "copper", "graphite", "lithium", "manganese", "nickel"]
                                                )

        save_fig(os.path.join(figures, "heatmap_ALL_minerals_constrained_minus_unconstrained_metal_content_relative_production_differences_2030_2040.png"))
        plt.close(fig)

    """
    Country and regional comparisons for 2030 and 2040 separately in subplots: annual production costs
    """
    make_plot = False
    if make_plot:
        multiply_factor = 1.0e-6
        results_file = os.path.join(output_data_path, "result_summaries", "energy_transport_totals_by_stage.xlsx")
        constraints = ["country_unconstrained", "region_unconstrained"]
        reference_minerals = ["cobalt","copper","graphite","lithium","manganese","nickel"]
        reference_minerals_short = ["Co","Cu", "Gr", "Li","Mn","Ni"]
        reference_mineral_colors = ["#fdae61","#f46d43","#66c2a5","#c2a5cf","#fee08b","#3288bd"]
        scenarios = [
            "2030_mid_min_threshold_metal_tons",
            "2030_mid_max_threshold_metal_tons",
            "2040_mid_min_threshold_metal_tons",
            "2040_mid_max_threshold_metal_tons"
        ]
        tons_column = "production_cost_usd"
        index_cols = ["reference_mineral", "iso3"]
        all_properties = mineral_properties()

        for rf in reference_minerals:
            dfs = []  # To store DataFrames for each scenario and constraint
            stages = all_properties[rf]["stage_labels"]
            stage_colors = all_properties[rf]["stage_label_colors"]

            for scenario in scenarios:
                for cs in constraints:
                    df = pd.read_excel(results_file, sheet_name=cs, index_col=[0, 1, 2, 3, 4])
                    df = df.reset_index()

                    # Filter by scenario, processing_stage, and mineral
                    filtered_df = df[(df["scenario"] == scenario) & 
                                    (df["processing_stage"] > 0) & 
                                    (df["reference_mineral"] == rf)]

                    if filtered_df.empty:
                        print(f"No data available for scenario '{scenario}' and constraint '{cs}'. Skipping...")
                        continue

                    m_df = pd.DataFrame(sorted(filtered_df["iso3"].unique()), columns=["iso3"])

                    # Aggregate and process stage-wise data
                    for st in stages:
                        s_df = filtered_df[filtered_df["processing_type"] == st]
                        if not s_df.empty:
                            s_df[tons_column] = multiply_factor * s_df[tons_column]
                            s_df = s_df.groupby(["iso3", "processing_type"])[tons_column].sum().reset_index()
                            s_df.rename(columns={tons_column: f"{st}"}, inplace=True)
                            m_df = pd.merge(m_df, s_df[["iso3", f"{st}"]], how="left", on="iso3").fillna(0)
                        else:
                            m_df[f"{st}"] = 0

                    dfs.append(m_df.set_index("iso3"))

            # Compute the differences for 2030 and 2040
            delta_df = []
            if len(dfs) >= 4:  # Ensure we have all the required DataFrames
                delta_df.append(dfs[1].subtract(dfs[0], fill_value=0))  # 2030 Regional - National
                delta_df.append(dfs[3].subtract(dfs[2], fill_value=0))  # 2040 Regional - National

            # Plot results for 2030 and 2040
            fig, axs = plt.subplots(1, 2, figsize=(19, 9), dpi=500, sharey=True)

            if not delta_df:
                print(f"No data differences to plot for {rf}. Skipping...")
                continue
            delta_df_2030 = pd.DataFrame(delta_df[0]) if delta_df else pd.DataFrame()
            delta_df_2040 = pd.DataFrame(delta_df[1]) if delta_df else pd.DataFrame()

            if delta_df_2030 is None or delta_df_2040 is None:
                print(f"Missing data for 2030 or 2040 for {rf}. Skipping...")
                continue
            
            if not delta_df_2030.empty:
                axs[0] = plot_clustered_stacked(
                    fig, axs[0], [delta_df_2030], stage_colors,
                    labels=["2030 difference"],
                    ylabel="Difference in annual production costs (million USD)",
                    title=f"{rf.title()} 2030 - Regional minus National (Unconstrained)",
                    df_style = 'multiple', 
                    orientation="horizontal",
                    scenario_labels = False,
                    shift_bars = True
                )

            if not delta_df_2040.empty:
                axs[1] = plot_clustered_stacked(
                    fig, axs[1], [delta_df_2040], stage_colors,
                    labels=["2040 difference"],
                    ylabel="Difference in annual production costs (million USD)",
                    title=f"{rf.title()} 2040 - Regional minus National (Unconstrained)",
                    df_style = 'multiple', 
                    orientation="horizontal",
                    scenario_labels = False,
                    shift_bars = True
                )

            # Configure gridlines
            for ax in axs:
                ax.grid(False, axis="y")
                ax.grid(which="major", axis="x", linestyle="-", zorder=0)
                ax.axvline(x=0, color="black", linewidth=1, zorder=3)


            plt.tight_layout()
            save_fig(os.path.join(figures, f"{rf}_MN_MR_costs_comparisons_2030_2040.png"))
            plt.close()

    """
    Country and regional comparisons for 2030 and 2040 separately in subplots: annual production costs
    """
    make_plot = False 
    if make_plot:
        multiply_factor = 1.0e-6 # make 1 for those w per tonne
        units = 'million USD per tonne'
        results_file = os.path.join(output_data_path, "result_summaries", "combined_transport_totals_by_stage.xlsx")
        constraints = ["country_unconstrained", "region_unconstrained"]
        reference_minerals = ["cobalt","copper","graphite","lithium","manganese","nickel"]
        reference_minerals_short = ["Co","Cu", "Gr", "Li","Mn","Ni"]
        reference_mineral_colors = ["#fdae61","#f46d43","#66c2a5","#c2a5cf","#fee08b","#3288bd"]
        scenarios = [
                    "2030_mid_min_threshold_metal_tons",
                    "2030_mid_max_threshold_metal_tons",
                    "2040_mid_min_threshold_metal_tons",
                    "2040_mid_max_threshold_metal_tons"
                ]
        cost_column = ['export_transport_cost_usd_per_tonne', 
                       'import_transport_cost_usd_per_tonne', # import aren't as useful but plotting just in case
                        # 'total_gcosts_per_tons', #  general total transport costs per tonnes along roads and rail routes within a country
                        'energy_opex'
                    ]
        index_cols = ["reference_mineral", "iso3"]
        en_tr = ['transport',
                'transport',
                # 'transport',
                'energy']
        all_properties = mineral_properties()
        
        for rf in reference_minerals:
            dfs = []  # To store DataFrames for each scenario and constraint
            stages = all_properties[rf]["stage_labels"]
            stage_colors = all_properties[rf]["stage_label_colors"]

            for scenario in scenarios:
                for cs in constraints:
                    df = pd.read_excel(results_file, sheet_name=cs, index_col=[0, 1, 2, 3, 4])
                    df = df.reset_index()

                    # Filter by scenario, processing_stage, and mineral
                    filtered_df = df[(df["scenario"] == scenario) & 
                                    (df["processing_stage"] > 0) & 
                                    (df["reference_mineral"] == rf)]

                    if filtered_df.empty:
                        print(f"No data available for scenario '{scenario}' and constraint '{cs}'. Skipping...")
                        continue

                    m_df = pd.DataFrame(sorted(filtered_df["iso3"].unique()), columns=["iso3"])

                    for c, et in zip(cost_column, en_tr):
                        if c != 'energy_opex':
                            multiply_factor = 1
                            units = 'USD per tonne'
                        # Aggregate and process stage-wise data
                        for st in stages:
                            s_df = filtered_df[filtered_df["processing_type"] == st]
                            if not s_df.empty:
                                s_df[c] = multiply_factor * s_df[c]
                                s_df = s_df.groupby(["iso3", "processing_type"])[c].sum().reset_index()
                                s_df.rename(columns={c: f"{st}"}, inplace=True)
                                m_df = pd.merge(m_df, s_df[["iso3", f"{st}"]], how="left", on="iso3").fillna(0)
                            else:
                                m_df[f"{st}"] = 0
                    # shifting one indent up from here

                        dfs.append(m_df.set_index("iso3"))

                        # Compute the differences for 2030 and 2040
                        delta_df = []
                        if len(dfs) >= 4:  # Ensure we have all the required DataFrames
                            delta_df.append(dfs[1].subtract(dfs[0], fill_value=0))  # 2030 Regional - National
                            # print('National')
                            # print(dfs[0])
                            delta_df.append(dfs[3].subtract(dfs[2], fill_value=0))  # 2040 Regional - National

                        # Plot results for 2030 and 2040
                        fig, axs = plt.subplots(1, 2, figsize=(19, 9), dpi=500, sharey=True)

                        if not delta_df:
                            print(f"No data differences to plot for {rf}. Skipping...")
                            continue
                        delta_df_2030 = pd.DataFrame(delta_df[0]) if delta_df else pd.DataFrame()
                        delta_df_2040 = pd.DataFrame(delta_df[1]) if delta_df else pd.DataFrame()

                        if delta_df_2030 is None or delta_df_2040 is None:
                            print(f"Missing data for 2030 or 2040 for {rf}. Skipping...")
                            continue
                        
                        if not delta_df_2030.empty:
                            axs[0] = plot_clustered_stacked(
                                fig, axs[0], delta_df_2030, stage_colors,
                                labels=["2030 difference"],
                                ylabel=f"Difference in annual {et} costs ({units})",
                                title=f"{rf.title()} 2030 - Regional minus National (Unconstrained)",
                                df_style = 'single_yr',
                                scenario_labels = False,
                            )

                        if not delta_df_2040.empty:
                            axs[1] = plot_clustered_stacked(
                                fig, axs[1], delta_df_2040, stage_colors,
                                labels=["2040 difference"],
                                ylabel=f"Difference in annual {et} costs ({units})",
                                title=f"{rf.title()} 2040 - Regional minus National (Unconstrained)",
                                df_style = 'single_yr',
                                scenario_labels = False
                            )

                        # Configure gridlines
                        for ax in axs:
                            ax.grid(False, axis="x")
                            ax.grid(which="major", axis="y", linestyle="-", zorder=0)
                            ax.axhline(y=0, color="black", linewidth=1, zorder=3)

                        plt.tight_layout()
                        save_fig(os.path.join(figures, f"{rf}_MN_MR_{et}_comparisons_2030_2040.png"))
                        plt.close()

    """
    Country and regional uncontrained comparisons for 2030 and 2040 separately in single plots:
    annual energy, production and transport costs, some unit costs and some total - TO DO SAVE TO EXCEL
    """
    make_plot = False 
    if make_plot:
        results_file = os.path.join(output_data_path, "result_summaries", "combined_energy_transport_totals_by_stage.xlsx")

        constraints = ["country_unconstrained", "region_unconstrained"]
        reference_minerals = ["cobalt","copper","graphite","lithium","manganese","nickel"]
        reference_minerals_short = ["Co","Cu", "Gr", "Li","Mn","Ni"]
        reference_mineral_colors = ["#fdae61","#f46d43","#66c2a5","#c2a5cf","#fee08b","#3288bd"]

        allowed_mineral_processing = {
                    "nickel": {
                        "processing_stage": [3, 5], # sulphate
                        "processing_type": ["Early refining", "Precursor related product"]
                    },
                    "copper": {
                        "processing_stage": [3, 5], # sulphate
                        "processing_type": ["Early refining", "Precursor related product"]
                    },
                    "cobalt": {
                        "processing_stage": [4.1, 5], # sulphate
                        "processing_type": ["Early refining", "Precursor related product"]
                    }
                    ,
                    "graphite": {
                        "processing_stage": [3, 4], # 
                        "processing_type": ["Early refining", "Precursor related product"]
                    }
                    ,
                    "manganese": {
                        "processing_stage": [3.1, 4.1], # 
                        "processing_type": ["Early refining", "Precursor related product"]
                    }
                    ,
                    "lithium": {
                        "processing_stage": [3, 4.2], # 
                        "processing_type": ["Early refining", "Precursor related product"] #
                    }
                }
        
        # Cost columns to plot
        cost_columns = [
            "export_transport_cost_usd_per_tonne",
            "import_transport_cost_usd_per_tonne",# import aren't as useful but plotting just in case
            "energy_opex", 
            "energy_opex_per_tonne",
            "production_cost_usd_per_tonne",
            "production_cost_usd",
            "transport_export_tonsCO2eq_pertonne",
            "water_usage_m3"
        ]
        en_tr = ["transport", 
                "transport", 
                "energy",
                "energy",
                "production",
                "production",
                "emissions",
                "water"
                ]
        short_col_name = [
                    "export transport cost",
                    "import transport cost",# import aren't as useful but plotting just in case
                    "energy OPEX",
                    "energy OPEX",
                    "production cost",
                    "production cost",
                    "export transport emissions",
                    "water use"
                        ]

        scenarios = [
            "2030_mid_min_threshold_metal_tons",
            "2030_mid_max_threshold_metal_tons",
            "2040_mid_min_threshold_metal_tons",
            "2040_mid_max_threshold_metal_tons"
        ]

        # 1) Define scenario→stage mapping
        scenario_stage_map = {
            "2030_mid_min_threshold_metal_tons": "Early refining",
            "2030_mid_max_threshold_metal_tons": "Early refining",
            "2040_mid_min_threshold_metal_tons": "Precursor related product",
            "2040_mid_max_threshold_metal_tons": "Precursor related product"
        }

        # If you have a function that returns stage labels/colors:
        all_properties = mineral_properties()

        for rf in reference_minerals:
            stages_full = all_properties[rf]["stage_labels"]
            stage_colors_full = all_properties[rf]["stage_label_colors"]

            # Build a dictionary: stage→color (for this mineral)
            stage_color_map = dict(zip(stages_full, stage_colors_full))

            for c, et, sh_c in zip(cost_columns, en_tr, short_col_name):
                print(c)
                if c in ['energy_opex', "production_cost_usd"]: # all those that are not unit costs
                            multiply_factor = 1.0e-6 
                            units = 'million USD per tonne'
                if c not in ['energy_opex', "production_cost_usd", 'transport_export_tonsCO2eq_pertonne', 'water_usage_m3']:# all those that are unit costs
                            multiply_factor = 1
                            units = 'USD per tonne'
                if c == "transport_export_tonsCO2eq_pertonne":
                    multiply_factor = 1#.0e-3
                    units = 'kilotonne CO2eq per tonne'
                if c == "water_usage_m3":
                    multiply_factor = 1.0e-6 
                    units = 'million m3'
            
                for cs in constraints:
                    print(cs)
                    # We'll accumulate data_2030 & data_2040
                    data_2030 = []
                    data_2040 = []
                    # Also keep track of colors for each scenario DataFrame
                    bar_colors_2030 = []
                    bar_colors_2040 = []
                    for scenario in scenarios:
                        # Determine year from scenario name
                        if "2030" in scenario:
                            scenario_year = 2030
                        elif "2040" in scenario:
                            scenario_year = 2040
                        else:
                            print(f"Skipping scenario '{scenario}' - cannot determine year.")
                            continue
                        
                        if "country" in cs:
                            cs_l = "country"
                        else:
                            cs_l = "region"
                        df = pd.read_excel(results_file, sheet_name=cs, index_col=[0,1,2,3,4])
                        df = df.reset_index()

                        filtered_df = df[
                            (df["scenario"] == scenario)
                            & (df["reference_mineral"] == rf)
                            # where is the cost column selection
                        ]

                        if rf in allowed_mineral_processing:
                            allowed_stages = allowed_mineral_processing[rf].get("processing_stage", [])
                            allowed_types = allowed_mineral_processing[rf].get("processing_type", [])
                            # Keep only rows that match both stage & type
                            filtered_df = filtered_df[
                                filtered_df["processing_stage"].isin(allowed_stages)
                                & filtered_df["processing_type"].isin(allowed_types)
                            ]
                        
                        if filtered_df.empty:
                            print(f"No data after dictionary filter for {rf}, scenario '{scenario}', constraint '{cs}'")
                            continue

                        # group by iso3 and sum up cost for whichever stage remains
                        m_df = pd.DataFrame(sorted(filtered_df["iso3"].unique()), columns=["iso3"])
                        s_df = filtered_df.copy()
                        s_df[c] = multiply_factor * s_df[c]

                        # group by iso3 and processing_type => pivot
                        grouped = s_df.groupby(["iso3", "processing_type"])[c].sum().reset_index()
                        pivoted = grouped.pivot(index="iso3", columns="processing_type", values=c).fillna(0)
                        pivoted.reset_index(inplace=True)

                        # Merge into m_df
                        m_df = pd.merge(m_df, pivoted, on="iso3", how="left").fillna(0)
                        m_df.set_index("iso3", inplace=True)
                        
                        if "2030" in scenario:
                            m_df = m_df[['Early refining']]
                            color_for_this_df = stage_color_map['Early refining']
                            data_2030.append(m_df)
                            bar_colors_2030.append(color_for_this_df)
                            # if cs_l == 'country':
                            #     color_for_this_df = '#fb724a'

                        if "2040" in scenario:
                            m_df = m_df[['Precursor related product']]
                            color_for_this_df = stage_color_map['Precursor related product']
                            data_2040.append(m_df)
                            bar_colors_2040.append(color_for_this_df)
                            # if cs_l == 'country':
                            #     color_for_this_df = '#7f4c00'
                           
                    
                    # moving one indent back again so that 2030 and 2040 data can be filled in the loops. Srote region/country separately?
                    fig, axs = plt.subplots(1, 2, figsize=(19, 9), dpi=500, sharey=True)

                    # LEFT: 2030
                    if data_2030:
                        plot_data_2030 = pd.concat(data_2030, axis=1)
                        axs[0] = plot_clustered_stacked(
                            fig, axs[0],
                            plot_data_2030,
                            bar_colors_2030,  
                            labels=[f"2030_{cs_l}"],
                            ylabel=f"Annual {sh_c} ({units})",
                            title=f"{rf.title()} 2030 - Unconstrained ({cs_l.capitalize()})", # \n  {sh_c} 
                            df_style='single_yr',
                            stacked = False,
                            shift_bars = False,
                            scenario_labels = False
                        )
                    else:
                        axs[0].set_title(f"{rf.title()} 2030 - Unconstrained No Data ({cs_l.capitalize()})")
                    
                    # RIGHT: 2040
                    if data_2040:
                        plot_data_2040 = pd.concat(data_2040, axis=1)
                        axs[1] = plot_clustered_stacked(
                            fig, axs[1],
                            plot_data_2040,
                            bar_colors_2040,
                            labels=[f"2040_{cs_l}"],
                            ylabel=f"Annual {sh_c} ({units})",
                            title=f"{rf.title()} 2040 - Unconstrained ({cs_l.capitalize()})",
                            df_style='single_yr',
                            stacked = False,
                            shift_bars = False,
                            scenario_labels = False
                        )
                    else:
                        axs[1].set_title(f"{rf.title()} 2040 - Unconstrained No Data ({cs_l.capitalize()})") #, fontsize=14

                    # Configure gridlines
                    for ax in axs:
                        ax.grid(which="major", axis="y", linestyle="-", zorder=0)
                        ax.grid(False, axis="x")

                    
                    plt.tight_layout()
                    # Save figure
                    filename = f"{rf}_{cs}_{c}_comparisons_2030_2040.png"
                    save_fig(os.path.join(figures, filename))
                    plt.close()

    """
    Country and regional uncontrained and constrained comparisons for 2030 and 2040 separately in plots for all minerals (only two stages) and countries:
    annual energy, production and transport tonnes and costs RUN AGAIN WHEN YOU'RE HAPPY WITH LABELS ON BARS
    """ 
    make_plot = False 
    if make_plot:
        results_file = os.path.join(output_data_path, "result_summaries", "combined_energy_transport_totals_by_stage.xlsx")

        constraints = ["country_unconstrained", "region_unconstrained",
                      "country_constrained", "region_constrained"]
        reference_minerals = ["cobalt","copper","graphite","lithium","manganese","nickel"]
        reference_minerals_short = ["Co","Cu", "Gr", "Li","Mn","Ni"]
        reference_mineral_colors = ["#fdae61","#f46d43","#66c2a5","#c2a5cf","#fee08b","#3288bd"]

        reference_mineral_colormap = dict(zip(reference_minerals, reference_mineral_colors))

        reference_mineral_namemap = dict(zip(reference_minerals, reference_minerals_short))
        reference_mineral_colormapshort = dict(zip(reference_minerals_short, reference_mineral_colors))

        allowed_mineral_processing = {
                    "nickel": {
                        "processing_stage": [3, 5], # sulphate
                        "processing_type": ["Early refining", "Precursor related product"]
                    },
                    "copper": {
                        "processing_stage": [3, 5], # sulphate
                        "processing_type": ["Early refining", "Precursor related product"]
                    },
                    "cobalt": {
                        "processing_stage": [4.1, 5], # sulphate
                        "processing_type": ["Early refining", "Precursor related product"]
                    }
                    ,
                    "graphite": {
                        "processing_stage": [3, 4], # 
                        "processing_type": ["Early refining", "Precursor related product"]
                    }
                    ,
                    "manganese": {
                        "processing_stage": [3.1, 4.1], # 
                        "processing_type": ["Early refining", "Precursor related product"]
                    }
                    ,
                    "lithium": {
                        "processing_stage": [3, 4.2], # 
                        "processing_type": ["Early refining", "Precursor related product"] #
                    }
                }
        
        # Cost columns to plot
        cost_columns = [
            # "export_transport_cost_usd_per_tonne", # removing unit cost ones as they're not useful shown together
            # "import_transport_cost_usd_per_tonne",# import aren't as useful but plotting just in case
            "energy_opex" ,
            'energy_investment_usd',
            # "production_cost_usd_per_tonne",
            "production_cost_usd",
            # "transport_export_tonsCO2eq_pertonne",
            "water_usage_m3",
            "production_tonnes"
        ]
        en_tr = [
            # "transport", 
            #     "transport", 
                "energy",
                "energy",
                # "production",
                "production",
                # "emissions",
                "water",
                "production"
                ]
        short_col_name = [
                    # "export transport cost",
                    # "import transport cost",# import aren't as useful but plotting just in case
                    "energy OPEX",
                    "energy investments" ,
                    # "production cost",
                    "production cost",
                    # "export transport emissions",
                    "water use",
                    "production"
                        ]

        scenarios = [
            "2030_mid_min_threshold_metal_tons",
            "2030_mid_max_threshold_metal_tons",
            "2040_mid_min_threshold_metal_tons",
            "2040_mid_max_threshold_metal_tons"
        ]

        # 1) Define scenario→stage mapping
        scenario_stage_map = {
            "2030_mid_min_threshold_metal_tons": "Early refining",
            "2030_mid_max_threshold_metal_tons": "Early refining",
            "2040_mid_min_threshold_metal_tons": "Precursor related product",
            "2040_mid_max_threshold_metal_tons": "Precursor related product"
        }

        # If you have a function that returns stage labels/colors:
        all_properties = mineral_properties()

        for c, et, sh_c in zip(cost_columns, en_tr, short_col_name):
            print(c)
            if c in ['energy_opex', "production_cost_usd", "energy_investment_usd"]:  # all those that are not unit costs
                        multiply_factor = 1.0e-6 
                        units = 'million USD'
            if c not in ['energy_opex', "production_cost_usd", 'transport_export_tonsCO2eq_pertonne', 'water_usage_m3']:# all those that are unit costs
                        multiply_factor = 1
                        units = 'USD per tonne'
            if c == "transport_export_tonsCO2eq_pertonne":
                multiply_factor = 1#.0e-3
                units = 'kilotonne CO2eq per tonne'
            if c == "water_usage_m3":
                multiply_factor = 1.0e-6 
                units = 'million m3'
            if c== 'production_tonnes':
                multiply_factor = 1.0e-6 
                units = 'million tonne'
        
            for cs in constraints:
                print(cs)
                # Figures with minerals on categorical axis; change row, column depending on horizontal or vertical
                # fig, axs = plt.subplots(1, 2, figsize=(15, 7), dpi=500, sharey=True) # vertical
                fig, axs = plt.subplots(2,1, figsize=(7,9), dpi=500, sharex=True) #horizontal
                # We'll accumulate data_2030 & data_2040
                data_2030 = pd.DataFrame()
                data_2040 = pd.DataFrame()

                for scenario in scenarios:
                    # Determine year from scenario name
                    if "2030" in scenario:
                        scenario_year = 2030
                    elif "2040" in scenario:
                        scenario_year = 2040
                    else:
                        print(f"Skipping scenario '{scenario}' - cannot determine year.")
                        continue
                    
                    if "country" in cs:
                        cs_l = "country"
                    else:
                        cs_l = "region"
                    df = pd.read_excel(results_file, sheet_name=cs, index_col=[0,1,2,3,4])
                    df = df.reset_index()

                    # Get unique countries
                    countries = df['iso3'].unique()
                    num_countries = len(countries)
                    # Get the colormap and sample distinct colors
                    colormap = plt.colormaps['tab20']  # Get the 'tab20' colormap
                    colors = colormap(np.linspace(0, 1, num_countries))  # Sample 'num_countries' colors

                    # Convert colors to hex format
                    country_colors = [plt.matplotlib.colors.rgb2hex(color) for color in colors]
                    country_color_map = dict(zip(countries, country_colors))

                    for rf in reference_minerals:
                        filtered_df = df[
                            (df["scenario"] == scenario)
                            & (df["reference_mineral"] == rf)
                        ]

                        if rf in allowed_mineral_processing:
                            allowed_stages = allowed_mineral_processing[rf].get("processing_stage", [])
                            allowed_types = allowed_mineral_processing[rf].get("processing_type", [])
                            # Keep only rows that match both stage & type
                            filtered_df = filtered_df[
                                filtered_df["processing_stage"].isin(allowed_stages)
                                & filtered_df["processing_type"].isin(allowed_types)
                            ]
                        
                        if filtered_df.empty:
                            print(f"No data after dictionary filter for {rf}, scenario '{scenario}', constraint '{cs}'")
                            continue

                        # group by iso3 and sum up cost for whichever stage remains
                        # m_df = pd.DataFrame(sorted(filtered_df["iso3"].unique()), columns=["iso3"])
                        s_df = filtered_df.copy()
                        s_df[c] = multiply_factor * s_df[c]

                        # group by iso3 and processing_type => pivot
                        grouped = s_df.groupby(['reference_mineral', "iso3", "processing_type"])[c].sum().reset_index()
                        # print(grouped)
                        
                        if "2030" in scenario:
                            m_df_2030 = grouped[grouped['processing_type']== 'Early refining']
                            data_2030 = pd.concat([data_2030, m_df_2030], ignore_index = True)

                        if "2040" in scenario:
                            m_df_2040 = grouped[grouped['processing_type']== 'Precursor related product']
                            data_2040 = pd.concat([data_2040, m_df_2040], ignore_index = True)
                cs_clean = cs.replace("_", " ").title()

                # Printing to compare why the 30-40 are the same
                # print("2030")
                # print(data_2030)
                # print("2040")
                # print(data_2040)
                data_2030['reference_mineral_short'] = data_2030['reference_mineral'].map(reference_mineral_namemap)
                data_2040['reference_mineral_short'] = data_2040['reference_mineral'].map(reference_mineral_namemap)

                data_2030.rename(columns = {'reference_mineral_short':'Mineral'}, inplace = True)
                data_2040.rename(columns = {'reference_mineral_short':'Mineral'}, inplace = True)
                
                # Comment or uncomment for vertical and horizontal below
                # axs[0] = plot_stacked_bar(data_2030, group_by='reference_mineral', stack_col='iso3', value_col=c,
                #                     orientation="vertical", ax=axs[0], colors=country_color_map, short_val_label = sh_c, 
                #                     units = units, grouped_categ = 'Country')
                # axs[0].set_title(f"2030 - Early refining- {cs_clean}")

                # axs[1] =plot_stacked_bar(data_2040, group_by='reference_mineral', stack_col='iso3', value_col=c,
                #                     orientation="vertical", ax=axs[1], colors=country_color_map, short_val_label = sh_c, 
                #                     units = units, grouped_categ = 'Country')
                # axs[1].set_title(f"2040 - Precursor related product - {cs_clean}")
                    
                # for ax in axs:
                #     ax.grid(which="major", axis="y", linestyle="-", zorder=0)
                #     ax.grid(False, axis="x")

                    
                # plt.tight_layout()
                # # Save figure
                # filename = f"{cs}_{c}_mineral_comparisons_2030_2040.png"
                # save_fig(os.path.join(figures, filename))
                # plt.close()

                # Horizontal

                axs[0] = plot_stacked_bar(data_2030, group_by='Mineral', stack_col='iso3', value_col=c,
                                    orientation="horizontal", ax=axs[0], colors=country_color_map, short_val_label = sh_c, 
                                    units = units, grouped_categ = 'Country')
                axs[0].set_title(f"2030 - Early refining- {cs_clean}")

                axs[1] =plot_stacked_bar(data_2040, group_by='Mineral', stack_col='iso3', value_col=c,
                                    orientation="horizontal", ax=axs[1], colors=country_color_map, short_val_label = sh_c, 
                                    units = units, grouped_categ = 'Country')
                axs[1].set_title(f"2040 - Precursor related product - {cs_clean}")
                
                axs[0].set_xlabel(f"{sh_c} ({units})")
                # axs[0].xaxis.set_label_coords(0.5, -0.1)  # Adjust position if needed
                axs[0].xaxis.label.set_visible(True)
                axs[0].tick_params(axis="x", which="both", labelsize=12, labelbottom=True)
                # axs[1].set_xlabel(f"{sh_c} ({units})")

                for ax in axs:
                    ax.grid(which="major", axis="x", linestyle="-", zorder=0)
                    ax.grid(False, axis="y")
                    
                plt.tight_layout()
                # Save figure
                filename = f"{cs}_{c}_mineral_comparisons_2030_2040H.png"
                save_fig(os.path.join(figures, filename))
                plt.close()

                # Figures with countries on categorical axis; change row, column depending on horizontal or vertical
                # fig, axs = plt.subplots(1, 2, figsize=(15, 7), dpi=500, sharey=True) # vertical
                fig, axs = plt.subplots(2,1, figsize=(7,13), dpi=500, sharex=True) #horizontal
                # We'll accumulate data_2030 & data_2040
                data_2030 = pd.DataFrame()
                data_2040 = pd.DataFrame()

                for scenario in scenarios:
                    # Determine year from scenario name
                    if "2030" in scenario:
                        scenario_year = 2030
                    elif "2040" in scenario:
                        scenario_year = 2040
                    else:
                        print(f"Skipping scenario '{scenario}' - cannot determine year.")
                        continue
                    
                    if "country" in cs:
                        cs_l = "country"
                    else:
                        cs_l = "region"
                    df = pd.read_excel(results_file, sheet_name=cs, index_col=[0,1,2,3,4])
                    df = df.reset_index()

                    for rf in reference_minerals:

                        filtered_df = df[
                            (df["scenario"] == scenario)
                            & (df["reference_mineral"] == rf)
                        ]

                        if rf in allowed_mineral_processing:
                            allowed_stages = allowed_mineral_processing[rf].get("processing_stage", [])
                            allowed_types = allowed_mineral_processing[rf].get("processing_type", [])
                            # Keep only rows that match both stage & type
                            filtered_df = filtered_df[
                                filtered_df["processing_stage"].isin(allowed_stages)
                                & filtered_df["processing_type"].isin(allowed_types)
                            ]
                        
                        if filtered_df.empty:
                            print(f"No data after dictionary filter for {rf}, scenario '{scenario}', constraint '{cs}'")
                            continue

                        # group by iso3 and sum up cost for whichever stage remains
                        # m_df = pd.DataFrame(sorted(filtered_df["iso3"].unique()), columns=["iso3"])
                        s_df = filtered_df.copy()
                        s_df[c] = multiply_factor * s_df[c]

                        # group by iso3 and processing_type => pivot
                        grouped = s_df.groupby(['reference_mineral', "iso3", "processing_type"])[c].sum().reset_index()
                        # print(grouped)
                        
                        if "2030" in scenario:
                            m_df_2030 = grouped[grouped['processing_type']== 'Early refining']
                            data_2030 = pd.concat([data_2030, m_df_2030], ignore_index = True)

                        if "2040" in scenario:
                            m_df_2040 = grouped[grouped['processing_type']== 'Precursor related product']
                            data_2040 = pd.concat([data_2040, m_df_2040], ignore_index = True)
                cs_clean = cs.replace("_", " ").title()
                data_2030['reference_mineral_short'] = data_2030['reference_mineral'].map(reference_mineral_namemap)
                data_2040['reference_mineral_short'] = data_2040['reference_mineral'].map(reference_mineral_namemap)

                data_2030.rename(columns = {'iso3':'Country'}, inplace = True)
                data_2040.rename(columns = {'iso3':'Country'}, inplace = True)

                # Comment or uncomment for vertical or horizontal below
                # axs[0] = plot_stacked_bar(data_2030, group_by='iso3', stack_col='reference_mineral', value_col=c,
                #                     orientation="vertical", ax=axs[0], colors= reference_mineral_colormap, short_val_label = sh_c, 
                #                     units = units, grouped_categ = 'Mineral')
                # axs[0].set_title(f"2030 - Early refining - {cs_clean}")

                # axs[1] =plot_stacked_bar(data_2040, group_by='iso3', stack_col='reference_mineral', value_col=c,
                #                     orientation="vertical", ax=axs[1], colors= reference_mineral_colormap, short_val_label = sh_c, 
                #                     units = units, grouped_categ = 'Mineral')
                # axs[1].set_title(f"2040 - 'Precursor related product' - {cs_clean}")
                    
                # for ax in axs:
                #     ax.grid(which="major", axis="y", linestyle="-", zorder=0)
                #     ax.grid(False, axis="x")

                    
                # plt.tight_layout()
                # # Save figure
                # filename = f"{cs}_{c}_country_comparisons_2030_2040.png"
                # save_fig(os.path.join(figures, filename))
                # plt.close()
                axs[0] = plot_stacked_bar(data_2030, group_by='Country', stack_col='reference_mineral_short', value_col=c,
                                    orientation="horizontal", ax=axs[0], colors= reference_mineral_colormapshort, short_val_label = sh_c, 
                                    units = units, grouped_categ = 'Mineral')
                axs[0].set_title(f"2030 - Early refining - {cs_clean}")

                axs[1] =plot_stacked_bar(data_2040, group_by='Country', stack_col='reference_mineral_short', value_col=c,
                                    orientation="horizontal", ax=axs[1], colors= reference_mineral_colormapshort, short_val_label = sh_c, 
                                    units = units, grouped_categ = 'Mineral')
                axs[1].set_title(f"2040 - Precursor related product - {cs_clean}")
                    
                axs[0].set_xlabel(f"{sh_c} ({units})")
                # axs[0].xaxis.set_label_coords(0.5, -0.1)  # Adjust position if needed
                axs[0].xaxis.label.set_visible(True)
                axs[0].tick_params(axis="x", which="both", labelsize=12, labelbottom=True)
                # axs[1].set_xlabel(f"{sh_c} ({units})")

                for ax in axs:
                    ax.grid(which="major", axis="x", linestyle="-", zorder=0)
                    ax.grid(False, axis="y")

                plt.tight_layout()
                # Save figure
                filename = f"{cs}_{c}_country_comparisons_2030_2040H.png"
                save_fig(os.path.join(figures, filename))
                plt.close()

    
    """
    Country and regional uncontrained and constrained comparisons for 2022, 2030 and 2040 separately in plots for all minerals (ALL stages) and countries:
    annual energy, production and transport costs RUN AGAIN WHEN YOU'RE HAPPY WITH LABELS ON BARS
    """ 
    make_plot = False 
    if make_plot:
        results_file = os.path.join(output_data_path, "result_summaries", "combined_energy_transport_totals_by_stage.xlsx")

        constraints = ["country_unconstrained", "region_unconstrained",
                      "country_constrained", "region_constrained"]
        figure_nrows = [3,2,2,2]
        reference_minerals = ["cobalt","copper","graphite","lithium","manganese","nickel"]
        reference_minerals_short = ["Co","Cu", "Gr", "Li","Mn","Ni"]
        reference_mineral_colors = ["#fdae61","#f46d43","#66c2a5","#c2a5cf","#fee08b","#3288bd"]

        reference_mineral_colormap = dict(zip(reference_minerals, reference_mineral_colors))

        reference_mineral_namemap = dict(zip(reference_minerals, reference_minerals_short))
        reference_mineral_colormapshort = dict(zip(reference_minerals_short, reference_mineral_colors))
        
        # Cost columns to plot
        cost_columns = [
            "energy_opex" ,
            'energy_investment_usd',
            "production_cost_usd",
            "water_usage_m3",
            "production_tonnes"
        ]
        en_tr = [
                "energy",
                "energy",
                "production",
                "water",
                "production"
                ]
        short_col_name = [
                    "energy OPEX",
                    "energy investments" ,
                    "production cost",
                    "water use",
                    "production"
                        ]

        scenarios = [
            "2022_baseline",
            "2030_mid_min_threshold_metal_tons",
            "2030_mid_max_threshold_metal_tons",
            "2040_mid_min_threshold_metal_tons",
            "2040_mid_max_threshold_metal_tons"
        ]

        # If you have a function that returns stage labels/colors:
        all_properties = mineral_properties()

        for c, et, sh_c in zip(cost_columns, en_tr, short_col_name):
            print(c)
            if c in ['energy_opex', "production_cost_usd", "energy_investment_usd"]:  # all those that are not unit costs
                multiply_factor = 1.0e-6 
                units = 'million USD'
            if c not in ['energy_opex', "production_cost_usd", "energy_investment_usd", 'transport_export_tonsCO2eq_pertonne', 'water_usage_m3']:# all those that are unit costs
                multiply_factor = 1
                units = 'USD per tonne'
            if c == "transport_export_tonsCO2eq_pertonne":
                multiply_factor = 1#.0e-3
                units = 'kilotonne CO2eq per tonne'
            if c == "water_usage_m3":
                multiply_factor = 1.0e-6 
                units = 'million m3'
            if c== 'production_tonnes':
                multiply_factor = 1.0e-6 
                units = 'million tonne'
        
            for cs, nr in zip(constraints, figure_nrows):
                print(cs)
                # Figures with minerals on categorical axis; change row, column depending on horizontal or vertical
                # fig, axs = plt.subplots(1, 2, figsize=(15, 7), dpi=500, sharey=True) # vertical
                fig, axs = plt.subplots(nr,1, figsize=(7,nr*5), dpi=500, sharex=True) #horizontal
                
                data_2022 = pd.DataFrame()
                data_2030 = pd.DataFrame()
                data_2040 = pd.DataFrame()

                for scenario in scenarios:
                    print(scenario)
                    if "2022" in scenario:
                        scenario_year = 2022
                    elif "2030" in scenario:
                        scenario_year = 2030
                    elif "2040" in scenario:
                        scenario_year = 2040
                    else:
                        print(f"Skipping scenario '{scenario}' - cannot determine year.")
                        continue
                    
                    if "country" in cs:
                        cs_l = "country"
                    else:
                        cs_l = "region"
                    df = pd.read_excel(results_file, sheet_name=cs, index_col=[0,1,2,3,4])
                    df = df.reset_index()

                    # Get unique countries
                    countries = sorted(df['iso3'].unique())
                    num_countries = len(countries)
                    # Get the colormap and sample distinct colors
                    colormap = plt.colormaps['tab20']  # Get the 'tab20' colormap
                    colors = colormap(np.linspace(0, 1, num_countries))  # Sample 'num_countries' colors

                    # Convert colors to hex format
                    country_colors = [plt.matplotlib.colors.rgb2hex(color) for color in colors]
                    country_color_map = dict(zip(countries, country_colors))

                    for rf in reference_minerals:
                        filtered_df = df[
                            (df["scenario"] == scenario)
                            & (df["reference_mineral"] == rf)
                            & (df["processing_stage"]>0 ) # all stages
                        ]
                        
                        if filtered_df.empty:
                            print(f"No data after dictionary filter for {rf}, scenario '{scenario}', constraint '{cs}'")
                            continue

                        # group by iso3 and sum up cost for whichever stage remains
                        # m_df = pd.DataFrame(sorted(filtered_df["iso3"].unique()), columns=["iso3"])
                        s_df = filtered_df.copy()
                        s_df[c] = multiply_factor * s_df[c]

                        # group by iso3 and processing_type => pivot
                        grouped = s_df.groupby(['reference_mineral', "iso3", "processing_type"])[c].sum().reset_index()
                        # print(grouped)

                        if "2022" in scenario:
                            m_df_2022 = grouped
                            data_2022 = pd.concat([data_2022, m_df_2022], ignore_index = True)
                        if "2030" in scenario:
                            m_df_2030 = grouped#[grouped['processing_type']== 'Early refining']
                            data_2030 = pd.concat([data_2030, m_df_2030], ignore_index = True)
                        if "2040" in scenario:
                            m_df_2040 = grouped#[grouped['processing_type']== 'Precursor related product']
                            data_2040 = pd.concat([data_2040, m_df_2040], ignore_index = True)
                cs_clean = cs.replace("_", " ").title()

                # Printing to compare why the 30-40 are the same
                # print("2030")
                # print(data_2030)
                # print("2040")
                # print(data_2040)
                if data_2022.empty:
                    print("No data for 2022")
                else:
                    data_2022['reference_mineral_short'] = data_2022['reference_mineral'].map(reference_mineral_namemap)
                    data_2022.rename(columns = {'reference_mineral_short':'Mineral'}, inplace = True)
                
                data_2030['reference_mineral_short'] = data_2030['reference_mineral'].map(reference_mineral_namemap)
                data_2040['reference_mineral_short'] = data_2040['reference_mineral'].map(reference_mineral_namemap)

                data_2030.rename(columns = {'reference_mineral_short':'Mineral'}, inplace = True)
                data_2040.rename(columns = {'reference_mineral_short':'Mineral'}, inplace = True)
                
                # Comment or uncomment for vertical and horizontal below
                # axs[0] = plot_stacked_bar(data_2030, group_by='reference_mineral', stack_col='iso3', value_col=c,
                #                     orientation="vertical", ax=axs[0], colors=country_color_map, short_val_label = sh_c, 
                #                     units = units, grouped_categ = 'Country')
                # axs[0].set_title(f"2030 - Early refining- {cs_clean}")

                # axs[1] =plot_stacked_bar(data_2040, group_by='reference_mineral', stack_col='iso3', value_col=c,
                #                     orientation="vertical", ax=axs[1], colors=country_color_map, short_val_label = sh_c, 
                #                     units = units, grouped_categ = 'Country')
                # axs[1].set_title(f"2040 - Precursor related product - {cs_clean}")
                    
                # for ax in axs:
                #     ax.grid(which="major", axis="y", linestyle="-", zorder=0)
                #     ax.grid(False, axis="x")

                    
                # plt.tight_layout()
                # # Save figure
                # filename = f"{cs}_{c}_mineral_comparisons_2030_2040.png"
                # save_fig(os.path.join(figures, filename))
                # plt.close()

                # Horizontal
                if data_2022.empty:
                    print("No data for 2022")
                    axs[0] = plot_stacked_bar(data_2030, group_by='Mineral', stack_col='iso3', value_col=c,
                                    orientation="horizontal", ax=axs[0], colors=country_color_map, short_val_label = sh_c, 
                                    units = units, grouped_categ = 'Country')
                    axs[0].set_title(f"2030 - All stages - {cs_clean}")

                    axs[1] =plot_stacked_bar(data_2040, group_by='Mineral', stack_col='iso3', value_col=c,
                                        orientation="horizontal", ax=axs[1], colors=country_color_map, short_val_label = sh_c, 
                                        units = units, grouped_categ = 'Country')
                    axs[1].set_title(f"2040 - All stages - {cs_clean}")
                if data_2022.empty == False:
                    axs[0] = plot_stacked_bar(data_2022, group_by='Mineral', stack_col='iso3', value_col=c,
                                        orientation="horizontal", ax=axs[0], colors=country_color_map, short_val_label = sh_c, 
                                        units = units, grouped_categ = 'Country')
                    axs[0].set_title(f"2022 - All stages - {cs_clean}")

                    axs[1] = plot_stacked_bar(data_2030, group_by='Mineral', stack_col='iso3', value_col=c,
                                        orientation="horizontal", ax=axs[1], colors=country_color_map, short_val_label = sh_c, 
                                        units = units, grouped_categ = 'Country')
                    axs[1].set_title(f"2030 - All stages - {cs_clean}")

                    axs[2] =plot_stacked_bar(data_2040, group_by='Mineral', stack_col='iso3', value_col=c,
                                        orientation="horizontal", ax=axs[2], colors=country_color_map, short_val_label = sh_c, 
                                        units = units, grouped_categ = 'Country')
                    axs[2].set_title(f"2040 - All stages - {cs_clean}")
                    
                axs[0].set_xlabel(f"{sh_c} ({units})")
                axs[0].xaxis.label.set_visible(True)
                axs[0].tick_params(axis="x", which="both", labelsize=12, labelbottom=True)

                axs[1].set_xlabel(f"{sh_c} ({units})")
                axs[1].xaxis.label.set_visible(True)
                axs[1].tick_params(axis="x", which="both", labelsize=12, labelbottom=True)

                for ax in axs:
                    ax.grid(which="major", axis="x", linestyle="-", zorder=0)
                    ax.grid(False, axis="y")

                    
                plt.tight_layout()
                # Save figure
                filename = f"{cs}_{c}_mineral_comparisons_2022_2030_2040H.png"
                save_fig(os.path.join(figures, filename))
                plt.close()

                # Figures with countries on categorical axis; change row, column depending on horizontal or vertical
                # fig, axs = plt.subplots(1, 2, figsize=(15, 7), dpi=500, sharey=True) # vertical
                fig, axs = plt.subplots(nr,1, figsize=(7,nr*5), dpi=500, sharex=True) #horizontal
                # We'll accumulate data_2030 & data_2040
                data_2022 = pd.DataFrame()
                data_2030 = pd.DataFrame()
                data_2040 = pd.DataFrame()

                for scenario in scenarios:
                    # Determine year from scenario name
                    if "2022" in scenario:
                        scenario_year = 2022
                    elif "2030" in scenario:
                        scenario_year = 2030
                    elif "2040" in scenario:
                        scenario_year = 2040
                    else:
                        print(f"Skipping scenario '{scenario}' - cannot determine year.")
                        continue
                    
                    if "country" in cs:
                        cs_l = "country"
                    else:
                        cs_l = "region"
                    df = pd.read_excel(results_file, sheet_name=cs, index_col=[0,1,2,3,4])
                    df = df.reset_index()

                    for rf in reference_minerals:

                        filtered_df = df[
                            (df["scenario"] == scenario)
                            & (df["reference_mineral"] == rf)
                            & (df["processing_stage"]>0 ) # all stages
                        ]

                        if filtered_df.empty:
                            print(f"No data after dictionary filter for {rf}, scenario '{scenario}', constraint '{cs}'")
                            continue

                        # group by iso3 and sum up cost for whichever stage remains
                        # m_df = pd.DataFrame(sorted(filtered_df["iso3"].unique()), columns=["iso3"])
                        s_df = filtered_df.copy()
                        s_df[c] = multiply_factor * s_df[c]

                        # group by iso3 and processing_type => pivot
                        grouped = s_df.groupby(['reference_mineral', "iso3", "processing_type"])[c].sum().reset_index()
                        # print(grouped)
                        
                        if "2022" in scenario:
                            m_df_2022 = grouped
                            data_2022 = pd.concat([data_2022, m_df_2022], ignore_index = True)
                        if "2030" in scenario:
                            m_df_2030 = grouped#[grouped['processing_type']== 'Early refining']
                            data_2030 = pd.concat([data_2030, m_df_2030], ignore_index = True)
                        if "2040" in scenario:
                            m_df_2040 = grouped#[grouped['processing_type']== 'Precursor related product']
                            data_2040 = pd.concat([data_2040, m_df_2040], ignore_index = True)
                cs_clean = cs.replace("_", " ").title()

                if data_2022.empty:
                    print("No data for 2022")
                else:
                    data_2022['reference_mineral_short'] = data_2022['reference_mineral'].map(reference_mineral_namemap)
                    data_2022.rename(columns = {'iso3':'Country'}, inplace = True)

                data_2030['reference_mineral_short'] = data_2030['reference_mineral'].map(reference_mineral_namemap)
                data_2040['reference_mineral_short'] = data_2040['reference_mineral'].map(reference_mineral_namemap)

                data_2030.rename(columns = {'iso3':'Country'}, inplace = True)
                data_2040.rename(columns = {'iso3':'Country'}, inplace = True)

                # Comment or uncomment for vertical or horizontal below
                # axs[0] = plot_stacked_bar(data_2030, group_by='iso3', stack_col='reference_mineral', value_col=c,
                #                     orientation="vertical", ax=axs[0], colors= reference_mineral_colormap, short_val_label = sh_c, 
                #                     units = units, grouped_categ = 'Mineral')
                # axs[0].set_title(f"2030 - Early refining - {cs_clean}")

                # axs[1] =plot_stacked_bar(data_2040, group_by='iso3', stack_col='reference_mineral', value_col=c,
                #                     orientation="vertical", ax=axs[1], colors= reference_mineral_colormap, short_val_label = sh_c, 
                #                     units = units, grouped_categ = 'Mineral')
                # axs[1].set_title(f"2040 - 'Precursor related product' - {cs_clean}")
                    
                # for ax in axs:
                #     ax.grid(which="major", axis="y", linestyle="-", zorder=0)
                #     ax.grid(False, axis="x")

                    
                # plt.tight_layout()
                # # Save figure
                # filename = f"{cs}_{c}_country_comparisons_2030_2040.png"
                # save_fig(os.path.join(figures, filename))
                # plt.close()
                if data_2022.empty:
                    print("No data for 2022")
                    axs[0] = plot_stacked_bar(data_2030, group_by='Country', stack_col='reference_mineral_short', value_col=c,
                                        orientation="horizontal", ax=axs[0], colors= reference_mineral_colormapshort, short_val_label = sh_c, 
                                        units = units, grouped_categ = 'Mineral')
                    axs[0].set_title(f"2030 - All stages - {cs_clean}")

                    axs[1] =plot_stacked_bar(data_2040, group_by='Country', stack_col='reference_mineral_short', value_col=c,
                                        orientation="horizontal", ax=axs[1], colors= reference_mineral_colormapshort, short_val_label = sh_c, 
                                        units = units, grouped_categ = 'Mineral')
                    axs[1].set_title(f"2040 - All stages - {cs_clean}")
                    
                if data_2022.empty == False:
                    axs[0] = plot_stacked_bar(data_2022, group_by='Country', stack_col='reference_mineral_short', value_col=c,
                                        orientation="horizontal", ax=axs[0], colors= reference_mineral_colormapshort, short_val_label = sh_c,
                                        units = units, grouped_categ = 'Mineral')   
                    axs[0].set_title(f"2022 - All stages - {cs_clean}")
                
                    axs[1] = plot_stacked_bar(data_2030, group_by='Country', stack_col='reference_mineral_short', value_col=c,
                                        orientation="horizontal", ax=axs[1], colors= reference_mineral_colormapshort, short_val_label = sh_c, 
                                        units = units, grouped_categ = 'Mineral')
                    axs[1].set_title(f"2030 - All stages - {cs_clean}")

                    axs[2] =plot_stacked_bar(data_2040, group_by='Country', stack_col='reference_mineral_short', value_col=c,
                                        orientation="horizontal", ax=axs[2], colors= reference_mineral_colormapshort, short_val_label = sh_c, 
                                        units = units, grouped_categ = 'Mineral')
                    axs[2].set_title(f"2040 - All stages - {cs_clean}")
                
                axs[0].set_xlabel(f"{sh_c} ({units})")
                axs[0].xaxis.label.set_visible(True)
                axs[0].tick_params(axis="x", which="both", labelsize=12, labelbottom=True)

                axs[1].set_xlabel(f"{sh_c} ({units})")
                axs[1].xaxis.label.set_visible(True)
                axs[1].tick_params(axis="x", which="both", labelsize=12, labelbottom=True)    
                for ax in axs:
                    ax.grid(which="major", axis="x", linestyle="-", zorder=0)
                    ax.grid(False, axis="y")

                    
                plt.tight_layout()
                # Save figure
                filename = f"{cs}_{c}_country_comparisons_2022_2030_2040H.png"
                save_fig(os.path.join(figures, filename))
                plt.close()
    
    # ------------------------------------------------------------------
    # Supply curves for each mineral and constraint: option to shade unit costs types to see how each variable influences the cost
    # ------------------------------------------------------------------
    make_plot = False 
    if make_plot:
        multiply_factor = 1   # no unit conversion here, costs already in USD per tonne
        units = 'USD per tonne'

        results_file = os.path.join(output_data_path, "result_summaries", "combined_energy_transport_totals_by_stage.xlsx")

        constraints = ["country_unconstrained", "region_unconstrained",
                       "country_constrained", "region_constrained"]
        reference_minerals = ["copper", "cobalt", "manganese", "lithium", "graphite", "nickel"]

        # allowed processing: only the desired processing stages and types will be used.
        allowed_mineral_processing = {
            "nickel": {
                "processing_stage": [1, 3, 5],
                "processing_type": ["Beneficiation", "Early refining", "Precursor related product"]
            },
            "copper": {
                "processing_stage": [1, 3, 5],
                "processing_type": ["Beneficiation", "Early refining", "Precursor related product"]
            },
            "cobalt": {
                "processing_stage": [1, 4.1, 5],
                "processing_type": ["Beneficiation", "Early refining", "Precursor related product"]
            },
            "graphite": {
                "processing_stage": [1, 3, 4],
                "processing_type": ["Beneficiation", "Early refining", "Precursor related product"]
            },
            "manganese": {
                "processing_stage": [1, 3.1, 4.1],
                "processing_type": ["Beneficiation", "Early refining", "Precursor related product"]
            },
            "lithium": {
                "processing_stage": [1, 3, 4.2],
                "processing_type": ["Beneficiation", "Early refining", "Precursor related product"]
            }
        }
            
        # Cost columns are used only to compute a unit cost per tonne.
        cost_columns = [
                        "export_transport_cost_usd_per_tonne",
                        "import_transport_cost_usd_per_tonne",
                        "production_cost_usd_per_tonne",
                        "energy_opex_per_tonne",
                        "energy_investment_usd_per_tonne"
                    ]
        tons_column = "production_tonnes_for_costs"# "production_tonnes"

        # For labeling purposes:
        en_tr = ["transport", "transport", "production", "energy"]
        short_col_name = ["export transport cost", "import transport cost", 
                            "production cost", "energy OPEX", "energy CAPEX"]

        # Scenarios that appear in the spreadsheet.
        scenarios = [
            "2022_baseline",
            "2030_mid_min_threshold_metal_tons",
            "2030_mid_max_threshold_metal_tons",
            "2040_mid_min_threshold_metal_tons",
            "2040_mid_max_threshold_metal_tons"
        ]

        all_properties = {}  
        for mineral in reference_minerals:
            all_properties[mineral] = {
                "stage_labels": ["Beneficiation", "Early refining", "Precursor related product"],
                "stage_label_colors": ["#a6cee3", "#1f78b4", "#b2df8a"]
            }

        # Now, for each mineral and constraint, we create a figure with one column and 3 rows.
        for rf in reference_minerals:
            # Get full stage labels and colors for the mineral.
            stages_full = all_properties[rf]["stage_labels"]
            stage_colors_full = all_properties[rf]["stage_label_colors"]
            # Build a dictionary from stage label to color.
            stage_color_map = dict(zip(stages_full, stage_colors_full))
            # The allowed processing types for this mineral come from the allowed_mineral_processing dict.
            allowed_types = allowed_mineral_processing.get(rf, {}).get("processing_type", [])
            
            for cs in constraints:
                print(f"Processing mineral: {rf}, constraint: {cs}")
                # Read the full data for this constraint.
                df = pd.read_excel(results_file, sheet_name=cs, index_col=[0,1,2,3,4]).reset_index()

                # Get unique countries
                countries = sorted(df['iso3'].unique())
                num_countries = len(countries)
                # Get the colormap and sample distinct colors
                colormap = plt.colormaps['tab20']  # Get the 'tab20' colormap
                colors = colormap(np.linspace(0, 1, num_countries))  # Sample 'num_countries' colors

                # Convert colors to hex format
                country_colors = [plt.matplotlib.colors.rgb2hex(color) for color in colors]
                country_color_map = dict(zip(countries, country_colors))

                for sc in scenarios:
                    # print(sc)
                    df_sc = df[df['scenario'] == sc]
                    if "2022" in sc:
                        year = '2022'
                    if "2030" in sc:
                        year = '2030'
                    if "2040" in sc:
                        year = '2040'

                    # Create the unit cost column as the sum over the cost columns.
                    df_sc['unit_cost_all_usd_tonne'] = df_sc[cost_columns].sum(axis=1)
                    # Filter for the current mineral 
                    df_sc = df_sc[df_sc["reference_mineral"] == rf]
                    if df_sc.empty:
                        print(f"No data for {rf} under {cs}. Skipping...")
                        continue

                    # Create a figure with one column and as many rows as allowed processing types.
                    n_types = len(allowed_types)
                    fig, axs = plt.subplots(n_types, 1, figsize=(11, 19), dpi=500, sharex=False)
                    # If there's only one subplot, wrap it in a list.
                    if n_types == 1:
                        axs = [axs]
                    
                    allowed_stages = allowed_mineral_processing.get(rf, {}).get("processing_stage", [])
                    allowed_types = allowed_mineral_processing.get(rf, {}).get("processing_type", [])

                    # For each allowed processing type, filter the data and create the supply curve.
                    for i, proc_type in enumerate(allowed_types):
                        # Filter data for the current processing type AND its allowed processing stages
                        df_type = df_sc[
                            (df_sc["processing_type"] == proc_type) &  # Specific processing type
                            (df_sc["processing_stage"].isin(allowed_stages))  # Allowed stages for this mineral
                        ]
                        if df_type.empty:
                            print(f"No data for {rf}, {cs}, processing type: {proc_type}. Skipping subplot.")
                            continue
                        unique_prices = df_type["price_usd_per_tonne"].dropna().astype(float).unique()
                        unique_prices = [p for p in unique_prices if p>0]
                        price_value = unique_prices[0] if len(unique_prices) > 0 else None
                        
                        if len(unique_prices) > 1:
                            print(f"Warning: Multiple prices found for {rf}, {cs}, {proc_type}. Using first value.")
                        price_value = unique_prices[0] if len(unique_prices) > 0 else None 
                        
                        # There's missing price values for minerals that do have production- I've told Raghav
                        grouped = df_type.groupby(["iso3", "processing_type"]).agg({
                                                    tons_column: "sum",  # Sum production
                                                    "unit_cost_all_usd_tonne": "sum"  # Sum unit costs
                                                }).reset_index()
                        
                        m_df = pd.merge(pd.DataFrame(sorted(df_type["iso3"].unique()), columns=["iso3"]), grouped, on="iso3", how="left").fillna(0)
                        m_df.set_index("iso3", inplace=True)
                        
                        # Plot the supply curve on the i-th subplot.
                        # The supply curve function will create a step plot where x is cumulative production and y is cost.
                        cs_clean = cs.replace("_", " ") 
                        axs[i] = plot_supply_curve_bars(m_df, tons_column, 'unit_cost_all_usd_tonne',
                                                     ax=axs[i], sort=True, color_map = country_color_map)
                        axs[i].set_title(f"{rf.title()} {year} - {cs_clean.capitalize()} - {proc_type}", fontsize=14)

                        # Add a horizontal dotted line for the price, if available
                        if price_value is not None:
                            axs[i].axhline(y=price_value, linestyle="dotted", color="red", linewidth=2)

                            # Determine the x-position for the label (rightmost production value)
                            max_production = m_df[tons_column].sum()  # Total production to position text
                            x_pos = max_production * 1.02  # Slightly beyond the last bar for clarity

                            # Place the label close to the dotted line
                            axs[i].text(x_pos, price_value, f"Price: {price_value:.2f} USD/t", 
                                        ha='left', va='center', fontsize=10, color="black", fontweight='bold')
                        axs[i].legend()  # Ensure the legend displays the price line
                    
                    # Save the figure with an appropriate filename.
                    filename = f"{rf}_{cs}_{year}_supply_curve.png"
                    # Ensure you have a defined 'figures' folder or path.
                    plt.tight_layout()
                    plt.savefig(os.path.join(figures, filename))
                    plt.close()

    # ------------------------------------------------------------------
    # Supply curves for 2022 Beneficiation, 2030 Early ref and 2040 Precursor-related: only stage costs
    # ------------------------------------------------------------------
    make_plot = False 
    if make_plot:
        multiply_factor = 1   # no unit conversion here, costs already in USD per tonne
        units = 'USD per tonne'

        results_file = os.path.join(output_data_path,  "all_data.xlsx")

        constraints = ["country_unconstrained", 
                      "region_unconstrained" #, "country_constrained", "region_constrained"
                       ]
        reference_minerals = ["copper", "cobalt", "manganese", "lithium", "graphite", "nickel"]

        # allowed processing: only the desired processing stages and types will be used.
        allowed_mineral_processing = {
            "nickel": {
                "processing_stage": [1, 3, 5],
                "processing_type": ["Beneficiation", "Early refining", "Precursor related product"],
                "processing_year": [2022, 2030, 2040]
            },
            "copper": {
                "processing_stage": [1, 3, 5],
                "processing_type": ["Beneficiation", "Early refining", "Precursor related product"],
                "processing_year": [2022, 2030, 2040]
            },
            "cobalt": {
                "processing_stage": [1, 4.1, 5],
                "processing_type": ["Beneficiation", "Early refining", "Precursor related product"],
                "processing_year": [2022, 2030, 2040]
            },
            "graphite": {
                "processing_stage": [1, 3, 4],
                "processing_type": ["Beneficiation", "Early refining", "Precursor related product"],
                "processing_year": [2022, 2030, 2040]
            },
            "manganese": {
                "processing_stage": [1, 3.1, 4.1],
                "processing_type": ["Beneficiation", "Early refining", "Precursor related product"],
                "processing_year": [2022, 2030, 2040]
            },
            "lithium": {
                "processing_stage": [1, 3, 4.2],
                "processing_type": ["Beneficiation", "Early refining", "Precursor related product"],
                "processing_year": [2022, 2030, 2040]
            }
        }
            
        # Cost columns are used only to compute a unit cost per tonne.
        cost_columns = [
                        "export_transport_cost_usd_per_tonne",
                        "import_transport_cost_usd_per_tonne",
                        "production_cost_usd_per_tonne",
                        "energy_opex_per_tonne",
                        "energy_investment_usd_per_tonne"
                    ]
        tons_column = "production_tonnes_for_costs"# "production_tonnes"

        # For labeling purposes:
        en_tr = ["transport", "transport", "production", "energy"]
        short_col_name = ["export transport cost", "import transport cost", 
                            "production cost", "energy OPEX", "energy CAPEX"]

        # Scenarios that appear in the spreadsheet.
        scenarios = [
            "2022_baseline",
            "2030_mid_min_threshold_metal_tons", # country unconstrained
            "2040_mid_min_threshold_metal_tons", # country unconstrained
        ]

        all_properties = {}  
        
        for mineral in reference_minerals:
            all_properties[mineral] = {
                "stage_labels": ["Beneficiation", "Early refining", "Precursor related product"],
                "stage_label_colors": ["#a6cee3", "#1f78b4", "#b2df8a"]
            }

        # Read the full data for this constraint.
        for constraint in constraints:
            df = pd.read_excel(results_file).reset_index()
            df = df[df['constraint'] == 'country_unconstrained']

            for rf in reference_minerals:
                print(rf)
                # Get full stage labels and colors for the mineral.
                stages_full = all_properties[rf]["stage_labels"]
                stage_colors_full = all_properties[rf]["stage_label_colors"]
                # Build a dictionary from stage label to color.
                stage_color_map = dict(zip(stages_full, stage_colors_full))
                # The allowed processing types for this mineral come from the allowed_mineral_processing dict.
                allowed_types = allowed_mineral_processing.get(rf, {}).get("processing_type", [])

                # Get unique countries
                countries = sorted(df['iso3'].unique())
                num_countries = len(countries)
                # Get the colormap and sample distinct colors
                colormap = plt.colormaps['tab20']  # Get the 'tab20' colormap
                colors = colormap(np.linspace(0, 1, num_countries))  # Sample 'num_countries' colors

                # Convert colors to hex format
                country_colors = [plt.matplotlib.colors.rgb2hex(color) for color in colors]
                country_color_map = dict(zip(countries, country_colors))
                plot_data = pd.DataFrame()
                price_values = {}
                for sc in scenarios:
                    df_sc = df[df['scenario'] == sc]
                    if "2022" in sc:
                        year = '2022'
                    if "2030" in sc:
                        year = '2030'
                    if "2040" in sc:
                        year = '2040'
                    if year not in price_values:
                        price_values[year] = {}  # Initialize empty dict for this year

                    # Create the unit cost column as the sum over the cost columns.
                    df_sc['unit_cost_all_usd_tonne'] = df_sc[cost_columns].sum(axis=1)
                    # Filter for the current mineral 
                    df_sc = df_sc[df_sc["reference_mineral"] == rf]
                    if df_sc.empty:
                        print(f"No data for {rf} under {cs}. Skipping...")
                        continue
                
                    allowed_stages = allowed_mineral_processing.get(rf, {}).get("processing_stage", [])
                    allowed_types = allowed_mineral_processing.get(rf, {}).get("processing_type", [])
                    allowed_years = allowed_mineral_processing.get(rf, {}).get("processing_year", [])
                    
                    # For each allowed processing type, filter the data and create the supply curve.
                    for proc_type, proc_year in zip(allowed_types, allowed_years):
                        # Filter data for the current processing type AND its allowed processing stages
                        df_type = df_sc[
                            (df_sc["processing_type"] == proc_type) &  # Specific processing type
                            (df_sc["processing_stage"].isin(allowed_stages))  # Allowed stages for this mineral
                            
                        ]
                        df_type_prices = df_sc[
                            (df_sc["processing_type"] == proc_type) &  # Specific processing type
                            (df_sc["processing_stage"].isin(allowed_stages)) & # Allowed stages for this mineral
                            (df_sc["year"].isin(allowed_years)) # Allowed years
                        ]
                        if df_type.empty:
                            print(f"No data for {rf}, processing type: {proc_type}. Skipping subplot.")
                            continue
                        if df_type_prices.empty:
                            price_values[year][proc_type] = 0  # Default value if no data
                            continue
                        unique_prices = df_type_prices["price_usd_per_tonne"].dropna().astype(float).unique()
                        unique_prices = [p for p in unique_prices if p>0]
                        
                        if len(unique_prices) > 1:
                            print(f"Warning: Multiple prices found for {rf}, {proc_type}. Using first value.")
                        price_values[year][proc_type] = unique_prices[0] if unique_prices else 0
                        
                        # There's missing price values for minerals that do have production- I've told Raghav
                        grouped = df_type.groupby(["iso3", "processing_type"]).agg({
                                                    tons_column: "sum",  # Sum production
                                                    "unit_cost_all_usd_tonne": "sum"  # Sum unit costs
                                                }).reset_index()
                        
                        m_df = pd.merge(pd.DataFrame(sorted(df_type["iso3"].unique()), columns=["iso3"]), grouped, on="iso3", how="left").fillna(0)
                        m_df.set_index("iso3", inplace=True)
                        m_df['year'] = year
                        # print(m_df)
                        plot_data = pd.concat([plot_data, m_df.reset_index()], ignore_index = True)
                
                # print("Final Price Values by Year:", price_values)

                allowed_types = allowed_mineral_processing.get(rf, {}).get("processing_type", [])
                allowed_years = allowed_mineral_processing.get(rf, {}).get("processing_year", [])
                final_price_list = []
                for y in ['2022', '2030', '2040']:  # Ensure year is an integer
                    if y in price_values:
                        # Select the price only for the specific allowed processing type for this year
                        final_price_list.append(price_values[y].get(allowed_types[allowed_years.index(int(y))], 0))
                    else:
                        final_price_list.append(0)  # Default if year missing
                # print("Final price list:", final_price_list)

                # Create a figure with one column and as many rows as allowed processing types.
                n_types = len(allowed_types)
                fig, axs = plt.subplots(n_types, 1, figsize=(11, 19), dpi=500, sharex=False)
                # If there's only one subplot, wrap it in a list.
                if n_types == 1:
                    axs = [axs]
                for i, (ptype, y, pricev) in enumerate(zip(allowed_types, allowed_years, final_price_list)):
                    axs[i] = plot_supply_curve_bars(plot_data[(plot_data['processing_type']== ptype) & (plot_data['year']== str(y))].set_index('iso3'), 
                                                    tons_column, 'unit_cost_all_usd_tonne',
                                                    ax=axs[i], sort=True, color_map = country_color_map)
                    axs[0].set_title(f"{rf.title()} 2022 - Beneficiation - Country Unconstrained", fontsize=14)
                    axs[1].set_title(f"{rf.title()} 2030 - Early refining - Country Unconstrained", fontsize=14)
                    axs[2].set_title(f"{rf.title()} 2040 - Precursor related product - Country Unconstrained", fontsize=14)

                    # Add a horizontal dotted line for the price, if available
                    if pricev is not None:
                        axs[i].axhline(y=pricev, linestyle="dotted", color="red", linewidth=2)

                        # Determine the x-position for the label (rightmost production value)
                        max_production = plot_data[plot_data['year']== y][tons_column].sum()  # Total production to position text
                        x_pos = max_production * 1.02  # Slightly beyond the last bar for clarity

                        # Place the label close to the dotted line
                        axs[i].text(x_pos, pricev, f"Price: {float(pricev):.0f} USD/t", 
                                    ha='left', va='center', fontsize=10, color="black", fontweight='bold')
                    axs[i].legend()  # Ensure the legend displays the price line
                
                # Save the figure with an appropriate filename.
                filename = f"{constraint}_{rf}_supply_curve_2022_2030_2040.png"
                # Ensure you have a defined 'figures' folder or path.
                plt.tight_layout()
                plt.savefig(os.path.join(figures, filename))
                plt.close()
    
    # ------------------------------------------------------------------
    # Supply curves for 2022 Beneficiation, 2030 Early ref and 2040 Precursor-related: WITH costs for earlier stages NO ratios THIS IS THE MAIN ONE
    # ------------------------------------------------------------------
    make_plot = False 
    if make_plot:
        multiply_factor = 1   # no unit conversion here, costs already in USD per tonne
        units = 'USD per tonne'

        results_file = os.path.join(output_data_path,  "all_data.xlsx")

        constraints = [
                        "country_unconstrained", 
                        "region_unconstrained" #, "country_constrained", "region_constrained"
                       ]
        reference_minerals = ["copper", "cobalt", "manganese", "lithium", "graphite", "nickel"]

        # allowed processing: only the desired processing stages and types will be used.
        allowed_mineral_processing = {
            "nickel": {
                "processing_stage": [1, 3,  5],
                "processing_type": ["Beneficiation", "Early refining",  "Precursor related product"],
                "processing_year": [2022, 2030, 2040],
            },
            "copper": {
                "processing_stage": [1, 3,  5],
                "processing_type": ["Beneficiation", "Early refining",  "Precursor related product"],
                "processing_year": [2022, 2030, 2040],
            },
            "cobalt": {
                "processing_stage": [1, 4.1, 5],
                "processing_type": ["Beneficiation", "Early refining", "Precursor related product"],
                "processing_year": [2022, 2030, 2040],
            },
            "graphite": {
                "processing_stage": [1, 3, 4],
                "processing_type": ["Beneficiation", "Early refining", "Precursor related product"],
                "processing_year": [2022, 2030, 2040],
            },
            "manganese": {
                "processing_stage": [1, 3.1, 4.1],
                "processing_type": ["Beneficiation", "Early refining", "Precursor related product"],
                "processing_year": [2022, 2030, 2040],
            },
            "lithium": {
                "processing_stage": [1, 3, 4.2],
                "processing_type": ["Beneficiation", "Early refining", "Precursor related product"],
                "processing_year": [2022, 2030, 2040],
            }
        }
            
        # Cost columns are used only to compute a unit cost per tonne.
        cost_columns = [
                        "export_transport_cost_usd_per_tonne",
                        "import_transport_cost_usd_per_tonne",
                        "production_cost_usd_per_tonne",
                        "energy_opex_per_tonne",
                        "energy_investment_usd_per_tonne"
                    ]
        tons_column = "production_tonnes_for_costs"# "production_tonnes"

        # For labeling purposes:
        en_tr = ["transport", "transport", "production", "energy"]
        short_col_name = ["export transport cost", "import transport cost", 
                            "production cost", "energy OPEX", "energy CAPEX"]

        scenarios_country_unconstrained = [
                                            "2022_baseline",
                                            "2030_mid_min_threshold_metal_tons",
                                            "2040_mid_min_threshold_metal_tons"
                                        ]

        scenarios_region_unconstrained = [
                                            "2022_baseline",
                                            "2030_mid_max_threshold_metal_tons",  # Use mid_max here
                                            "2040_mid_max_threshold_metal_tons"   # Use mid_max here
                                        ]

        all_properties = {}  
        
        for mineral in reference_minerals:
            all_properties[mineral] = {
                "stage_labels": ["Beneficiation", "Early refining", "Precursor related product"],
                "stage_label_colors": ["#a6cee3", "#1f78b4", "#b2df8a"]
            }

        for constraint in constraints:
            df = pd.read_excel(results_file).reset_index()
            df = df[df['constraint'] == constraint]

            # Get unique countries
            countries = sorted(df['iso3'].unique())
            num_countries = len(countries)
            # Get the colormap and sample distinct colors
            colormap = plt.colormaps['tab20']  # Get the 'tab20' colormap
            colors = colormap(np.linspace(0, 1, num_countries))  # Sample 'num_countries' colors

            # Convert colors to hex format
            country_colors = [plt.matplotlib.colors.rgb2hex(color) for color in colors]
            country_color_map = dict(zip(countries, country_colors))

            if constraint == "region_unconstrained":
                df_2022 = pd.read_excel(results_file).reset_index()
                df_2022 = df_2022[(df_2022['constraint'] == "country_unconstrained")&(df_2022['scenario'] == '2022_baseline')]
                df_2022['constraint'] = 'region_unconstrained'
                df = pd.concat([df, df_2022], ignore_index = True)
                df.drop(columns = 'index', inplace = True)
                print("merged data for 2022")
                print(df['year'].unique())
            
            if constraint == "country_unconstrained":
                scenarios = scenarios_country_unconstrained
            elif constraint == "region_unconstrained":
                scenarios = scenarios_region_unconstrained

            
            for rf in reference_minerals:
                plot_data = pd.DataFrame()
                print(rf)
                # Filter for the current mineral 
                df_rf = df[df["reference_mineral"] == rf]
                print(df_rf)
                
                # Get full stage labels and colors for the mineral.
                stages_full = all_properties[rf]["stage_labels"]
                stage_colors_full = all_properties[rf]["stage_label_colors"]
                # Build a dictionary from stage label to color.
                stage_color_map = dict(zip(stages_full, stage_colors_full))
                # The allowed processing types for this mineral come from the allowed_mineral_processing dict.
                allowed_types = allowed_mineral_processing.get(rf, {}).get("processing_type", [])
                
                price_values = {}
                for sc in scenarios:
                    df_sc = df_rf[df_rf['scenario'] == sc]
                    if "2022" in sc:
                        year = '2022'
                    elif "2030" in sc:
                        year = '2030'
                    elif "2040" in sc:
                        year = '2040'
                    if year not in price_values:
                        price_values[year] = {}  # Initialize empty dict for this year

                    # Create the unit cost column as the sum over the cost columns.
                    df_sc['unit_cost_all_usd_tonne'] = df_sc[cost_columns].sum(axis=1)

                    if df_sc.empty:
                        print(f"No data for {rf} under {constraint}. Skipping...")
                        continue
                
                    allowed_stages = allowed_mineral_processing.get(rf, {}).get("processing_stage", [])
                    allowed_types = allowed_mineral_processing.get(rf, {}).get("processing_type", [])
                    allowed_years = allowed_mineral_processing.get(rf, {}).get("processing_year", [])

                    print(f"Allowed years for {rf}: {allowed_years}")
                    print(f"Unique years in df_sc: {df_sc['year'].unique()}")
                    
                    # For each allowed processing type, filter the data and create the supply curve.
                    for proc_type, proc_year in zip(allowed_types, allowed_years):
                        # Filter data for the current processing type AND its allowed processing stages
                        df_type = df_sc[
                                (df_sc["processing_type"] == proc_type) &  # Specific processing type
                                (df_sc["processing_stage"].isin(allowed_stages))  # Allowed stages for this mineral
                                    ]
                        df_type_prices = df_sc[
                                    (df_sc["processing_type"] == proc_type) &  # Specific processing type
                                    (df_sc["processing_stage"].isin(allowed_stages)) & # Allowed stages for this mineral
                                    (df_sc["year"].isin(allowed_years)) # Allowed years
                                    ]
                        if df_type.empty:
                            print(f"No data for {rf}, processing type: {proc_type}. Skipping subplot.")
                            continue
                        if df_type_prices.empty:
                            price_values[year][proc_type] = 0  # Default value if no data
                            continue
                        unique_prices = df_type_prices["price_usd_per_tonne"].dropna().astype(float).unique()
                        unique_prices = [p for p in unique_prices if p>0]
                        
                        if len(unique_prices) > 1:
                            print(f"Warning: Multiple prices found for {rf}, {proc_type}. Using first value.")
                        price_values[year][proc_type] = unique_prices[0] if unique_prices else 0
                        
                        # There's missing price values for minerals that do have production- I've told Raghav
                        grouped = df_type.groupby(["iso3", "processing_stage", "processing_type"]).agg({
                                                    tons_column: "sum",  # Sum production
                                                    "unit_cost_all_usd_tonne": "sum"  # Sum unit costs
                                                }).reset_index()
                        
                        m_df = pd.merge(pd.DataFrame(sorted(df_type["iso3"].unique()), columns=["iso3"]), grouped, on="iso3", how="left").fillna(0)
                        m_df.set_index("iso3", inplace=True)
                        m_df['year'] = year
                        plot_data = pd.concat([plot_data, m_df.reset_index()], ignore_index = True)
                
                # print("Final Price Values by Year:", price_values)

                allowed_types = allowed_mineral_processing.get(rf, {}).get("processing_type", [])
                allowed_years = allowed_mineral_processing.get(rf, {}).get("processing_year", [])
                
                final_price_list = []
                for y in ['2022', '2030', '2040']:  # Ensure year is an integer
                    if y in price_values:
                        # Select the price only for the specific allowed processing type for this year
                        final_price_list.append(price_values[y].get(allowed_types[allowed_years.index(int(y))], 0))
                    else:
                        final_price_list.append(0)  # Default if year missing
                # print("Final price list:", final_price_list)

                # Create a figure with one column and as many rows as allowed processing types.
                n_types = len(allowed_types)
                fig, axs = plt.subplots(n_types, 1, figsize=(11, 19), dpi=500, sharex=False)
                # If there's only one subplot, wrap it in a list.
                if n_types == 1:
                    axs = [axs]
                
                processing_sums = {
                                    '2022': ["Beneficiation"],
                                    '2030': ["Beneficiation", "Early refining"],
                                    '2040': ["Beneficiation", "Early refining", "Precursor related product"],
                                }
                print(rf)
                                        
                # aggregating the 3 stages
                # if not plot_data.empty:
                plot_data_agg = plot_data.drop(columns = 'processing_stage').groupby(['iso3', 'year', 'processing_type']).agg({
                                            tons_column: "sum",  # Sum production
                                            "unit_cost_all_usd_tonne": "sum"
                                        }).reset_index()
                
                # Create a copy of the unit cost column for updating
                plot_data_agg["updated_unit_cost_usd_tonne"] = plot_data_agg["unit_cost_all_usd_tonne"]

                # Iterate over the defined years and update unit costs where necessary
                for year, stages in processing_sums.items():
                    df_year = plot_data_agg[plot_data_agg["year"] == year]
                    
                    for iso in df_year["iso3"].unique():
                        df_iso = df_year[df_year["iso3"] == iso]
                        cumulative_cost = 0
                        for stage in stages:
                            stage_cost = df_iso[df_iso["processing_type"] == stage]["updated_unit_cost_usd_tonne"].sum()
                            cumulative_cost += stage_cost
                            plot_data_agg.loc[(plot_data_agg["iso3"] == iso) & (plot_data_agg["year"] == year) & (plot_data_agg["processing_type"] == stage), 
                                                                        "updated_unit_cost_usd_tonne"] = cumulative_cost

                for i, (ptype, y, pricev) in enumerate(zip(allowed_types, allowed_years, final_price_list)):
                    # if not plot_data_agg[plot_data_agg['year']== str(y)].empty:
                    print(plot_data_agg[(plot_data_agg['year']== str(y))&(plot_data_agg['processing_type']== ptype)].set_index('iso3'))
                    axs[i] = plot_supply_curve_bars(plot_data_agg[(plot_data_agg['year']== str(y))&(plot_data_agg['processing_type']== ptype)].set_index('iso3'), 
                                                    tons_column, 'updated_unit_cost_usd_tonne',
                                                    ax=axs[i], sort=True, color_map = country_color_map)
                    axs[0].set_title(f"{rf.title()} 2022 - Beneficiation - {constraint.replace('_', ' ').title()}", fontsize=14)
                    axs[1].set_title(f"{rf.title()} 2030 - Early refining - {constraint.replace('_', ' ').title()}", fontsize=14)
                    axs[2].set_title(f"{rf.title()} 2040 - Precursor related product - {constraint.replace('_', ' ').title()}", fontsize=14)

                    # Add a horizontal dotted line for the price, if available
                    if pricev is not None:
                        axs[i].axhline(y=pricev, linestyle="dotted", color="red", linewidth=2)

                        # Determine the x-position for the label (rightmost production value)
                        max_production = plot_data_agg[plot_data_agg['year']== y][tons_column].sum()  # Total production to position text
                        x_pos = max_production * 1.02  # Slightly beyond the last bar for clarity

                        # Place the label close to the dotted line
                        axs[i].text(x_pos, pricev, f"Price: {float(pricev):.0f} USD/t", 
                                    ha='left', va='center', fontsize=10, color="black", fontweight='bold')
                    axs[i].legend()  # Ensure the legend displays the price line
                
                # Save the figure with an appropriate filename.
                filename = f"{constraint}_{rf}_supply_curve_sum_costs_2022_2030_2040.png"
                # Ensure you have a defined 'figures' folder or path.
                plt.tight_layout()
                plt.savefig(os.path.join(figures, filename))
                plt.close()
    
    # ------------------------------------------------------------------
    # Supply curves for 2022 Beneficiation, 2030 Early ref and 2040 Precursor-related: WITH costs for earlier stages and ratios: not quite right
    # THIS SHOULD BE USED IF IT'S FIXED WELL
    # ------------------------------------------------------------------
    make_plot = False 
    if make_plot:
        multiply_factor = 1   # no unit conversion here, costs already in USD per tonne
        units = 'USD per tonne'

        results_file = os.path.join(output_data_path,  "all_data.xlsx")

        constraints = ["country_unconstrained", 
                    #     "region_unconstrained", "country_constrained", "region_constrained"
                       ]
        reference_minerals = ["copper", "cobalt", "manganese", "lithium", "graphite", "nickel"]

        # allowed processing: only the desired processing stages and types will be used.
        allowed_mineral_processing = {
            "nickel": {
                "processing_stage": [1, 2, 3, 4, 5],
                "processing_type": ["Beneficiation", "Early refining", "Precursor related product", "Precursor related product"],
                "processing_year": [2022, 2030, 2030, 2040, 2040],
                "processing_ratios": [1/8.0707, 1, 1/1.07, 1, 1/0.399]
            },
            "copper": {
                "processing_stage": [1, 3, 4.3, 5],
                "processing_type": ["Beneficiation", "Early refining", "Precursor related product", "Precursor related product"],
                "processing_year": [2022, 2030, 2040, 2040],
                "processing_ratios": [1/53.29, 1/1.39, 1/0.815, 1/0.525]
            },
            "cobalt": {
                "processing_stage": [1, 4.1, 5],
                "processing_type": ["Beneficiation", "Early refining", "Precursor related product"],
                "processing_year": [2022, 2030, 2040],
                "processing_ratios": [1, 1/5, 1/0.6]
            },
            "graphite": {
                "processing_stage": [1, 3, 4],
                "processing_type": ["Beneficiation", "Early refining", "Precursor related product"],
                "processing_year": [2022, 2030, 2040],
                "processing_ratios": [1/9.59, 1/2.2, 1/1.13] # check the first against Raghav's table, could be 1.05
            },
            "manganese": {
                "processing_stage": [1, 3.1, 4.1],
                "processing_type": ["Beneficiation", "Early refining", "Precursor related product"],
                "processing_year": [2022, 2030, 2040],
                "processing_ratios": [1/0.613, 1/1.71, 1/0.906]
            },
            "lithium": {
                "processing_stage": [1, 3, 4.2],
                "processing_type": ["Beneficiation", "Early refining", "Precursor related product"],
                "processing_year": [2022, 2030, 2040],
                "processing_ratios": [1/16.919, 1/17.86, 1/1.62]
            }
        }
            
        # Cost columns are used only to compute a unit cost per tonne.
        cost_columns = [
                        "export_transport_cost_usd_per_tonne",
                        "import_transport_cost_usd_per_tonne",
                        "production_cost_usd_per_tonne",
                        "energy_opex_per_tonne",
                        "energy_investment_usd_per_tonne"
                    ]
        tons_column = "production_tonnes_for_costs"# "production_tonnes"

        # For labeling purposes:
        en_tr = ["transport", "transport", "production", "energy"]
        short_col_name = ["export transport cost", "import transport cost", 
                            "production cost", "energy OPEX", "energy CAPEX"]

        # Scenarios that appear in the spreadsheet.
        scenarios = [
            "2022_baseline",
            "2030_mid_min_threshold_metal_tons", # country unconstrained
            "2040_mid_min_threshold_metal_tons", # country unconstrained
        ]

        all_properties = {}  
        
        for mineral in reference_minerals:
            all_properties[mineral] = {
                "stage_labels": ["Beneficiation", "Early refining", "Precursor related product"],
                "stage_label_colors": ["#a6cee3", "#1f78b4", "#b2df8a"]
            }

        # Now, for each mineral and constraint, we create a figure with one column and 3 rows.
        for rf in reference_minerals:
            print(rf)
            # Get full stage labels and colors for the mineral.
            stages_full = all_properties[rf]["stage_labels"]
            stage_colors_full = all_properties[rf]["stage_label_colors"]
            # Build a dictionary from stage label to color.
            stage_color_map = dict(zip(stages_full, stage_colors_full))
            # The allowed processing types for this mineral come from the allowed_mineral_processing dict.
            allowed_types = allowed_mineral_processing.get(rf, {}).get("processing_type", [])
            
            # Read the full data for this constraint.
            df = pd.read_excel(results_file).reset_index()
            df = df[df['constraint'] == 'country_unconstrained']

            # Get unique countries
            countries = sorted(df['iso3'].unique())
            num_countries = len(countries)
            # Get the colormap and sample distinct colors
            colormap = plt.colormaps['tab20']  # Get the 'tab20' colormap
            colors = colormap(np.linspace(0, 1, num_countries))  # Sample 'num_countries' colors

            # Convert colors to hex format
            country_colors = [plt.matplotlib.colors.rgb2hex(color) for color in colors]
            country_color_map = dict(zip(countries, country_colors))
            plot_data = pd.DataFrame()
            price_values = {}
            for sc in scenarios:
                df_sc = df[df['scenario'] == sc]
                if "2022" in sc:
                    year = '2022'
                if "2030" in sc:
                    year = '2030'
                if "2040" in sc:
                    year = '2040'
                if year not in price_values:
                    price_values[year] = {}  # Initialize empty dict for this year

                # Create the unit cost column as the sum over the cost columns.
                df_sc['unit_cost_all_usd_tonne'] = df_sc[cost_columns].sum(axis=1)
                # Filter for the current mineral 
                df_sc = df_sc[df_sc["reference_mineral"] == rf]
                if df_sc.empty:
                    print(f"No data for {rf} under {cs}. Skipping...")
                    continue
              
                allowed_stages = allowed_mineral_processing.get(rf, {}).get("processing_stage", [])
                allowed_types = allowed_mineral_processing.get(rf, {}).get("processing_type", [])
                allowed_years = allowed_mineral_processing.get(rf, {}).get("processing_year", [])
                
                filtered = [
                        (s, y, t)
                        for s, y, t in zip(allowed_stages, allowed_years, allowed_types)
                        if not ((rf == "nickel" and s == 4) or (rf == "copper" and s == 4.3))
                    ]

                allowed_relevant_stages = [s for (s, y, t) in filtered]
                allowed_relevant_years  = [y for (s, y, t) in filtered]
                allowed_relevant_types  = [t for (s, y, t) in filtered]
                
                for proc_type, proc_year in zip(allowed_relevant_types, allowed_relevant_years):
                    df_type_prices = df_sc[
                                (df_sc["processing_type"] == proc_type) &  # Specific processing type
                                (df_sc["processing_stage"].isin(allowed_relevant_stages)) & # Allowed stages for this mineral
                                (df_sc["year"].isin(allowed_relevant_years)) # Allowed years
                                ]
                    if df_type_prices.empty:
                        price_values[year][proc_type] = 0  # Default value if no data
                        continue
                    unique_prices = df_type_prices["price_usd_per_tonne"].dropna().astype(float).unique()
                    unique_prices = [p for p in unique_prices if p>0]
                    
                    if len(unique_prices) > 1:
                        print(f"Warning: Multiple prices found for {rf}, {proc_type}. Using first value.")
                    price_values[year][proc_type] = unique_prices[0] if unique_prices else 0
                    
                # For each allowed processing type, filter the data and create the supply curve.
                for proc_type, proc_year in zip(allowed_types, allowed_years):
                    # Filter data for the current processing type AND its allowed processing stages
                    df_type = df_sc[
                            (df_sc["processing_type"] == proc_type) &  # Specific processing type
                            (df_sc["processing_stage"].isin(allowed_stages))  # Allowed stages for this mineral
                                ]
                    if df_type.empty:
                        print(f"No data for {rf}, processing type: {proc_type}. Skipping subplot.")
                        continue
                    
                    # There's missing price values for minerals that do have production- I've told Raghav
                    grouped = df_type.groupby(["iso3", "processing_stage", "processing_type"]).agg({
                                                tons_column: "sum",  # Sum production
                                                "unit_cost_all_usd_tonne": "sum"  # Sum unit costs
                                            }).reset_index()
                    
                    m_df = pd.merge(pd.DataFrame(sorted(df_type["iso3"].unique()), columns=["iso3"]), grouped, 
                                    on="iso3", how="left").fillna(0)
                    m_df.set_index("iso3", inplace=True)
                    m_df['year'] = year
                    plot_data = pd.concat([plot_data, m_df.reset_index()], ignore_index = True)
            
            # print("Final Price Values by Year:", price_values)

            allowed_types = allowed_mineral_processing.get(rf, {}).get("processing_type", [])
            allowed_years = allowed_mineral_processing.get(rf, {}).get("processing_year", [])
            allowed_ratios = allowed_mineral_processing.get(rf, {}).get("processing_ratios", []) 

            final_price_list = []
            for y in ['2022', '2030', '2040']:  # Ensure year is an integer
                if y in price_values:
                    # Select the price only for the specific allowed processing type for this year
                    final_price_list.append(price_values[y].get(allowed_types[allowed_years.index(int(y))], 0))
                else:
                    final_price_list.append(0)  # Default if year missing
            # print("Final price list:", final_price_list)

            # Create a figure with one column and as many rows as allowed processing types.
            n_types = len(allowed_types)
            fig, axs = plt.subplots(3, 1, figsize=(11, 19), dpi=500, sharex=False)
            # If there's only one subplot, wrap it in a list.
            if n_types == 1:
                axs = [axs]
            
            processing_sums = {
                                '2022': ["Beneficiation"],
                                '2030': ["Beneficiation", "Early refining"],
                                '2040': ["Beneficiation", "Early refining", "Precursor related product"],
                            }
            print(rf)
            
            # to do: need to multiply by intensities 
            # Function to map processing ratios
            def get_processing_ratio(row, rf):
                allowed_ratios = allowed_mineral_processing.get(rf, {})
                for stage, ptype, year, ratio in zip(allowed_ratios.get("processing_stage", []), 
                                                    allowed_ratios.get("processing_type", []), 
                                                    allowed_ratios.get("processing_year", []), 
                                                    allowed_ratios.get("processing_ratios", [])):
                    if row["processing_stage"] == stage and row["processing_type"] == ptype and row["year"] == str(year):
                        return ratio
                return None

            # Apply processing ratios for a reference mineral
            plot_data["ratio"] = plot_data.apply(lambda row: get_processing_ratio(row, rf), axis=1)
            #  multiply ratio and unit cost only when calculating the updated unit cost, i.e. when aggregating the 3 stages
            plot_data["adjusted_unit_cost_usd_tonne"] = plot_data.apply(
                                        lambda row: row["unit_cost_all_usd_tonne"] * row["ratio"]
                                        if row["processing_type"] != "Beneficiation" and pd.notnull(row["ratio"]) else row["unit_cost_all_usd_tonne"], axis=1
                                    )
            # print(plot_data[plot_data['year'] == '2022'])
            # aggregating the 3 stages
            plot_data_agg = plot_data.drop(columns = 'processing_stage').groupby(['iso3', 'year', 'processing_type']).agg({
                                        tons_column: "sum",  # Sum production
                                        "adjusted_unit_cost_usd_tonne": "sum",  # Sum unit costs
                                        "unit_cost_all_usd_tonne": "sum"
                                    }).reset_index()
            # print(plot_data_agg)
            # Create a copy of the unit cost column for updating
            plot_data_agg["updated_unit_cost_usd_tonne"] = plot_data_agg["adjusted_unit_cost_usd_tonne"]
            # print(plot_data_agg)

            # Iterate over the defined years and update unit costs where necessary
            for year, stages in processing_sums.items():
                df_year = plot_data_agg[plot_data_agg["year"] == year]
                
                for iso in df_year["iso3"].unique():
                    df_iso = df_year[df_year["iso3"] == iso]
                    cumulative_cost = 0
                    for stage in stages:
                        stage_cost = df_iso[df_iso["processing_type"] == stage]["adjusted_unit_cost_usd_tonne"].sum()
                        cumulative_cost += stage_cost
                        plot_data_agg.loc[(plot_data_agg["iso3"] == iso) & (plot_data_agg["year"] == year) & (plot_data_agg["processing_type"] == stage), 
                                                                    "updated_unit_cost_usd_tonne"] = cumulative_cost

            # Preserve original column names and values
            plot_data_agg["total_unit_cost_usd_tonne"] = plot_data_agg["updated_unit_cost_usd_tonne"]
            plot_data_agg.drop(columns=["updated_unit_cost_usd_tonne"], inplace=True)
            
            # Save to a new Excel file
            output_file = os.path.join(output_data_path, "unit_cost_data.xlsx")  
            plot_data_agg.to_excel(output_file, index=False)

            for i, (ptype, y, pricev) in enumerate(zip(allowed_types, allowed_years, final_price_list)):
                print(plot_data_agg[(plot_data_agg['year']== str(y))&(plot_data_agg['processing_type']== ptype)].set_index('iso3'))
                axs[i] = plot_supply_curve_bars(plot_data_agg[(plot_data_agg['year']== str(y))&(plot_data_agg['processing_type']== ptype)].set_index('iso3'), 
                                                tons_column, 'total_unit_cost_usd_tonne',
                                                ax=axs[i], sort=True, color_map = country_color_map)
                axs[0].set_title(f"{rf.title()} 2022 - Beneficiation - Country Unconstrained", fontsize=14)
                axs[1].set_title(f"{rf.title()} 2030 - Early refining - Country Unconstrained", fontsize=14)
                axs[2].set_title(f"{rf.title()} 2040 - Precursor related product - Country Unconstrained", fontsize=14)

                # Add a horizontal dotted line for the price, if available
                if pricev is not None:
                    axs[i].axhline(y=pricev, linestyle="dotted", color="red", linewidth=2)

                    # Determine the x-position for the label (rightmost production value)
                    max_production = plot_data_agg[plot_data_agg['year']== y][tons_column].sum()  # Total production to position text
                    x_pos = max_production * 1.02  # Slightly beyond the last bar for clarity

                    # Place the label close to the dotted line
                    axs[i].text(x_pos, pricev, f"Price: {float(pricev):.0f} USD/t", 
                                ha='left', va='center', fontsize=10, color="black", fontweight='bold')
                axs[i].legend()  # Ensure the legend displays the price line
            
            # Save the figure with an appropriate filename.
            filename = f"{rf}_supply_curve_sum_costs_ratios_2022_2030_2040.png"
            # Ensure you have a defined 'figures' folder or path.
            plt.tight_layout()
            plt.savefig(os.path.join(figures, filename))
            plt.close()

    # ------------------------------------------------------------------
    # Revenue share of GDP per mineral and constraint: needs fixing some overlap of LABELS ON BARS AND SORTED VALUES OF BARS
    # ------------------------------------------------------------------
    make_plot = False 
    if make_plot:
        # multiply_factor = 1   # no unit conversion here, costs already in USD per tonne
        # units = 'USD per tonne'

        results_file = os.path.join(output_data_path, "result_summaries", "combined_transport_totals_by_stage.xlsx")

        constraints = ["country_unconstrained", "region_unconstrained",
                       "country_constrained", "region_constrained"]
        scenarios = [
            "2030_mid_min_threshold_metal_tons",
            "2030_mid_max_threshold_metal_tons",
            "2040_mid_min_threshold_metal_tons",
            "2040_mid_max_threshold_metal_tons"
        ]

        reference_minerals = ["cobalt","copper","graphite","lithium","manganese","nickel"]
        reference_minerals_short = ["Co","Cu", "Gr", "Li","Mn","Ni"]
        reference_mineral_colors = ["#fdae61","#f46d43","#66c2a5","#c2a5cf","#fee08b","#3288bd"]

        reference_mineral_colormap = dict(zip(reference_minerals, reference_mineral_colors))

        reference_mineral_namemap = dict(zip(reference_minerals, reference_minerals_short))
        reference_mineral_colormapshort = dict(zip(reference_minerals_short, reference_mineral_colors))

        # allowed processing: only the desired processing stages and types will be used.
        allowed_mineral_processing = {
            "nickel": {
                "processing_stage": [1, 3, 5],
                "processing_type": ["Beneficiation", "Early refining", "Precursor related product"]
            },
            "copper": {
                "processing_stage": [1, 3, 5],
                "processing_type": ["Beneficiation", "Early refining", "Precursor related product"]
            },
            "cobalt": {
                "processing_stage": [1, 4.1, 5],
                "processing_type": ["Beneficiation", "Early refining", "Precursor related product"]
            },
            "graphite": {
                "processing_stage": [1, 3, 4],
                "processing_type": ["Beneficiation", "Early refining", "Precursor related product"]
            },
            "manganese": {
                "processing_stage": [1, 3.1, 4.1],
                "processing_type": ["Beneficiation", "Early refining", "Precursor related product"]
            },
            "lithium": {
                "processing_stage": [1, 3, 4.2],
                "processing_type": ["Beneficiation", "Early refining", "Precursor related product"]
            }
        }
            
        # Cost columns are used only to compute a unit cost per tonne.
        cost_columns = [ "revenue_gdp_share_usd"  ]
        short_col_name = ['Revenue share of GDP']
        units = "%"

        for c, sh_c in zip(cost_columns, short_col_name):
            print(c)
                    
            for cs in constraints:
                print(cs)
                # Figures with minerals on categorical axis; change row, column depending on horizontal or vertical
                # fig, axs = plt.subplots(1, 2, figsize=(15, 7), dpi=500, sharey=True) # vertical
                fig, axs = plt.subplots(2,1, figsize=(8,12), dpi=500, sharex=True) #horizontal
                # We'll accumulate data_2030 & data_2040
                data_2030 = pd.DataFrame()
                data_2040 = pd.DataFrame()

                for scenario in scenarios:
                    # Determine year from scenario name
                    if "2030" in scenario:
                        scenario_year = 2030
                    elif "2040" in scenario:
                        scenario_year = 2040
                    else:
                        print(f"Skipping scenario '{scenario}' - cannot determine year.")
                        continue
                    
                    if "country" in cs:
                        cs_l = "country"
                    else:
                        cs_l = "region"
                    df = pd.read_excel(results_file, sheet_name=cs, index_col=[0,1,2,3,4])
                    df = df.reset_index()

                    # Adjust GDP for inflation in 2030 and 2040 - TO DO
                    # assumption of annual inflation of 2.5% in perpetuity, this means that you would need to scale-up 
                    # 2030 GDP by 1.22 (=1.025^8) and 2040 GDP by 1.56 (=1.025^18)
                    # Adjust GDP for inflation in 2030 and 2040
                    df["gdp_usd"] = df.apply(lambda row: row["gdp_usd"] * 1.22 if "2030" in row["scenario"] else 
                                                        row["gdp_usd"] * 1.56 if "2040" in row["scenario"] else row["gdp_usd"], axis=1)

                    df[c] = [x / y * 100 if y != 0 else 0 for x, y in zip(df["revenue_usd"], df["gdp_usd"])]

                    # Get unique countries
                    countries = sorted(df['iso3'].unique())
                    num_countries = len(countries)
                    # Get the colormap and sample distinct colors
                    colormap = plt.colormaps['tab20']  # Get the 'tab20' colormap
                    colors = colormap(np.linspace(0, 1, num_countries))  # Sample 'num_countries' colors

                    # Convert colors to hex format
                    country_colors = [plt.matplotlib.colors.rgb2hex(color) for color in colors]
                    country_color_map = dict(zip(countries, country_colors))

                    for rf in reference_minerals:
                        filtered_df = df[
                            (df["scenario"] == scenario)
                            & (df["reference_mineral"] == rf)
                        ]
                        # group by iso3 and processing_type => pivot
                        grouped = filtered_df.groupby(['reference_mineral', "iso3", "processing_type"])[c].sum().reset_index()
                        # print(grouped)
                        
                        if "2030" in scenario:
                            m_df_2030 = grouped#[grouped['processing_type']== 'Early refining']
                            data_2030 = pd.concat([data_2030, m_df_2030], ignore_index = True)

                        if "2040" in scenario:
                            m_df_2040 = grouped#[grouped['processing_type']== 'Precursor related product']
                            data_2040 = pd.concat([data_2040, m_df_2040], ignore_index = True)
                cs_clean = cs.replace("_", " ").title()

                data_2030['reference_mineral_short'] = data_2030['reference_mineral'].map(reference_mineral_namemap)
                data_2040['reference_mineral_short'] = data_2040['reference_mineral'].map(reference_mineral_namemap)
                data_2030.rename(columns = {'iso3':'Country'}, inplace = True)
                data_2040.rename(columns = {'iso3':'Country'}, inplace = True)

                # **Sort countries by total value (highest first)**
                sorted_countries_2030 = data_2030.groupby('Country')[c].sum().sort_values(ascending=False).index
                sorted_countries_2040 = data_2040.groupby('Country')[c].sum().sort_values(ascending=False).index

                # Reorder data based on sorted order
                data_2030 = data_2030.set_index('Country').loc[sorted_countries_2030].reset_index()
                data_2040 = data_2040.set_index('Country').loc[sorted_countries_2040].reset_index()

                axs[0] = plot_stacked_bar(data_2030, group_by='Country', stack_col='reference_mineral_short', value_col=c,
                                    orientation="horizontal", ax=axs[0], colors= reference_mineral_colormapshort, short_val_label = sh_c, 
                                    units = units, grouped_categ = 'Mineral')
                axs[0].set_title(f"2030 - All stages - {cs_clean}", fontsize=22)
                axs[0].tick_params(axis='both', labelsize=22)

                axs[1] =plot_stacked_bar(data_2040, group_by='Country', stack_col='reference_mineral_short', value_col=c,
                                    orientation="horizontal", ax=axs[1], colors= reference_mineral_colormapshort, short_val_label = sh_c, 
                                    units = units, grouped_categ = 'Mineral')
                axs[1].set_title(f"2040 - All stages - {cs_clean}", fontsize=22)
                axs[1].legend().remove()
                axs[1].tick_params(axis='both', labelsize=22)
                    
                for ax in axs:
                    ax.grid(which="major", axis="x", linestyle="-", zorder=0)
                    ax.grid(False, axis="y")
                    ax.set_ylabel("Country", fontsize=22)
                    ax.set_xlabel("Revenue share of GDP (%)", fontsize=22)

                axs[0].xaxis.label.set_visible(True)
                axs[0].tick_params(axis="x", which="both", labelsize=22, labelbottom=True)    
                plt.tight_layout()
                # Save figure
                filename = f"{cs}_{c}_country_comparisons_2030_2040H.png"
                save_fig(os.path.join(figures, filename))
                plt.close()

                            
    # ------------------------------------------------------------------
    # Value addition per mineral and constraint: needs fixing some overlap of LABELS ON BARS AND SORTED VALUES OF BARS
    # ------------------------------------------------------------------
    make_plot = False  # THE NUMBERS CHANGED A LOT FROM FIRST VERSION IN CURRENT REPORT, FIX REPORT
    if make_plot:
        results_file = os.path.join(output_data_path, "result_summaries", "combined_transport_totals_by_stage.xlsx")
        output_excel_path = os.path.join(output_data_path, "value_addition.xlsx")  

        # Constraints and scenarios to iterate over
        constraints = ["country_unconstrained", "region_unconstrained",
                    "country_constrained", "region_constrained"]
        scenarios = [
                    "2022_baseline",
                    "2030_mid_min_threshold_metal_tons",
                    "2030_mid_max_threshold_metal_tons",
                    "2040_mid_min_threshold_metal_tons",
                    "2040_mid_max_threshold_metal_tons"
                ]

        reference_minerals = ["cobalt","copper","graphite","lithium","manganese","nickel"]
        reference_minerals_short = ["Co","Cu", "Gr", "Li","Mn","Ni"]
        reference_mineral_colors = ["#fdae61","#f46d43","#66c2a5","#c2a5cf","#fee08b","#3288bd"]

        reference_mineral_colormap = dict(zip(reference_minerals, reference_mineral_colors))

        reference_mineral_namemap = dict(zip(reference_minerals, reference_minerals_short))
        reference_mineral_colormapshort = dict(zip(reference_minerals_short, reference_mineral_colors))

        # Cost column for value addition calculations
        cost_column = "value_addition_gdp_share_usd"

        # Dictionary to store results per constraint
        all_results = {}

        for cs in constraints:
            fig, axs = plt.subplots(2,1, figsize=(8,12), dpi=500, sharex=True) #horizontal
            data_2030 = pd.DataFrame()
            data_2040 = pd.DataFrame()
            print(f"Processing constraint: {cs}")

            data_list = []  # Store results for each constraint before saving

            for scenario in scenarios:
                print(f"  Processing scenario: {scenario}")

                # Determine year from scenario name
                scenario_year = 2030 if "2030" in scenario else 2040 if "2040" in scenario else 2020 if "2022" in scenario else None
                if not scenario_year:
                    print(f"  Skipping scenario '{scenario}' - invalid year detected.")
                    continue

                # Read data
                df = pd.read_excel(results_file, sheet_name=cs, index_col=[0, 1, 2, 3, 4]).reset_index()

                # Adjust GDP for inflation
                df["gdp_usd"] = df.apply(lambda row: row["gdp_usd"] * 1.22 if "2030" in row["scenario"] else 
                                                row["gdp_usd"] * 1.56 if "2040" in row["scenario"] else row["gdp_usd"], axis=1)

                # Process each mineral
                for rf in reference_minerals:
                    print(f"    Processing mineral: {rf}")

                    filtered_df = df[
                        (df["scenario"] == scenario) &
                        (df["reference_mineral"] == rf)
                    ]

                    # Ensure proper data types
                    filtered_df["processing_stage"] = filtered_df["processing_stage"].astype(float)
                    filtered_df["processing_stage"] = pd.Categorical(filtered_df["processing_stage"], ordered=True)

                    if filtered_df.empty:
                        print(f"  No data found for {rf} in scenario '{scenario}', constraint '{cs}'")
                        continue

                    # Group data
                    # print(filtered_df[["iso3", "reference_mineral", "processing_stage", "gdp_usd"]])
                    grouped_df = filtered_df[["iso3", "reference_mineral", "processing_stage", "production_cost_usd_per_tonne",
                                               "price_usd_per_tonne","production_tonnes", "gdp_usd" ]].groupby(["iso3", 
                                                                                                    "reference_mineral"]).apply(calculate_value_added)
                    grouped_df = grouped_df.reset_index(drop=True)

                    # Compute value addition percentage
                    grouped_df[cost_column] = grouped_df.apply(lambda row: (row["value_added"] / row["gdp_usd"]) * 100 if row["gdp_usd"] != 0 else 0, axis=1)

                    # Store relevant columns
                    grouped_df = grouped_df[["iso3", "reference_mineral", "processing_stage", "value_added", cost_column]]
                    grouped_df["Scenario"] = scenario
                    grouped_df["Constraint"] = cs
                    print(grouped_df.groupby(['reference_mineral','Constraint', 'processing_stage'])[cost_column].sum())

                    # Append results
                    data_list.append(grouped_df)

                    # group by iso3 and processing_type => pivot
                    grouped_all = grouped_df[['reference_mineral',
                                                 'iso3', cost_column]].groupby(['reference_mineral', "iso3"])[cost_column].sum().reset_index()
                    
                    if "2030" in scenario:
                        m_df_2030 = grouped_all
                        data_2030 = pd.concat([data_2030, m_df_2030], ignore_index = True)

                    if "2040" in scenario:
                        m_df_2040 = grouped_all
                        data_2040 = pd.concat([data_2040, m_df_2040], ignore_index = True)

            cs_clean = cs.replace("_", " ").title()

            data_2030['reference_mineral_short'] = data_2030['reference_mineral'].map(reference_mineral_namemap)
            data_2040['reference_mineral_short'] = data_2040['reference_mineral'].map(reference_mineral_namemap)

            data_2030.rename(columns = {'iso3':'Country'}, inplace = True)
            data_2040.rename(columns = {'iso3':'Country'}, inplace = True)

            # **Sort countries by total value (highest first)**
            sorted_countries_2030 = data_2030.groupby('Country')[cost_column].sum().sort_values(ascending=False).index
            sorted_countries_2040 = data_2040.groupby('Country')[cost_column].sum().sort_values(ascending=False).index

            # Reorder data based on sorted order
            data_2030 = data_2030.set_index('Country').loc[sorted_countries_2030].reset_index()
            data_2040 = data_2040.set_index('Country').loc[sorted_countries_2040].reset_index()

            axs[0] = plot_stacked_bar(data_2030, group_by='Country', stack_col='reference_mineral_short', value_col=cost_column,
                                orientation="horizontal", ax=axs[0], colors= reference_mineral_colormapshort, 
                                short_val_label = "Value addition share of GDP", units = "%", grouped_categ = 'Mineral')
            axs[0].set_title(f"2030 - All stages - {cs_clean}", fontsize=22)
            axs[0].tick_params(axis='both', labelsize=22)

            axs[1] = plot_stacked_bar(data_2040, group_by='Country', stack_col='reference_mineral_short', value_col=cost_column,
                                orientation="horizontal", ax=axs[1], colors= reference_mineral_colormapshort, 
                                short_val_label = "Value addition share of GDP", units = "%", grouped_categ = 'Mineral')
            axs[1].set_title(f"2040 - All stages - {cs_clean}", fontsize=22)
            axs[1].legend().remove()
            axs[1].tick_params(axis='both', labelsize=22)
                
            for ax in axs:
                ax.grid(which="major", axis="x", linestyle="-", zorder=0)
                ax.grid(False, axis="y")
                ax.set_xlabel("Value addition share of GDP (%)", fontsize=22)
                ax.set_ylabel("Country", fontsize=22)
            
            # axs[0].set_xlabel(f"Value addition share of GDP (%)", fontsize=22)
            axs[0].xaxis.label.set_visible(True)
            axs[0].tick_params(axis="x", which="both", labelsize=22, labelbottom=True)

            plt.tight_layout()
            # Save figure
            filename = f"{cs}_{cost_column}_country_comparisons_2030_2040H.png"
            save_fig(os.path.join(figures, filename))
            plt.close()

            # Save results for this constraint
            if data_list:
                all_results[cs] = pd.concat(data_list, ignore_index=True)

        # Save all results to an Excel file with multiple sheets
        if all_results:
            with pd.ExcelWriter(output_excel_path, engine="xlsxwriter") as writer:
                for sheet_name, df in all_results.items():
                    df.to_excel(writer, sheet_name=sheet_name, index=False)

            print(f" Data successfully saved to {output_excel_path}")
        else:
            print(" No valid data to save.")

    # ------------------------------------------------------------------
    # Weighted production based on ESG indicators: using z scores for production
    # ------------------------------------------------------------------ 
    make_plot = False
    if make_plot:
        # production data
        output_excel_path = os.path.join(output_data_path, "all_data.xlsx")  
        # Read data
        all_data = pd.read_excel(output_excel_path)

        all_data_clean = all_data[all_data['processing_stage'] > 0]

        production_by_stage = all_data_clean.groupby(['constraint','year','iso3', 'reference_mineral', 
                                                'processing_type', 'processing_stage'])['production_tonnes_for_costs'].sum().reset_index()
        # Z scores - removing country so that's where the z score is calculated
        production_by_stage["production_tonnes_for_costs_z"] = production_by_stage.groupby(
                                                                                        ['constraint', 'year', 'reference_mineral',
                                                                                        'processing_type', 'processing_stage']
                                                                                )["production_tonnes_for_costs"].transform(
                                                                            lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
                                                                        )

        country_scores = production_by_stage.groupby(["constraint","iso3", 'reference_mineral'])["production_tonnes_for_costs_z"].mean().reset_index()
                                                                                          
        data = {
                "Country": [
                            "DRC", "Burundi", "Uganda", "Mozambique", "Madagascar", "Angola", "Zimbabwe", "Kenya", "Tanzania", 
                            "Malawi", "Zambia", "South Africa", "Namibia", "Botswana"
                            ],
                "iso3": [
                        "COD", "BDI", "UGA", "MOZ", "MDG", "AGO", "ZWE", "KEN", "TZA", "MWI", "ZMB", "ZAF", "NAM", "BWA"
                        ],
                "Lebre": [
                        38.7, 41.5, 37.5, 41.7, 44.1, 45.3, 39.1, 47.6, 46.7, 46.0, 48.3, 47.6, 51.7, 56.7
                        ],
                "EPI-PPI-RGI": [
                                27.1, 26.6, 32.3, 32.2, 30.7, 29.8, 35.9, 33.7, 35.8, 37.5, 36.8, 44.5, 49.4, 55.9
                                ]
                }

        indicators_df = pd.DataFrame(data)

        merged_df = country_scores.merge(indicators_df[['iso3', 'EPI-PPI-RGI']], on="iso3", how="left")

        merged_df["Composite_Score"] = (merged_df["production_tonnes_for_costs_z"] + merged_df["EPI-PPI-RGI"]) / 2

        # ranked_df = merged_df.sort_values("Composite_Score", ascending=False)

        print(merged_df.head())

        reference_minerals = ["cobalt", "copper", "graphite", "lithium", "manganese", "nickel"]
        reference_mineral_colors = ["#fdae61", "#f46d43", "#66c2a5", "#c2a5cf", "#fee08b", "#3288bd"]

        # Map mineral names to colors
        color_map = dict(zip(reference_minerals, reference_mineral_colors))

        for cs in merged_df['constraint'].unique():
            merged_df_c = merged_df[merged_df['constraint'] == cs]

            # Set up a 3x3 subplot grid
            fig, axes = plt.subplots(2, 3, figsize=(15, 13), sharex=True, sharey=True)
            axes = axes.flatten()

            for idx, mineral in enumerate(reference_minerals):
                ax = axes[idx]
                subset = merged_df_c[merged_df_c['reference_mineral'] == mineral]

                if not subset.empty:
                    ax.scatter(
                        subset["EPI-PPI-RGI"],
                        subset["production_tonnes_for_costs_z"],
                        color=color_map[mineral],
                        alpha=0.9,
                        edgecolors='k',
                        s=80
                    )

                    # Add labels
                    for _, row in subset.iterrows():
                        ax.text(
                            row["EPI-PPI-RGI"] + 0.02,
                            row["production_tonnes_for_costs_z"] + 0.02,
                            row["iso3"],
                            fontsize=8
                        )

                ax.axhline(0, color='gray', linestyle='--', linewidth=0.7)
                ax.axvline(0, color='gray', linestyle='--', linewidth=0.7)

                ax.set_title(mineral.capitalize())
                ax.set_xlabel("EPI-PPI-RGI average")
                ax.set_ylabel("Standardised production")
                ax.grid(True)

            # Hide unused subplots (cells 6–8 for a 3x3 grid)
            for idx in range(len(reference_minerals), 6):
                fig.delaxes(axes[idx])

            # General title
            overall_title = cs.replace("_", " ").title() + " ESG Indicator vs. Production by Mineral"
            fig.suptitle(overall_title, fontsize=16)

            plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave space for suptitle
            filename = f"{cs}_subplots_ESG_and_production_by_mineral.png"
            save_fig(os.path.join(figures, filename))
            plt.close()
    
    # ------------------------------------------------------------------
    # Weighted production based on ESG indicators: using percentages of production and only relevant 2030, 2040 stages.
    # Needs fixing which stages are used per mineral so it works well. Not needed in the end for the ppt.
    # ------------------------------------------------------------------ 
    make_plot = False
    if make_plot:
        # production data
        output_excel_path = os.path.join(output_data_path, "all_data.xlsx")  

        allowed_mineral_processing = {
            "nickel": {
                "processing_stage": [1, 3,  5],
                "processing_type": ["Beneficiation", "Early refining",  "Precursor related product"],
                "processing_year": [2022, 2030, 2040],
            },
            "copper": {
                "processing_stage": [1, 3,  5],
                "processing_type": ["Beneficiation", "Early refining",  "Precursor related product"],
                "processing_year": [2022, 2030, 2040],
            },
            "cobalt": {
                "processing_stage": [1, 4.1, 5],
                "processing_type": ["Beneficiation", "Early refining", "Precursor related product"],
                "processing_year": [2022, 2030, 2040],
            },
            "graphite": {
                "processing_stage": [1, 3, 4],
                "processing_type": ["Beneficiation", "Early refining", "Precursor related product"],
                "processing_year": [2022, 2030, 2040],
            },
            "manganese": {
                "processing_stage": [1, 3.1, 4.1],
                "processing_type": ["Beneficiation", "Early refining", "Precursor related product"],
                "processing_year": [2022, 2030, 2040],
            },
            "lithium": {
                "processing_stage": [1, 3, 4.2],
                "processing_type": ["Beneficiation", "Early refining", "Precursor related product"],
                "processing_year": [2022, 2030, 2040],
            }
        }
        # Read data
        all_data = pd.read_excel(output_excel_path)

        all_data_clean = all_data[all_data['processing_stage'] > 0]

        production_by_stage = all_data_clean.groupby(['constraint','year','iso3', 'reference_mineral', 
                                                'processing_type', 'processing_stage'])['production_tonnes_for_costs'].sum().reset_index()
        
        # Apply filtering based on allowed_mineral_processing
        filtered_rows = []

        for mineral, filters in allowed_mineral_processing.items():
            allowed_stages = filters["processing_stage"]
            allowed_types = filters["processing_type"]
            allowed_years = filters["processing_year"]
            
            subset = production_by_stage[
                (production_by_stage['reference_mineral'] == mineral) &
                (production_by_stage['processing_stage'].isin(allowed_stages)) &
                (production_by_stage['processing_type'].isin(allowed_types)) &
                (production_by_stage['year'].isin(allowed_years))
            ]
            
            filtered_rows.append(subset)

        # Concatenate all filtered subsets
        production_by_stage = pd.concat(filtered_rows, ignore_index=True)
        
        # Z scores - removing country so that's where the z score is calculated
        production_by_stage["production_tonnes_for_costs_z"] = production_by_stage.groupby(
                                                                                        ['constraint', 'year', 'reference_mineral',
                                                                                        'processing_type', 'processing_stage']
                                                                                )["production_tonnes_for_costs"].transform(
                                                                            lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
                                                                        )
        # Calculate total production for each constraint, year, reference_mineral
        total_production = production_by_stage.groupby(
                                                        ['constraint', 'year', 'reference_mineral', 'processing_type', 'processing_stage']
                                                    )["production_tonnes_for_costs"].transform('sum')

        # Calculate percentage contribution per iso3 (country)
        production_by_stage["production_percentage"] = (
                                                            production_by_stage["production_tonnes_for_costs"] / total_production
                                                        ) * 100
        production_by_stage.to_csv(os.path.join(output_data_path,
                                 f"ESG_and_production_prodonly.csv"), index=False)

        country_scores = production_by_stage.groupby(
                                                    ["constraint", "year", "iso3", "reference_mineral",'processing_type', 'processing_stage']
                                                ).agg({
                                                    "production_tonnes_for_costs_z": "mean",
                                                    "production_percentage": "mean"  
                                                }).reset_index()                                                                          
        data = {
                "Country": [
                            "DRC", "Burundi", "Uganda", "Mozambique", "Madagascar", "Angola", "Zimbabwe", "Kenya", "Tanzania", 
                            "Malawi", "Zambia", "South Africa", "Namibia", "Botswana"
                            ],
                "iso3": [
                        "COD", "BDI", "UGA", "MOZ", "MDG", "AGO", "ZWE", "KEN", "TZA", "MWI", "ZMB", "ZAF", "NAM", "BWA"
                        ],
                "Lebre": [
                        38.7, 41.5, 37.5, 41.7, 44.1, 45.3, 39.1, 47.6, 46.7, 46.0, 48.3, 47.6, 51.7, 56.7
                        ],
                "EPI-PPI-RGI": [
                                27.1, 26.6, 32.3, 32.2, 30.7, 29.8, 35.9, 33.7, 35.8, 37.5, 36.8, 44.5, 49.4, 55.9
                                ]
                }

        indicators_df = pd.DataFrame(data)

        merged_df = country_scores.merge(indicators_df[['iso3', 'EPI-PPI-RGI']], on="iso3", how="left")

        merged_df["Composite_zScore"] = (merged_df["production_tonnes_for_costs_z"] + merged_df["EPI-PPI-RGI"]) / 2
        merged_df["Composite_prod_share_score"] = (merged_df["production_percentage"] + merged_df["EPI-PPI-RGI"]) / 2

        merged_df.to_csv(os.path.join(output_data_path,
                                 f"ESG_and_production.csv"), index=False)

        print(merged_df)

        reference_minerals = ["cobalt", "copper", "graphite", "lithium", "manganese", "nickel"]
        reference_mineral_colors = ["#fdae61", "#f46d43", "#66c2a5", "#c2a5cf", "#fee08b", "#3288bd"]

        # Map mineral names to colors
        color_map = dict(zip(reference_minerals, reference_mineral_colors))

        # for cs in merged_df['constraint'].unique():
        #     merged_df_c = merged_df[merged_df['constraint'] == cs]
        #     for y in merged_df['year'].unique():
        #         merged_df_y = merged_df_c[merged_df_c['year'] == y]                

        #         # Set up a 3x3 subplot grid
        #         fig, axes = plt.subplots(2, 3, figsize=(15, 13), sharex=True, sharey=True)
        #         axes = axes.flatten()

        #         for idx, mineral in enumerate(reference_minerals):
        #             ax = axes[idx]
        #             subset = merged_df_y[merged_df_y['reference_mineral'] == mineral]

        #             if not subset.empty:
        #                 ax.scatter(
        #                     subset["EPI-PPI-RGI"],
        #                     subset["production_tonnes_for_costs_z"],
        #                     color=color_map[mineral],
        #                     alpha=0.9,
        #                     edgecolors='k',
        #                     s=80
        #                 )

        #                 # Add labels
        #                 for _, row in subset.iterrows():
        #                     ax.text(
        #                         row["EPI-PPI-RGI"] + 0.02,
        #                         row["production_tonnes_for_costs_z"] + 0.02,
        #                         row["iso3"],
        #                         fontsize=8
        #                     )

        #             ax.axhline(0, color='gray', linestyle='--', linewidth=0.7)
        #             ax.axvline(0, color='gray', linestyle='--', linewidth=0.7)

        #             ax.set_title(mineral.capitalize())
        #             ax.set_xlabel("EPI-PPI-RGI average")
        #             ax.set_ylabel("Standardised production")
        #             ax.grid(True)

        #         # Hide unused subplots (cells 6–8 for a 3x3 grid)
        #         for idx in range(len(reference_minerals), 6):
        #             fig.delaxes(axes[idx])

        #         # General title
        #         overall_title = str(y) +" " + cs.replace("_", " ").title() + " ESG Indicator vs. Production by Mineral"
        #         fig.suptitle(overall_title, fontsize=16)

        #         plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave space for suptitle
        #         filename = f"{cs}_{y}_subplots_ESG_and_production_by_mineral_relevant_stages.png"
        #         save_fig(os.path.join(figures, filename))
        #         plt.close()

        # Plotting with percentages
        for cs in merged_df['constraint'].unique():
            merged_df_c = merged_df[merged_df['constraint'] == cs]
            
            for y in merged_df['year'].unique():
                    
                if y == 2022:     
                    merged_df_y = merged_df_c[(merged_df_c['year'] == y) & (merged_df_c['processing_type'] == "Beneficiation")]     
                if y == 2030:     
                    merged_df_y = merged_df_c[(merged_df_c['year'] == y) & (merged_df_c['processing_type'] == "Early refining")]  
                if y == 2040:     
                    merged_df_y = merged_df_c[(merged_df_c['year'] == y) & (merged_df_c['processing_type'] == "Precursor related product")]  

                # Set up a 2x3 subplot grid
                fig, axes = plt.subplots(6, 1, figsize=(6, 15), sharex=True, sharey=True)
                axes = axes.flatten()

                for idx, mineral in enumerate(reference_minerals):
                    ax = axes[idx]
                    subset = merged_df_y[merged_df_y['reference_mineral'] == mineral]

                    if not subset.empty:
                        ax.scatter(
                            subset["EPI-PPI-RGI"],
                            subset["production_percentage"],  
                            color=color_map[mineral],
                            alpha=0.9,
                            edgecolors='k',
                            s=100
                        )

                        # # Add labels
                        # for _, row in subset.iterrows():
                        #     ax.text(
                        #         row["EPI-PPI-RGI"] + 0.05,
                        #         row["production_percentage"] + 0.05,  
                        #         row["iso3"],
                        #         fontsize=12
                        #     )
                        # Only label top producers (> 5% share)
                        top_producers = subset[subset["production_percentage"] >= 5]

                        for _, row in top_producers.iterrows():
                            ax.text(
                                row["EPI-PPI-RGI"] + 0.5,  # Move label slightly further right
                                row["production_percentage"] + 0.5,
                                row["iso3"],
                                fontsize=12,  # <<< Bigger font for labeled countries
                                fontweight='bold'
                            )
                    ax.set_ylim(0, 100)
                    ax.set_xlim(0, 60)
                    
                    ax.axhline(0, color='gray', linestyle='--', linewidth=0.7)
                    ax.axvline(0, color='gray', linestyle='--', linewidth=0.7)

                    ax.set_title(mineral.capitalize(), fontsize=14)  # <<< Bigger subplot titles
                    ax.set_xlabel("EPI-PPI-RGI average", fontsize=14)  # <<< Bigger x-axis label
                    ax.set_ylabel("Production share (%)", fontsize=14)  # <<< Bigger y-axis label
                    ax.tick_params(axis='both', which='major', labelsize=12)  # <<< Bigger tick labels
                    ax.grid(True)

                # Hide unused subplots
                for idx in range(len(reference_minerals), 6):
                    fig.delaxes(axes[idx])

                overall_title = str(y) +" " + cs.replace("_", " ").title()
                fig.suptitle(overall_title, fontsize=18)

                plt.tight_layout(rect=[0, 0, 1, 0.96])
                filename = f"{cs}_{y}_subplots_ESG_and_percentage_production_by_mineral_relevant_stages.png"
                save_fig(os.path.join(figures, filename))
                plt.close()





if __name__ == '__main__':
    CONFIG = load_config()
    main(CONFIG)
