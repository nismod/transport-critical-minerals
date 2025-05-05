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
tqdm.pandas()

# mpl.style.use('ggplot')

mpl.rcParams["font.family"] = "DejaVu Sans"
mpl.rcParams["font.size"] = 12
mpl.rcParams["axes.spines.top"] = False
mpl.rcParams["axes.spines.right"] = False
mpl.rcParams["axes.facecolor"] = "#FFFFFF"
mpl.rcParams["grid.alpha"] = 0.3
mpl.rcParams["grid.linestyle"] = "--"
# Increase figure DPI globally
mpl.rcParams["figure.dpi"] = 120


def plot_clustered_stacked(fig,axe,
                            dfall,
                            bar_colors,
                            labels=None,
                            stacked = True,
                            ylabel="Y-values",
                            title="multiple stacked bar plot",
                            H="/",**kwargs):
    """ 
        Source: https://stackoverflow.com/questions/22787209/how-to-have-clusters-of-stacked-bars
        Given a list of dataframes, with identical columns and index, create a clustered stacked bar plot. 
        labels is a list of the names of the dataframe, used for the legend
        title is a string for the title of the plot
        H is the hatch used for identification of the different dataframe
    """

    """
        # create fake dataframes
        df1 = pd.DataFrame(np.random.rand(4, 5),
                           index=["A", "B", "C", "D"],
                           columns=["I", "J", "K", "L", "M"])
        df2 = pd.DataFrame(np.random.rand(4, 5),
                           index=["A", "B", "C", "D"],
                           columns=["I", "J", "K", "L", "M"])
        df3 = pd.DataFrame(np.random.rand(4, 5),
                           index=["A", "B", "C", "D"], 
                           columns=["I", "J", "K", "L", "M"])

        # Then, just call :
        plot_clustered_stacked([df1, df2, df3],["df1", "df2", "df3"])
    """

    n_df = len(dfall)
    n_col = len(dfall[0].columns) 
    n_ind = len(dfall[0].index)
    # axe = plt.subplot(111)

    for df in dfall : # for each data frame
        bar_colors = list(islice(cycle(bar_colors), None, n_col))
        # bar_colors = list(zip(bar_colors["stages"],bar_colors["stage_colors"]))
        axe = df.plot(kind="bar",
                      linewidth=1.0,
                      stacked=True,
                      ax=axe,
                      legend=False,
                      grid=False,
                      color = bar_colors,
                      edgecolor = 'white',
                      **kwargs)  # make bar plots

    h,l = axe.get_legend_handles_labels() # get the handles we want to modify
    for i in range(0, n_df * n_col, n_col): # len(h) = n_col * n_df
        for j, pa in enumerate(h[i:i+n_col]):
            for rect in pa.patches: # for each index
                rect.set_x(rect.get_x() + 1 / float(n_df + 1) * i / float(n_col))
                # rect.set_hatch(H * int(i / n_col)) #edited part 
                rect.set_width(1 / float(n_df + 1))

    # sec.set_xticks([5, 15, 25], labels=['\nOughts', '\nTeens', '\nTwenties'])
    if labels is not None:
        locs = []
        lbs = []
        for i in range(len(labels)):
            lb = labels[i]
            locs += list((np.arange(0, 2 * n_ind, 2) + (2*i-1)*(1/ float(n_df + 1))) / 2.)
            if i == 1:
                lbs += [f"{lb[:4]}\n\n{d}" for d in df.index.values.tolist()]
            else:
                lbs += [lb[:4]]*len(df.index)
        
        axe.set_xticks(locs)
        axe.set_xticklabels(lbs, rotation = 0,fontsize=14,fontweight="bold")
    
    # axe.set_xticks((np.arange(0, 2 * n_ind, 2) + 1 / float(n_df + 1)) / 2.)
    # axe.set_xticklabels(
    #                 [f"\n{d}" for d in df.index.values.tolist()], 
    #                 rotation = 0,fontsize=15,fontweight="bold"
    #                 )
    axe.set_xlabel('')
    axe.set_ylabel(ylabel,fontweight='bold',fontsize=15)
    axe.tick_params(axis='y',labelsize=15)
    axe.set_title(title,fontweight='bold',fontsize=18)
    ymin, ymax = axe.get_ylim()
    axe.set_ylim(ymin,1.2*ymax)

    legend_handles = []
    titles = ["$\\bf{Mineral \, processing \, stages}$","$\\bf Scenarios$"]
    # legend_handles.append(axe.plot([],[],
    #                                 color="none",
    #                                 label="$\\bf{Mineral \, processing \, stages}$")[0])
    for idx,(bc,bl) in enumerate(zip(bar_colors,l[:n_col])):
        legend_handles.append(mpatches.Patch(color=bc,
                                        label=bl))

    # l1 = axe.legend(h[:n_col], l[:n_col], loc="upper right",prop={'size':15,'weight':'bold'})
    # legend_handles.append(axe.plot([],[],
    #                                 color="none",
    #                                 label="$\\bf Scenarios$")[0])
    
    # for idx in range(len(labels)):
    #     legend_handles.append(mpatches.Patch(facecolor="#000000",edgecolor='white',
    #                                     label=labels[idx],hatch=H*idx))
    leg = axe.legend(
                handles=legend_handles, 
                fontsize=15, 
                loc='upper left',
                # bbox_to_anchor=(0.7,-0.1),
                ncols=3,
                title="$\\bf{Mineral \, processing \, stages}$",
                title_fontsize=15,
                frameon=False)

    # Move titles to the left 
    for item, label in zip(leg.legend_handles, leg.texts):
        if label._text in titles:
            width = item.get_window_extent(fig.canvas.get_renderer()).width
            label.set_ha('left')
            label.set_position((-10.0*width,0))
    return axe


def new_plot_clustered_stacked(fig,axe,
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
    titles = ["$\\bf{Mineral \\, processing \\, stages}$", "$\\bf Scenarios$"]
    legend_handles.append(axe.plot([],[], color="none", label="$\\bf{Mineral \\, processing \\, stages}$")[0])

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
    else:
        titles = ["$\\bf{Mineral \\, processing \\, stages}$"]
    if orientation == "horizontal":
        leg = axe.legend(handles=legend_handles, fontsize=12, loc='upper left', frameon=False)

    if orientation == "vertical":
        leg = axe.legend(handles=legend_handles, fontsize=12, loc='lower left', frameon=False)

    if shift_bars == False:
        leg = axe.legend(handles=legend_handles, fontsize=12, loc='upper left', frameon=False)

    # Optionally move titles in legend
    for item, label_ in zip(leg.legend_handles, leg.texts):
        if label_._text in titles:
            # width = item.get_window_extent(fig.canvas.get_renderer()).width
            # label_.set_ha('left')
            # label_.set_position((-10.0 * width, 0))
            # width_item = item.get_window_extent(fig.canvas.get_renderer()).width # new for width
            label_.set_ha('left')
            # label_.set_position((-10.0 * width_item, 0)) # new for width

    return axe


# def plot_stacked_bar(df, group_by, stack_col, value_col, orientation="vertical", 
#                      ax=None, colors=None, short_val_label=None, units=None, 
#                      annotate_totals=False, percentage=False, grouped_categ="", 
#                      short_labels=None, **kwargs):
#     """
#     Creates a stacked bar plot from a DataFrame.
    
#     Data is aggregated by the 'group_by' column and 'stack_col' so that each group 
#     is represented as one bar, and each bar is split into segments according to the unique 
#     values in stack_col. The numeric values are summed from value_col for each combination.
    
#     If annotate_totals is True, the function calculates the total for each group and places
#     the total value as text at the top (for vertical plots) or to the right (for horizontal plots)
#     of each bar.
    
#     Additionally, this function annotates each stacked segment with its category label. If a 
#     segment is sufficiently large, the label is drawn inside the bar. If the segment is small,
#     no external annotation is added.
    
#     Parameters:
#       - df: The input DataFrame.
#       - group_by: The column name to group by (defines the categories on the primary axis).
#       - stack_col: The column name that defines the stacking (each unique value becomes a segment).
#       - value_col: The column name containing the numeric values.
#       - orientation: "vertical" (default) for vertical bars or "horizontal" for horizontal bars.
#       - ax: Optional matplotlib Axes to plot on. If None, a new figure and Axes are created.
#       - colors: Optional. Either a dictionary mapping each unique value in stack_col to a color,
#                 or a list/tuple of colors to use in order.
#       - short_val_label: A short label for the value column. Defaults to value_col if not provided.
#       - units: The units to be displayed alongside the value label (e.g. "USD per tonne").
#                Defaults to an empty string if not provided.
#       - annotate_totals: Boolean. If True, annotate each bar with the total for that group.
#       - percentage: Boolean. If True, the values in the pivot are converted to percentages.
#       - grouped_categ: Title for the legend.
#       - short_labels: Optional list of short names to use for the legend labels.
#       - **kwargs: Additional keyword arguments to pass to the plotting function.
    
#     Returns:
#       The matplotlib Axes object containing the stacked bar plot.
#     """
#     # Set defaults for short_val_label and units.
#     if short_val_label is None:
#         short_val_label = value_col
#     if units is None:
#         units = ""
#     custom_val_label = short_val_label
#     if units:
#         custom_val_label += " (" + units + ")"
    
#     # Create the pivot table.
#     df_pivot = df.pivot_table(index=group_by, columns=stack_col, values=value_col, 
#                               aggfunc='sum', fill_value=0)
    
#     if percentage:
#         # Convert absolute values to percentages.
#         df_shares = df_pivot.div(df_pivot.sum(axis=1), axis=0)
#         df_plot = df_shares * 100
#     else:
#         df_plot = df_pivot.copy()
    
#     # Process the colors.
#     if isinstance(colors, dict):
#         color_list = [colors.get(col, None) for col in df_plot.columns]
#     elif isinstance(colors, (list, tuple)):
#         color_list = colors
#     else:
#         color_list = None

#     # Create the axes if not provided.
#     if ax is None:
#         fig, ax = plt.subplots()
    
#     # Plot the stacked bar chart.
#     if orientation == "horizontal":
#         df_plot.plot(kind='barh', stacked=True, ax=ax, color=color_list, **kwargs)
#         ax.set_xlabel(custom_val_label)
#         ax.set_ylabel(group_by.replace("_", " "))
#     elif orientation == "vertical":
#         df_plot.plot(kind='bar', stacked=True, ax=ax, color=color_list, **kwargs)
#         ax.set_xlabel(group_by.replace("_", " "))
#         ax.set_ylabel(custom_val_label)
#     else:
#         raise ValueError("Orientation must be either 'vertical' or 'horizontal'.")
    
#     ax.set_title(f"{value_col} by {group_by} and {stack_col}")
    
#     # Annotate totals if requested.
#     if annotate_totals:
#         group_totals = df_pivot.sum(axis=1)

#         if percentage:
#             offset = 100 * 0.01  # Ensures text appears just after 100%
#         else:
#             offset = group_totals.max() * 0.05  # Dynamic offset for absolute values

#         if orientation == "vertical":
#             xticks = ax.get_xticks()
#             for i, group in enumerate(df_plot.index):
#                 total = group_totals.loc[group]
#                 x = xticks[i] if i < len(xticks) else i
                
#                 # Ensure correct placement for percentage-based bars
#                 y_pos = 100 + offset if percentage else df_pivot.loc[group].sum() + offset
                
#                 ax.text(x, y_pos, f"{total:,.0f}kt CO2e", ha='center', va='bottom', fontsize=10, fontweight='bold', color="black")
        
#         else:  # Horizontal orientation
#             yticks = ax.get_yticks()
#             for i, group in enumerate(df_plot.index):
#                 total = group_totals.loc[group]
#                 y = yticks[i] if i < len(yticks) else i

#                 # Ensure correct placement for percentage-based bars
#                 x_pos = 100 + offset if percentage else df_pivot.loc[group].sum() + offset
                
#                 ax.text(x_pos, y, f"{total:,.0f}kt CO2e", ha='left', va='center', fontsize=10, fontweight='bold', color="black")
    
#     # Add the legend outside the plot area.
#     if short_labels is not None:
#         handles, _ = ax.get_legend_handles_labels()
#         ax.legend(handles, short_labels, title=grouped_categ, loc='center left', bbox_to_anchor=(1, 0.5))
#     else:
#         ax.legend(title=grouped_categ, loc='center left', bbox_to_anchor=(1, 0.5))
    
#     # Compute threshold per group to ensure consistent annotation
#     threshold = df_pivot.abs().sum(axis=1).max() * 0.05  # Use absolute values for correct thresholding

#     # Annotate each stacked segment with its category label.
#     if orientation == "vertical":
#         x_positions = np.arange(len(df_plot.index))
        
#         for i, group in enumerate(df_plot.index):
#             row = df_plot.loc[group]
#             bottom_positive = 0  # Tracks stacking position for positive values
#             bottom_negative = 0  # Tracks stacking position for negative values
#             has_negative = any(row[col] < 0 for col in df_plot.columns)  # Check for negative values

#             for col in df_plot.columns:
#                 val = row[col]
#                 abs_val = abs(val)  # Absolute value for threshold comparison
                
#                 # Only annotate if the absolute value is above the threshold
#                 if abs_val >= threshold:
#                     x = x_positions[i]
#                     label_text = str(col)

#                     # Use short labels if available
#                     if short_labels is not None and col in df_plot.columns:
#                         label_text = short_labels[df_plot.columns.get_loc(col)]

#                     # Adjust positioning for both cases:
#                     if has_negative:  
#                         # Mixed positive & negative case
#                         if val > 0:
#                             y = bottom_positive + val / 2  # Center inside positive stack
#                             bottom_positive += val  # Move stacking up for positive side
#                         else:
#                             y = bottom_negative + val / 2  # Center inside negative stack
#                             bottom_negative += val  # Move stacking down for negative side
#                     else:  
#                         # Fully positive case
#                         y = bottom_positive + val / 2  # Center inside segment
#                         bottom_positive += val  # Move stacking up
                    
#                     # Ensure text is only added if there's enough space
#                     if abs_val >= threshold * 1.5:  
#                         ax.text(x, y, label_text, ha='center', va='center', fontsize=10, color='white', rotation=90)

#     else:  # Horizontal orientation
#         y_positions = np.arange(len(df_plot.index))
        
#         for i, group in enumerate(df_plot.index):
#             row = df_plot.loc[group]
#             left_positive = 0  # Tracks stacking position for positive values
#             left_negative = 0  # Tracks stacking position for negative values
#             has_negative = any(row[col] < 0 for col in df_plot.columns)  # Check for negative values
            
#             for col in df_plot.columns:
#                 val = row[col]
#                 abs_val = abs(val)  # Absolute value for threshold comparison
                
#                 # Only annotate if the absolute value is above the threshold
#                 if abs_val >= threshold:
#                     y = y_positions[i]
#                     label_text = str(col)

#                     # Use short labels if available
#                     if short_labels is not None and col in df_plot.columns:
#                         label_text = short_labels[df_plot.columns.get_loc(col)]

#                     # Adjust positioning for both cases:
#                     if has_negative:  
#                         # Mixed positive & negative case
#                         if val > 0:
#                             x = left_positive + val / 2  # Center inside positive stack
#                             left_positive += val  # Move stacking right for positive side
#                         else:
#                             x = left_negative + val / 2  # Center inside negative stack
#                             left_negative += val  # Move stacking left for negative side
#                     else:  
#                         # Fully positive case
#                         x = left_positive + val / 2  # Center inside segment
#                         left_positive += val  # Move stacking right
                    
#                     # Ensure text is only added if there's enough space
#                     if abs_val >= threshold * 1.5:  
#                         ax.text(x, y, label_text, ha='center', va='center', fontsize=10, color='white')

#     if percentage:
#         ax.figure.subplots_adjust(right=0.8)  # Expands the right side so the legend is fully visible
#     return ax


def plot_stacked_bar(df, group_by, stack_col, value_col, orientation="vertical", 
                     ax=None, colors=None, short_val_label=None, units=None, 
                     annotate_totals=False, percentage=False, grouped_categ="", 
                     short_labels=None, annotate_labels=True, **kwargs):
    """
    Creates a stacked bar plot from a DataFrame with optional category labels inside bars.

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
      - annotate_labels: Boolean. If True (default), annotate each stacked segment with its category label.
      - **kwargs: Additional keyword arguments to pass to the plotting function.

    Returns:
      The matplotlib Axes object containing the stacked bar plot.
    """
    # Set defaults for short_val_label and units
    if short_val_label is None:
        short_val_label = value_col
    if units is None:
        units = ""
    custom_val_label = short_val_label
    if units:
        custom_val_label += " (" + units + ")"
    
    # Create the pivot table
    df_pivot = df.pivot_table(index=group_by, columns=stack_col, values=value_col, aggfunc='sum', fill_value=0)
    
    if percentage:
        df_shares = df_pivot.div(df_pivot.sum(axis=1), axis=0)
        df_plot = df_shares * 100
    else:
        df_plot = df_pivot.copy()

    # Process the colors
    if isinstance(colors, dict):
        color_list = [colors.get(col, None) for col in df_plot.columns]
    elif isinstance(colors, (list, tuple)):
        color_list = colors
    else:
        color_list = None

    # Create the axes if not provided
    if ax is None:
        fig, ax = plt.subplots()
    
    # Plot the stacked bar chart
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

    # Annotate totals if requested
    if annotate_totals:
        group_totals = df_pivot.sum(axis=1)
        offset = 100 * 0.01 if percentage else group_totals.max() * 0.05

        if orientation == "vertical":
            xticks = ax.get_xticks()
            for i, group in enumerate(df_plot.index):
                total = group_totals.loc[group]
                x = xticks[i] if i < len(xticks) else i
                y_pos = 100 + offset if percentage else df_pivot.loc[group].sum() + offset
                ax.text(x, y_pos, f"{total:,.0f}", ha='center', va='bottom', fontsize=10, fontweight='bold', color="black")
        else:  # Horizontal orientation
            yticks = ax.get_yticks()
            for i, group in enumerate(df_plot.index):
                total = group_totals.loc[group]
                y = yticks[i] if i < len(yticks) else i
                x_pos = 100 + offset if percentage else df_pivot.loc[group].sum() + offset
                ax.text(x_pos, y, f"{total:,.0f}", ha='left', va='center', fontsize=10, fontweight='bold', color="black")
    
    # Add the legend
    if short_labels is not None:
        handles, _ = ax.get_legend_handles_labels()
        ax.legend(handles, short_labels, title=grouped_categ, loc='center left', bbox_to_anchor=(1, 0.5))
    else:
        ax.legend(title=grouped_categ, loc='center left', bbox_to_anchor=(1, 0.5))

    # **Annotate each stacked segment with its category label (only if annotate_labels=True)**
    if annotate_labels:
        threshold = df_pivot.abs().sum(axis=1).max() * 0.05  # Threshold for text visibility

        if orientation == "vertical":
            x_positions = np.arange(len(df_plot.index))
            
            for i, group in enumerate(df_plot.index):
                row = df_plot.loc[group]
                bottom = 0  # Tracks stacking position

                for col in df_plot.columns:
                    val = row[col]
                    if abs(val) >= threshold:  # Only annotate if the value is significant
                        x = x_positions[i]
                        y = bottom + val / 2
                        label_text = short_labels[df_plot.columns.get_loc(col)] if short_labels else str(col)
                        ax.text(x, y, label_text, ha='center', va='center', fontsize=10, color='white', rotation=90)
                    bottom += val  # Move stacking up
                    
        else:  # Horizontal orientation
            y_positions = np.arange(len(df_plot.index))
            
            for i, group in enumerate(df_plot.index):
                row = df_plot.loc[group]
                left = 0  # Tracks stacking position

                for col in df_plot.columns:
                    val = row[col]
                    if abs(val) >= threshold:  # Only annotate if the value is significant
                        y = y_positions[i]
                        x = left + val / 2
                        label_text = short_labels[df_plot.columns.get_loc(col)] if short_labels else str(col)
                        ax.text(x, y, label_text, ha='center', va='center', fontsize=10, color='white')
                    left += val  # Move stacking right

    if percentage:
        ax.figure.subplots_adjust(right=0.8)  # Expand right side so the legend is fully visible
    return ax


def main(config):
    processed_data_path = config['paths']['data']
    output_data_path = config['paths']['results']
    figures_data_path = config['paths']['figures']

    multiply_factor = 1.0
    country_codes = ["ZMB"]

    figures = os.path.join(figures_data_path,f"{'_'.join(country_codes)}_figures")
    if os.path.exists(figures) is False:
        os.mkdir(figures)

    figures = os.path.join(figures_data_path,f"{'_'.join(country_codes)}_figures","bar_plots")
    if os.path.exists(figures) is False:
        os.mkdir(figures)

    columns = [
                "production_tonnes", 
                # "export_tonnes",
                # "export_transport_cost_usd", 
                # "transport_total_tonsCO2eq", #used to be "tonsCO2eq",
                # "revenue_usd",
                # "production_cost_usd",
                "all_cost_usd"
            ]
    column_titles = [
                        "production volume (kilotonne)",
                        # "export volume (000' tonnes)",
                        # "transport costs (million USD)",
                        # "transport carbon emissions (tonsCO2eq)",
                        # "revenue (million USD)",
                        # "production costs (million USD)",
                        "total costs (million USD)"
                    ]
    multiply_factors = [1.0e-3,1.0e-3,1.0e-6,1,1.0e-6,1.0e-6,1.0e-6]
    stage_colors = [
                        "#1f78b4",  # Strong Blue
                        "#33a02c",  # Green
                        "#e31a1c",  # Red
                        "#ff7f00",  # Orange
                        "#6a3d9a",  # Purple
                        "#b15928",  # Brown
                        "#a6cee3",  # Light Blue
                        "#b2df8a",  # Light Green
                        "#fb9a99"   # Soft Red (for balance)
                    ]
    type_colors = ['#fed976', '#fb6a4a', '#7f0000']

    column_titles = [f"Annual {c}" for c in column_titles]
    scenarios_descriptions = [
                                {
                                    "scenario_type":"country_unconstrained",
                                    "scenario_name":"Mid National",
                                    "scenarios":[
                                                    "2022_baseline",
                                                    "2030_mid_min_threshold_metal_tons",
                                                    "2040_mid_min_threshold_metal_tons"
                                                ],
                                    "scenario_labels":[
                                                    "2022",
                                                    "2030",
                                                    "2040"
                                                ]
                                },
                                {
                                    "scenario_type":"region_unconstrained",
                                    "scenario_name":"Mid Regional",
                                    "scenarios":[
                                                    "2022_baseline",
                                                    "2030_mid_max_threshold_metal_tons",
                                                    "2040_mid_max_threshold_metal_tons"
                                                ],
                                    "scenario_labels":[
                                                    "2022",
                                                    "2030",
                                                    "2040"
                                                ]
                                },
                                {
                                    "scenario_type":"country_constrained",
                                    "scenario_name":"Mid National",
                                    "scenarios":[
                                                    "2022_baseline",
                                                    "2030_mid_min_threshold_metal_tons",
                                                    "2040_mid_min_threshold_metal_tons"
                                                ],
                                    "scenario_labels":[
                                                    "2022",
                                                    "2030",
                                                    "2040"
                                                ]
                                },
                                {
                                    "scenario_type":"region_constrained",
                                    "scenario_name":"Mid Regional",
                                    "scenarios":[
                                                    "2022_baseline",
                                                    "2030_mid_max_threshold_metal_tons",
                                                    "2040_mid_max_threshold_metal_tons"
                                                ],
                                    "scenario_labels":[
                                                    "2022",
                                                    "2030",
                                                    "2040"
                                                ]
                                },
                    ]
    ######## From Raghav
    make_plot = False
    if make_plot is True:
        results_file = os.path.join(output_data_path,
                                "result_summaries",
                                "combined_energy_transport_totals_by_stage.xlsx")
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
            data_df = data_df[(data_df["iso3"].isin(country_codes)) & (data_df["production_tonnes"] > 0)]
            data_df["reference_mineral"] = data_df["reference_mineral"].str.title()
            reference_minerals = sorted(list(set(data_df["reference_mineral"].values.tolist())))
            for cdx,(col,col_t,m_t) in enumerate(zip(columns,column_titles,multiply_factors)):
                fig, ax = plt.subplots(1,1,figsize=(18,9),dpi=500)
                dfall = []
                df = data_df[data_df["processing_stage"] > 0]
                stages = sorted(list(set(df["processing_stage"].values.tolist())))
                stage_colors = stage_colors[:len(stages)]
                for sc in scenarios:
                    m_df = pd.DataFrame(reference_minerals,columns=["reference_mineral"])
                    for st in stages:
                        s_df = df[(df["processing_stage"] == st) & (df["scenario"] == sc)]
                        if len(s_df.index) > 0:
                            s_df[col] = m_t*s_df[col]
                            s_df.rename(columns={col:f"Stage {st}"},inplace=True)
                            m_df = pd.merge(
                                            m_df,s_df[["reference_mineral",f"Stage {st}"]],
                                            how="left",on=["reference_mineral"]).fillna(0)
                        else:
                            m_df[f"Stage {st}"] = 0.0
                    dfall.append(m_df.set_index(["reference_mineral"]))
                ax = plot_clustered_stacked(
                                            fig,ax,dfall,stage_colors,
                                            labels=sc_l,
                                            ylabel=col_t, 
                                            title=f"{sc_n} scenario")
                plt.grid()
                plt.tight_layout()
                save_fig(os.path.join(results_folder,
                            f"{col}_{sc_t}.png"))
                plt.close()
    
    ######
    # Saving results: metal content
    make_plot = False
    if make_plot:
        results_file = os.path.join(output_data_path, "result_summaries", "combined_energy_transport_totals_by_stage.xlsx")
        output_excel_path = os.path.join(output_data_path, "regional_national_metal_content.xlsx")

        # Create an empty list to store all results before saving
        all_results = {}
        
        for sd in scenarios_descriptions:
            sc_t = sd["scenario_type"]
            sc_n = sd["scenario_name"]
            scenarios = sd["scenarios"]
            sc_l = sd["scenario_labels"]

            # Read scenario data
            data_df = pd.read_excel(results_file, sheet_name=sc_t, index_col=[0, 1, 2, 3, 4]).reset_index()

            # If it's not "country_unconstrained", add the 2022 baseline
            if sc_t != "country_unconstrained":
                baseline_df = pd.read_excel(results_file, sheet_name="country_unconstrained", index_col=[0, 1, 2, 3, 4]).reset_index()
                data_df = pd.concat([baseline_df[baseline_df["year"] == 2022], data_df], axis=0, ignore_index=True)

            # Filter data for the required country
            country_codes = ["ZMB"]  # Replace with actual ISO3 country code
            data_df = data_df[(data_df["iso3"].isin(country_codes)) & (data_df["production_tonnes"] == 0)]  # metal content
            data_df["reference_mineral"] = data_df["reference_mineral"].str.title()
            reference_minerals = sorted(data_df["reference_mineral"].unique())

            # Store results in a list
            results_list = []

            col_values = columns[0] if isinstance(columns[0], list) else [columns[0]]
            col_titles = column_titles[0] if isinstance(column_titles[0], list) else [column_titles[0]]
            mult_factors = multiply_factors[0] if isinstance(multiply_factors[0], list) else [multiply_factors[0]]

            for col, col_t, m_t in zip(col_values, col_titles, mult_factors):  # only metal content
                dfall = []
                df = data_df[data_df["processing_stage"] == 0]
                stages = sorted(df["processing_stage"].unique())

                for sc in scenarios:
                    m_df = pd.DataFrame(reference_minerals, columns=["reference_mineral"])
                    
                    for st in stages:
                        s_df = df[(df["processing_stage"] == st) & (df["scenario"] == sc)]

                        if not s_df.empty:
                            # Check if column exists before modifying
                            if col in s_df.columns:
                                s_df[col] = m_t * s_df[col]
                                s_df.rename(columns={col: f"Stage {st}"}, inplace=True)
                            else:
                                print(f"⚠ Warning: Column '{col}' not found in data for scenario '{sc}', stage {st}")
                                continue  # Skip to the next stage

                            # Ensure reference_mineral exists before merging
                            if "reference_mineral" in s_df.columns:
                                m_df = m_df.merge(s_df[["reference_mineral", f"Stage {st}"]], how="left", on="reference_mineral")
                            else:
                                print(f"⚠ Warning: 'reference_mineral' column missing in scenario '{sc}', stage {st}")
                                continue  # Skip to the next stage

                        else:
                            print(f"⚠ Warning: No data found for scenario '{sc}', stage {st}")
                            m_df[f"Stage {st}"] = 0.0  # Ensure column exists
                        
                    # Add metadata columns
                    m_df["Scenario"] = sc
                    m_df["Constraint"] = sc_t  # Store scenario type as constraint
                    m_df["Variable"] = col_t  # Store column title
                    
                    dfall.append(m_df.set_index(["reference_mineral", "Scenario", "Constraint", "Variable"]))

                # Ensure non-empty DataFrame is stored
                if dfall:
                    results_list.append(pd.concat(dfall))
                else:
                    print(f"⚠ Warning: No results for '{sc_t}' - skipping.")

            # Store the results for this scenario type in a dictionary
            if results_list:
                all_results[sc_t] = pd.concat(results_list)
            else:
                print(f"⚠ Warning: No valid data for scenario type '{sc_t}'.")

        # Write all results to an Excel file with multiple sheets
        if all_results:
            with pd.ExcelWriter(output_excel_path, engine="xlsxwriter") as writer:
                for sheet_name, df in all_results.items():
                    df.to_excel(writer, sheet_name=sheet_name)

            print(f" Data successfully saved to {output_excel_path}")
        else:
            print("⚠ No valid data to save.")

    """
    Country and regional uncontrained and constrained comparisons for 2030 and 2040 separately in plots for all minerals (ALL stages):
    annual energy, production and total costs: works
    """ 
    make_plot = False 
    if make_plot:
        results_file = os.path.join(output_data_path,  "all_data.xlsx")

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
            # "energy_opex" ,
            # 'energy_investment_usd',
            # "production_cost_usd",
            # "water_usage_m3",
            "production_tonnes",
            'all_cost_usd',
            'energy_req_capacity_kW'
        ]
        en_tr = [
                # "energy",
                # "energy",
                # "production",
                # "water",
                "production",
                "cost",
                'energy'
                ]
        short_col_name = [
                    # "energy OPEX",
                    # "energy investments" ,
                    # "production cost",
                    # "water use",
                    "Production",
                    "Total cost",
                    "Energy capacity"
                        ]

        scenarios = [
            "2022_baseline",
            "2030_mid_min_threshold_metal_tons",
            "2030_mid_max_threshold_metal_tons",
            "2040_mid_min_threshold_metal_tons",
            "2040_mid_max_threshold_metal_tons"
        ]


        for c, et, sh_c in zip(cost_columns, en_tr, short_col_name):
            print(c)
            if c == "all_cost_usd":  # all those that are not unit costs
                multiply_factor = 1.0e-6 
                units = 'million USD'
            if c == "water_usage_m3":
                multiply_factor = 1.0e-6 
                units = 'million m3'
            if c == 'production_tonnes':
                multiply_factor = 1.0e-6 
                units = 'million tonne'
            if c == 'energy_req_capacity_kW':
                multiply_factor = 1.0e-6
                units = 'GW'
            # Reding data
            df = pd.read_excel(results_file, index_col=[0,1,2,3,4])
            df = df.reset_index()

            # Select country
            df_country = df[df["iso3"].isin(country_codes)]
            
            for cs, nr in zip(constraints, figure_nrows):
                df_con = df_country[df_country["constraint"] == cs]
                print(cs)
                # Figures with minerals on categorical axis; change row, column depending on horizontal or vertical
                # fig, axs = plt.subplots(1, 2, figsize=(15, 7), dpi=500, sharey=True) # vertical
                fig, axs = plt.subplots(nr,1, figsize=(9,nr*3), dpi=500, sharex=True) #horizontal
                
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
                    

                    # Make stage color map
                    stage_color_map = dict(zip(df_con[df_con["processing_stage"]>0]["processing_stage"].unique(), stage_colors))

                    type_colors = ['#fed976', '#fb6a4a', '#7f0000']
                    stage_type_color_map = dict(zip(df_con[df_con["processing_stage"]>0]["processing_type"].unique(), type_colors))

                    for rf in reference_minerals:
                        filtered_df = df_con[
                            (df_con["scenario"] == scenario)
                            & (df_con["reference_mineral"] == rf)
                            & (df_con["processing_stage"]>0 ) # all stages
                        ]
                        
                        if filtered_df.empty:
                            print(f"No data after dictionary filter for {rf}, scenario '{scenario}', constraint '{cs}'")
                            continue

                        # group by iso3 and sum up cost for whichever stage remains
                        # m_df = pd.DataFrame(sorted(filtered_df["iso3"].unique()), columns=["iso3"])
                        s_df = filtered_df.copy()
                        s_df[c] = multiply_factor * s_df[c]

                        # group by iso3 and processing_type => pivot
                        grouped = s_df.groupby(['reference_mineral', "processing_type"])[c].sum().reset_index()
                        # print(grouped)

                        if "2022" in scenario:
                            m_df_2022 = grouped
                            data_2022 = pd.concat([data_2022, m_df_2022], ignore_index = True)
                        if "2030" in scenario:
                            m_df_2030 = grouped
                            data_2030 = pd.concat([data_2030, m_df_2030], ignore_index = True)
                        if "2040" in scenario:
                            m_df_2040 = grouped
                            data_2040 = pd.concat([data_2040, m_df_2040], ignore_index = True)
                cs_clean = cs.replace("_", " ").title()

                if data_2022.empty:
                    print("No data for 2022")
                else:
                    data_2022['reference_mineral_short'] = data_2022['reference_mineral'].map(reference_mineral_namemap)
                    data_2022.rename(columns = {'reference_mineral_short':'Mineral'}, inplace = True)
                
                data_2030['reference_mineral_short'] = data_2030['reference_mineral'].map(reference_mineral_namemap)
                data_2040['reference_mineral_short'] = data_2040['reference_mineral'].map(reference_mineral_namemap)

                data_2030.rename(columns = {'reference_mineral_short':'Mineral'}, inplace = True)
                data_2040.rename(columns = {'reference_mineral_short':'Mineral'}, inplace = True)
                

                # Horizontal
                if data_2022.empty:
                    print("No data for 2022")
                    axs[0] = plot_stacked_bar(data_2030, group_by='Mineral', stack_col='processing_type', value_col=c,
                                    orientation="horizontal", ax=axs[0], colors=stage_type_color_map, short_val_label = sh_c, 
                                    units = units, grouped_categ = 'Processing stage', annotate_labels = False)
                    axs[0].set_title(f"2030 - All stages - {cs_clean}")

                    axs[1] =plot_stacked_bar(data_2040, group_by='Mineral', stack_col='processing_type', value_col=c,
                                        orientation="horizontal", ax=axs[1], colors=stage_type_color_map, short_val_label = sh_c, 
                                        units = units, grouped_categ = 'Processing stage', annotate_labels = False)
                    axs[1].set_title(f"2040 - All stages - {cs_clean}")
                if data_2022.empty == False:
                    axs[0] = plot_stacked_bar(data_2022, group_by='Mineral', stack_col='processing_type', value_col=c,
                                        orientation="horizontal", ax=axs[0], colors=stage_type_color_map, short_val_label = sh_c, 
                                        units = units, grouped_categ = 'Processing stage', annotate_labels = False)
                    axs[0].set_title(f"2022 - All stages - {cs_clean}")

                    axs[1] = plot_stacked_bar(data_2030, group_by='Mineral', stack_col='processing_type', value_col=c,
                                        orientation="horizontal", ax=axs[1], colors=stage_type_color_map, short_val_label = sh_c, 
                                        units = units, grouped_categ = 'Processing stage', annotate_labels = False)
                    axs[1].set_title(f"2030 - All stages - {cs_clean}")

                    axs[2] =plot_stacked_bar(data_2040, group_by='Mineral', stack_col='processing_type', value_col=c,
                                        orientation="horizontal", ax=axs[2], colors=stage_type_color_map, short_val_label = sh_c, 
                                        units = units, grouped_categ = 'Processing stage', annotate_labels = False)
                    axs[2].set_title(f"2040 - All stages - {cs_clean}")
                    
                for ax in axs:
                    ax.grid(which="major", axis="x", linestyle="-", zorder=0)
                    ax.grid(False, axis="y")

                    
                plt.tight_layout()
                # Save figure
                filename = f"{country_codes[0]}_{cs}_{c}_mineral_comparisons_2022_2030_2040H.png"
                save_fig(os.path.join(figures, filename))
                plt.close()

                # Figures with countries on categorical axis; change row, column depending on horizontal or vertical
                # fig, axs = plt.subplots(1, 2, figsize=(15, 7), dpi=500, sharey=True) # vertical
                fig, axs = plt.subplots(nr,1, figsize=(9,nr*2.5), dpi=500, sharex=True) #horizontal
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

                    for rf in reference_minerals:

                        filtered_df = df_con[
                            (df_con["scenario"] == scenario)
                            & (df_con["reference_mineral"] == rf)
                            & (df_con["processing_stage"]>0 ) # all stages
                        ]

                        if filtered_df.empty:
                            print(f"No data after dictionary filter for {rf}, scenario '{scenario}', constraint '{cs}'")
                            continue

                        # group by iso3 and sum up cost for whichever stage remains
                        # m_df = pd.DataFrame(sorted(filtered_df["iso3"].unique()), columns=["iso3"])
                        s_df = filtered_df.copy()
                        s_df[c] = multiply_factor * s_df[c]

                        # group by iso3 and processing_type => pivot
                        grouped = s_df.groupby(['reference_mineral', "processing_type"])[c].sum().reset_index()
                        # print(grouped)
                        
                        if "2022" in scenario:
                            m_df_2022 = grouped
                            data_2022 = pd.concat([data_2022, m_df_2022], ignore_index = True)
                        if "2030" in scenario:
                            m_df_2030 = grouped
                            data_2030 = pd.concat([data_2030, m_df_2030], ignore_index = True)
                        if "2040" in scenario:
                            m_df_2040 = grouped
                            data_2040 = pd.concat([data_2040, m_df_2040], ignore_index = True)
                cs_clean = cs.replace("_", " ").title()

                if data_2022.empty:
                    print("No data for 2022")
                else:
                    data_2022['reference_mineral_short'] = data_2022['reference_mineral'].map(reference_mineral_namemap)

                data_2030['reference_mineral_short'] = data_2030['reference_mineral'].map(reference_mineral_namemap)
                data_2040['reference_mineral_short'] = data_2040['reference_mineral'].map(reference_mineral_namemap)


                if data_2022.empty:
                    print("No data for 2022")
                    axs[0] = plot_stacked_bar(data_2030, group_by='processing_type', stack_col='reference_mineral_short', value_col=c,
                                        orientation="horizontal", ax=axs[0], colors= reference_mineral_colormapshort, short_val_label = sh_c, 
                                        units = units, grouped_categ = 'Mineral')
                    axs[0].set_title(f"2030 - All stages - {cs_clean}")

                    axs[1] =plot_stacked_bar(data_2040, group_by='processing_type', stack_col='reference_mineral_short', value_col=c,
                                        orientation="horizontal", ax=axs[1], colors= reference_mineral_colormapshort, short_val_label = sh_c, 
                                        units = units, grouped_categ = 'Mineral')
                    axs[1].set_title(f"2040 - All stages - {cs_clean}")
                if data_2022.empty == False:
                    axs[0] = plot_stacked_bar(data_2022, group_by='processing_type', stack_col='reference_mineral_short', value_col=c,
                                        orientation="horizontal", ax=axs[0], colors= reference_mineral_colormapshort, short_val_label = sh_c,
                                        units = units, grouped_categ = 'Mineral')   
                    axs[0].set_title(f"2022 - All stages - {cs_clean}")
                
                    axs[1] = plot_stacked_bar(data_2030, group_by='processing_type', stack_col='reference_mineral_short', value_col=c,
                                        orientation="horizontal", ax=axs[1], colors= reference_mineral_colormapshort, short_val_label = sh_c, 
                                        units = units, grouped_categ = 'Mineral')
                    axs[1].set_title(f"2030 - All stages - {cs_clean}")

                    axs[2] =plot_stacked_bar(data_2040, group_by='processing_type', stack_col='reference_mineral_short', value_col=c,
                                        orientation="horizontal", ax=axs[2], colors= reference_mineral_colormapshort, short_val_label = sh_c, 
                                        units = units, grouped_categ = 'Mineral')
                    axs[2].set_title(f"2040 - All stages - {cs_clean}")
                    
                for ax in axs:
                    ax.grid(which="major", axis="x", linestyle="-", zorder=0)
                    ax.grid(False, axis="y")

                    
                plt.tight_layout()
                # Save figure
                filename = f"{country_codes[0]}_{cs}_{c}_comparisons_2022_2030_2040H.png"
                save_fig(os.path.join(figures, filename))
                plt.close()           

    #########
    # Differences for most variables
    #########
    make_plot = True
    if make_plot:
        results_file = os.path.join(output_data_path, "all_data.xlsx")

        # Define scenario pairs for comparison
        scenario_pairs = [
            ("2030_mid_min_threshold_metal_tons", "2030_mid_max_threshold_metal_tons"),
            ("2040_mid_min_threshold_metal_tons", "2040_mid_max_threshold_metal_tons"),
        ]

        for col, col_t, m_t in zip(columns, column_titles, multiply_factors):
            print(col)
            # Read scenario data once
            data_df = pd.read_excel(results_file, index_col=[0, 1, 2, 3, 4]).reset_index()

            # Filter data for relevant country and valid production
            country_codes = ["ZMB"]
            data_df = data_df[(data_df["iso3"].isin(country_codes)) & (data_df["production_tonnes"] > 0)& (data_df["processing_stage"] > 0)]
            data_df["reference_mineral"] = data_df["reference_mineral"].str.title()

            # Apply the multiplication factor
            data_df[col] *= m_t

            # Pivot the DataFrame to compare constraints
            pivot_df = data_df.pivot_table(
                index=["scenario", "reference_mineral", "processing_type", "processing_stage"],
                columns="constraint",
                values=col,
                aggfunc="sum",
                fill_value=0  # Ensures missing stages don't cause issues
            )

            # Reset index to make it easy to manipulate
            pivot_df.reset_index(inplace=True)

            # Compute differences only for the defined scenario pairs
            delta_dfs = {}  # Dictionary to store results

            for scenario_min, scenario_max in scenario_pairs:
                delta_col_name = f"delta_{scenario_max}_vs_{scenario_min}"
                
                # Extract only the relevant scenarios
                df_min = pivot_df[pivot_df["scenario"] == scenario_min].set_index(["reference_mineral", "processing_type", "processing_stage"])
                df_min.drop(columns="scenario", inplace=True)  # Drop the scenario column
                df_max = pivot_df[pivot_df["scenario"] == scenario_max].set_index(["reference_mineral", "processing_type", "processing_stage"])
                df_max.drop(columns="scenario", inplace=True)  # Drop the scenario column

                # Ensure both dataframes have the same index structure
                common_index = df_min.index.union(df_max.index)  # Get all possible index values
                df_min = df_min.reindex(common_index, fill_value=0)  # Fill missing stages with 0
                df_max = df_max.reindex(common_index, fill_value=0)

                df_min = df_min[['country_constrained', 'country_unconstrained']]
                df_min.rename(columns = {'country_constrained': 'constrained', 'country_unconstrained': 'unconstrained'}, inplace = True)
                df_max = df_max[['region_constrained', 'region_unconstrained']]
                df_max.rename(columns = {'region_constrained': 'constrained', 'region_unconstrained': 'unconstrained'}, inplace = True)

                # Compute the delta
                delta_df = df_max - df_min # Region minus country
                delta_df['year'] = scenario_max.split("_")[0]  # Store the year for reference      
                delta_df = delta_df.reset_index()

                # Add year column properly
                delta_df["year"] = scenario_max.split("_")[0]  

                # Store the results in a dictionary
                delta_dfs[delta_col_name] = delta_df

            # **Generate Separate Plots for Constrained and Unconstrained**
            for const in ['constrained', 'unconstrained']:
                fig, axs = plt.subplots(1, 2, figsize=(12, 7), dpi=500, sharey=True)
                
                # Set bar width dynamically based on minerals and stages
                bar_width = 0.2

                for delta_col_name, delta_df in delta_dfs.items():
                    # **Verify Year Column Exists Before Accessing**
                    if "year" not in delta_df.columns:
                        print(f"Warning: 'year' column missing in {delta_col_name}. Skipping plot.")
                        continue  # Skip if the year column is missing

                    year = str(delta_df["year"].iloc[0])  # Extract correct year

                    # Convert wide format to long format for easier plotting
                    df_long = delta_df.melt(
                        id_vars=["reference_mineral", "year", "processing_type", "processing_stage"],
                        var_name="Constraint", value_name="Difference"
                    )

                    df_long.rename(columns={"processing_type": "Type", "processing_stage": "Stage"}, inplace=True)
                    
                    # Aggregate data by mineral and processing type
                    df_long = df_long[df_long["Constraint"] == const].drop(columns=['Stage', 'Constraint']).groupby(['year','reference_mineral', 'Type']).sum().reset_index()

                    # Determine the correct subplot index
                    ax_idx = 0 if year == "2030" else 1  # 2030 -> axs[0], 2040 -> axs[1]

                    # X-axis positions based on unique minerals
                    minerals = sorted(df_long["reference_mineral"].unique())  # Ensure sorting for consistency
                    types = sorted(df_long["Type"].unique())

                    x_positions = {mineral: i for i, mineral in enumerate(minerals)}  # Map minerals to fixed x positions

                    # Plot bars for each processing type
                    for i, ty in enumerate(types):
                        subset = df_long[df_long["Type"] == ty]
                        subset["x_pos"] = subset["reference_mineral"].map(x_positions)
                        print(subset)

                        # Filter out missing values
                        subset = subset.dropna(subset=["x_pos"])  # Ensure x positions are valid

                        axs[ax_idx].bar(
                            subset["x_pos"], subset["Difference"], 
                            width=bar_width, label=f"{ty}", color=type_colors[i % len(type_colors)]
                        )

                    # Formatting subplot
                    axs[ax_idx].set_title(f"{year}: Difference (Region - Country) \n {const.title()}", 
                                        fontsize=14, fontweight="bold")
                    axs[ax_idx].grid(axis="y", linestyle="--", alpha=0.6)
                
                axs[0].set_ylabel(col_t, fontsize=14, fontweight="bold")

                # X-axis formatting (apply to both subplots)
                for ax in axs:
                    ax.set_xticks(range(len(minerals)))  # Match positions to minerals
                    ax.set_xticklabels(minerals, rotation=45, fontsize=12, fontweight="bold")

                # Legend only in the top subplot
                axs[0].legend(fontsize=10, loc="upper left", ncol=3)

                # Remove x-axis grid
                axs[0].grid(False, axis="x")
                axs[1].grid(False, axis="x")

                # Save plot with constraint name
                results_folder = os.path.join(figures_data_path, f"MR_MN")
                os.makedirs(results_folder, exist_ok=True)
                plt.tight_layout()
                save_fig(os.path.join(results_folder, f"{col}_differences_subplot_{const}.png"))
                plt.close()

    #########
    # Differences for most variables: unconstrained minus constrained
    #########
    make_plot = True
    if make_plot:
        results_file = os.path.join(output_data_path, "all_data.xlsx")

        # Define scenario pairs for comparison
        scenario_pairs = [
                            ("2030_mid_min_threshold_metal_tons", "2030_mid_max_threshold_metal_tons" ), # country, region
                            ("2040_mid_min_threshold_metal_tons", "2040_mid_max_threshold_metal_tons"), 
                        ]

        for col, col_t, m_t in zip(columns, column_titles, multiply_factors):
            print(col)
            # Read scenario data once
            data_df = pd.read_excel(results_file, index_col=[0, 1, 2, 3, 4]).reset_index()

            # Filter data for relevant country and valid production
            country_codes = ["ZMB"]
            data_df = data_df[(data_df["iso3"].isin(country_codes)) & (data_df["production_tonnes"] > 0)& (data_df["processing_stage"] > 0)]
            data_df["reference_mineral"] = data_df["reference_mineral"].str.title()

            # Apply the multiplication factor
            data_df[col] *= m_t

            # Pivot the DataFrame to compare constraints
            pivot_df = data_df.pivot_table(
                index=["scenario", "reference_mineral", "processing_type", "processing_stage"],
                columns="constraint",
                values=col,
                aggfunc="sum",
                fill_value=0  # Ensures missing stages don't cause issues
            )

            # Reset index to make it easy to manipulate
            pivot_df.reset_index(inplace=True)

            # Compute differences only for the defined scenario pairs
            delta_dfs = {}  # Dictionary to store results

            for scenario_min, scenario_max in scenario_pairs:
                delta_col_name = f"delta_{scenario_max}_vs_{scenario_min}"
                
                # Extract only the relevant scenarios
                df_min = pivot_df[pivot_df["scenario"] == scenario_min].set_index(["reference_mineral", "processing_type", "processing_stage"])
                df_min.drop(columns="scenario", inplace=True)  # Drop the scenario column, all country
                df_max = pivot_df[pivot_df["scenario"] == scenario_max].set_index(["reference_mineral", "processing_type", "processing_stage"])
                df_max.drop(columns="scenario", inplace=True)  # Drop the scenario column, all region

                # Ensure both dataframes have the same index structure
                common_index = df_min.index.union(df_max.index)  # Get all possible index values
                df_min = df_min.reindex(common_index, fill_value=0)  # Fill missing stages with 0
                df_max = df_max.reindex(common_index, fill_value=0)

                # df_constrained = pd.concat(df_min[['country_constrained']], df_max[['region_constrained']])
                df_constrained = pd.merge(df_min[['country_constrained']], df_max[['region_constrained']], 
                                                left_index=True, right_index=True, how='outer')
                df_constrained.rename(columns = {'country_constrained': 'country', 
                                        'region_constrained': 'region'}, inplace = True)

                df_unconstrained = pd.merge(df_min[['country_unconstrained']], df_max[['region_unconstrained']], 
                                                left_index=True, right_index=True, how='outer')
                df_unconstrained.rename(columns = {'country_unconstrained': 'country', 
                                        'region_unconstrained': 'region'}, inplace = True)

                # Compute the delta
                delta_df = df_constrained - df_unconstrained # constrained minus unconstrained
                delta_df['year'] = scenario_max.split("_")[0]  # Store the year for reference      
                delta_df = delta_df.reset_index()

                # Add year column properly
                delta_df["year"] = scenario_max.split("_")[0]  
                print(delta_df)

                # Store the results in a dictionary
                delta_dfs[delta_col_name] = delta_df


            # **Generate Separate Plots for Constrained and Unconstrained**
            for const in ['country', 'region']:
                fig, axs = plt.subplots(1, 2, figsize=(13, 7), dpi=500, sharey=True)
                
                # Set bar width dynamically based on minerals and stages
                bar_width = 0.2

                for delta_col_name, delta_df in delta_dfs.items():
                    # **Verify Year Column Exists Before Accessing**
                    if "year" not in delta_df.columns:
                        print(f"Warning: 'year' column missing in {delta_col_name}. Skipping plot.")
                        continue  # Skip if the year column is missing

                    year = str(delta_df["year"].iloc[0])  # Extract correct year

                    # Convert wide format to long format for easier plotting
                    df_long = delta_df.melt(
                        id_vars=["reference_mineral", "year", "processing_type", "processing_stage"],
                        var_name="Constraint", value_name="Difference"
                    )

                    df_long.rename(columns={"processing_type": "Type", "processing_stage": "Stage"}, inplace=True)
                    
                    # Aggregate data by mineral and processing type
                    df_long = df_long[df_long["Constraint"] == const].drop(columns=['Stage', 'Constraint']).groupby(['year',
                                                                                                                    'reference_mineral',
                                                                                                                     'Type']).sum().reset_index()

                    # Determine the correct subplot index
                    ax_idx = 0 if year == "2030" else 1  # 2030 -> axs[0], 2040 -> axs[1]

                    # X-axis positions based on unique minerals
                    minerals = sorted(df_long["reference_mineral"].unique())  # Ensure sorting for consistency
                    types = sorted(df_long["Type"].unique())

                    x_positions = {mineral: i for i, mineral in enumerate(minerals)}  # Map minerals to fixed x positions

                    # Plot bars for each processing type
                    for i, ty in enumerate(types):
                        subset = df_long[df_long["Type"] == ty]
                        subset["x_pos"] = subset["reference_mineral"].map(x_positions)
                        print(subset)

                        # Filter out missing values
                        subset = subset.dropna(subset=["x_pos"])  # Ensure x positions are valid

                        axs[ax_idx].bar(
                            subset["x_pos"], subset["Difference"], 
                            width=bar_width, label=f"{ty}", color=type_colors[i % len(type_colors)]
                        )

                    # Formatting subplot
                    axs[ax_idx].set_title(f"{year}: Difference (Constrained - Unconstrained) \n {const.title()}", 
                                        fontsize=14, fontweight="bold")
                    axs[ax_idx].grid(axis="y", linestyle="--", alpha=0.6)
                
                axs[0].set_ylabel(col_t, fontsize=14, fontweight="bold")

                # X-axis formatting (apply to both subplots)
                for ax in axs:
                    ax.set_xticks(range(len(minerals)))  # Match positions to minerals
                    ax.set_xticklabels(minerals, rotation=45, fontsize=12, fontweight="bold")

                # Legend only in the top subplot
                axs[0].legend(fontsize=10, loc="lower left", ncol=3)
                # handles, labels = axs[0].get_legend_handles_labels()
                # fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.1), ncol=3)

                # Remove x-axis grid
                axs[0].grid(False, axis="x")
                axs[1].grid(False, axis="x")

                # Save plot with constraint name
                results_folder = os.path.join(figures_data_path, f"MR_MN")
                os.makedirs(results_folder, exist_ok=True)
                plt.tight_layout() #rect=[0, 0.05, 1, 1]
                save_fig(os.path.join(results_folder, f"{col}_differences_subplot_{const}.png"))
                plt.close()
                     
    #### 
    # Differences for most variables as shares: not really shares, need to fix again
    ######
    make_plot = False
    if make_plot:
        results_file = os.path.join(output_data_path, "result_summaries", "combined_energy_transport_totals_by_stage.xlsx")
        output_excel_path = os.path.join(output_data_path, "regional_national_differences.xlsx")

        # Create an empty list to store all results before saving
        all_results = []
        
        for sd in scenarios_descriptions:
            sc_t = sd["scenario_type"]
            sc_n = sd["scenario_name"]
            scenarios = sd["scenarios"]
            sc_l = sd["scenario_labels"]

            # Read scenario data
            data_df = pd.read_excel(results_file, sheet_name=sc_t, index_col=[0, 1, 2, 3, 4]).reset_index()

            # If it's not "country_unconstrained", add the 2022 baseline
            if sc_t != "country_unconstrained":
                baseline_df = pd.read_excel(results_file, sheet_name="country_unconstrained", index_col=[0, 1, 2, 3, 4]).reset_index()
                data_df = pd.concat([baseline_df[baseline_df["year"] == 2022], data_df], axis=0, ignore_index=True)

            # Filter data for the required country
            country_codes = ["ZMB"]  # Replace with actual ISO3 country code
            data_df = data_df[(data_df["iso3"].isin(country_codes)) & (data_df["production_tonnes"] > 0)]
            data_df["reference_mineral"] = data_df["reference_mineral"].str.title()
            reference_minerals = sorted(data_df["reference_mineral"].unique())

            for col, col_t, m_t in zip(columns, column_titles, multiply_factors):
                dfall = []
                df = data_df[data_df["processing_stage"] > 0]
                stages = sorted(df["processing_stage"].unique())
                stage_colors = stage_colors[:len(stages)]  # Adjust colors dynamically

                for sc in scenarios:
                    m_df = pd.DataFrame(reference_minerals, columns=["reference_mineral"])
                    for st in stages:
                        s_df = df[(df["processing_stage"] == st) & (df["scenario"] == sc)]
                        if not s_df.empty:
                            s_df[col] = m_t * s_df[col]
                            s_df.rename(columns={col: f"Stage {st}"}, inplace=True)
                            m_df = pd.merge(m_df, s_df[["reference_mineral", f"Stage {st}"]], how="left", on="reference_mineral").fillna(0)
                        else:
                            m_df[f"Stage {st}"] = 0.0
                    dfall.append(m_df.set_index(["reference_mineral"]))

                # Compute differences: Regional - National (for 2030 and 2040)
                if len(dfall) >= 3:  # Ensure at least 3 scenarios exist
                    delta_2030 = dfall[1].subtract(dfall[0], fill_value=0)  # 2030 Regional - National
                    delta_2040 = dfall[2].subtract(dfall[1], fill_value=0)  # 2040 Regional - National
                else:
                    continue  # Skip if we don't have enough data

                # Convert DataFrames to Long Format for Plotting
                df_long_2030 = delta_2030.reset_index().melt(id_vars=["reference_mineral"], var_name="Stage", value_name="Difference")
                df_long_2040 = delta_2040.reset_index().melt(id_vars=["reference_mineral"], var_name="Stage", value_name="Difference")

                # **Plotting with Subplots**
                fig, axs = plt.subplots(2, 1, figsize=(12, 12), dpi=500, sharex=True)

                # Set bar width dynamically based on minerals and stages
                bar_width = 0.2
                minerals = df_long_2030["reference_mineral"].unique()
                stages = df_long_2030["Stage"].unique()
                x = np.arange(len(minerals))  # Base positions for each mineral

                # Plot for 2030
                for i, stage in enumerate(stages):
                    subset = df_long_2030[df_long_2030["Stage"] == stage]
                    positions = x - bar_width * (len(stages) / 2) + i * bar_width
                    axs[0].bar(positions, subset["Difference"], width=bar_width, label=f"{stage}", color=stage_colors[i])

                axs[0].set_title(f"{sc_n} Scenario: 2030 Difference (Regional - National)", fontsize=16, fontweight="bold")
                axs[0].grid(axis="y", linestyle="--", alpha=0.6)  # Grid only on Y-axis
                axs[0].set_ylabel(col_t, fontsize=14, fontweight="bold")

                # Plot for 2040
                for i, stage in enumerate(stages):
                    subset = df_long_2040[df_long_2040["Stage"] == stage]
                    positions = x - bar_width * (len(stages) / 2) + i * bar_width
                    axs[1].bar(positions, subset["Difference"], width=bar_width, label=f"{stage}", color=stage_colors[i])

                axs[1].set_title(f"{sc_n} Scenario: 2040 Difference (Regional - National)", fontsize=16, fontweight="bold")
                axs[1].grid(axis="y", linestyle="--", alpha=0.6)  # Grid only on Y-axis
                axs[1].set_ylabel(col_t, fontsize=14, fontweight="bold")

                # X-axis formatting
                axs[1].set_xticks(x)
                axs[1].set_xticklabels(minerals, rotation=45, fontsize=12, fontweight="bold")

                # Legend only in the top subplot
                axs[0].legend(fontsize=10, loc="upper left", ncol=3)

                # Remove x-axis grid
                axs[0].grid(False, axis="x")
                axs[1].grid(False, axis="x")

                # Save plot
                results_folder = os.path.join(figures_data_path, f"{sc_n}_{sc_t}")
                os.makedirs(results_folder, exist_ok=True)
                plt.tight_layout()
                save_fig(os.path.join(results_folder, f"{col}_{sc_t}_differences_subplot.png"))
                plt.close()
    
   
    """Unconstrained country and regional comparisons: annual production
    """
    make_plot = False # not useful
    if make_plot is True:
        multiply_factor = 1.0e-3
        results_file = os.path.join(output_data_path,
                                "result_summaries",
                                "combined_transport_totals_by_stage.xlsx")
        constraints = ["country_unconstrained","region_unconstrained"]
        reference_mineral_colors = ["#f46d43","#fdae61","#fee08b","#c2a5cf","#66c2a5","#3288bd"]
        reference_minerals = ["copper","cobalt","manganese","lithium","graphite","nickel"]
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
                    results_folder = os.path.join(figures,f"{cs}")
                    if os.path.exists(results_folder) == False:
                        os.mkdir(results_folder)
                    df = pd.read_excel(
                                        results_file,
                                        sheet_name=cs,
                                        index_col=[0,1,2,3,4])
                    df = df.reset_index()
                    countries = sorted(list(set(df["iso3"].values.tolist())))
                    df = df[(df["scenario"] == scenario) & 
                        (df["processing_stage"] > 0) & 
                        (df["reference_mineral"] == rf) & 
                        (df["iso3"].isin(country_codes))]  
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
                        df_c = pd.concat(dfall)
                        df_c = df_c[df_c.index.isin(country_codes)]
                    
            fig, ax = plt.subplots(1,1,figsize=(18,9),dpi=500)   
            ax = new_plot_clustered_stacked(
                                        fig,ax,df_c,stage_colors,
                                        labels=["National 2030","Regional 2030","National 2040","Regional 2040"],
                                        ylabel="Annual export volumes (000' tonnes)", 
                                        title=f"{rf.title()} Mid National and Mid Regional scenario comparisons")
            plt.grid()
            plt.tight_layout()
            save_fig(os.path.join(results_folder, f"{rf}_MN_MR_export_comparisions_side_by_side.png"))
            plt.close()
    
    ############
    # DELTA Constrained and unconstrained country and regional comparisons: annual production
    ############
    make_plot = False
    if make_plot:
        multiply_factor = 1.0e-3
        results_file = os.path.join(output_data_path, "result_summaries", "combined_energy_transport_totals_by_stage.xlsx")
        constraints = ["country_unconstrained", "region_unconstrained",
                        "country_constrained", "region_constrained"]
        reference_minerals = ["cobalt", "copper", "graphite", "lithium", "manganese", "nickel"]
        reference_mineral_colors = ["#fdae61", "#f46d43", "#66c2a5", "#c2a5cf", "#fee08b", "#3288bd"]
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
                    results_folder = os.path.join(figures,f"{cs}")
                    if os.path.exists(results_folder) == False:
                        os.mkdir(results_folder)
                    print(f"{cs}")
                    df = pd.read_excel(results_file, sheet_name=cs, index_col=[0, 1, 2, 3, 4])
                    df = df.reset_index()

                    # Filter by scenario, processing_stage, and mineral
                    filtered_df = df[(df["scenario"] == scenario) & 
                                    (df["processing_stage"] > 0) & 
                                    (df["reference_mineral"] == rf)&
                                    (df["iso3"] == country_codes)]

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
                    df_c = dfs[dfs['iso3'].isin(country_codes)]
                    # print("dfs", dfs)

            # Compute the differences for 2030 and 2040
            delta_df = []
            if len(dfs) >= 4:  # Ensure we have all the required DataFrames
                delta_df.append(df_c[1].subtract(df_c[0], fill_value=0))  # 2030 Regional - National 
                delta_df.append(df_c[3].subtract(df_c[2], fill_value=0))  # 2040 Regional - National
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
                axs[0] = new_plot_clustered_stacked(
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

            if not delta_df_2040.empty:
                axs[1] = new_plot_clustered_stacked(
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
            save_fig(os.path.join(results_folder, f"{rf}_unconstrained_MN_MR_production_comparisons_2030_2040.png"))
            plt.close()

            for scenario in scenarios:
                for cs in constraints[3:]:
                    df = pd.read_excel(results_file, sheet_name=cs, index_col=[0, 1, 2, 3, 4])
                    df = df.reset_index()

                    # Filter by scenario, processing_stage, and mineral
                    filtered_df = df[(df["scenario"] == scenario) & 
                                    (df["processing_stage"] > 0) & 
                                    (df["reference_mineral"] == rf) &
                                    (df["iso3"].isin(country_codes))]

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
                    df_c = pd.concat(dfs)
                    df_c = df_c[df_c.index.isin(country_codes)]

            # Compute the differences for 2030 and 2040
            delta_df = []
            if len(dfs) >= 4:  # Ensure we have all the required DataFrames
                delta_df.append(df_c[1].subtract(df_c[0], fill_value=0))  # 2030 Regional - National 
                delta_df.append(df_c[3].subtract(df_c[2], fill_value=0))  # 2040 Regional - National

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
                axs[0] = new_plot_clustered_stacked(
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

            if not delta_df_2040.empty:
                axs[1] = new_plot_clustered_stacked(
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

            plt.tight_layout()
            save_fig(os.path.join(results_folder, f"{rf}_constrained_MN_MR_production_comparisons_2030_2040.png"))
            plt.close()
    

if __name__ == '__main__':
    CONFIG = load_config()
    main(CONFIG)
