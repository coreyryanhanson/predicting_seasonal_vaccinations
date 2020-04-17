import re
import math
import colorsys
import matplotlib
import numpy as np
import pandas as pd
import scipy.stats as scs
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix


def calc_z_from_proportions(v1, v2, n1, n2):
    p1, p2, p = v1/n1, v2/n1, (v1+v2)/(n1+n2)
    se1, se2 = p*(1-p)/n1, p*(1-p)/n2
    num = abs(p1 - p2)
    denom = math.sqrt(se1 + se2)
    return num/denom

def calc_chi_squared(df, column, target):
    table = pd.crosstab(df[column], df[target])
    print(table)
    return scs.chi2_contingency(table)

def chi_squared_loop(df, target, alpha):
    failed_cols, pass_cols = [], []
    for column in df.columns:
        chi = calc_chi_squared(df, column, target)
        if chi[1] > alpha:
            failed_cols.append([column, chi[1]])
        else:
            pass_cols.append([column, chi[1]])
    return failed_cols, pass_cols

def correlated_columns_by_threshold(df, columns, threshhold):
    corr = df[columns].corr()
    results = pd.Index([])
    for column in corr.columns:
        results = results.union(corr[column][df_test_threshhold(corr, column, threshhold)].index.drop(column))
    return results

def df_test_threshhold(df,column, threshhold):
    return (df[column] >= threshhold) & (df[column] >= -threshhold)

def z_test_proportions(df, column, target, target_val, alpha):
    failed_vals, passed_vals = [], []
    table = pd.crosstab(df[column], df[target])
    n, n_val = df[target].size, table[target_val].sum()
    for i in np.arange(0, table[target_val].size):
        row = table.iloc[i]
        n1, v1 = row.sum(), row[target_val]
        n2, v2 = n - n1, n_val - v1
        z = calc_z_from_proportions(v1, v2, n1, n2)
        p = scs.norm.cdf(-z) * 2
        if p > alpha:
            failed_vals.append([table.index[i], z, p])
        else:
            passed_vals.append([table.index[i], z, p])
    return failed_vals, passed_vals



def dynamic_heatmap(df, columns, fontsize=20, annot=False, palette=None, figsize=(15, 10), squaresize=500):
    """Plots a heatmap that changes size values depending on correlation Adapted from:
    https://towardsdatascience.com/better-heatmaps-and-correlation-matrix-plots-in-python-41445d0f2bec"""

    plt.figure(figsize=figsize)
    corr = df[columns].corr()
    sns.set(style="dark")
    grid_bg_color = sns.axes_style()['axes.facecolor']

    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    corr = pd.melt(corr.reset_index(),
                   id_vars='index')  # Unpivot the dataframe, so we can get pair of arrays for x and y
    corr.columns = ['x', 'y', 'value']

    x = corr['x']
    y = corr['y']
    size = corr['value'].abs()

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=figsize)
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        horizontalalignment='right');

    # Mapping from column names to integer coordinates
    x_labels = [v for v in sorted(x.unique())]
    y_labels = [v for v in sorted(y.unique())]
    x_to_num = {p[1]: p[0] for p in enumerate(x_labels)}
    y_to_num = {p[1]: p[0] for p in enumerate(y_labels)}

    size_scale = squaresize

    if palette:
        n_colors = len(palette)
    else:
        n_colors = 256  # Use 256 colors for the diverging color palette
        palette = sns.diverging_palette(20, 220, n=n_colors) # Create the palette
    color_min, color_max = [-1,
                            1]  # Range of values that will be mapped to the palette, i.e. min and max possible correlation
    color = corr["value"]

    def value_to_color(val):
        val_position = float((val - color_min)) / (
                    color_max - color_min)  # position of value in the input range, relative to the length of the input range
        ind = int(val_position * (n_colors - 1))  # target index in the color palette
        return palette[ind]

    plot_grid = plt.GridSpec(1, 15, hspace=0.2, wspace=0.1)  # Setup a 1x15 grid
    ax = plt.subplot(plot_grid[:, :-1])  # Use the leftmost 14 columns of the grid for the main plot

    ax.scatter(
        x=x.map(x_to_num),  # Use mapping for x
        y=y.map(y_to_num),  # Use mapping for y
        s=size * size_scale,  # Vector of square sizes, proportional to size parameter
        c=color.apply(value_to_color),  # Vector of square colors, mapped to color palette
        marker='s'  # Use square as scatterplot marker
    )

    # Show column labels on the axes
    ax.set_xticks([x_to_num[v] for v in x_labels])
    ax.set_xticklabels(x_labels, rotation=45, horizontalalignment='right')
    ax.set_yticks([y_to_num[v] for v in y_labels])
    ax.set_yticklabels(y_labels)
    #     ax.set_fontsize(font_scale)
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(fontsize)

    numbers = corr['value'].round(decimals=2)

    if annot:
        for i, txt in enumerate(numbers):
            annot_font_size = int(fontsize * size[i] * annot)
            ax.annotate(txt, (x.map(x_to_num)[i], y.map(x_to_num)[i]),
                        horizontalalignment="center", verticalalignment="center",
                        color=grid_bg_color, fontweight="black", fontsize=annot_font_size)

    ax.grid(False, 'major')
    ax.grid(True, 'minor')
    ax.set_xticks([t + 0.5 for t in ax.get_xticks()], minor=True)
    ax.set_yticks([t + 0.5 for t in ax.get_yticks()], minor=True)
    ax.set_xlim([-0.5, max([v for v in x_to_num.values()]) + 0.5])
    ax.set_ylim([-0.5, max([v for v in y_to_num.values()]) + 0.5])

    # Add color legend on the right side of the plot
    ax = plt.subplot(plot_grid[:, -1])  # Use the rightmost column of the plot

    col_x = [0] * len(palette)  # Fixed x coordinate for the bars
    bar_y = np.linspace(color_min, color_max, n_colors)  # y coordinates for each of the n_colors bars

    bar_height = bar_y[1] - bar_y[0]
    ax.barh(
        y=bar_y,
        width=[5] * len(palette),  # Make bars 5 units wide
        left=col_x,  # Make bars start at 0
        height=bar_height,
        color=palette,
        linewidth=0
    )
    ax.set_xlim(1, 2)  # Bars are going from 0 to 5, so lets crop the plot somewhere in the middle
    ax.grid(False)  # Hide grid
    ax.set_xticks([])  # Remove horizontal ticks
    ax.set_yticks(np.linspace(min(bar_y), max(bar_y), 3))  # Show vertical ticks for min, middle and max
    ax.yaxis.tick_right()  # Show vertical ticks on the right
    plt.show()

def hex_to_hsl(hex_color, dec=False):
    rgb = hex_to_rgb(hex_color, True)
    raw_hls = colorsys.rgb_to_hls(*rgb)
    if dec:
        return raw_hsl
    else:
        h, l, s = int(raw_hls[0]*360), int(raw_hls[1]*100), int(raw_hls[2]*100)
        return [h, s, l]

def hex_to_rgb(hex_color, dec=False):
    if hex_color[0] and len(hex_color) == 7:
        search = re.search("#(\w{2})(\w{2})(\w{2})", hex_color)
        if dec:
            return [int(search.group(i), 16)/256 for i in np.arange(1,4)]
        else:
            return [int(search.group(i), 16) for i in np.arange(1,4)]
    else:
        print(f"{hex_color} is not formatted correctly")


def confusion_matrix_graph(model, X_test, y_test, font_scale=2, style="darkgrid", palette=None, figsize=(14, 12)):
    sns.set_style(style)
    sns.set(font_scale=font_scale)
    f, ax = plt.subplots(figsize=figsize)
    converted_pal = matplotlib.colors.ListedColormap(sns.color_palette(palette))
    plot_confusion_matrix(model, X_test, y_test, cmap=converted_pal, ax=ax)
    for text in ax.texts:
        label = text.get_text()
        label = int(float(label))
        text.set_text(f"{label}")
    plt.show()

def independant_palette(color1, color2, sep=10, n_colors=256):
    hsl1, hsl2 = hex_to_hsl(color1), hex_to_hsl(color2)
    pal1 = sns.diverging_palette(hsl1[0], hsl2[0], s=hsl1[1], l=hsl1[2], sep=sep, n=n_colors)
    pal2 = sns.diverging_palette(hsl1[0], hsl2[0], s=hsl2[1], l=hsl2[2], sep=sep, n=n_colors)
    return [*pal1[0:n_colors//2], *pal2[n_colors//2:]]

def show_feature_importances(model, df, figsize=(14, 12), palette=None, font_scale=1, ascending=False, rows=12, style="darkgrid"):
    sns.set_style(style)
    f, ax = plt.subplots(figsize=figsize)
    sns.set(font_scale=font_scale)
    importance = pd.DataFrame(model.feature_importances_, index=df.columns).reset_index()
    importance.columns = pd.Index(["Feature", "Importance"])
    sns.barplot(y="Feature", x="Importance", data=importance.sort_values("Importance",ascending=ascending).iloc[0:rows],
                palette=palette, ax=ax)


def imbal_graph(df, target, title):
    print('Target Variable: ' + target)
    print('\n')
    summary = df.groupby([target])[target].count()
    print(summary)
    print('\n')
    print('Percentages')
    print(summary/summary.sum())
    pal = sns.color_palette(("#102ca8", "#ee823e"))
    plt.figure(figsize = (10,5))
    sns.countplot(df[target], palette=pal)
    plt.title(target + ":  " + title)
    plt.ylabel('Count')
    plt.show()
    plt.savefig(f'graphs/{target}_imbalance_check_barchart.png')


def stacked(df, columns, target, target_lab):
    for column in columns:
        title = column + " " + target + " summary"
        summary = df.groupby([column,target])[column].count().unstack()
        print(title)
        print('\n')
        print(summary)
        print(summary/summary.sum(axis=0))
        print(summary.divide(summary.sum(axis=1),axis=0))
        pal = sns.color_palette(("#102ca8", "#ee823e"))
        sns.set_palette(pal)
        p = summary.plot(kind = 'bar', stacked = True, title = title,)
        p.set_xlabel(column)
        p.set_ylabel('Count')
        p.legend(target_lab)
        plt.show()
        plt.savefig(f'graphs/{target}_{column}_barchart.png')