import matplotlib.pyplot as plt
import pandas as pd
import os


"""
2018-10-19 - Kai Kammer
Library containing shared plotting functions
"""

def create_plots_dir(out_dir = 'plots'):
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def save_fig(out_name, out_dir='plots'):
    out_dir = create_plots_dir(out_dir)
    # plt.tight_layout()
    figure = plt.gcf()  # get current figure
    figure.set_size_inches(19,12)
    plt.tight_layout()
    plt.tight_layout()
    plt.tight_layout()
    plt.savefig("{0}/plot_{1}.png".format(out_dir, out_name))


def save_g(fg, out_name, out_dir='plots', **kwargs):
    out_dir = create_plots_dir(out_dir)
    fg.savefig("{0}/plot_{1}.png".format(out_dir, out_name), **kwargs)


def save_n_show_fig(out_name, out_dir = 'plots'):
    out_dir = create_plots_dir(out_dir)
    plt.tight_layout()
    figure = plt.gcf()  # get current figure
    figure.set_size_inches(19,12)
    plt.tight_layout()
    plt.tight_layout()
    plt.tight_layout()
    plt.savefig("{0}/plot_{1}.png".format(out_dir, out_name))
    plt.show()

def facet_grid_vertical_label(fg, size=None):
    for row in fg.axes:
        for ax in row:
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90, size=size)

def label_point(x, y, val, ax):
    a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
    for i, point in a.iterrows():
        ax.text(point['x'], point['y']+0.15, str(point['val']), fontsize=14, fontweight=600)

def map_point(x,y, annot, **kwargs):
    # x = x.unique()
    # print(x)
    # print(y)
    y_anno_dict = {y.values[n]:annot.values[n] for n in range(len((y)))}
    print(y_anno_dict)
    for i, y_val in enumerate(y_anno_dict.keys()):
        # print(str(annot.values[i]), x.values[i], y.values[i])
        plt.annotate(str(y_anno_dict[y_val]), xy=(i, y_val), fontsize=8,
                    color=kwargs.get("color", "k"),
                    bbox=dict(pad=.9, alpha=1, fc='w', color='none'),
                    va='center', ha='center', weight='bold')

def horizontal_mean_line(y, **kwargs):
    plt.axhline(y.mean(), **kwargs)