import matplotlib.pyplot as plt
import pandas as pd
import os


"""
2018-10-19 - Kai Kammer
Library containing shared plotting functions
"""

def create_plots_dir(out_dir = 'plots'):
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(out_dir + '/svg', exist_ok=True)
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
    plt.savefig("{0}/plot_{1}.svg".format(out_dir + '/svg', out_name))


def save_g(fg, out_name, out_dir='plots', **kwargs):
    out_dir = create_plots_dir(out_dir)
    fg.savefig("{0}/plot_{1}.png".format(out_dir, out_name), **kwargs)
    fg.savefig("{0}/plot_{1}.svg".format(out_dir + '/svg', out_name), **kwargs)


def save_n_show_fig(out_name, out_dir = 'plots'):
    out_dir = create_plots_dir(out_dir)
    plt.tight_layout()
    figure = plt.gcf()  # get current figure
    figure.set_size_inches(19,12)
    plt.tight_layout()
    plt.tight_layout()
    plt.tight_layout()
    plt.savefig("{0}/plot_{1}.png".format(out_dir, out_name))
    plt.savefig("{0}/plot_{1}.svg".format(out_dir + '/svg', out_name))
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
    # create a ordered list of unique value triplets
    zip_y_annotate = list(dict.fromkeys(zip(x,y,annot)))
    # get the x_labels from matplotlib
    x_locs, x_labels = plt.xticks()
    x_labels = [t.get_text() for t in x_labels]
    # dict with x label:pos association
    x_label_loc_dict = dict(zip(x_labels, x_locs))
    x_unique = [e[0] for e in zip_y_annotate]
    y_unique =[e[1] for e in zip_y_annotate]
    annot_unique = [e[2] for e in zip_y_annotate]
    # only retain the x positions which actually have y values to annotate
    x_pos_valid = [x_label_loc_dict[str(n)] for n in x_unique]
    # again create a ordered list of unique triplets; this only containing valid x values
    zip_y_annotate = list(dict.fromkeys(zip(x_pos_valid, y_unique, annot_unique)))
    for x, y, annot in zip_y_annotate:
        plt.annotate(str(annot), xy=(x, y), fontsize=10,
                    color='black',
                    bbox=dict(pad=.9, alpha=.9, fc='w', color='none'),
                    va='center', ha='center', weight='bold')
        # color: text color; bbox: dict defining the bounding box; fc: fill color;
        # zorder: draw order of the element; we need a high number here; otherwise the hue parameter of sns overwrites

def horizontal_mean_line(y, **kwargs):
    plt.axhline(y.mean(), **kwargs)