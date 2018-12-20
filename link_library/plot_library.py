import matplotlib.pyplot as plt
import os


"""
2018-10-19 - Kai Kammer
Library containing shared plotting functions
"""

def create_plots_dir(out_dir = 'plots'):
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def save_fig(out_name, out_dir = 'plots'):
    out_dir = create_plots_dir(out_dir)
    # plt.tight_layout()
    figure = plt.gcf()  # get current figure
    figure.set_size_inches(19,12)
    plt.tight_layout()
    plt.tight_layout()
    plt.tight_layout()
    plt.savefig("{0}/plot_{1}.png".format(out_dir, out_name))


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