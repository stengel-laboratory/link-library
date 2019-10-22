import matplotlib.pyplot as plt
import pandas as pd
import shutil
import os


"""
2018-10-19 - Kai Kammer
Library containing shared plotting functions
"""

def create_plots_dir(out_dir = 'plots'):
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(out_dir + '/svg', exist_ok=True)
    os.makedirs(out_dir + '/csv', exist_ok=True)
    return out_dir


def save_fig(out_name, df=None, out_dir='plots'):
    out_dir = create_plots_dir(out_dir)
    # plt.tight_layout()
    figure = plt.gcf()  # get current figure
    figure.set_size_inches(19,12)
    plt.tight_layout()
    plt.tight_layout()
    plt.tight_layout()
    plt.savefig("{0}/plot_{1}.png".format(out_dir, out_name))
    plt.savefig("{0}/plot_{1}.svg".format(out_dir + '/svg', out_name))
    if df is not None:
        df.to_csv("{0}/plot_{1}.csv".format(out_dir + '/csv', out_name))

def get_altair_driver():
    chromedriver = 'chromdriver'
    chrome = 'chrome'
    chromium = 'chromium'
    geckodriver = 'geckodriver'
    ff = 'firefox'
    if shutil.which(ff):
        if shutil.which(geckodriver):
            return ff
        else:
            print('Warning: Firefox is installed but geckodriver is not. Cannot save via Firefox')
    if shutil.which(chrome) or shutil.which(chromium):
        if shutil.which(chromedriver):
            return chrome
        else:
            print('Warning: Chrome is installed but chromedriver is not. Cannot save via Chrome')
            print('Exiting without plotting')
            exit(1)


def save_g(fg, out_name, df_list=None, out_dir='plots', **kwargs):
    out_dir = create_plots_dir(out_dir)
    if 'altair' in str(type(fg)):
        driver = get_altair_driver()
        fg.save("{0}/plot_{1}.png".format(out_dir, out_name), scale_factor=2, webdriver=driver)
        fg.save("{0}/plot_{1}.svg".format(out_dir + '/svg', out_name), scale_factor=2, webdriver=driver)
    else:
        fg.savefig("{0}/plot_{1}.png".format(out_dir, out_name), **kwargs)
        fg.savefig("{0}/plot_{1}.svg".format(out_dir + '/svg', out_name), **kwargs)
    if df_list is not None:
        # is it really a list?
        if isinstance(df_list, list):
            for n, df_list in enumerate(df_list):
                if isinstance(df_list, pd.DataFrame):
                    df_list.to_csv(f"{out_dir + '/csv'}/plot_{out_name}_{n}.csv")
                else:
                    print(f"ERROR: Unknown object passed for saving: {type(df_list)}")
        # it's just a single dataframe then
        elif isinstance(df_list, pd.DataFrame):
            df_list.to_csv(f"{out_dir + '/csv'}/plot_{out_name}.csv")
        else:
            print(f"ERROR: Unknown object passed for saving: {type(df_list)}")


def save_n_show_fig(out_name, df=None, out_dir = 'plots'):
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
    if df is not None:
        df.to_csv("{0}/plot_{1}.csv".format(out_dir + '/csv', out_name))


def facet_grid_vertical_label(fg, size=None):
    for row in fg.axes:
        for ax in row:
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90, size=size)


def label_point(x, y, val, ax):
    a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
    for i, point in a.iterrows():
        ax.text(point['x'], point['y']+0.15, str(point['val']), fontsize=14, fontweight=600)


def map_point(x,y, annot, boxplot=False, **kwargs):
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
    if boxplot:
        color = 'white'
        bbox=dict(boxstyle='circle',pad=.1, alpha=1., fc='black', color='none')
    else:
        color = 'black'
        bbox=dict(boxstyle='circle',pad=.1, alpha=.9, fc='w', color='none')
    for x, y, annot in zip_y_annotate:
        plt.annotate(str(annot), xy=(x, y), fontsize=10,
                    color=color,
                    bbox=bbox,
                    va='center', ha='center', weight='bold',zorder=100)
        # color: text color; bbox: dict defining the bounding box; fc: fill color;
        # zorder: draw order of the element; we need a high number here; otherwise the hue parameter of sns overwrites

def horizontal_mean_line(y, **kwargs):
    plt.axhline(y.mean(), **kwargs)
