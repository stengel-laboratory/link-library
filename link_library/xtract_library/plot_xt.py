import link_library.plot_library as plib
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import link_library as ll


class PlotMaster(object):

    def __init__(self, out_folder='plots'):
        self.out_folder = out_folder
        self.xt_db = ll.xTractDB()

    def plot_associated_mono_links(self, df, df_dist=None):
        # sns.set_style("whitegrid")
        if df_dist is not None:
            df = pd.merge(df, df_dist, how='outer', on=self.xt_db.uxid_string)
            # filter link groups without at least one distance value (i.e. the xlink should have a distance)
            df = df.groupby(self.xt_db.link_group_string).filter(lambda x: x[self.xt_db.dist_string].count() > 0)

        ax = sns.scatterplot(x=self.xt_db.link_group_string, y=self.xt_db.log2_string, style=self.xt_db.type_string, hue=self.xt_db.type_string, data=df,
                             palette="Set1", s=150)
        ax.xaxis.set_tick_params(rotation=90)
        # ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
        num_x = df[self.xt_db.link_group_string].nunique()
        b_alt = True
        for x in range(num_x):
            if b_alt:
                c = 'lightcoral'
                b_alt = False
            else:
                c = 'skyblue'
                b_alt = True
            ax.vlines(x=x, ymin=min(df[self.xt_db.log2_string]), ymax=max(df[self.xt_db.log2_string]), linestyles=':', colors=c)
        # ax.hlines(y=0, xmin=min(df[link_group_string]), xmax=max(df[link_group_string]), linestyles='-', colors='grey')
        ax.hlines(y=1, xmin=0, xmax=num_x-1, linestyles='--', colors='grey')
        ax.hlines(y=-1, xmin=0, xmax=num_x-1, linestyles='--', colors='grey')
        if df_dist is not None:
            df_xtract_xl_only = df[~df[self.xt_db.dist_string].isnull()].copy()
            df_xtract_xl_only[self.xt_db.dist_string] = df_xtract_xl_only[self.xt_db.dist_string].round().astype('int32')
            plib.label_point(df_xtract_xl_only[self.xt_db.link_group_string], df_xtract_xl_only[self.xt_db.log2_string], df_xtract_xl_only[self.xt_db.dist_string], ax)
        plib.save_fig("xtract_monolinks", self.out_folder)
        plt.clf()


    def plot_dist_vs_quant(self, df, df_dist):
        df = pd.merge(df, df_dist, how='outer', on=self.xt_db.uxid_string)
        # filter link groups without at least one distance value (i.e. the xlink should have a distance)
        df = df.groupby(self.xt_db.link_group_string).filter(lambda x: x[self.xt_db.dist_string].count() > 0)
        sns.regplot(data=df, y=self.xt_db.dist_string, x=self.xt_db.log2_string)
        plib.save_fig("xtract_dist_vs_quant", self.out_folder)


    def plot_mono_vs_xlink_quant(self, df):
        df_xlinks = df[df[self.xt_db.type_string] == self.xt_db.type_xlink_string]
        df_monos = df[df[self.xt_db.type_string] == self.xt_db.type_mono_string]
        # df_xlinks = df_xlinks.rename(index=str, columns={log2_string: log2_string+"_xlink"})
        df_monos = df_monos.rename(index=str, columns={self.xt_db.log2_string: self.xt_db.log2_string + "_mono"})
        df_monos = df_monos[[self.xt_db.link_group_string, self.xt_db.log2_string+"_mono"]]
        df = pd.merge(df_xlinks, df_monos, on=self.xt_db.link_group_string)
        ax = sns.regplot(x=self.xt_db.log2_string+"_mono", y=self.xt_db.log2_string, data=df)
        plib.save_fig("xtract_xlink_vs_monolink", self.out_folder)
        plt.clf()
        df['mono_mean'] = df.groupby([self.xt_db.uxid_string])[self.xt_db.log2_string + "_mono"].transform('mean')
        ax = sns.regplot(x="mono_mean", y=self.xt_db.log2_string, data=df)
        plib.save_fig("xtract_xlink_vs_monolink_mean", self.out_folder)
        plt.clf()