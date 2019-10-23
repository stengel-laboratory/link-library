import seaborn as sns
import pandas as pd
import numpy as np
import link_library.plot_library as plib
import link_library as ll
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
from scipy.stats import norm
from functools import reduce
import altair as alt
import scipy.stats as stats


class PlotMaster(object):

    def __init__(self, bag_cont, out_folder='plots'):
        self.out_folder = out_folder
        self.bag_cont = bag_cont

    # TODO: separation by link type only works properly when the types have different uxids
    # TODO: this means loop ind xlinks with the same u(x)id will be merged into one link
    def plot_clustermap(self):
        def cluster_formatter(cg, df_links):
            # create invisible plot to create a legend for the row labeling; adapted from https://stackoverflow.com/a/27992943
            for label in df_links[self.bag_cont.col_link_type].unique():
                cg.ax_col_dendrogram.bar(0, 0, color=lut[label],
                                         label=label, linewidth=0)
            cg.ax_col_dendrogram.legend(loc="upper right")
            ax = cg.ax_heatmap
            # hiding the y labels as they are really not informative
            ax.tick_params(axis='y', which='both', right=False, labelright=False)

        # how far up we want to sum up values; done first (understand it as an exclusion list)
        sum_list = [self.bag_cont.col_exp, self.bag_cont.col_bio_rep,
                    self.bag_cont.col_tech_rep]  # self.bag_cont.col_tech_rep, self.bag_cont.col_charge self.bag_cont.col_weight_type
        # how far do we take the mean of values; done second
        mean_list = [self.bag_cont.col_exp, self.bag_cont.col_bio_rep]
        # the following two lines will take the mean of the tech replicates; should always be used if don't want separate tech reps
        # mean_list = list(sum_list)
        # mean_list.remove(self.bag_cont.col_tech_rep)
        df = self.bag_cont.get_pivot(sum_list, mean_list, pivot_on=self.bag_cont.col_area_sum_total, log2=True)
        df_links = self.bag_cont.df_orig.drop_duplicates(subset=[self.bag_cont.col_level])
        df_links = df_links[[self.bag_cont.col_level, self.bag_cont.col_link_type]]
        df_label = df_links.set_index(self.bag_cont.col_level)
        df_label = df_label[self.bag_cont.col_link_type]
        # up to 3 colors are defined right now; more are possible
        lut = dict(zip(df_label.unique(), [(129 / 255, 201 / 255, 191 / 255), (80 / 255, 124 / 255, 216 / 255),
                                           (255 / 255, 221 / 255, 153 / 255)]))  # alternatively: 'cbr'
        row_colors = df_label.map(lut)
        # df, dist_orig, dist_imputed = self.bag_cont.fillna_with_normal_dist(df)
        # sns.distplot(dist_orig, kde=True, fit=norm)
        # self.plot_fig(name='cluster', extra="orig_dist")
        plt.clf()
        # if len(dist_imputed) > 1:
        #     sns.distplot(dist_imputed, kde=True, fit=norm)
        #     self.plot_fig(name='cluster', extra="imputed_dist")
        # print("Non imputed values: {0}. Imputed values: {1}".format(len(dist_orig),len(dist_imputed)))
        cg = sns.clustermap(data=df, cmap="mako_r", metric='canberra', row_colors=row_colors, xticklabels=True)
        cluster_formatter(cg, df_links)
        self.plot_fig(name="cluster", g=cg, dpi=300, df_list=df)
        cg = sns.clustermap(data=df, cmap="mako_r", z_score=0, metric='canberra', row_colors=row_colors,
                            xticklabels=True)
        cluster_formatter(cg, df_links)
        self.plot_fig(name='cluster', extra="z_row", g=cg, dpi=300)
        cg = sns.clustermap(data=df, cmap="mako_r", z_score=1, metric='canberra', row_colors=row_colors,
                            xticklabels=True)
        cluster_formatter(cg, df_links)
        self.plot_fig(name='cluster', extra="z_col", g=cg, dpi=300)

    def plot_bar(self):
        sum_list = [self.bag_cont.col_exp, self.bag_cont.col_bio_rep, self.bag_cont.col_tech_rep]
        mean_list = [self.bag_cont.col_exp]
        df = self.bag_cont.get_group(sum_list, mean_list, self.bag_cont.col_area_sum_total)
        # filter zero ms1 intensities
        df = df.loc[df[self.bag_cont.col_area_sum_total] > 0]
        # filter ids not found in at least two experiments (as each id exists only once for each experiment)
        df = df.groupby(self.bag_cont.col_level).filter(lambda x: len(x) > 1)
        df = df.sort_values([self.bag_cont.col_area_sum_total], ascending=False)
        # filter by log2ratio
        # df = df[(df[col_log2ratio] > 3) | (df[col_log2ratio] < -3)]
        df = df.reset_index(drop=True)
        # following line can be used to only plot partial data
        # df = df.loc[df[index_string] < len(df.index)]
        ax = sns.barplot(x=self.bag_cont.col_level, y=self.bag_cont.col_area_sum_total, hue=self.bag_cont.col_exp,
                         data=df, ci='sd')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        ax.set(title="{0} col_level".format(self.bag_cont.col_level), yscale='log')
        self.plot_fig(name="bar", df_list=df)

    def plot_ms1_area_std(self):
        sum_list = [self.bag_cont.col_exp, self.bag_cont.col_bio_rep, self.bag_cont.col_tech_rep,
                    self.bag_cont.col_weight_type, self.bag_cont.col_link_type]
        mean_list = [self.bag_cont.col_exp, self.bag_cont.col_link_type]
        df = self.bag_cont.get_stats(sum_list, mean_list, log2=True)
        df = df.sort_values([self.bag_cont.col_exp, self.bag_cont.col_level])
        # print(df)
        fg = sns.relplot(data=df, y='std', x='mean', hue=self.bag_cont.col_link_type, col=self.bag_cont.col_exp)
        for row in fg.axes:
            for ax in row:
                # ax.set_yscale('log', basey=2)
                # ax.set_xscale('log', basex=10)
                ax.set_ylim(ymin=0)
                ax.hlines(y=0.5, xmin=df['mean'].min(), xmax=df['mean'].max(), label="std=0.5", colors=['grey'])
        self.plot_fig(name="std", g=fg, df_list=df)
        # df = df[(df[self.bag_cont.col_log2ratio] < 1)&(df[self.bag_cont.col_log2ratio] > -1)]

    def plot_lowess(self):
        import statsmodels.nonparametric.smoothers_lowess as lw
        sum_list = [self.bag_cont.col_exp, self.bag_cont.col_bio_rep, self.bag_cont.col_tech_rep]
        mean_list = [self.bag_cont.col_exp]
        df_stats = self.bag_cont.get_stats(sum_list, mean_list)
        # n-way merge
        df_stats['snr'] = np.log10(df_stats['snr'])
        df_stats['lowess'] = lw.lowess(df_stats['mean'], df_stats['snr'], return_sorted=False)
        df_stats['lowdiff'] = df_stats['mean'] / df_stats['lowess']
        fg = sns.relplot(data=df_stats, x='snr', y='lowdiff', col=self.bag_cont.col_exp)
        self.plot_fig(name="log2ratio_l", g=fg, df_list=df_stats)

    def plot_log2ma(self):
        import statsmodels.nonparametric.smoothers_lowess as lw
        sum_list = [self.bag_cont.col_exp, self.bag_cont.col_bio_rep, self.bag_cont.col_tech_rep]
        mean_list = [self.bag_cont.col_exp]
        exp_ref = ll.input_log2_ref(self.bag_cont.exp_list)
        df = self.bag_cont.get_log2ratio(sum_list, mean_list, ref=exp_ref)
        df['lowess'] = lw.lowess(df[self.bag_cont.col_log2ratio], df[self.bag_cont.col_log2avg], return_sorted=False)
        df['lowdiff'] = df[self.bag_cont.col_log2ratio] - df['lowess']
        # n-way merge
        df = df.sort_values([self.bag_cont.col_exp, self.bag_cont.col_level])
        fg = sns.lmplot(data=df, y='lowdiff', x=self.bag_cont.col_log2avg, col=self.bag_cont.col_exp, lowess=True)
        for row in fg.axes:
            for ax in row:
                ax.hlines(y=0, xmin=df[self.bag_cont.col_log2avg].min(),
                          xmax=df[self.bag_cont.col_log2avg].max(), label="log2ratio=0", colors=['purple'])
        self.plot_fig(name="log2_ma", g=fg, df_list=df)

    def plot_log2ratio(self):
        def get_std_group(x):
            std = x.values[0]
            if 0 <= std < 0.1:
                return 0.1
            elif 0.1 <= std < 0.5:
                return 0.5
            elif 0.5 <= std < 1:
                return 1
            elif 1 <= std < 2:
                return 2
            elif 2 <= std < 3:
                return 3
            return 4

        sum_list = [self.bag_cont.col_exp, self.bag_cont.col_bio_rep, self.bag_cont.col_tech_rep, self.bag_cont.col_link_type]
        mean_list = [self.bag_cont.col_exp, self.bag_cont.col_link_type]
        exp_ref = ll.input_log2_ref(self.bag_cont.exp_list)
        df_pval = self.bag_cont.get_two_sided_ttest(sum_list, sum_list, ref=exp_ref)
        df_stats = self.bag_cont.get_stats(sum_list, mean_list)
        df_log2 = self.bag_cont.get_log2ratio(sum_list, mean_list, ref=exp_ref)
        # n-way merge
        df = reduce(lambda left, right: pd.merge(left, right, on=[self.bag_cont.col_level, self.bag_cont.col_exp]),
                    [df_stats, df_log2, df_pval])
        df = df.sort_values([self.bag_cont.col_exp, self.bag_cont.col_level])
        df['std_grp'] = df.groupby(self.bag_cont.col_level)['std'].transform(get_std_group)
        n_unique = df['std_grp'].nunique()
        print(df.groupby(self.bag_cont.col_exp).agg([pd.Series.mean, pd.Series.median]))
        fg = sns.relplot(data=df, y='qval', x=self.bag_cont.col_log2ratio, col=self.bag_cont.col_exp,
                         palette=sns.cubehelix_palette(n_unique, start=.5, rot=-.75), hue='std_grp')
        for row in fg.axes:
            for ax in row:
                ax.set_ylim(ymin=0)
                ax.hlines(y=0.05, xmin=df[self.bag_cont.col_log2ratio].min(),
                          xmax=df[self.bag_cont.col_log2ratio].max(), label="qval=0.05", colors=['purple'])
        self.plot_fig(name="log2ratio", g=fg, df_list=df)

    def plot_link_overview(self, exp_percentage=50, convert_to_log2=True):
        sum_list = [self.bag_cont.col_exp, self.bag_cont.col_bio_rep, self.bag_cont.col_weight_type,
                    self.bag_cont.col_tech_rep, self.bag_cont.col_link_type]  # self.bag_cont.col_tech_rep
        mean_list = [self.bag_cont.col_exp, self.bag_cont.col_bio_rep, self.bag_cont.col_weight_type,
                     self.bag_cont.col_link_type]
        cnt_column = 'count'
        mean_column = self.bag_cont.col_area_sum_total + '_mean'
        outname = "link_overview"
        df = self.bag_cont.get_group(sum_list, mean_list, self.bag_cont.col_area_sum_total, log2=convert_to_log2)
        # print("Mean STD", np.mean(df.groupby(self.bag_cont.col_level)[self.bag_cont.col_area_sum_total].apply(pd.Series.std)))
        num_exp = df[self.bag_cont.col_exp].nunique()
        # filter links which were not found in x percent of the total experiments
        df = df.groupby([self.bag_cont.col_level]).filter(
            lambda x: x[self.bag_cont.col_exp].nunique() / num_exp >= exp_percentage / 100)

        if self.bag_cont.col_domain:
            df = self.bag_cont.get_prot_name_and_link_pos(df)
            df = df.sort_values([self.bag_cont.col_domain, self.bag_cont.col_exp, self.bag_cont.col_level])
        else:
            df = df.sort_values([self.bag_cont.col_exp, self.bag_cont.col_level])
        fg = sns.catplot(data=df, x=self.bag_cont.col_exp, y=self.bag_cont.col_area_sum_total,
                         col=self.bag_cont.col_level, kind='point', col_wrap=5, ci='sd', sharey=False,
                         hue=self.bag_cont.col_domain, sharex=False, legend=False)
        # force scientific notation on y-axis for the original distribution (which is in the range of 1e7 to 1e10)
        if not convert_to_log2:
            for ax in fg.axes.flat:
                ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        # fg = sns.relplot(data=df, x=self.bag_cont.col_exp, y=self.bag_cont.col_area_sum_total,
        #                  col=self.bag_cont.col_level, kind='line', col_wrap=5, ci='sd',
        #                  hue=self.bag_cont.col_domain, facet_kws={'sharey':False, 'sharex':False})
        df[cnt_column] = df.groupby([self.bag_cont.col_level, self.bag_cont.col_exp, self.bag_cont.col_link_type])[
            self.bag_cont.col_level].transform('count')
        df[mean_column] = df.groupby([self.bag_cont.col_level, self.bag_cont.col_exp, self.bag_cont.col_link_type])[
            self.bag_cont.col_area_sum_total].transform('mean')

        fg.map(plib.map_point, self.bag_cont.col_exp, mean_column, cnt_column)
        fg.add_legend()  # in order to properly draw the legend after using fg.map, it has to be drawn after fg.map
        if not convert_to_log2:
            outname += "_non_log2"
        self.plot_fig(name=outname, g=fg, df_list=df)

    def plot_domain_single_link(self, exp_percentage=50, ratio_mean=False, log2ratio=False):
        value_column = self.bag_cont.col_ratio
        if log2ratio:
            value_column = self.bag_cont.col_log2ratio
        sum_list = [self.bag_cont.col_exp, self.bag_cont.col_bio_rep, self.bag_cont.col_weight_type,
                    self.bag_cont.col_tech_rep, self.bag_cont.col_link_type]  # self.bag_cont.col_tech_rep
        if ratio_mean:
            mean_list = [self.bag_cont.col_exp, self.bag_cont.col_bio_rep, self.bag_cont.col_weight_type]
        else:
            mean_list = [self.bag_cont.col_exp, self.bag_cont.col_link_type]
        cnt_column = 'count'
        mean_column = value_column + '_mean_exp'
        df = self.bag_cont.get_log2ratio(sum_list, mean_list, ratio_only=(not log2ratio), keep_ref=True)
        # print("Mean STD", np.mean(df.groupby(self.bag_cont.col_level)[self.bag_cont.col_area_sum_total].apply(pd.Series.std)))
        num_exp = df[self.bag_cont.col_exp].nunique()
        # filter links which were not found in x percent of the total experiments
        df = df.groupby([self.bag_cont.col_level]).filter(
            lambda x: x[self.bag_cont.col_exp].nunique() / num_exp >= exp_percentage / 100)

        if self.bag_cont.col_domain:
            df = self.bag_cont.get_prot_name_and_link_pos(df)
            df = df.sort_values([self.bag_cont.col_domain, self.bag_cont.col_exp, self.bag_cont.col_level])
            df[mean_column] = df.groupby([self.bag_cont.col_exp, self.bag_cont.col_domain])[
                value_column].transform('mean')
            df[cnt_column] = df.groupby([self.bag_cont.col_exp, self.bag_cont.col_domain])[
                self.bag_cont.col_level].transform('count')
            fg = sns.catplot(data=df, x=self.bag_cont.col_exp, y=value_column,
                             hue=self.bag_cont.col_level, kind='point', col_wrap=5, ci='sd', sharey=True,
                             col=self.bag_cont.col_domain, sharex=False, legend=False, color='lightgrey')
        else:
            df = df.sort_values([self.bag_cont.col_exp, self.bag_cont.col_level])
            df[mean_column] = df.groupby([self.bag_cont.col_exp])[
                value_column].transform('mean')
            fg = sns.catplot(data=df, x=self.bag_cont.col_exp, y=value_column,
                             hue=self.bag_cont.col_level, kind='point', ci='sd', sharey=True,
                             sharex=False, legend=False, color='lightgrey')
        # for ax in fg.axes:
        #     ax.set_yscale('log', basey=2)
        # fg = sns.relplot(data=df, x=self.bag_cont.col_exp, y=self.bag_cont.col_area_sum_total,
        #                  col=self.bag_cont.col_level, kind='line', col_wrap=5, ci='sd',
        #                  hue=self.bag_cont.col_domain, facet_kws={'sharey':False, 'sharex':False})

        # ADDITIONAL PLOT FORMATTING
        backgroundartists = []
        for ax in fg.axes.flat:
            for l in ax.lines + ax.collections:
                l.set_zorder(1)
                backgroundartists.append(l)
        if df[self.bag_cont.col_exp].dtype.name == 'category':
            order = df[self.bag_cont.col_exp].cat.categories
        else:
            order = sorted(df[self.bag_cont.col_exp].unique())
        # MAP THE MEAN LINE PLOT
        fg.map(sns.pointplot, self.bag_cont.col_exp, value_column, ci='sd', color='black',
               order=order)

        for ax in fg.axes.flat:
            for l in ax.lines + ax.collections:
                if l not in backgroundartists:
                    l.set_zorder(5)
        # MAP THE NO. OF OBSERVATIONS GOING INTO THE MEAN
        fg.map(plib.map_point, self.bag_cont.col_exp, mean_column, cnt_column)
        # fg.add_legend()  # no legend for this plot
        if log2ratio:
            for ax in fg.axes.flat:
                ax.set_ylabel("log2ratio")
        outname = "domain_single_link"
        if ratio_mean:
            outname += "_ratio_mean"
        if log2ratio:
            outname += "_log2ratio"

        self.plot_fig(name=outname, g=fg, df_list=df)

    def plot_domain_overview(self, exp_percentage=50):
        if self.bag_cont.col_domain:
            sum_list = [self.bag_cont.col_exp, self.bag_cont.col_bio_rep, self.bag_cont.col_weight_type,
                        self.bag_cont.col_tech_rep]
            mean_list = [self.bag_cont.col_exp]
            cnt_column = 'count'
            mean_column = self.bag_cont.col_area_sum_total + '_mean'
            median_column = self.bag_cont.col_area_sum_total + '_median'
            mean_column_z = self.bag_cont.col_area_z_score + '_mean_z'
            median_column_z = self.bag_cont.col_area_z_score + '_median_z'
            df = self.bag_cont.get_group(sum_list, mean_list, self.bag_cont.col_area_sum_total, z_score=True)
            df = self.bag_cont.get_prot_name_and_link_pos(df)
            num_exp = df[self.bag_cont.col_exp].nunique()
            # filter links which were not found in x percent of the total experiments
            df = df.groupby([self.bag_cont.col_level]).filter(
                lambda x: x[self.bag_cont.col_exp].nunique() / num_exp >= exp_percentage / 100)

            df = df.sort_values([self.bag_cont.col_exp, self.bag_cont.col_domain])

            # this will count all unique links per domain and experiment (makes more sense tbh)
            df[cnt_column] = df.groupby([self.bag_cont.col_exp, self.bag_cont.col_domain])[
                self.bag_cont.col_domain].transform('count')
            df[mean_column] = df.groupby([self.bag_cont.col_domain, self.bag_cont.col_exp])[
                self.bag_cont.col_area_sum_total].transform('mean')
            df[median_column] = df.groupby([self.bag_cont.col_domain, self.bag_cont.col_exp])[
                self.bag_cont.col_area_sum_total].transform('median')
            df[mean_column_z] = df.groupby([self.bag_cont.col_domain, self.bag_cont.col_exp])[
                self.bag_cont.col_area_z_score].transform('mean')
            df[median_column_z] = df.groupby([self.bag_cont.col_domain, self.bag_cont.col_exp])[
                self.bag_cont.col_area_z_score].transform('median')

            fg = sns.catplot(data=df, x=self.bag_cont.col_exp, y=self.bag_cont.col_area_z_score,
                             col=self.bag_cont.col_domain, kind='point', col_wrap=5, ci='sd',
                             hue=self.bag_cont.col_domain, sharey=False, sharex=False)
            fg.map(plib.map_point, self.bag_cont.col_exp, mean_column_z, cnt_column)
            self.plot_fig(name="domain_overview_z", g=fg, facet_kws={'sharey': False, 'sharex': False}, df_list=df)

            fg = sns.catplot(data=df, x=self.bag_cont.col_exp, y=self.bag_cont.col_area_z_score,
                             col=self.bag_cont.col_domain, kind='box', col_wrap=5,
                             hue=self.bag_cont.col_domain, sharey=False, sharex=False, dodge=False)
            fg.map(sns.swarmplot, self.bag_cont.col_exp, self.bag_cont.col_area_z_score, color=".1")
            fg.map(plib.map_point, self.bag_cont.col_exp, median_column_z, cnt_column, boxplot=True)
            self.plot_fig(name="domain_overview_z_box", g=fg, facet_kws={'sharey': False, 'sharex': False}, df_list=df)

            fg = sns.catplot(data=df, x=self.bag_cont.col_exp, y=self.bag_cont.col_area_sum_total,
                             col=self.bag_cont.col_domain, kind='point', col_wrap=5, ci='sd',
                             hue=self.bag_cont.col_domain, sharey=False, sharex=False)
            fg.map(plib.map_point, self.bag_cont.col_exp, mean_column, cnt_column)
            self.plot_fig(name="domain_overview", g=fg, facet_kws={'sharey': False, 'sharex': False}, df_list=df)

            fg = sns.catplot(data=df, x=self.bag_cont.col_exp, y=self.bag_cont.col_area_sum_total,
                             col=self.bag_cont.col_domain, kind='box', col_wrap=5,
                             hue=self.bag_cont.col_domain, sharey=False, sharex=False, dodge=False)
            fg.map(sns.swarmplot, self.bag_cont.col_exp, self.bag_cont.col_area_sum_total, color=".1")
            fg.map(plib.map_point, self.bag_cont.col_exp, median_column, cnt_column, boxplot=True)
            self.plot_fig(name="domain_overview_box", g=fg, facet_kws={'sharey': False, 'sharex': False}, df_list=df)
        else:
            print("ERROR: No domains specified for plotting")
            exit(1)

    def plot_dilution_series(self):
        sns.set(font_scale=1.75)
        sns.set_style("whitegrid")
        loc = plticker.MultipleLocator(base=1.0)  # this locator puts ticks at regular intervals
        sum_list = [self.bag_cont.col_exp, self.bag_cont.col_bio_rep, self.bag_cont.col_tech_rep]
        mean_list = [self.bag_cont.col_exp]
        df = self.bag_cont.get_group(sum_list, mean_list, self.bag_cont.col_area_sum_total)
        df = df.sort_values([self.bag_cont.col_exp, self.bag_cont.col_level])
        exp_ref = ll.input_log2_ref(self.bag_cont.exp_list)
        df_log2ratio = self.bag_cont.get_log2ratio(sum_list, mean_list, exp_ref)
        df[self.bag_cont.col_area_sum_total] = np.log2(df[self.bag_cont.col_area_sum_total])
        ax = sns.boxplot(data=df, x=self.bag_cont.col_exp, y=self.bag_cont.col_area_sum_total)
        ax.yaxis.set_major_locator(loc)
        # ax.set(yscale='log')
        self.plot_fig(name="dilution_series", df_list=df)
        ax = sns.boxplot(data=df_log2ratio, x=self.bag_cont.col_exp, y=self.bag_cont.col_log2ratio)
        ax.yaxis.set_major_locator(loc)
        print(df.groupby(self.bag_cont.col_exp).agg([pd.Series.mean, pd.Series.median]))
        print(df_log2ratio.groupby(self.bag_cont.col_exp).agg([np.mean, np.median]))
        self.plot_fig(name="dilution_series_log2ratio", df_list=df_log2ratio)

    def plot_scatter(self):
        sum_list = [self.bag_cont.col_exp, self.bag_cont.col_bio_rep, self.bag_cont.col_tech_rep,
                    self.bag_cont.col_link_type]
        mean_list = [self.bag_cont.col_exp, self.bag_cont.col_link_type]
        df = self.bag_cont.get_group(sum_list, mean_list, self.bag_cont.col_area_sum_total, log2=True)
        exp_list = sorted(list(set(df[self.bag_cont.col_exp])))
        if len(exp_list) > 2:
            print("More than two experiments found. Please select which ones to plot.")
            print("{0}".format({no: exp for no, exp in enumerate(exp_list)}))
            exp1 = int(input("Please select first experiment: "))
            exp2 = int(input("Please select second experiment: "))
            exp1, exp2 = exp_list[exp1], exp_list[exp2]
        elif len(exp_list) == 2:
            exp1, exp2 = exp_list[0], exp_list[1]
        else:
            print("ERROR: Too few experiments: {0}".format(exp_list))
            exit(1)
        # df = df.loc[df[col_area_sum_norm_total] > 0]  # filter zero intensities

        df_x = df.loc[df[self.bag_cont.col_exp] == exp1]
        df_y = df.loc[df[self.bag_cont.col_exp] == exp2]
        df = pd.merge(df_x, df_y, on=[self.bag_cont.col_level, self.bag_cont.col_link_type],
                      how='inner')  # inner: only merge intersection of keys
        df = df.dropna()
        df = df.reset_index()
        # df = df.loc[df[index_string] < len(df.index)]

        # note that regplot (underlying lmplot) will automatically remove zero values when using log scale
        fg = sns.lmplot(x=self.bag_cont.col_area_sum_total + '_x', y=self.bag_cont.col_area_sum_total + '_y',
                        hue=self.bag_cont.col_link_type, data=df,
                        fit_reg=True, robust=False, ci=None, lowess=True)
        p_min = df[self.bag_cont.col_area_sum_total + '_x'].min()
        p_max = df[self.bag_cont.col_area_sum_total + '_y'].min()
        fg.set(xlabel="{0} ({1})".format(self.bag_cont.col_area_sum_total,
                                         df[self.bag_cont.col_exp + '_x'][0]),  # KK_26S_merged_final.analyzer.quant
               ylabel="{0} ({1})".format(self.bag_cont.col_area_sum_total,
                                         df[self.bag_cont.col_exp + '_y'][0]),
               title="{0} col_level".format(self.bag_cont.col_level))
        # draw horizontal line for all possible plots
        for row in fg.axes:
            for ax in row:
                ax.plot(ax.get_xlim(), ax.get_xlim(), ls="--", c=".3")
        self.plot_fig(name="scatter", df_list=df)

    def plot_light_heavy_scatter(self):
        sum_list = [self.bag_cont.col_exp, self.bag_cont.col_bio_rep, self.bag_cont.col_tech_rep,
                    self.bag_cont.col_link_type, self.bag_cont.col_origin]
        mean_list = [self.bag_cont.col_exp, self.bag_cont.col_link_type, self.bag_cont.col_origin]
        df = self.bag_cont.get_group(sum_list, mean_list,
                                     [self.bag_cont.col_area_sum_total, self.bag_cont.col_area_sum_light,
                                      self.bag_cont.col_area_sum_heavy])
        fg = sns.lmplot(x=self.bag_cont.col_area_sum_light, y=self.bag_cont.col_area_sum_heavy,
                        hue=self.bag_cont.col_link_type,
                        col=self.bag_cont.col_exp, row=self.bag_cont.col_origin, data=df, fit_reg=False, sharex=True,
                        sharey=True, robust=True, ci=None, legend_out=False, )
        df_new = df[(df[self.bag_cont.col_area_sum_heavy] > 0) & (df[self.bag_cont.col_area_sum_light] > 0)]
        min_val = np.min(df_new[[self.bag_cont.col_area_sum_light, self.bag_cont.col_area_sum_heavy]].min())
        min_val -= min_val / 2
        max_val = np.max(df_new[[self.bag_cont.col_area_sum_light, self.bag_cont.col_area_sum_heavy]].max())
        max_val += max_val / 2
        # note that not setting x,ylim to auto (the default) leads to strange scaling bugs with a log scale
        # therefore using the same limits for all subplots; also makes comparisons easier
        fg.set(xscale='log', yscale='log', xlim=(min_val, max_val), ylim=(min_val, max_val))
        # draw horizontal line for all possible plots
        for row in fg.axes:
            for ax in row:
                ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3")
        self.plot_fig(name='light_heavy_scatter', df_list=df)

    def plot_bio_rep_scatter(self):
        sum_list = [self.bag_cont.col_exp, self.bag_cont.col_bio_rep, self.bag_cont.col_tech_rep,
                    self.bag_cont.col_link_type]
        mean_list = [self.bag_cont.col_exp, self.bag_cont.col_bio_rep, self.bag_cont.col_link_type]
        df = self.bag_cont.get_group(sum_list, mean_list, [self.bag_cont.col_area_sum_total], log2=True)
        for n_outer in self.bag_cont.bio_rep_list:
            for n_inner in self.bag_cont.bio_rep_list:
                if n_inner > n_outer:
                    df_inner = df[df[self.bag_cont.col_bio_rep] == n_inner]
                    df_outer = df[df[self.bag_cont.col_bio_rep] == n_outer]
                    rep_inner = '{0}_rep_{1}'.format(self.bag_cont.col_area_sum_total, n_inner)
                    rep_outer = '{0}_rep_{1}'.format(self.bag_cont.col_area_sum_total, n_outer)
                    df_inner = df_inner.rename(index=str, columns={self.bag_cont.col_area_sum_total: rep_inner})
                    df_outer = df_outer.rename(index=str, columns={self.bag_cont.col_area_sum_total: rep_outer})
                    df_inner = df_inner.drop(columns=[self.bag_cont.col_bio_rep])
                    df_outer = df_outer.drop(columns=[self.bag_cont.col_bio_rep])
                    df_f = pd.merge(df_outer, df_inner,
                                    on=[self.bag_cont.col_level, self.bag_cont.col_exp, self.bag_cont.col_link_type],
                                    how='outer')
                    fg = sns.lmplot(x=rep_outer, y=rep_inner, hue=self.bag_cont.col_link_type,
                                    row=self.bag_cont.col_exp, data=df_f, fit_reg=True, sharex=True, sharey=True,
                                    lowess=True, legend_out=False)
                    # fg.set(xscale='log', yscale='log')
                    # draw horizontal line for all possible plots
                    for row in fg.axes:
                        for ax in row:
                            ax.plot(ax.get_xlim(), ax.get_xlim(), ls="--", c=".3")
                    self.plot_fig(name='rep', extra="{0}_vs_{1}".format(n_outer, n_inner), df_list=df_f)

    def plot_bio_rep_ma_scatter(self):
        import statsmodels.nonparametric.smoothers_lowess as lw
        def norm_lowess(x):
            x['lowess'] = lw.lowess(x[self.bag_cont.col_log2ratio], x[self.bag_cont.col_log2avg],
                                    return_sorted=False)
            x['lowdiff'] = x[self.bag_cont.col_log2ratio] - x['lowess']
            return x

        sum_list = [self.bag_cont.col_exp, self.bag_cont.col_bio_rep, self.bag_cont.col_tech_rep,
                    self.bag_cont.col_weight_type]
        mean_list = [self.bag_cont.col_exp, self.bag_cont.col_bio_rep]
        df = self.bag_cont.getlog2ratio_r(sum_list, mean_list, ref=1)
        # print(df)
        # df['lowess'] = lw.lowess(df[self.bag_cont.col_log2ratio], df[self.bag_cont.col_log2avg], return_sorted=False)
        # df['lowdiff'] = df[self.bag_cont.col_log2ratio]-df['lowess']
        df = df.groupby([self.bag_cont.col_exp, self.bag_cont.col_bio_rep]).apply(norm_lowess)
        # n-way merge
        df = df.sort_values([self.bag_cont.col_exp, self.bag_cont.col_level])
        fg = sns.lmplot(data=df, y='lowdiff', x=self.bag_cont.col_log2avg, col=self.bag_cont.col_bio_rep,
                        row=self.bag_cont.col_exp, lowess=True)
        for row in fg.axes:
            for ax in row:
                ax.hlines(y=0, xmin=df[self.bag_cont.col_log2avg].min(),
                          xmax=df[self.bag_cont.col_log2avg].max(), label="log2ratio=0", colors=['purple'])
        self.plot_fig(name="rep_ma", g=fg, df_list=df)

    def plot_mono_vs_xlink_quant(self):
        link_group_string = "link_group"
        df = self.bag_cont.get_matching_monos
        df = df.groupby(
            [self.bag_cont.col_level, self.bag_cont.col_exp, self.bag_cont.col_bio_rep, self.bag_cont.col_tech_rep,
             self.bag_cont.col_weight_type, self.bag_cont.col_link_type, link_group_string])[
            self.bag_cont.col_area_sum_total].sum().reset_index()
        df[self.bag_cont.col_area_sum_total] = np.log2(df[self.bag_cont.col_area_sum_total])
        df = df.groupby(
            [self.bag_cont.col_level, self.bag_cont.col_exp, self.bag_cont.col_link_type,
             link_group_string]).mean().reset_index()
        df_xlinks = df[df[self.bag_cont.col_link_type] == self.bag_cont.row_xlink_string]
        df_monos = df[df[self.bag_cont.col_link_type] == self.bag_cont.row_monolink_string]
        # df_xlinks = df_xlinks.rename(index=str, columns={log2_string: log2_string+"_xlink"})
        df_monos = df_monos.rename(index=str, columns={
            self.bag_cont.col_area_sum_total: self.bag_cont.col_area_sum_total + "_mono"})
        df_monos = df_monos[[link_group_string, self.bag_cont.col_area_sum_total + "_mono"]]
        df = pd.merge(df_xlinks, df_monos, on=link_group_string)
        # print(df.groupby([self.bag_cont.col_exp, link_group_string, self.bag_cont.col_link_type]).mean())
        fg = sns.lmplot(data=df, x=self.bag_cont.col_area_sum_total + "_mono", y=self.bag_cont.col_area_sum_total,
                        col=self.bag_cont.col_exp)
        self.plot_fig(name="mono_quant", g=fg, df_list=df)
        df['mono_mean'] = df.groupby([self.bag_cont.col_level])[self.bag_cont.col_area_sum_total + "_mono"].transform(
            'mean')
        fg = sns.lmplot(data=df, x="mono_mean", y=self.bag_cont.col_area_sum_total, col=self.bag_cont.col_exp)
        self.plot_fig(name="mono_quant_mean", g=fg, df_list=df)
        df['mono_min'] = df.groupby([self.bag_cont.col_level])[self.bag_cont.col_area_sum_total + "_mono"].transform(
            'min')
        fg = sns.lmplot(data=df, x="mono_min", y=self.bag_cont.col_area_sum_total, col=self.bag_cont.col_exp)
        self.plot_fig(name="mono_quant_min", g=fg, df_list=df)
        df['mono_max'] = df.groupby([self.bag_cont.col_level])[self.bag_cont.col_area_sum_total + "_mono"].transform(
            'max')
        fg = sns.lmplot(data=df, x="mono_max", y=self.bag_cont.col_area_sum_total, col=self.bag_cont.col_exp)
        self.plot_fig(name="mono_quant_max", g=fg, df_list=df)

    def plot_bio_rep_bar(self):
        df = pd.DataFrame(self.bag_cont.df_orig)
        # filter zero ms1 intensities
        # df = df.loc[df[col_area_sum_total] > 0]
        # filter ids not found in at least two experiments
        # df = df.groupby(col_level).filter(lambda x: len(x) > 1)
        # filter by log2ratio
        # df = df[(df[col_log2ratio] > 3) | (df[col_log2ratio] < -3)]
        # df = df.reset_index()
        # following line can be used to only plot partial data
        df = df.reset_index()
        df[self.bag_cont.col_index] = df[self.bag_cont.col_index].astype(int)
        df = df.loc[df[self.bag_cont.col_index] < len(df.index) / 4]
        fg = sns.catplot(kind="bar", x=self.bag_cont.col_level, y=self.bag_cont.col_area_sum_total,
                         hue=self.bag_cont.col_bio_rep, data=df, row=self.bag_cont.col_exp, ci=None)
        fg.set(yscale='log')
        for row in fg.axes:
            for ax in row:
                ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha='right', size=9)
        self.plot_fig(name='rep_bar', df_list=df)

    def get_lowdiff(self):
        import statsmodels.nonparametric.smoothers_lowess as lw
        def norm_lowess(x):
            x['lowess'] = lw.lowess(x['mean'], x['snr'],
                                    return_sorted=False)
            x['lowdiff'] = x['mean'] - x['lowess']
            return x

        sum_list = [self.bag_cont.col_exp, self.bag_cont.col_bio_rep, self.bag_cont.col_tech_rep]
        mean_list = [self.bag_cont.col_exp]
        df_stats = self.bag_cont.get_stats(sum_list, mean_list, log2=True)
        # n-way merge
        df_stats['snr'] = np.log10(df_stats['snr'])
        df_stats = df_stats.groupby(self.bag_cont.col_exp).apply(norm_lowess)
        return df_stats

    def plot_dist_vs_quant_log2(self):
        sum_list = [self.bag_cont.col_exp, self.bag_cont.col_bio_rep, self.bag_cont.col_tech_rep,
                    self.bag_cont.col_weight_type]
        mean_list = [self.bag_cont.col_exp]
        df_log2 = self.bag_cont.get_log2ratio(sum_list, mean_list)
        df_dist = self.bag_cont.get_distance_delta_df(sum_list + self.bag_cont.distance_list,
                                                      mean_list + self.bag_cont.distance_list)
        df = pd.merge(df_log2, df_dist, on=[self.bag_cont.col_level, self.bag_cont.col_exp])
        for col_dist in self.bag_cont.distance_list:
            col_dist += '_delta'
            fg = sns.lmplot(data=df, x=col_dist, y=self.bag_cont.col_log2ratio)
            # fg.set(xscale='log')
            self.plot_fig("quant_vs_{0}".format(col_dist), df)
        fg = sns.heatmap(df[[self.bag_cont.col_log2ratio] + [col for col in df.columns if 'delta' in col]].corr(),
                         annot=True, fmt=".2f", xticklabels=True, yticklabels=True)
        self.plot_fig("corr_log2ratio_vs_dist", df)

    def plot_dist_vs_quant(self):
        for col_dist in self.bag_cont.distance_list:
            sum_list = [self.bag_cont.col_exp, self.bag_cont.col_bio_rep, self.bag_cont.col_tech_rep, col_dist,
                        self.bag_cont.col_weight_type]
            mean_list = [self.bag_cont.col_exp, col_dist]
            # df = self.bag_cont.df_orig.copy()
            # dfs = self.get_lowdiff()
            # df = pd.merge(df, dfs, on=[self.bag_cont.col_level, self.bag_cont.col_exp])

            # df[self.bag_cont.col_area_sum_total + '_sum'] = df.groupby(sum_list)[
            #    self.bag_cont.col_area_sum_total].transform('sum')
            # df[self.bag_cont.col_area_sum_total + '_mean'] = df.groupby(mean_list)[self.bag_cont.col_area_sum_total + '_sum'].transform('mean')
            df = self.bag_cont.get_group(sum_list, mean_list, self.bag_cont.col_area_sum_total, log2=True)
            df = df.groupby(self.bag_cont.col_level).filter(lambda x: x[col_dist].count() > 0)
            # df =  df[df[col_dist] < 0]
            # df[col_dist] =  abs(df[col_dist])
            fg = sns.lmplot(data=df, x=col_dist, y=self.bag_cont.col_area_sum_total, col=self.bag_cont.col_exp)
            # fg.set(xscale='log')
            self.plot_fig("quant_vs_{0}".format(col_dist), df)

    def plot_dist_vs_quant_log2_alt(self):
        sum_list = [self.bag_cont.col_exp, self.bag_cont.col_bio_rep, self.bag_cont.col_tech_rep,
                    self.bag_cont.col_weight_type]
        mean_list = [self.bag_cont.col_exp]
        df_log2 = self.bag_cont.get_log2ratio(sum_list, mean_list)
        df_dist = self.bag_cont.get_distance_delta_df(sum_list + self.bag_cont.distance_list,
                                                      mean_list + self.bag_cont.distance_list)
        df = pd.merge(df_log2, df_dist, on=[self.bag_cont.col_level, self.bag_cont.col_exp])
        for col_dist in self.bag_cont.distance_list:
            col_dist += '_delta'
            df_merge = self._get_fit_merge_df(df, col_dist, self.bag_cont.col_log2ratio)
            alt_chart = self._get_alt_lin_fit_chart(df_merge, col_dist, self.bag_cont.col_log2ratio)
            self.plot_fig("quant_vs_{0}".format(col_dist), df, g=alt_chart)
        fg = sns.heatmap(df[[self.bag_cont.col_log2ratio] + [col for col in df.columns if 'delta' in col]].corr(),
                         annot=True, fmt=".2f", xticklabels=True, yticklabels=True)
        self.plot_fig("corr_log2ratio_vs_dist", df)

    def plot_dist_vs_quant_alt(self):
        for col_dist in self.bag_cont.distance_list:
            sum_list = [self.bag_cont.col_exp, self.bag_cont.col_bio_rep, self.bag_cont.col_tech_rep, col_dist,
                        self.bag_cont.col_weight_type]
            mean_list = [self.bag_cont.col_exp, col_dist]
            df = self.bag_cont.get_group(sum_list, mean_list, self.bag_cont.col_area_sum_total, log2=True)
            df = df.groupby(self.bag_cont.col_level).filter(lambda x: x[col_dist].count() > 0)
            df_merge = self._get_fit_merge_df(df, col_dist, self.bag_cont.col_area_sum_total)

            alt_chart = self._get_alt_lin_fit_chart(df_merge, col_dist, self.bag_cont.col_area_sum_total).facet(
                column=self.bag_cont.col_exp
            )  # .resolve_scale(x='independent', y='independent')
            self.plot_fig("quant_vs_{0}".format(col_dist), df_list=df, g=alt_chart)

    def plot_reaction_state(self):
        sum_list = [self.bag_cont.col_exp, self.bag_cont.col_bio_rep, self.bag_cont.col_tech_rep,
                    self.bag_cont.col_weight_type, self.bag_cont.col_reaction_state]
        mean_list = [self.bag_cont.col_exp, self.bag_cont.col_reaction_state]
        df = self.bag_cont.get_group(sum_list, mean_list, self.bag_cont.col_area_sum_total, log2=False)
        # df[self.bag_cont.col_area_sum_total] = np.log(df[self.bag_cont.col_area_sum_total])
        chart_single = alt.Chart(df).mark_bar().encode(
            # x=self.bag_cont.col_exp,
            x=self.bag_cont.col_level,
            # y=self.bag_cont.col_area_sum_total,
            y=alt.Y(self.bag_cont.col_area_sum_total, stack='normalize',
                    axis=alt.Axis(format='%', title='Normalized Area')),
            color=self.bag_cont.col_reaction_state,
        ).facet(
            row=self.bag_cont.col_exp,
            # column=self.bag_cont.col_level
        )

        df_mean_overall = df.groupby([self.bag_cont.col_exp, self.bag_cont.col_reaction_state]) \
            .apply(pd.DataFrame.mean).reset_index()
        chart_exp_overall = alt.Chart(df_mean_overall).mark_bar().encode(
            y=alt.Y(self.bag_cont.col_area_sum_total, stack='normalize',
                    axis=alt.Axis(format='%', title='Normalized Overall Area')),
            color=self.bag_cont.col_reaction_state,
        ).facet(
            row=self.bag_cont.col_exp
        )
        chart_combined = (chart_exp_overall | chart_single).properties(
            title='Normalized MS1 area overview',
        ).configure_title(
            fontSize=20,
            font='Courier',
            anchor='middle',
            color='grey'
        )
        self.plot_fig("reaction_state", df_list=[df, df_mean_overall], g=chart_combined)

    def plot_reaction_state_log2ratio(self):
        # maps which state was upregulated if ref is quenched
        def _map_upregulation(x):
            if x >= 0: return self.bag_cont.row_hydrolyzed
            return self.bag_cont.row_quenched
        sum_list = [self.bag_cont.col_exp, self.bag_cont.col_bio_rep, self.bag_cont.col_tech_rep,
                    self.bag_cont.col_weight_type, self.bag_cont.col_reaction_state]
        mean_list = [self.bag_cont.col_exp, self.bag_cont.col_reaction_state]
        df = self.bag_cont.get_log2ratio(sum_list, mean_list, ref=self.bag_cont.row_quenched,
                                         ratio_between=self.bag_cont.col_reaction_state)
        df[self.bag_cont.col_reaction_state] = df[self.bag_cont.col_log2ratio].transform(_map_upregulation)
        chart_single = alt.Chart(df).mark_bar().encode(
            x=self.bag_cont.col_level,
            # y=self.bag_cont.col_area_sum_total,
            y=alt.Y(self.bag_cont.col_log2ratio),
            color=self.bag_cont.col_reaction_state,
            # color=alt.condition(
            #     alt.FieldGTEPredicate(field=self.bag_cont.col_log2ratio, gte=0),
            #     alt.value('#1f77b4'),
            #     alt.value('#ff7f0e'),
            # ),
        ).facet(
            row=self.bag_cont.col_exp
        )

        df_mean_overall = df.groupby([self.bag_cont.col_exp]) \
            .apply(pd.DataFrame.mean).reset_index()
        df_mean_overall[self.bag_cont.col_reaction_state] = df_mean_overall[
            self.bag_cont.col_log2ratio].transform(_map_upregulation)
        chart_exp_overall_base = alt.Chart(df_mean_overall).mark_bar().encode(
            y=alt.Y(self.bag_cont.col_log2ratio),
            color=self.bag_cont.col_reaction_state,
        )
        text_overall = chart_exp_overall_base.mark_text(
            dx=30,
        ).encode(
            text=alt.Text(self.bag_cont.col_log2ratio, format='.3f'),
        )
        chart_exp_overall_color = chart_exp_overall_base.encode(
            color=alt.condition(
                alt.FieldGTEPredicate(field=self.bag_cont.col_log2ratio, gte=0),
                alt.value('#1f77b4'),
                alt.value('#ff7f0e'),
            ),
        )
        chart_overall_final = (chart_exp_overall_color + text_overall).facet(
            row=self.bag_cont.col_exp,
        )

        chart_combined = (chart_overall_final | chart_single).properties(
            title='Log2 Ratio hydrolized vs quenched (negative values mean up regulation of quenched links)',
        ).configure_title(
            fontSize=20,
            font='Courier',
            anchor='middle',
            color='grey'
        ).resolve_scale(
            y='shared'
        )

        self.plot_fig("reaction_state_log2ratio", df_list=[df, df_mean_overall], g=chart_combined)
        # self.plot_fig("reaction_state_avg", df=df_mean_overall, g=chart_exp_overall)

    def plot_dist_quant_corr(self):
        dist_list = self.bag_cont.distance_list
        for exp in self.bag_cont.exp_list:
            sum_list = [self.bag_cont.col_level, self.bag_cont.col_exp, self.bag_cont.col_bio_rep,
                        self.bag_cont.col_tech_rep, self.bag_cont.col_weight_type] + dist_list
            mean_list = [self.bag_cont.col_level, self.bag_cont.col_exp] + dist_list
            df = self.bag_cont.df_orig.copy()
            df = df[df[self.bag_cont.col_exp] == exp]
            # exp_ref = ll.input_log2_ref(self.bag_cont.exp_list)
            # df_log2 = self.bag_cont.getlog2ratio(sum_list, mean_list, exp_ref)
            df[self.bag_cont.col_area_sum_total + '_sum'] = df.groupby(sum_list)[
                self.bag_cont.col_area_sum_total].transform('sum')
            df[self.bag_cont.col_area_sum_total + '_mean'] = df.groupby(mean_list)[
                self.bag_cont.col_area_sum_total + '_sum'].transform('mean')
            df[self.bag_cont.col_area_sum_total + '_std'] = df.groupby(mean_list)[
                self.bag_cont.col_area_sum_total + '_sum'].transform('std')
            df[self.bag_cont.col_area_sum_total + '_min'] = df.groupby(mean_list)[
                self.bag_cont.col_area_sum_total + '_sum'].transform('min')
            df[self.bag_cont.col_area_sum_total + '_max'] = df.groupby(mean_list)[
                self.bag_cont.col_area_sum_total + '_sum'].transform('max')
            df[self.bag_cont.col_area_sum_total + '_z'] = (df[self.bag_cont.col_area_sum_total] - df[
                self.bag_cont.col_area_sum_total + '_mean']) / df[self.bag_cont.col_area_sum_total + '_std']
            # df = df.groupby(self.bag_cont.col_level).filter(lambda x: x[self.bag_cont.col_dist].count() > 0)
            dfs = self.get_lowdiff()
            df = pd.merge(df, dfs, on=[self.bag_cont.col_level, self.bag_cont.col_exp])
            # df = pd.merge(df, df_log2, on=[self.bag_cont.col_level, self.bag_cont.col_exp])
            df = df[dist_list + ['lowdiff', 'lowess', self.bag_cont.col_area_sum_total,
                                 self.bag_cont.col_area_sum_total + '_sum', self.bag_cont.col_area_sum_total + '_mean',
                                 self.bag_cont.col_area_sum_total + '_min', self.bag_cont.col_area_sum_total + '_max',
                                 self.bag_cont.col_area_sum_total + '_z']]
            fg = sns.heatmap(df.corr(), annot=True, fmt=".2f", xticklabels=True, yticklabels=True)
            self.plot_fig("dist_vs_quant_corr_{0}".format(exp), df)

    def plot_fig(self, name, df_list=None, extra='', g=None, **kwargs):
        filter = ""
        if self.bag_cont.filter:
            filter = '_' + self.bag_cont.filter
        if extra:
            extra = '_' + extra
        save_string = "bag_{0}_{1}{2}{3}".format(name, self.bag_cont.col_level, filter, extra)
        if g is None:
            plib.save_fig(save_string, df_list=df_list, out_dir=self.out_folder)
        else:
            plib.save_g(g, save_string, df_list=df_list, out_dir=self.out_folder, **kwargs)
        plt.clf()

    def _lin_func(self, x, slope, intercept):
        return intercept + x * slope

    def _get_fit_df(self, x, x_axis, y_axis):
        slope, intercept, rval, pval, err = stats.linregress(x[x_axis], x[y_axis])
        return pd.Series({'slope': slope, 'intercept': intercept, 'rval': rval, 'pval': pval, 'err': err})

    def _get_fit_merge_df(self, df, x_axis, y_axis):
        df_fit = df.groupby(self.bag_cont.col_exp).apply(lambda x: self._get_fit_df(x, x_axis, y_axis)).reset_index()
        df_fit_vals = df.groupby(self.bag_cont.col_exp).apply(lambda x: self._get_fit_val_df(x, df_fit, x_axis))
        df_merge = pd.merge(df, df_fit_vals, on=[self.bag_cont.col_level, self.bag_cont.col_exp], how='inner')
        return df_merge

    def _get_fit_val_df(self, x, df_fit, x_axis):
        exp = x[self.bag_cont.col_exp].iloc[0]
        df_sel = df_fit[df_fit[self.bag_cont.col_exp] == exp]
        # lin_fit_x = np.linspace(x[col_dist].min(), x[col_dist].max(), len(x))
        lin_fit_y = self._lin_func(x[x_axis], df_sel['slope'].iloc[0], df_sel['intercept'].iloc[0])
        return pd.DataFrame(
            {'yfit': lin_fit_y, self.bag_cont.col_level: x[self.bag_cont.col_level], self.bag_cont.col_exp: exp,
             'rval': df_sel['rval'].iloc[0], 'pval': df_sel['pval'].iloc[0], 'err': df_sel['err'].iloc[0]})

    def _get_alt_lin_fit_chart(self, df, x_axis, y_axis):
        alt_point = alt.Chart(df).mark_circle(color='black').encode(
            x=alt.X(x_axis, scale=alt.Scale(zero=False), axis=alt.Axis(title=x_axis)),
            y=alt.Y(y_axis, scale=alt.Scale(zero=False),
                    axis=alt.Axis(title=y_axis)),
        )

        alt_lin_fit = alt_point.mark_line().encode(
            y=alt.Y('yfit'),
        )
        rval = 'rval'
        pval = 'pval'
        alt_text_rval = alt_point.mark_text(dy=0).encode(
            x=f'max({x_axis})',  # alt.value(df_merge[col_dist].max()),#
            y=alt.value(df[y_axis].max()),
            # f'max({self.bag_cont.col_area_sum_total})', #
            text=alt.Text('label:N'),
        ).transform_calculate(label=f'"{rval}: " + format(datum.{rval}, ".2f")')

        alt_text_pval = alt_text_rval.mark_text(dy=10).encode(
        ).transform_calculate(label=f'"{pval}: " + format(datum.{pval}, ".2f")')

        return alt.layer(alt_point, alt_lin_fit, alt_text_rval, alt_text_pval)
