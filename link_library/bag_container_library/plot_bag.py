import seaborn as sns
import pandas as pd
import numpy as np
import link_library.plot_library as plib
import link_library as ll
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
from scipy.stats import norm
from functools import reduce

class PlotMaster(object):

    def __init__(self, bag_cont, out_folder='plots'):
        self.out_folder = out_folder
        self.bag_cont = bag_cont

    def plot_clustermap(self):
        # how far up we want to sum up values; done first (understand it as an exclusion list)
        sum_list = [self.bag_cont.col_exp, self.bag_cont.col_bio_rep, self.bag_cont.col_tech_rep] # self.bag_cont.col_tech_rep, self.bag_cont.col_charge self.bag_cont.col_weight_type
        # how far do we take the mean of values; done second
        mean_list = [self.bag_cont.col_exp, self.bag_cont.col_bio_rep]
        # the following two lines will take the mean of the tech replicates; should always be used if don't want separate tech reps
        # mean_list = list(sum_list)
        # mean_list.remove(self.bag_cont.col_tech_rep)
        df = self.bag_cont.get_pivot(sum_list, mean_list, pivot_on=self.bag_cont.col_area_sum_total)
        df_links = pd.DataFrame(self.bag_cont.df_orig)
        df_links = df_links.groupby([self.bag_cont.col_level, self.bag_cont.col_link_type])[self.bag_cont.col_area_sum_total].sum().reset_index()
        df_links = df_links.set_index(self.bag_cont.col_level)
        df_links = df_links[self.bag_cont.col_link_type]
        lut = dict(zip(df_links.unique(), "cb"))
        row_colors = df_links.map(lut)
        # impute na values with a downshifted normal distribution
        df, dist_orig, dist_imputed = self.bag_cont.fillna_with_normal_dist(df)
        sns.distplot(dist_orig, kde=True, fit=norm)
        self.plot_fig(name='cluster', extra="orig_dist")
        plt.clf()
        if len(dist_imputed) > 1:
            sns.distplot(dist_imputed, kde=True, fit=norm)
            self.plot_fig(name='cluster', extra="imputed_dist")
        print("Non imputed values: {0}. Imputed values: {1}".format(len(dist_orig),len(dist_imputed)))
        cg = sns.clustermap(data=df,cmap="mako_r",metric='canberra', row_colors=row_colors)
        ax = cg.ax_heatmap
        # hiding the y labels as they are really not informative
        ax.tick_params(axis='y', which='both', right=False, labelright=False)
        self.plot_fig(name="cluster", g=cg, dpi=300)
        cg = sns.clustermap(data=df,cmap="mako_r", z_score=0, metric='canberra', row_colors=row_colors)
        ax = cg.ax_heatmap
        # hiding the y labels as they are really not informative
        ax.tick_params(axis='y', which='both', right=False, labelright=False)
        self.plot_fig(name='cluster', extra="z_row", g=cg, dpi=300)
        cg = sns.clustermap(data=df,cmap="mako_r", z_score=1, metric='canberra', row_colors=row_colors)
        ax = cg.ax_heatmap
        # hiding the y labels as they are really not informative
        ax.tick_params(axis='y', which='both', right=False, labelright=False)
        self.plot_fig(name='cluster', extra="z_col", g=cg, dpi=300)

    def plot_bar(self):
        df = pd.DataFrame(self.bag_cont.df_new)
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
        ax = sns.barplot(x=self.bag_cont.col_level, y=self.bag_cont.col_area_sum_total, hue=self.bag_cont.col_exp, data=df, ci='sd')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        ax.set(title="{0} col_level".format(self.bag_cont.col_level), yscale='log')
        self.plot_fig(name="bar")

    def plot_ms1_area_std(self):
        sum_list = [self.bag_cont.col_exp, self.bag_cont.col_bio_rep, self.bag_cont.col_tech_rep]
        mean_list = [self.bag_cont.col_exp]
        df = self.bag_cont.get_stats(sum_list, mean_list)
        df = df.sort_values([self.bag_cont.col_exp])
        print(df)
        fg = sns.relplot(data=df, x='mean', y='std', col=self.bag_cont.col_exp)
        for row in fg.axes:
            for ax in row:
                ax.set_ylim(ymin=0)
                ax.hlines(y=0.5, xmin=df['mean'].min(),
                          xmax=df['mean'].max(), label="std=0.5", colors=['grey'])
        self.plot_fig(name="std", g=fg)
        # df = df[(df[self.bag_cont.col_log2ratio] < 1)&(df[self.bag_cont.col_log2ratio] > -1)]

    def plot_log2ratio(self):
        sum_list = [self.bag_cont.col_exp, self.bag_cont.col_bio_rep, self.bag_cont.col_tech_rep]
        mean_list = [self.bag_cont.col_exp]
        pval_list = [self.bag_cont.col_exp, self.bag_cont.col_bio_rep, self.bag_cont.col_tech_rep]
        exp_ref = ll.input_log2_ref(self.bag_cont.exp_list)
        df_pval = self.bag_cont.get_two_sided_ttest(pval_list, [self.bag_cont.col_exp, self.bag_cont.col_bio_rep], ref=exp_ref)
        df_stats = self.bag_cont.get_stats(sum_list, mean_list)
        df_log2 = self.bag_cont.getlog2ratio(sum_list, mean_list, ref=exp_ref)
        # n-way merge
        df = reduce(lambda left, right: pd.merge(left, right, on=[self.bag_cont.col_level, self.bag_cont.col_exp]), [df_stats, df_log2, df_pval])
        df = df.sort_values([self.bag_cont.col_exp, self.bag_cont.col_level])
        print(df.groupby(self.bag_cont.col_exp).agg([pd.Series.mean, pd.Series.median]))
        fg = sns.relplot(data=df, y='qval', x=self.bag_cont.col_log2ratio, col=self.bag_cont.col_exp, palette="brg", hue='std')
        for row in fg.axes:
            for ax in row:
                ax.set_ylim(ymin=0)
                ax.hlines(y=0.05, xmin=df[self.bag_cont.col_log2ratio].min(),
                          xmax=df[self.bag_cont.col_log2ratio].max(), label="qval=0.05", colors=['purple'])
        self.plot_fig(name="log2ratio", g=fg)

    def plot_link_overview(self):
        sum_list = [self.bag_cont.col_exp, self.bag_cont.col_bio_rep, self.bag_cont.col_tech_rep]
        mean_list = [self.bag_cont.col_exp, self.bag_cont.col_bio_rep]
        df = self.bag_cont.get_group(sum_list, mean_list)
        df[self.bag_cont.col_area_sum_total] = np.log2(df[self.bag_cont.col_area_sum_total])
        # df[self.bag_cont.col_exp] = pd.Categorical(
        #     df[self.bag_cont.col_exp],
        #     categories=['T5', 'T3', 'T4', 'T2', 'T1', 'T6'],
        #     ordered=True
        # )
        # reverse sort order for exp
        df[self.bag_cont.col_exp] = pd.Categorical(
            df[self.bag_cont.col_exp],
            categories=sorted(df[self.bag_cont.col_exp].unique(), reverse=True),
            ordered=True
        )
        if self.bag_cont.col_domain:
            df = self.bag_cont.get_prot_name_and_link_pos(df)
            df = df.sort_values([self.bag_cont.col_domain, self.bag_cont.col_exp, self.bag_cont.col_level])
            df_domain_ov = self.bag_cont.get_group(sum_list, [self.bag_cont.col_exp])
            df_domain_ov[self.bag_cont.col_area_sum_total] = np.log2(df_domain_ov[self.bag_cont.col_area_sum_total])
            df_domain_ov = self.bag_cont.get_prot_name_and_link_pos(df_domain_ov)
            # this will count all unique links per domain
            # df_domain_ov['count'] = df_domain_ov.groupby([self.bag_cont.col_domain])[self.bag_cont.col_level].transform(
            #     lambda x: len(x.unique()))
            # this will count all unique links per domain and experiment (makes more sense tbh)
            df_domain_ov['count'] = df_domain_ov.groupby([self.bag_cont.col_exp, self.bag_cont.col_domain])[
                self.bag_cont.col_domain].transform('count')
            # fg = sns.relplot(data=df_domain_ov, x=self.bag_cont.col_exp, y=self.bag_cont.col_area_sum_total,
            #                  col=self.bag_cont.col_domain, kind='line', col_wrap=5, ci='sd',
            #                  hue=self.bag_cont.col_domain, facet_kws={'sharey':False, 'sharex':False})
            # df_domain_ov[self.bag_cont.col_exp] = pd.Categorical(
            #     df_domain_ov[self.bag_cont.col_exp],
            #     categories=['T5', 'T3', 'T2', 'T1', 'T6'],
            #     ordered=True
            # )
            df_domain_ov = df_domain_ov.sort_values([self.bag_cont.col_domain, self.bag_cont.col_exp])
            fg = sns.catplot(data=df_domain_ov, x=self.bag_cont.col_exp, y=self.bag_cont.col_area_sum_total,
                             col=self.bag_cont.col_domain, kind='point', col_wrap=5, ci='sd',
                             hue=self.bag_cont.col_domain, sharey=False, sharex=False)
            df_domain_ov['mean'] = df_domain_ov.groupby([self.bag_cont.col_domain, self.bag_cont.col_exp])[self.bag_cont.col_area_sum_total].transform('mean')
            print(df_domain_ov[df_domain_ov[self.bag_cont.col_level] == 'FUS-K9:297:x:FUS-K9:543'])
            df['mean'] = df.groupby([self.bag_cont.col_domain, self.bag_cont.col_exp])[
                self.bag_cont.col_area_sum_total].transform('mean')
            print(df[df[self.bag_cont.col_level] == 'FUS-K9:297:x:FUS-K9:543'])
            self.bag_cont.df_orig[
                              self.bag_cont.df_orig[self.bag_cont.col_level] == 'FUS-K9:297:x:FUS-K9:543'].groupby(
                self.bag_cont.col_exp)[self.bag_cont.col_area_sum_total].apply(lambda x: print(np.log2(x)))
            print(np.log2(self.bag_cont.df_orig[self.bag_cont.df_orig[self.bag_cont.col_level] == 'FUS-K9:297:x:FUS-K9:543'].groupby(self.bag_cont.col_exp)[self.bag_cont.col_area_sum_total].mean()))
            exit()
            # print(df_domain_ov.groupby(self.bag_cont.col_domain).apply(lambda x: print(x)))

            fg.map(plib.map_point, self.bag_cont.col_exp, 'mean', 'count')
            self.plot_fig(name="domain_overview", g=fg, facet_kws={'sharey':False, 'sharex':False})
        else:
            df = df.sort_values([self.bag_cont.col_exp, self.bag_cont.col_level])
        fg = sns.catplot(data=df, x=self.bag_cont.col_exp, y=self.bag_cont.col_area_sum_total, col=self.bag_cont.col_level, kind='point', col_wrap=5, ci='sd', sharey=False, hue=self.bag_cont.col_domain, sharex=False)
        # fg = sns.relplot(data=df, x=self.bag_cont.col_exp, y=self.bag_cont.col_area_sum_total,
        #                  col=self.bag_cont.col_level, kind='line', col_wrap=5, ci='sd',
        #                  hue=self.bag_cont.col_domain, facet_kws={'sharey':False, 'sharex':False})
        self.plot_fig(name="link_overview",g=fg)


    def plot_dilution_series(self):
        sns.set(font_scale=1.75)
        sns.set_style("whitegrid")
        loc = plticker.MultipleLocator(base=1.0)  # this locator puts ticks at regular intervals
        sum_list = [self.bag_cont.col_exp, self.bag_cont.col_bio_rep, self.bag_cont.col_tech_rep]
        mean_list = [self.bag_cont.col_exp]
        df = self.bag_cont.get_group(sum_list, mean_list)
        df = df.sort_values([self.bag_cont.col_exp, self.bag_cont.col_level])
        exp_ref = ll.input_log2_ref(self.bag_cont.exp_list)
        df_log2ratio = self.bag_cont.getlog2ratio(sum_list, mean_list, exp_ref)
        df[self.bag_cont.col_area_sum_total] = np.log2(df[self.bag_cont.col_area_sum_total])
        ax = sns.boxplot(data=df, x=self.bag_cont.col_exp, y=self.bag_cont.col_area_sum_total)
        ax.yaxis.set_major_locator(loc)
        # ax.set(yscale='log')
        self.plot_fig(name="dilution_series")
        ax = sns.boxplot(data=df_log2ratio, x=self.bag_cont.col_exp, y=self.bag_cont.col_log2ratio)
        ax.yaxis.set_major_locator(loc)
        print(df.groupby(self.bag_cont.col_exp).agg([pd.Series.mean, pd.Series.median]))
        print(df_log2ratio.groupby(self.bag_cont.col_exp).agg([np.mean, np.median]))
        self.plot_fig(name="dilution_series_log2ratio")

    def plot_scatter(self):
        df = pd.DataFrame(self.bag_cont.df_new)
        exp_list = sorted(list(set(df[self.bag_cont.col_exp])))
        if len(exp_list) > 2:
            print("More than two experiments found. Please select which ones to plot.")
            print("{0}".format({no:exp for no,exp in enumerate(exp_list)}))
            exp1 = int(input("Please select first experiment: "))
            exp2 = int(input("Please select second experiment: "))
            exp1, exp2 = exp_list[exp1], exp_list[exp2]
        elif len(exp_list) == 2:
            exp1, exp2 =exp_list[0], exp_list[1]
        else:
            print("ERROR: Too few experiments: {0}".format(exp_list))
            exit(1)
        #df = df.loc[df[col_area_sum_norm_total] > 0]  # filter zero intensities

        df_x = df.loc[df[self.bag_cont.col_exp] == exp1]
        df_y = df.loc[df[self.bag_cont.col_exp] == exp2]
        df = pd.merge(df_x, df_y, on=[self.bag_cont.col_level, self.bag_cont.col_link_type], how='inner')  # inner: only merge intersection of keys
        df = df.dropna()
        df = df.reset_index()
        # df = df.loc[df[index_string] < len(df.index)]

        # note that regplot (underlying lmplot) will automatically remove zero values when using log scale
        fg = sns.lmplot(x=self.bag_cont.col_area_sum_total + '_x', y=self.bag_cont.col_area_sum_total + '_y', hue=self.bag_cont.col_link_type, data=df,
                        fit_reg=False, robust=False, ci=None)
        min_x = df[self.bag_cont.col_area_sum_total + '_x'].min()
        min_y = df[self.bag_cont.col_area_sum_total + '_y'].min()
        min_min = min(min_x, min_y)
        # using same minimum value for x and y and offset it by half to not cut off values at the limits
        min_min -= min_min/2
        fg.set(xlabel="{0} ({1})".format(self.bag_cont.col_area_sum_total,
                                         df[self.bag_cont.col_exp+'_x'][0]),  #KK_26S_merged_final.analyzer.quant
               ylabel="{0} ({1})".format(self.bag_cont.col_area_sum_total,
                                         df[self.bag_cont.col_exp+'_y'][0]),
               xscale='log', yscale='log', title="{0} col_level".format(self.bag_cont.level),
               xlim=min_min,ylim=min_min)
        # draw horizontal line for all possible plots
        for row in fg.axes:
            for ax in row:
                ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3")
        self.plot_fig(name="scatter")

    def plot_light_heavy_scatter(self):
        df = pd.DataFrame(self.bag_cont.df_new)
        # df_new = df_new[df_new[col_link_type] == row_xlink_string]
        fg = sns.lmplot(x=self.bag_cont.col_area_sum_light, y=self.bag_cont.col_area_sum_heavy, hue=self.bag_cont.col_link_type,
                        col=self.bag_cont.col_exp_original, row=self.bag_cont.col_origin, data=df, fit_reg=False, sharex=True, sharey=True, robust=True, ci=None, legend_out=False, )
        df_new = df[(df[self.bag_cont.col_area_sum_heavy] > 0) & (df[self.bag_cont.col_area_sum_light] > 0)]
        min_val = np.min(df_new[[self.bag_cont.col_area_sum_light, self.bag_cont.col_area_sum_heavy]].min())
        min_val -= min_val/2
        max_val = np.max(df_new[[self.bag_cont.col_area_sum_light, self.bag_cont.col_area_sum_heavy]].max())
        max_val += max_val/2
        # note that not setting x,ylim to auto (the default) leads to strange scaling bugs with a log scale
        # therefore using the same limits for all subplots; also makes comparisons easier
        fg.set(xscale='log', yscale='log',xlim=(min_val,max_val), ylim=(min_val,max_val))
        # draw horizontal line for all possible plots
        for row in fg.axes:
            for ax in row:
                ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3")
        self.plot_fig(name='light_heavy_scatter')

    def plot_bio_rep_scatter(self):
        df = pd.DataFrame(self.bag_cont.df_new)
        bio_rep_dict = self.bag_cont.bio_rep_dict
        # df_new = df_new[df_new[col_link_type] == row_xlink_string]
        for n_outer, bio_rep_outer in enumerate(bio_rep_dict.keys()):
            for n_inner, bio_rep_inner in enumerate(bio_rep_dict.keys()):
                if n_inner > n_outer:
                    fg = sns.lmplot(x=bio_rep_outer, y=bio_rep_inner, #hue=col_link_type,
                                    row=self.bag_cont.col_exp_original, data=df, fit_reg=False, sharex=True, sharey=True,
                                    ci=None, legend_out=False, hue=self.bag_cont.col_link_type)
                    df_new = df[df[self.bag_cont.col_area_sum_total] > 0]
                    min_val = np.min(df_new[[self.bag_cont.col_area_sum_total]].min())
                    min_val -= min_val/2
                    max_val = np.max(df_new[[self.bag_cont.col_area_sum_total]].max())
                    max_val += max_val/2
                    # note that not setting x,ylim to auto (the default) leads to strange scaling bugs with a log scale
                    # therefore using the same limits for all subplots; also makes comparisons easier
                    fg.set(xscale='log', yscale='log',xlim=(min_val,max_val), ylim=(min_val,max_val))
                    # draw horizontal line for all possible plots
                    for row in fg.axes:
                        for ax in row:
                            ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3")
                    self.plot_fig(name='rep', extra="{0}_vs_{1}".format(bio_rep_dict[bio_rep_outer], bio_rep_dict[bio_rep_inner]))


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
        df = df.loc[df[self.bag_cont.col_index] < len(df.index)/4]
        fg = sns.catplot(kind="bar", x=self.bag_cont.col_level, y=self.bag_cont.col_area_sum_total, hue=self.bag_cont.col_area_bio_repl, data=df, row=self.bag_cont.col_exp_original, ci=None)
        fg.set(yscale='log')
        for row in fg.axes:
            for ax in row:
                ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha='right', size=9)
            self.plot_fig(name='rep_bar',extra="{0}".format(row))


    def plot_fig(self, name, extra='', g = None, **kwargs):
        filter = ""
        if self.bag_cont.filter:
            filter = '_' + self.bag_cont.filter
        if extra:
            extra = '_' + extra
        save_string = "bag_{0}_{1}{2}{3}".format(name, self.bag_cont.col_level, filter, extra)
        if g is None:
            plib.save_fig(save_string, self.out_folder)
        else:
            plib.save_g(g, save_string, self.out_folder, **kwargs)
        plt.clf()