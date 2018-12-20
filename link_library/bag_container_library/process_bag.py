import pandas as pd
import numpy as np
import copy
from statsmodels.sandbox.stats import multicomp
from statsmodels.stats import weightstats
from scipy.stats import zscore


# TODO: create the filter options for the df_orig; eventually replace the df_new completely

class BagContainer(object):

    def __init__(self, level, df_list, filter=''):
        self.col_uid = "uid"
        self.col_uxid = "uxid"
        self.col_exp = "exp_name"
        self.col_exp_file = "file_name"
        self.col_exp_original = 'exp_name_original'
        self.col_origin = 'origin'
        self.col_link_type = 'link_type'
        self.col_weight_type = 'weight_type'
        self.col_bag_type = 'bag_container_type'
        self.col_area_sum_total = 'ms1_area_sum'
        self.col_area_sum_norm_total = self.col_area_sum_total + '_norm'
        self.col_area_sum_light = 'ms1_area_sum_light'
        self.col_area_sum_heavy = 'ms1_area_sum_heavy'
        self.col_area_bio_repl = 'ms1_area_sum_bio_rep_'
        self.col_var = 'ms1_area_variance'
        self.col_std = 'ms1_area_std'
        self.col_index = 'index'
        self.col_log2ratio = 'log2ratio'
        self.col_lh_log2ratio = 'light_heavy_log2ratio'
        self.col_bio_rep = 'exp_bio_rep'
        self.col_tech_rep = 'exp_tech_rep'
        self.col_charge = 'charge'
        self.col_pval = 'pval'
        self.col_fdr = 'qval'
        self.col_mean = 'mean'
        self.col_std = 'std'
        self.row_monolink_string = 'monolink'
        self.row_xlink_string = 'xlink'
        self.row_light_string = 'light'
        self.row_heavy_string = 'heavy'
        self.row_details_string = 'details'
        self.cont_type = self.row_details_string
        self.uxid_string = 'b_peptide_uxID'
        self.uid_string = 'b_peptide_uID'
        self.seq_string = 'b_peptide_seq'
        self.exp_string = 'd_exp_name'
        self.fraction_string = 'd_exp_fraction'
        self.repl_bio_string = 'd_exp_biol_rep'
        self.repl_tech_string = 'd_exp_tech_rep'
        self.vio_string = 'a_bag_container_violations'
        self.sum_string = 'c_pg_area_sum_isotopes'
        self.valid_string = "b_peptide_var_valid"
        self.type_string = "b_peptide_type"
        self.charge_string = "b_peptide_z"
        if level.lower() == self.col_uid:
            self.level_string = self.uid_string
            self.col_level = self.col_uid
        elif level.lower() == self.col_uxid:
            self.level_string = self.uxid_string
            self.col_level = self.col_uxid
        else:
            print("ERROR: Improper ms1 col_level entered: {0}".format(level))
            exit(1)
        self.filter = ""
        if filter.lower() == self.row_monolink_string:
            self.filter = self.row_monolink_string
        elif filter.lower() == self.row_xlink_string:
            self.filter = self.row_xlink_string
        elif filter:
            print("ERROR: Improper link filter entered: {0}".format(filter))
            exit(1)
        self.type = 'BagContainer_{0}'.format(level)
        self.df_orig = pd.DataFrame()
        self.df_new = pd.DataFrame()
        for df in df_list:
            df = self.prepare_df(df)
            if self.filter:
                df = self.filter_df(df)
            self.df_orig = self.df_orig.append(copy.deepcopy(self.rename_columns(df)))
            self.df_new = self.df_new.append(self.get_sum_ms1_intensities_df(copy.deepcopy(df)))
        self.df_new = self.df_new.sort_values(
            [self.col_origin, self.col_exp_original, self.col_link_type, self.col_area_sum_total])
        self.df_orig = self.remove_violations_orig(self.df_orig)
        # dict which stores {replicate_area_column: replicate identifier}
        self.bio_rep_dict = {h: h.replace(self.col_area_bio_repl, "") for h in self.df_new.columns if
                             self.col_area_bio_repl in h}
        self.bio_rep_num = len(self.bio_rep_dict)
        # df orig should contain the original columns as given by xquest/xtract (just renamed and a few additions)
        # df new contains (id | ms1 sum | ms1 sum rep1| ...) columns; id corresponds to the bag_cont col_level (uid/uxid)

    def get_group(self, sum_list, mean_list):
        df = pd.DataFrame(self.df_orig)
        df = df.groupby([self.col_level] + sum_list)[self.col_area_sum_total].sum().reset_index()
        df = df.groupby([self.col_level] + mean_list)[self.col_area_sum_total].mean().reset_index()
        return df

    def remove_violations_orig(self, df):
        name = set(df[self.col_origin])
        # filtering these two means we get exactly the same results as from the regular bag container
        # removes violations (but no violations are calculated for monolinks)
        print("Shape of {0} before filtering xTract violations: {1}.".format(name, df.shape))
        df = df.loc[df[self.vio_string] == 0]
        df = df.loc[df[self.valid_string] == 1]
        print("Shape of {0} after filtering xTract violations: {1}.".format(name, df.shape))
        return df

    def get_stats(self, sum_list, mean_list):
        df = pd.DataFrame(self.df_orig)
        df = df.groupby([self.col_level] + sum_list)[self.col_area_sum_total].sum().reset_index()
        df_log2 = self.getlog2ratio(sum_list, mean_list)
        # print(df)
        df = df.replace(0, np.nan)
        df[self.col_area_sum_total] = np.log2(df[self.col_area_sum_total])
        # print(pd.pivot_table(df, values=self.col_area_sum_total, index=[self.col_level], columns=sum_list,
        #                      aggfunc=np.sum))
        df = df.groupby([self.col_level] + mean_list)[self.col_area_sum_total].agg(
            [pd.Series.mean, pd.Series.std])
        df = df.dropna()
        df = df.reset_index(level=[0, 1, 2])

        df = pd.merge(df, df_log2, on=self.col_level)

        # print(df)
        return df

    def getlog2ratio(self, sum_list, mean_list):
        df = self.get_group(sum_list, mean_list)
        exp_list = sorted(df[self.col_exp].unique())
        # print(df)
        df = df.replace(0, np.nan)
        df = df.dropna()
        # filter ids not found in at least two experiments (as each id exists only once for each experiment)
        df = df.groupby(self.col_level).filter(lambda x: len(x) > 1)
        df = df.groupby([self.col_level]).apply(lambda x: pd.Series({self.col_log2ratio: (np.log2(x[self.col_area_sum_total].loc[x[self.col_exp] == exp_list[0]].values / x[self.col_area_sum_total].loc[
                x[self.col_exp] == exp_list[1]].values)) if len(x[self.col_area_sum_total].loc[x[self.col_exp] == exp_list[0]].values) == len(x[self.col_area_sum_total].loc[
                x[self.col_exp] == exp_list[1]].values) else [np.nan]})).reset_index()
        # df = df.groupby([self.col_level]).apply(lambda x: print(x))
        # df = df.groupby([self.col_level]).apply(lambda x: print(
        #     (np.log2(x[self.col_area_sum_total].loc[x[self.col_exp] == exp_list[0]].values / x[self.col_area_sum_total].loc[
        #         x[self.col_exp] == exp_list[1]].values)) if len(x[self.col_area_sum_total].loc[x[self.col_exp] == exp_list[0]].values) == len(x[self.col_area_sum_total].loc[
        #         x[self.col_exp] == exp_list[1]].values) else [np.nan]))
        df[self.col_log2ratio] = df[self.col_log2ratio].apply(lambda y: np.nan if len(y) == 0 else y[0])
        return df

    def get_pivot(self, sum_list, mean_list):
        df = self.get_group(sum_list, mean_list)
        df = pd.pivot_table(df, values=self.col_area_sum_total, index=[self.col_level],
                            columns=mean_list, aggfunc=np.sum)
        return df

    def log2_and_fillna_with_normal_dist(self, df):
        # ref: http://www.coxdocs.org/doku.php?id=perseus:user:activities:matrixprocessing:imputation:replacemissingfromgaussian
        # ref: https://github.com/JurgenCox/perseus-plugins/blob/master/PerseusPluginLib/Impute/ReplaceMissingFromGaussian.cs
        # default values from mq (shift=1.8, width=0.3) for a normal dist
        # the underlying distribution is log-normal; log2 makes it normal
        shift = 1.8
        width = 0.3
        df = df.replace(0, np.nan)
        df = np.log2(df)
        a = df.values
        b = a[~np.isnan(a)]
        m = np.isnan(a)  # mask of NaNs
        mu, sigma = np.mean(b), np.std(b)  # mean and std of total matrix
        a[m] = np.random.normal(mu - shift * sigma, width * sigma, size=m.sum())
        # the following two lines shift the replicate mean to the experiment mean; works but needs more evaluation
        # df_mean_diff =  df.mean().mean(level=self.col_exp) - df.mean()
        # df = df + df_mean_diff
        # a[:] = zscore(a, axis=0) # normalize along columns via z-score
        return df, b, a[m]

    def get_two_sided_ttest(self, sum_list, mean_list):
        df = self.get_group(sum_list, mean_list)
        df = df.replace(0, np.nan)
        exp_list = df[self.col_exp].unique()
        df[self.col_area_sum_total] = np.log2(df[self.col_area_sum_total])

        df = df.groupby([self.col_level]).apply(lambda x: pd.Series(
            {self.col_pval: weightstats.ttest_ind(x[self.col_area_sum_total].loc[x[self.col_exp] == exp_list[0]],
                                           x[self.col_area_sum_total].loc[
                                               x[self.col_exp] == exp_list[1]], usevar='pooled')[1]}))
        df = df.dropna()
        df[self.col_fdr] = multicomp.multipletests(np.array(df[self.col_pval]), method='fdr_bh')[1]
        return df

    def rename_columns(self, df):
        if self.cont_type == self.row_details_string:
            df = df.rename(index=str, columns={self.exp_string: self.col_exp, self.level_string: self.col_level,
                                               self.sum_string: self.col_area_sum_total,
                                               self.repl_bio_string: self.col_bio_rep,
                                               self.repl_tech_string: self.col_tech_rep,
                                               self.charge_string: self.col_charge})
        return df

    # create new columns and modifies others to facilitate working with the df
    def prepare_df(self, df):
        # create two separate columns for link type and weight (i.e. heavy or light)
        df[self.col_link_type], df[self.col_weight_type] = df[self.type_string].str.split(':').str
        df[self.col_exp_file] = df.name[:df.name.rfind('_')]
        df[self.col_origin] = df.name
        # df_abs_pos = df[self.uxid_string].str.split(':', expand=True)
        # let's remove the weight from uid string; so we have the same uids for both container types
        if self.cont_type == self.row_details_string and self.level_string == self.uid_string:
            # using regex to match either :heavy or :light and all the following string (.*)
            df[self.uid_string] = df[self.uid_string].str.replace(":({0}|{1}).*"
                                                                  .format(self.row_light_string, self.row_heavy_string),
                                                                  "")
            # doing the same for the heavy/light string
            df[self.uid_string] = df[self.uid_string].str.replace(":({0}|{1}).*"
                                                                  .format(self.row_monolink_string,
                                                                          self.row_xlink_string), "")
            # and again for the charge string
            df[self.uid_string] = df[self.uid_string].str.replace(":({0}).*".format("::*"), "")
        return df

    def filter_df(self, df):
        # optionally filter for link type
        if self.filter:
            df = df.loc[df[self.col_link_type] == self.filter]
        return df

    # function to get ms1 intensities grouped by experiment; works with bag_container.stats and .details
    # returns new dataframe
    def get_sum_ms1_intensities_df(self, df):
        # we split up the input by experiment and compute all ms1 intensities separately
        exp_list = list(set(df[self.exp_string]))
        # this list stores the dataframes by experiment
        df_exp_list = []

        # all ids (either uxid or uid) are put into one set and used for constructing the initial dataframe
        # this also allows for detection of ids missing from one of the two experiments
        id_list = list(set(df[self.level_string]))
        # iterating by experiment
        for exp in exp_list:
            # since we use the experiment to group the plotting, the exp_new name will also include the original file name
            exp_new = "{0} ({1}): {2}".format(exp, self.cont_type, df.name[:df.name.rfind('_')])
            # creating the results dataframe
            df_res = pd.DataFrame()
            kwargs = {self.col_level: id_list,
                      self.col_exp: pd.Series([exp_new for l in range(len(id_list))]),
                      self.col_exp_original: pd.Series([exp for l in range(len(id_list))]),
                      self.col_bag_type: pd.Series([self.cont_type for l in range(len(id_list))]),
                      self.col_origin: pd.Series([df.name for l in range(len(id_list))])}
            df_res = df_res.assign(**kwargs)

            # filtering the input dataframe by experiment name
            df_exp = df.loc[df[self.exp_string] == exp]
            # processing bag_container.details
            # filtering these two means we get exactly the same results as from the regular bag container
            # removes violations (but no violations are calculated for monolinks)
            df_exp = df_exp.loc[df_exp[self.vio_string] == 0]
            df_exp = df_exp.loc[df_exp[self.valid_string] == 1]

            df_exp = df_exp.rename(index=str, columns={self.level_string: self.col_level})
            # summing up the total ms1 sum for the same id; again only really sums at uxid col_level
            df_ms1_tot = df_exp.groupby(
                [self.col_level, self.col_link_type])[self.sum_string].sum().reset_index(name=self.col_area_sum_total)
            df_res = pd.merge(df_res, df_ms1_tot, on=[self.col_level], how='inner')
            # the intensities for light and heavy ids have to calculated explicitly for details container
            # we group by weight and sum up the itensities
            df_ms1_lh = df_exp.groupby(
                [self.col_level, self.col_weight_type])[self.sum_string].sum().reset_index(name=self.col_area_sum_total)
            # we want the heavy and light intensities as separate columns and therefore pivot the dataframe here
            df_ms1_lh = pd.pivot_table(df_ms1_lh, values=self.col_area_sum_total, index=[self.col_level],
                                       columns=self.col_weight_type)
            df_ms1_lh = df_ms1_lh.reset_index()
            df_ms1_lh = df_ms1_lh.rename(
                index=str, columns={self.row_light_string: self.col_area_sum_light,
                                    self.row_heavy_string: self.col_area_sum_heavy})
            df_res = pd.merge(df_res, df_ms1_lh, on=[self.col_level], how='inner')
            df_ms1_bio_rep = df_exp.groupby(
                [self.col_level, self.repl_bio_string])[self.sum_string].sum().reset_index(name=self.col_area_sum_total)
            df_ms1_bio_rep = pd.pivot_table(df_ms1_bio_rep, values=self.col_area_sum_total, index=[self.col_level],
                                            columns=self.repl_bio_string)
            df_ms1_bio_rep = df_ms1_bio_rep.reset_index()
            # df_ms1_bio_rep = df_ms1_bio_rep.fillna(-1)
            rep_list = sorted(set(df_exp[self.repl_bio_string]))
            rep_name_dict = {x: self.col_area_bio_repl + str(x) for x in rep_list}
            df_ms1_bio_rep = df_ms1_bio_rep.rename(index=str, columns=rep_name_dict)
            df_res = pd.merge(df_res, df_ms1_bio_rep, on=[self.col_level], how='inner')
            df_res[self.col_var] = df_res[list(rep_name_dict.values())].var(axis=1)
            df_res[self.col_std] = df_res[list(rep_name_dict.values())].std(axis=1)
            # not sure if we should fill up na here with 0
            # if we do it would mean an id not detected in one experiment gets a 0 intensity
            # df_res = df_res.fillna(0)

            # computes a normalized by maximum ms1 intensity; not used atm
            df_res[self.col_area_sum_norm_total] = df_res[self.col_area_sum_total] / df_res[
                self.col_area_sum_total].max()
            # computes the light/heavy log2ratio
            df_res[self.col_lh_log2ratio] = np.log2(df_res[self.col_area_sum_light] / df_res[self.col_area_sum_heavy])
            # df_res[col_area_sum_norm_total] = (df_res[col_area_sum_total] - df_res[col_area_sum_total].min()) / (df_res[col_area_sum_total].max() - df_res[col_area_sum_total].min())
            df_exp_list.append(df_res)
        df_final = pd.DataFrame()
        for dfl in df_exp_list:
            df_final = df_final.append(dfl)
        # computing the same violations as extract and removing them; right now only works on uid col_level
        df_final = self.remove_violations(df_final)

        # computing the log2ratio between the two experiments; hardcoded right now; at least reference should be a user setting
        df_log2 = pd.pivot_table(df_final, values=self.col_area_sum_total, index=[self.col_level],
                                 columns=self.col_exp).reset_index()
        df_log2 = df_log2.dropna()
        df_log2[self.col_log2ratio] = np.log2(df_log2.iloc[:, 2] / df_log2.iloc[:, 1])
        df_final = pd.merge(df_final, df_log2[[self.col_level, self.col_log2ratio]], on=[self.col_level], how='left')
        return df_final

    # removes uids with violations; only checks for light/heavy log2ratio violations
    # TODO: implement violation which requires all charge states of a peptide to be present in both experiments
    # TODO: split up into violation assign (use separate columns) and filtering
    # TODO: keep violation columns from xTract for comparison
    def remove_violations(self, df):
        name = set(df[self.col_origin])
        print("Shape of {0} before filtering zero intensities: {1}.".format(name, df.shape))
        df = df[(df[self.col_area_sum_light] > 0) | (df[self.col_area_sum_heavy] > 0)]
        print("Shape of {0} before filtering lh log2ratio: {1}.".format(name, df.shape))
        df = df[(df[self.col_lh_log2ratio] < 1) & (df[self.col_lh_log2ratio] > -1)]
        print("Shape of {0} after filtering: {1}.".format(name, df.shape))
        return df

    def divide_int_by_reps(self, df):
        # TODO: check if this is really necessary
        # ms1 intensities have to be divided by tech_replicates*bio_replicates when coming from a details container
        if self.cont_type == self.row_details_string:
            df[[self.col_area_sum_total, self.col_area_sum_light, self.col_area_sum_heavy]] = \
                df[[self.col_area_sum_total, self.col_area_sum_light, self.col_area_sum_heavy]].apply(
                    lambda x: x / self.bio_rep_num)
