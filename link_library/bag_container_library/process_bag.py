import pandas as pd
import numpy as np
import copy
from statsmodels.sandbox.stats import multicomp
from statsmodels.stats import weightstats
from scipy.stats import levene
from scipy.stats import zscore


# TODO: right now log2ratio is grouped by link type but pvals are not leading to different results in case of loop links
# TODO: create the filter options for the df_orig; eventually replace the df_new completely
# TODO: check whether xTract filters for the peptide fdr or if it's already done for the bag container
# TODO: add list of columns to preserve; i.e. get_pivot/get_group should always preserve uxid, uid and violations

class BagContainer(object):

    def __init__(self, level, df_list, filter=None, sel_exp=False, impute_missing=False, norm_exps='yes',
                 norm_reps=False, df_domains=None, df_dist=None, whitelist=None, sortlist=None, vio_list=('lh', 'xt')):
        self.impute_missing = impute_missing
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
        self.col_area_sum_light = self.col_area_sum_total + '_light'
        self.col_area_sum_heavy = self.col_area_sum_total + '_heavy'
        self.col_area_z_score = 'z_val'
        self.col_area_rep = self.col_area_sum_total + '_rep_mean'
        self.col_area_exp = self.col_area_sum_total + '_exp_mean'
        self.rep_exp_ratio_string = self.col_area_sum_total + '_rep_exp_ratio'
        self.col_index = 'index'
        self.col_log2ratio = 'log2ratio'
        self.col_ratio = 'ratio'
        self.col_log2avg = 'log2avg'
        self.col_log2ratio_ref = 'log2ratio_ref_exp'
        self.col_lh_log2ratio = 'light_heavy_log2ratio'
        self.col_bio_rep = 'exp_bio_rep'
        self.col_tech_rep = 'exp_tech_rep'
        self.col_charge = 'charge'
        self.col_pval = 'pval'
        self.col_fdr = 'qval'
        self.col_mean = 'mean'
        self.col_std = 'std'
        self.col_proteins = 'proteins'
        self.col_positions = 'link_positions'
        self.col_reaction_state = 'reaction_state'
        self.col_imputed = 'imputed'
        if df_domains is not None:
            self.col_domain = 'domain'
        else:
            self.col_domain = None
        self.distance_list = []
        if df_dist is not None:
            for col in df_dist.columns:
                if col != self.col_exp and col != self.col_uxid:
                    self.distance_list.append(col)
        self.dom_prot = 'protein'
        self.dom_range = 'range'
        self.norm_exp = norm_exps
        self.bag_container_index_string = 'a_bag_container_db_index'
        self.row_monolink_string = 'monolink'
        self.row_xlink_string = 'xlink'
        self.row_loop_link = 'intralink'
        self.row_light_string = 'light'
        self.row_heavy_string = 'heavy'
        self.row_details_string = 'details'
        self.row_quenched = 'quenched'
        self.row_hydrolyzed = 'hydrolyzed'
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
        self.sum_string_orig = 'c_pg_area_sum_isotopes_org'
        self.first_iso_string = 'c_pg_max_int_first_iso'
        self.first_iso_string_orig = 'c_pg_max_int_first_iso_org'
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
            print(f"ERROR: Improper ms1 col_level entered: {level}")
            exit(1)
        self.filter = ""
        if filter is not None:
            if filter.lower() == self.row_monolink_string:
                self.filter = self.row_monolink_string
            elif filter.lower() == self.row_xlink_string:
                self.filter = self.row_xlink_string
            elif filter.lower() == self.row_loop_link:
                self.filter = self.row_loop_link
            elif filter:
                print(f"ERROR: Improper link filter entered: {filter}")
                exit(1)
        self.type = f'BagContainer_{level}'
        # df orig should contain the original columns as given by xquest/xtract (just renamed and a few additions)
        self.df_orig = pd.DataFrame()
        self.df_domains = df_domains
        for df in df_list:
            df = self.prepare_df(df)
            if self.filter:
                df = self.filter_link_type(df)
            self.df_orig = self.df_orig.append(copy.deepcopy(self.rename_columns(df)))
        if sel_exp:
            exp_dict = {no: exp for no, exp in enumerate(sorted(self.df_orig[self.col_exp].unique()))}
            print("Please select the experiments you want to exclude")
            print(f"{exp_dict}")
            sel = input("Enter a numbers separated by spaces): ")
            sel = sel.split(" ")
            sel = [int(s) for s in sel]
            sel = [exp_dict[s] for s in exp_dict.keys() if s not in sel]
            print(f"The following experiments were selected: {sel}")
            self.df_orig = self.filter_exp(self.df_orig, sel)
        self.df_orig = self.remove_invalid_ids(self.df_orig)
        self.df_orig = self.compute_lh_log2ratio(self.df_orig)
        if 'lh' in vio_list:
            self.df_orig = self.remove_lh_violations(self.df_orig)
        if 'xt' in vio_list:
            self.df_orig = self.remove_xtract_violations(self.df_orig)
        if 'lh' not in vio_list and 'xt' not in vio_list:
            'WARNING: No violations removal was selected. Results will be unfiltered'
        if whitelist is not None:
            self.df_orig = self.filter_linky_by_whitelist(self.df_orig, whitelist)
        self.df_orig = self.compute_rep_and_exp_mean(self.df_orig)
        if norm_reps:
            self.df_orig = self.normalize_replicates(self.df_orig)
        if norm_exps == 'yes':
            self.df_orig = self.normalize_experiments(self.df_orig)
        if self.distance_list:
            self.df_orig = self.add_link_distances(self.df_orig, df_dist)
        self.df_orig = self.get_reaction_state(self.df_orig)
        self.minimal_groups = [self.col_exp, self.col_bio_rep, self.col_tech_rep, self.col_weight_type,
                               self.col_link_type, self.col_origin]
        self.area_columns = [self.col_area_sum_total, self.col_area_sum_light, self.col_area_sum_heavy]
        if self.col_reaction_state in self.df_orig.columns:
            self.minimal_groups.append(self.col_reaction_state)
        self.df_orig = self.df_orig.groupby([self.col_level] + self.minimal_groups)[self.area_columns].sum().reset_index()
        if self.impute_missing:
            self.df_orig = self.get_imputed_df(self.df_orig)
            # some plots need the light/heavy area columns to function
            self.df_orig = self.compute_lh_log2ratio(self.df_orig, force_groups=[self.col_level, self.col_exp, self.col_link_type])
        self.bio_rep_list = self.df_orig[self.col_bio_rep].unique()
        self.exp_list = sorted(self.df_orig[self.col_exp].unique())
        if sortlist is not None:
            sortlist = sortlist[self.col_exp].values
            sortlist = [e for e in sortlist if e in self.exp_list]
            self.df_orig[self.col_exp] = pd.Categorical(
                self.df_orig[self.col_exp],
                categories=sortlist,
                ordered=True
            )
        self.bio_rep_num = len(self.bio_rep_list)
        self.tech_rep_list = self.df_orig[self.col_tech_rep].unique()
        self.tech_rep_num = len(self.tech_rep_list)

        self.print_info(self.df_orig)
        # self.df_orig[self.col_area_sum_total] = np.log2(self.df_orig[self.col_area_sum_total])
        # self.df_orig = self.divide_int_by_reps(self.df_orig)

    def print_info(self, df):
        print("The following experiments and number of replicates were found in the bag container")
        print(df.groupby(self.col_exp)[self.col_bio_rep, self.col_tech_rep].nunique(), '\n')

    # normalizes inter-experiment abundance in the xTract way: use one experiment as the reference
    # calculate the mean intensities and their ratio compared to the reference
    # divide/multiply all experiments by these factors
    def normalize_experiments(self, df):
        norm_string = 'norm_factor'
        norm_string_inverted = norm_string + '_inverted'
        # sum_list = [self.bag_container_index_string, self.col_exp, self.col_uid]
        mean_list = [self.col_exp]
        # df_sum = df.groupby(sum_list)[self.col_area_sum_total].sum().reset_index()
        # df[self.col_area_sum_total] = np.log2(df[self.col_area_sum_total])
        df_norm = df.groupby(mean_list)[self.col_area_sum_total].mean().reset_index(name=norm_string)
        # df_norm[self.col_area_sum_total] = np.log2(df_norm[self.col_area_sum_total])
        # reference factor to normalize to
        # as an alternative to the mean one could also use a reference experiment (like xTract does) or the max() or the min()
        exp_ref_factor = df_norm[norm_string].mean()
        df_norm[norm_string] /= exp_ref_factor
        df_norm[norm_string_inverted] = df_norm[norm_string] ** -1
        print("\nExperiment abundance normalization found the following normalization factors:")
        print(df_norm, '\n')
        for exp in df_norm[self.col_exp]:
            df.loc[df[self.col_exp] == exp, [self.col_area_sum_total]] /= \
                df_norm.loc[df_norm[self.col_exp] == exp, [norm_string]].values[0]
        print("Experiment areas after normalization:")
        print(df.groupby(mean_list)[self.col_area_sum_total].mean(), '\n')
        return df

    def compute_rep_and_exp_mean(self, df):
        mean_list = [self.col_exp, self.col_bio_rep, self.col_tech_rep]

        df[self.col_area_rep] = df.groupby(mean_list)[self.col_area_sum_total].transform(np.mean)
        df[self.col_area_exp] = df.groupby([self.col_exp])[self.col_area_rep].transform(np.mean)

        return df

    def normalize_replicates(self, df):
        rep_exp_ratio_string = self.col_area_sum_total + '_rep_exp_ratio'
        mean_list = [self.col_exp, self.col_bio_rep, self.col_tech_rep]
        df[self.col_area_sum_total + '_before_rep_norm'] = df[self.col_area_sum_total]

        df_mean_rep = df.groupby(mean_list)[self.col_area_sum_total].mean()
        print("Mean replicate areas before normalization: ")
        print(df_mean_rep, '\n')
        df_mean_exp = df.groupby([self.col_exp])[self.col_area_sum_total].mean()
        print("Mean experiment areas: ")
        print(df_mean_exp, '\n')

        df[rep_exp_ratio_string] = df[self.col_area_rep] / df[self.col_area_exp]
        df[self.col_area_sum_total] = df[self.col_area_sum_total] / df[rep_exp_ratio_string]
        # print(df.groupby([self.col_exp, self.col_bio_rep, self.col_tech_rep]).apply(lambda x:print(x[[self.col_exp, self.col_bio_rep, self.col_tech_rep, self.col_uid, self.col_area_sum_total, self.col_area_sum_total + '_mean', self.col_area_sum_total + '_exp_mean', self.col_area_sum_total + '_diff', self.col_area_sum_total + '_norm']])))
        print("Mean replicate areas after normalization: ")
        print(df.groupby(mean_list)[self.col_area_sum_total].mean(), '\n')
        return df

    def rename_columns(self, df):
        # the ms1 area string depends on the kind of the experiment normalization
        ms1_string = ""
        if self.norm_exp == 'yes' or self.norm_exp == 'no':
            # use the non_normalized ms1 area if it exists
            # this is the case if xTract's normalization was used
            if self.sum_string_orig in df:
                ms1_string = self.sum_string_orig
                print("Experiment Normalization: Found column containing non-normalized values")
            # if the string does not exist we assume the default column is not normalized
            else:
                ms1_string = self.sum_string
                print("Experiment Normalization: Values have not been normalized by xTract. Using default column")
        elif self.norm_exp == 'xt':
            # if we're using xTract's normalization the sum_orig must exist we don't want to use it
            if self.sum_string_orig in df:
                ms1_string = self.sum_string
                print("Experiment Normalization: Using values calculated by xTract")
            # if the string does not exist we assume something is not right
            else:
                print(
                    "ERROR: xTract's experiment normalization was specified but the normalized values are not in the bag container. Exiting")
                exit(1)
        else:
            print(f"ERROR: Unknown normalization method specified: {self.norm_exp}. Exiting")
            exit(1)

        df = df.rename(index=str, columns={self.exp_string: self.col_exp, self.uid_string: self.col_uid,
                                           self.uxid_string: self.col_uxid,
                                           ms1_string: self.col_area_sum_total,
                                           self.repl_bio_string: self.col_bio_rep,
                                           self.repl_tech_string: self.col_tech_rep,
                                           self.charge_string: self.col_charge, })
        return df

    # create new columns and modify others to facilitate working with the df
    def prepare_df(self, df):
        # create two separate columns for link type and weight (i.e. heavy or light)
        df[self.col_link_type], df[self.col_weight_type] = df[self.type_string].str.split(':').str
        df[self.col_exp_file] = df.name[:df.name.rfind('_')]
        df[self.col_origin] = df.name
        # df_abs_pos = df[self.uxid_string].str.split(':', expand=True)
        # let's remove the weight, link type and charge from uid string; so we have the same uids for both container types
        # using regex to match either :heavy or :light and all the following string (.*)
        df[self.uid_string] = df[self.uid_string].str.replace(
            f":({self.row_light_string}|{self.row_heavy_string}).*", "")
        # doing the same for the link type string
        df[self.uid_string] = df[self.uid_string].str.replace(
            f":({self.row_monolink_string}|{self.row_xlink_string}|{self.row_loop_link}).*", "")
        # and again for the charge string
        df[self.uid_string] = df[self.uid_string].str.replace(f":({'::*'}).*", "")
        return df

    def filter_link_type(self, df):
        # optionally filter for link type
        if self.filter:
            df = df.loc[df[self.col_link_type] == self.filter]
        return df

    def filter_exp(self, df, sel_list):
        # optionally filter experiments
        df = df.loc[df[self.col_exp].isin(sel_list)]
        return df

    def filter_linky_by_whitelist(self, df, df_white_list):
        # uxid and and exp are mandatory; further columns are possible but optional
        for col in df_white_list.columns:
            if col not in df.columns:
                print(f"ERROR: column \"{col}\" was"
                      f" found inside the whitelist but not in bag container. Exiting")
                exit(1)
            for entry in df_white_list[col].unique():
                if entry not in df[col].unique():
                    print(f"WARNING: the whitelist column {col} contains an entry called {entry} which is not a valid "
                          f"entry in the bag container. The row containing this entry will be IGNORED")
        name = set(df[self.col_origin])
        print(f"The whitelist contains {len(df_white_list)} entries")
        print(f"Shape of {name} before filtering via whitelist: {df.shape}.")
        df = pd.merge(df, df_white_list, on=list(df_white_list.columns))
        print(f"Shape of {name} after filtering via whitelist: {df.shape}.")
        return df

    def get_group(self, sum_list, mean_list, group_on, log2=False, z_score=False):
        df = pd.DataFrame(self.df_orig)
        # turn groupon into a list if it is not already
        if not isinstance(group_on, list):
            group_on = [group_on]
        if self.col_imputed in df.columns:
            group_on.append(self.col_imputed)
        df = df.groupby([self.col_level] + sum_list)[group_on].sum().reset_index()
        df = df.groupby([self.col_level] + mean_list)[group_on].mean().reset_index()
        if log2 or z_score:
            df[self.col_area_sum_total] = df[self.col_area_sum_total].map(np.log2)
        if z_score:
            df[self.col_area_z_score] = df.groupby([self.col_level])[self.col_area_sum_total].transform(zscore)
        return df

    def remove_invalid_ids(self, df):
        name = set(df[self.col_origin])
        print(f"Shape of {name} before filtering invalid ids: {df.shape}.")
        df = df.loc[df[self.valid_string] == 1]
        print(f"Shape of {name} after filtering invalid ids: {df.shape}.")
        return df

    def remove_xtract_violations(self, df):
        name = set(df[self.col_origin])
        # filtering these two means we get exactly the same results as from the regular bag container
        # removes violations (but no violations are calculated for monolinks)
        df = df.loc[df[self.vio_string] == 0]
        print(f"Shape of {name} after removing xTract violations: {df.shape}.")
        return df

    def compute_lh_log2ratio(self, df, force_groups=None):
        # xtract filters per uid, experiment and charge state and link type
        # note that this will compute across all replicates
        def get_lh_ratio(x):
            log2 = np.nan
            log2series = x.groupby([self.col_weight_type])[self.col_area_sum_total].sum()
            if len(log2series) == 2:
                log2 = np.log2(log2series[1] / log2series[0])
            x[self.col_lh_log2ratio] = log2
            if not np.isnan(log2):
                x[self.col_area_sum_light] = log2series[self.row_light_string]
                x[self.col_area_sum_heavy] = log2series[self.row_heavy_string]
            else:
                x[self.col_area_sum_light] = np.nan
                x[self.col_area_sum_heavy] = np.nan
            return x
        groups = [self.col_uid, self.col_exp, self.col_charge, self.col_link_type]
        if force_groups is not None:
            groups = force_groups
        df = df.groupby(groups).apply(get_lh_ratio)
        return df

    # TODO: for xTract: instead of filling missing values with a shifted distribution it will assign the detection limit
    # TODO: for the missing value (10E3) with a var of 500 then use the maximum intensity of the first isotope as ref
    # TODO: don't impute links with violations ?
    # replace missing observations by drawing random values from a normal distribution
    # 1) pivot and unstack -> creates the missing entries
    # 2) convert to log2 scale in order to have a normal distribution of our observations
    # 3) determine mean and sd of our normal distribution
    # 4) draw random numbers for the missing values using a downward shifted normal distribution
    # ref: http://www.coxdocs.org/doku.php?id=perseus:user:activities:matrixprocessing:imputation:replacemissingfromgaussian
    # ref: https://github.com/JurgenCox/perseus-plugins/blob/master/PerseusPluginLib/Impute/ReplaceMissingFromGaussian.cs
    # default values from mq (shift=1.8, width=0.3) for a normal dist
    def get_imputed_df(self, df):
        # helper function which first pivots the dataframe and then unstacks it again
        # this will automatically create missing values based on the minimal groups
        # for example a values was not observed in one the technical replicates but not in another
        # missing values will be returned as NaNs
        def _stacker(df_tmp):
            df_pivot = pd.pivot_table(df_tmp, values=self.col_area_sum_total, index=[self.col_level],
                                      columns=self.minimal_groups, aggfunc=np.sum)
            df_unstack = df_pivot.unstack().reset_index(name=self.col_area_sum_total).sort_values(
                [self.col_level, self.col_exp])
            return df_unstack
        # setting the seed makes the number drawing reproducible
        np.random.seed(5029)
        # parameters for the distribution shift; using the MaxQuant defaults
        shift = 1.8
        width = 0.3
        # grouping by link type and reaction state (i.e. quenched or hydrolyzed monolinks) prevents creating duplicate
        # entries for theses groups (i.e. a uxID for monolink would otherwise create a crosslink entry)
        # also monolinks can either have the reaction state hydrolyzed or quenched while all other link types
        # are neither. Without the groupby this would create artificial quenched/hydrolyzed states for other link types
        df = df.groupby([self.col_link_type, self.col_reaction_state], as_index=False).apply(
            _stacker).reset_index(drop=True)
        #possible futture code for separate monolink treatment
        # df_mono = df[df[self.col_link_type] == self.row_monolink_string]
        # df_rest = df[~(df[self.col_link_type] == self.row_monolink_string)]
        # df_mono = _stacker(df_mono).reset_index(drop=True)
        # print(df_mono[df_mono[self.col_area_sum_total].isna()])
        # grp_list = self.minimal_groups.copy()
        # grp_list.remove(self.col_reaction_state)
        # df_mono = df_mono.groupby([self.col_level] + grp_list).filter(lambda x: x[self.col_area_sum_total].isna().sum() < 2).reset_index(drop=True)
        # print(df_mono[df_mono[self.col_area_sum_total].isna()])
        # exit()
        # df_rest = df_rest.groupby([self.col_link_type, self.col_reaction_state], as_index=False).apply(_stacker).reset_index(drop=True)
        # df = pd.concat([df_rest, df_mono]).reset_index(drop=True)
        #df = df.groupby([self.col_link_type, self.col_reaction_state], as_index=False).apply(_stacker).reset_index(drop=True)
        # original distribution; replacing 0 with NaNs and then dropping all NaNs
        values_org = df[self.col_area_sum_total].replace(0, np.nan).dropna().values
        # original distribution in log2 scale
        values_org_log2 = np.log2(values_org)
        # column denoting whether a value was imputed
        df[self.col_imputed] = False
        a = df[self.col_area_sum_total].values
        # mask of NaNs or 0s
        m = np.isnan(a) | (a == 0)
        # label imputed values
        df.loc[m, self.col_imputed] = True
        # mean and std of total matrix (in log2 scale)
        mu, sigma = np.mean(values_org_log2), np.std(values_org_log2)
        # draw random numbers
        a[m] = np.random.normal(mu - shift * sigma, width * sigma, size=m.sum())
        # since log2 values are not desired we go back to the original distribution
        df.loc[m, self.col_area_sum_total] = 2 ** a[m]
        # the following two lines shift the replicate mean to the experiment mean; works but needs more evaluation
        # df_mean_diff =  df.mean().mean(level=self.col_exp) - df.mean()
        # df = df + df_mean_diff
        # a[:] = zscore(a, axis=0) # normalize along columns via z-score
        print(f"Imputed {np.sum(m)} values of {len(df)} total values")
        return df

    def add_link_distances(self, df, df_dist):
        df_dist = df_dist.drop_duplicates()
        print(df.shape)
        df = pd.merge(df, df_dist, how='left', on=[self.col_exp, self.col_uxid])
        print(df.shape)
        return df

    def remove_lh_violations(self, df):
        # xtract filters per uid, experiment and charge state
        name = set(df[self.col_origin])
        print(f"Shape of {name} before light/heavy log2 filter: {df.shape}.")
        df = df[(df[self.col_lh_log2ratio] > -1) & (df[self.col_lh_log2ratio] < 1)]
        print(f"Shape of {name} after light/heavy log2 filter: {df.shape}")
        return df

    def get_stats(self, sum_list, mean_list, log2=False):
        df = pd.DataFrame(self.df_orig)
        df = df.groupby([self.col_level] + sum_list)[self.col_area_sum_total].sum().reset_index()
        if log2:
            df[self.col_area_sum_total] = np.log2(df[self.col_area_sum_total])
        # print(df)
        # df = df.replace(0, np.nan)

        # print(pd.pivot_table(df, values=self.col_area_sum_total, index=[self.col_level], columns=sum_list,
        #                      aggfunc=np.sum))

        df = df.groupby([self.col_level] + mean_list)[self.col_area_sum_total].agg([pd.Series.mean, pd.Series.std])
        df['snr'] = df['mean'] / df['std']
        # 'iqr': lambda x: x.quantile(0.75)-x.quantile(0.25),  'median': pd.Series.median, 'q25': lambda x: x.quantile(0.25), 'q75': lambda y: y.quantile(0.75)})

        df = df.dropna()
        len_index = len(df.index.names)
        levels = [i for i in range(len_index)]
        df = df.reset_index(level=levels)
        print("DF Stats mean")
        print(df.groupby(self.col_exp).mean())
        return df

    def get_pivot(self, sum_list, mean_list, pivot_on, log2=False):
        df = self.get_group(sum_list, mean_list, pivot_on, log2=log2)
        df = pd.pivot_table(df, values=pivot_on, index=[self.col_level],
                            columns=mean_list, aggfunc=np.sum)
        return df

    def get_log2ratio(self, sum_list, mean_list, ref=None, ratio_only=False, keep_ref=False, ratio_between=None):
        ref_string = '_ref'
        area_sum_ref = self.col_area_sum_total + ref_string
        if self.col_link_type not in sum_list:
            sum_list.append(self.col_link_type)
        if self.col_link_type not in mean_list:
            mean_list.append(self.col_link_type)
        df = self.get_group(sum_list, mean_list, group_on=self.col_area_sum_total)
        if ratio_between is None:
            ratio_between = self.col_exp
        if not ref:
            if df[ratio_between].dtype.name == 'category':
                ref = df[ratio_between].cat.categories[0]
            else:
                ref = sorted(df[ratio_between].unique())[0]
        # group list contains the columns to differentiate by
        grp_list = [self.col_level, self.col_link_type]
        # if we are not computing the log2ratio based on the experiments we should differentiate them
        if ratio_between != self.col_exp:
            grp_list.append(self.col_exp)
        df_ref = df[df[ratio_between] == ref]
        if self.impute_missing:
            df_ref = df_ref[[self.col_area_sum_total, self.col_imputed] + grp_list]
        else:
            df_ref = df_ref[[self.col_area_sum_total] + grp_list]

        df = pd.merge(df, df_ref, on=grp_list, suffixes=('', ref_string))
        if ratio_only:
            df[self.col_ratio] = df[self.col_area_sum_total] / df[area_sum_ref]
        else:
            df[self.col_log2ratio] = np.log2(df[self.col_area_sum_total] / df[area_sum_ref])
        df[self.col_log2avg] = np.log2(df[self.col_area_sum_total] * df[area_sum_ref]) / 2
        df[self.col_log2ratio_ref] = ref
        if not keep_ref:
            df = df.loc[df[ratio_between] != ref]
        return df

    def get_distance_delta_df(self, sum_list, mean_list, ref=None):
        if self.col_link_type not in sum_list:
            sum_list.append(self.col_link_type)
        if self.col_link_type not in mean_list:
            mean_list.append(self.col_link_type)
        df = self.get_group(sum_list, mean_list, group_on=self.col_area_sum_total, log2=True)
        grp_list = [self.col_level, self.col_link_type]
        if not ref:
            if df[self.col_exp].dtype.name == 'category':
                ref = df[self.col_exp].cat.categories[0]
            else:
                ref = sorted(df[self.col_exp].unique())[0]
        df_ref = df[df[self.col_exp] == ref]
        df_ref = df_ref[self.distance_list + grp_list]
        ref_rename_dict = {n: n + '_exp_ref' for n in self.distance_list}
        df_ref = df_ref.rename(index=str, columns=ref_rename_dict)
        df = pd.merge(df, df_ref, on=grp_list)
        for dist in self.distance_list:
            df[dist + '_delta'] = df[dist] - df[dist + '_exp_ref']
        df['distance_ref'] = ref
        df = df.loc[df[self.col_exp] != ref]
        return df

    def getlog2ratio_r(self, sum_list, mean_list, ref):
        def get_loggi(x):
            log2col = pd.Series(np.log2(
                x[self.col_area_sum_total] / x[self.col_area_sum_total].loc[x[self.col_bio_rep] == ref].values[0]))
            log2col = log2col.rename(self.col_log2ratio)
            log2avgcol = pd.Series(np.log2(
                x[self.col_area_sum_total] * x[self.col_area_sum_total].loc[x[self.col_bio_rep] == ref].values[0]) / 2)
            log2avgcol = log2avgcol.rename(self.col_log2avg)
            df = pd.concat([x, log2col, log2avgcol], axis=1)
            kwargs = {self.col_log2ratio_ref: ref}
            df = df.assign(**kwargs)
            return df

        df = self.df_orig.groupby([self.col_exp, self.col_level]).apply(get_loggi).reset_index(drop=True)
        df = df.drop(columns=[self.col_area_sum_total])  # makes no sense to return this column for a single experiment
        df = df.loc[df[self.col_bio_rep] != ref]
        return df

    def get_two_sided_ttest(self, sum_list, mean_list, ref):
        # function takes a dataframe grouped by ids and calculates pvalues against a reference
        def get_ttest(x):
            # takes two group values, determines whether their variances are equal (via levene test)
            # and computes and independent t-test with either pooled (Student) or non-pooled (Welsh) variances
            def compare_variances(vals_exp, vals_ref):
                if not np.isnan(np.var(vals_exp)) and not np.isnan(np.var(vals_ref)):
                    eq_var_pval = levene(vals_exp, vals_ref)[1]
                    if eq_var_pval > 0.01:
                        var = 'pooled'
                    else:
                        var = 'unequal'
                        # print("unequal_ref\n", vals_ref,)
                        # print("unequal_exp\n", vals_exp)
                        # print("unequal_var\n", np.var(vals_ref), np.var(vals_exp))
                    return weightstats.ttest_ind(vals_exp, vals_ref, usevar=var)[1]
                else:
                    return np.nan

            y = x.groupby(self.col_exp).apply(
                lambda t: pd.Series({self.col_pval: compare_variances(t[self.col_area_sum_total],
                                                                      x[self.col_area_sum_total].loc[
                                                                          (x[self.col_exp] == ref)])}))
            return y

        def get_fdr(x):
            qvals = multicomp.multipletests(np.array(x[self.col_pval]), method='fdr_bh')[1]
            x = x.assign(**{self.col_fdr: qvals})
            return x

        df = self.get_group(sum_list, mean_list, group_on=self.col_area_sum_total)
        df = df.groupby([self.col_link_type, self.col_level]).apply(get_ttest).reset_index()
        df = df.dropna()
        df = df.groupby([self.col_exp]).apply(get_fdr).reset_index(drop=True)
        df = df.loc[df[self.col_exp] != ref]
        print("qvals smaller than 0.05", len(df[df[self.col_fdr] <= 0.05]))
        print("pvals smaller than 0.01", len(df[df[self.col_pval] <= 0.01]))
        return df

    def get_prot_name_and_link_pos(self, df):
        # temp df; splitting uxid into positions and protein names
        def get_pos_and_prot(entry_list):
            pos_list = []
            prot_list = []
            for entry in entry_list:
                if entry == '':
                    # found a link to amino acid one; xQuest just assigns an empty string here
                    entry = '1'
                if entry.isdigit():
                    pos_list.append(int(entry))
                elif not entry.isdigit() and entry != 'x':
                    prot_list.append(entry)
            return pd.Series([pos_list, prot_list])

        df_tmp = df[self.col_uxid].str.split(':').apply(get_pos_and_prot)
        # direct assignment does not work as it takes just the column headers as values
        df[self.col_positions], df[self.col_proteins] = df_tmp[0], df_tmp[1]
        # if the bag container was given a domains list try to match it
        if self.df_domains is not None:
            # first level of apply function; will return df with columns uxid and matching domain, if any
            def match_domains(df_tmp):
                # second level of apply function; will return a boolean list with True for a matching range
                def filter_range(range_list, pos):
                    assert len(range_list) == 2, "ERROR: Domain range should include two values (start and end)"
                    pos1 = int(range_list[0])
                    pos2 = int(range_list[1])
                    return pos1 <= pos <= pos2

                domain_list = []
                pos_list = df_tmp[self.col_positions].iloc[0]
                prot_list = df_tmp[self.col_proteins].iloc[0]
                assert len(pos_list) == len(prot_list)
                for n, prot in enumerate(prot_list):
                    pos = pos_list[n]
                    df_dom = self.df_domains[self.df_domains[self.dom_prot] == prot]
                    range_series = df_dom[self.dom_range].str.split('-')
                    valid_domains = range_series.apply(lambda x: filter_range(x, pos))
                    df_dom = df_dom[valid_domains]
                    # should be a list containing a single value
                    if len(df_dom) == 1:
                        domain = df_dom[self.col_domain].values[0]
                    elif len(df_dom) == 0:
                        domain = 'Unknown'
                    else:
                        print(
                            f'ERROR: Multiple domains found: {df_dom.values}. Protein: {prot}. Position: {pos}.\nExiting.')
                        exit(1)
                    domain_list.append(domain)
                # remove duplicates
                domain_list = list(set(domain_list))
                # always sort so that the order of domains is identical for inversed crosslinks
                domain_list.sort()
                # join on - to get domain1-domain2 etc.
                domain_all = '-'.join(domain_list)
                return domain_all

            df_dom = df.groupby(self.col_uxid).apply(match_domains).reset_index(name=self.col_domain)
            df = pd.merge(df, df_dom, on=self.col_uxid)
        return df

    # removes uids with violations; only checks for light/heavy log2ratio violations
    # TODO: implement violation which requires all charge states of a peptide to be present in both experiments
    # TODO: split up into violation assign (use separate columns) and filtering
    # TODO: keep violation columns from xTract for comparison
    def remove_violations(self, df):
        name = set(df[self.col_origin])
        print(f"Shape of {name} before filtering zero intensities: {df.shape}.")
        df = df[(df[self.col_area_sum_light] > 0) | (df[self.col_area_sum_heavy] > 0)]
        print(f"Shape of {name} before filtering lh log2ratio: {df.shape}.")
        df = df[(df[self.col_lh_log2ratio] < 1) & (df[self.col_lh_log2ratio] > -1)]
        print(f"Shape of {name} after filtering: {df.shape}.")
        return df

    def divide_int_by_reps(self, df):
        # TODO: check if this is really necessary
        # ms1 intensities have to be divided by tech_replicates*bio_replicates when coming from a details container
        # if self.cont_type == self.row_details_string:
        #     df[[self.col_area_sum_total, self.col_area_sum_light, self.col_area_sum_heavy]] = \
        #         df[[self.col_area_sum_total, self.col_area_sum_light, self.col_area_sum_heavy]].apply(
        #             lambda x: x / self.bio_rep_num)
        if self.cont_type == self.row_details_string:
            df[self.col_area_sum_total] = \
                df[self.col_area_sum_total].apply(lambda x: x / self.bio_rep_num * self.tech_rep_num)
        return df

    def get_matching_monos(self):
        import link_library as ll
        df = self.df_orig.copy()
        print(df.groupby(self.col_link_type)[self.col_uxid].nunique())
        pos_string = "Positions"
        prot_string = "Proteins"
        link_group_string = "link_group"
        df_tmp = df[self.col_uxid].str.split(':').apply(ll.get_prot_name_and_link_pos)
        # direct assignment does not work as it takes just the column headers as values
        df[pos_string], df[prot_string] = df_tmp[0], df_tmp[1]

        def renumber_groups(x):
            if not hasattr(renumber_groups, "counter"):
                renumber_groups.counter = 0  # it doesn't exist yet, so initialize it
            x[link_group_string] = renumber_groups.counter
            renumber_groups.counter += 1
            return x

        def filter_link_groups(df, no_monos=2):
            df = df.groupby(link_group_string).filter(
                lambda x: x[x[self.col_link_type] == self.row_xlink_string][self.col_uxid].nunique() == 1)
            df = df.groupby(link_group_string).filter(
                lambda x: x[x[self.col_link_type] == self.row_monolink_string][self.col_uxid].nunique() >= no_monos)
            df = df.reset_index(drop=True)
            df = df.groupby(link_group_string).apply(renumber_groups)
            return df

        def find_associated_link(x, df_monos):
            def is_equal(a, b):  # given two links a and b check whether they link the same protein and position
                prots_a = (a[prot_string].iloc[0])
                pos_a = (a[pos_string].iloc[0])
                prot_pos_a = set(zip(prots_a, pos_a))  # put prot name and pos into tuple for easy set intersection
                prots_b = (b[prot_string].iloc[0])
                pos_b = (b[pos_string].iloc[0])
                prot_pos_b = set(zip(prots_b, pos_b))
                link_intersect = prot_pos_a & prot_pos_b  # get intersecting links
                if len(link_intersect) > 0:
                    return True
                return False

            tmp = df_monos.groupby(self.col_uxid).filter(lambda y: is_equal(x, y))

            if len(tmp) > 0:
                tmp[link_group_string] = x[link_group_string].iloc[0]

                # x[associated_link_string] = [tmp[uid_string].values]  # assignment is buggy and not needed anyway
                x = pd.concat([x, tmp], sort=True)

            return x

        df_xlinks = df[
            df[self.col_link_type] == self.row_xlink_string].copy()  # using a copy since I set values in the next line
        df_xlinks = df_xlinks.groupby(self.col_uxid).apply(renumber_groups)
        df_monos = df[df[self.col_link_type] == self.row_monolink_string]
        df_new = df_xlinks.groupby(self.col_uxid).apply(lambda x: find_associated_link(x, df_monos)).reset_index(
            drop=True)
        num_mono1 = df_new.groupby(link_group_string).filter(
            lambda x: x[x[self.col_link_type] == self.row_monolink_string][self.col_uxid].nunique() >= 1 and
                      x[x[self.col_link_type] == self.row_xlink_string][self.col_uxid].nunique() == 1)[
            link_group_string].nunique()
        num_mono2 = df_new.groupby(link_group_string).filter(
            lambda x: x[x[self.col_link_type] == self.row_monolink_string][self.col_uxid].nunique() == 2 and
                      x[x[self.col_link_type] == self.row_xlink_string][self.col_uxid].nunique() == 1)[
            link_group_string].nunique()
        num_total = df_new[link_group_string].nunique()
        print(f"Link groups with 1 monolink: {num_mono1} ({num_mono1 / num_total:.0%})")
        print(f"Link groups with 2 monolinks: {num_mono2} ({num_mono2 / num_total:.0%})")
        df_new = filter_link_groups(df_new)
        return df_new

    def get_reaction_state(self, df):
        def _get_reaction_state(x):
            if "-155" in x:
                return "quenched"
            elif "-156" in x:
                return "hydrolized"
            else:
                return 'crosslinked'
        # only calculate if there are monolinks in the bag container
        if self.row_monolink_string in df[self.col_link_type].unique():
            df[self.col_reaction_state] = df[self.col_uid].transform(_get_reaction_state)
            df[self.col_uid] = df[self.col_uid].str.replace(
                f"-15(5|6).*", "")
        return df
