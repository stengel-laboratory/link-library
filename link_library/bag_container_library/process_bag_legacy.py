import pandas as pd
import numpy as np

setting_filter = ""  # filter is set in __main__ if given

col_level = ""  # sets actual col_level for run in __main__ (i.e. uid or uxid)

col_uid = "uid"
col_uxid = "uxid"
col_exp = "exp_name"
col_exp_original = 'exp_name_original'
col_origin = 'origin'
col_link_type = 'link_type'
col_weight_type = 'weight_type'
col_bag_type = 'bag_container_type'
col_area_sum_total = 'ms1_area_sum'
col_area_sum_norm_total = col_area_sum_total + '_norm'
col_area_sum_light = 'ms1_area_sum_light'
col_area_sum_heavy = 'ms1_area_sum_heavy'
col_area_bio_repl = 'ms1_area_sum_bio_rep_'
col_var = 'ms1_area_variance'
col_std = 'ms1_area_std'
col_index = 'index'
col_log2ratio = 'log2ratio'
col_lh_log2ratio = 'light_heavy_log2ratio'
col_bio_rep = 'exp_bio_rep'
row_monolink_string = 'monolink'
row_xlink_string = 'xlink'
row_light_string = 'light'
row_heavy_string = 'heavy'
row_regular_string = 'regular'
row_details_string = 'details'

class BagContainer(object):

    def __init__(self, mode, level, filter=''):
        self.col_uid = "uid"
        self.col_uxid = "uxid"
        self.col_exp = "exp_name"
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
        self.row_monolink_string = 'monolink'
        self.row_xlink_string = 'xlink'
        self.row_light_string = 'light'
        self.row_heavy_string = 'heavy'
        self.row_regular_string = 'regular'
        self.row_details_string = 'details'
        if mode == self.row_regular_string:
            self.cont_type = self.row_regular_string
            self.uxid_string = 'a_uxID'
            self.uid_string = 'a_uID'
            self.exp_string = 'a_experimentname'
            self.vio_string = 'c_violations'
            self.sum_light_string = 'b_light_msum_area_sum_isotopes'
            self.sum_heavy_string = 'b_heavy_msum_area_sum_isotopes'

        else:
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
        if level == self.col_uid:
            self.level = self.uid_string
        elif level == self.col_uxid:
            self.level = self.uxid_string
        else:
            print("ERROR: Improper ms1 col_level entered: {0}".format(level))
            exit(1)
        if filter.lower() == row_monolink_string:
            self.filter = row_monolink_string
        elif filter.lower() == row_xlink_string:
            self.filter = row_xlink_string
        elif filter:
            print("ERROR: Improper link filter entered: {0}".format(filter))
            exit(1)
        self.type = 'BagContainer_{0}_{1}'.format(mode, level)

def fillna_with_normal_dist(df):
    # ref: http://www.coxdocs.org/doku.php?id=perseus:user:activities:matrixprocessing:imputation:replacemissingfromgaussian
    # ref: https://github.com/JurgenCox/perseus-plugins/blob/master/PerseusPluginLib/Impute/ReplaceMissingFromGaussian.cs
    shift = 1.8
    width = 0.3
    a = df.values
    m = np.isnan(a) # mask of NaNs
    mean_columns = df.mean()
    print(mean_columns)
    mu, sigma = np.mean(mean_columns), np.std(mean_columns) # mean and std of total matrix
    a[m] = np.random.normal(mu-shift*sigma, width*sigma, size=m.sum())
    return df, a[m]


def rename_columns(df, bag_cont):
    if bag_cont.cont_type == row_details_string:
        df = df.rename(index=str, columns={bag_cont.exp_string: col_exp_original,bag_cont.level: col_level,
                                           bag_cont.sum_string: col_area_sum_total,
                                           bag_cont.repl_bio_string: col_bio_rep,})
    return df


# function to get ms1 intensities grouped by experiment; works with bag_container.stats and .details
# returns new dataframe
def get_sum_ms1_intensities_df(df, bag_cont):
    # we split up the input by experiment and compute all ms1 intensities separately
    exp_list = list(set(df[bag_cont.exp_string]))
    # this list stores the dataframes by experiment
    df_exp_list = []

    # let's remove the weight from uid string; so we have the same uids for both container types
    if bag_cont.cont_type == row_details_string and col_level == col_uid:
        # using regex to match either :heavy or :light and all the following string (.*)
        df[bag_cont.uid_string] = df[bag_cont.uid_string].str.replace(":({0}|{1}).*"
                                                                      .format(row_light_string, row_heavy_string),"")
    # all ids (either uxid or uid) are put into one set and used for constructing the initial dataframe
    # this also allows for detection of ids missing from one of the two experiments
    id_list = list(set(df[bag_cont.level]))
    # iterating by experiment
    for exp in exp_list:
        # since we use the experiment to group the plotting, the exp_new name will also include the original file name
        exp_new = "{0} ({1}): {2}".format(exp, bag_cont.cont_type, df.name[:df.name.rfind('_')])
        # creating the results dataframe
        df_res = pd.DataFrame()
        kwargs = {col_level: id_list,
                  col_exp: pd.Series([exp_new for l in range(len(id_list))]),
                  col_exp_original: pd.Series([exp for l in range(len(id_list))]),
                  col_bag_type: pd.Series([bag_cont.cont_type for l in range(len(id_list))]),
                  col_origin: pd.Series([df.name for l in range(len(id_list))])}
        df_res = df_res.assign(**kwargs)

        # filtering the input dataframe by experiment name
        df_exp = df.loc[df[bag_cont.exp_string] == exp]
        # processing bag_container.stats
        if bag_cont.cont_type == row_regular_string:
            # first filtering violations found by xTract
            df_exp = df_exp.loc[df_exp[bag_cont.vio_string] == 0]
            # putting the link type (mono/xlink) into its own column
            df_exp[col_link_type] = df_exp[bag_cont.uid_string].str.split('::').str[2]
            # selecting the columns necessary to sum up ms1 intensities which are separated into heavy and light
            df_ms1_tot = df_exp[[bag_cont.level, bag_cont.sum_light_string, bag_cont.sum_heavy_string, col_link_type]]
            # renaming the columns before merging them into our results df
            df_ms1_tot = df_ms1_tot.rename(index=str, columns={bag_cont.level: col_level,
                                                               bag_cont.sum_light_string: col_area_sum_light,
                                                               bag_cont.sum_heavy_string: col_area_sum_heavy})
            # computing the total ms1 sum by computing the light+heavy sum along the x-axis (i.e. row-wise)
            df_ms1_tot[col_area_sum_total] = df_ms1_tot[[col_area_sum_light, col_area_sum_heavy]].sum(axis=1)
            # this will sum up the intensities for the same ids;
            # i.e. does nothing at uid col_level, sums up uids at uxid col_level
            df_ms1_tot = df_ms1_tot.groupby(
                [col_level, col_link_type], as_index=False)[col_area_sum_total, col_area_sum_light, col_area_sum_heavy].sum()
            # merging ms1 intensities with our results df
            # outer join: union of keys; inner: intersection
            df_res = pd.merge(df_res, df_ms1_tot, on=[col_level], how='inner')
        # processing bag_container.details
        else:
            # filtering these two means we get exactly the same results as from the regular bag container
            # removes violations (but no violations are calculated for monolinks)
            df_exp = df_exp.loc[df_exp[bag_cont.vio_string] == 0]
            df_exp = df_exp.loc[df_exp[bag_cont.valid_string] == 1]

            df_exp = df_exp.rename(index=str, columns={bag_cont.level: col_level})
            # create two separate columns for link type and weight (i.e. heavy or light)
            df_exp[col_link_type], df_exp[col_weight_type] = df_exp[bag_cont.type_string].str.split(':').str
            # summing up the total ms1 sum for the same id; again only really sums at uxid col_level
            df_ms1_tot = df_exp.groupby(
                [col_level, col_link_type])[bag_cont.sum_string].sum().reset_index(name=col_area_sum_total)
            df_res = pd.merge(df_res, df_ms1_tot, on=[col_level], how='inner')
            # the intensities for light and heavy ids have to calculated explicitly for details container
            # we group by weight and sum up the itensities
            df_ms1_lh = df_exp.groupby(
                [col_level, col_weight_type])[bag_cont.sum_string].sum().reset_index(name=col_area_sum_total)
            # we want the heavy and light intensities as separate columns and therefore pivot the dataframe here
            df_ms1_lh = pd.pivot_table(df_ms1_lh, values=col_area_sum_total, index=[col_level], columns=col_weight_type)
            df_ms1_lh = df_ms1_lh.reset_index()
            df_ms1_lh = df_ms1_lh.rename(
                index=str, columns={row_light_string: col_area_sum_light, row_heavy_string: col_area_sum_heavy})
            df_res = pd.merge(df_res, df_ms1_lh, on=[col_level], how='inner')
            df_ms1_bio_rep = df_exp.groupby(
                [col_level, bag_cont.repl_bio_string])[bag_cont.sum_string].sum().reset_index(name=col_area_sum_total)
            df_ms1_bio_rep = pd.pivot_table(df_ms1_bio_rep, values=col_area_sum_total, index=[col_level], columns=bag_cont.repl_bio_string)
            df_ms1_bio_rep = df_ms1_bio_rep.reset_index()
            # df_ms1_bio_rep = df_ms1_bio_rep.fillna(-1)
            rep_list = sorted(set(df_exp[bag_cont.repl_bio_string]))
            rep_name_dict = {x: col_area_bio_repl + str(x) for x in rep_list}
            df_ms1_bio_rep = df_ms1_bio_rep.rename(index=str, columns=rep_name_dict)
            df_res = pd.merge(df_res, df_ms1_bio_rep, on=[col_level], how='inner')
            df_res[col_var] = df_res[list(rep_name_dict.values())].var(axis=1)
            df_res[col_std] = df_res[list(rep_name_dict.values())].std(axis=1)
        # not sure if we should fill up na here with 0
        # if we do it would mean an id not detected in one experiment gets a 0 intensity
        # df_res = df_res.fillna(0)
        # optionally filter for link type; would be more efficient to do this earlier
        if setting_filter:
            df_res = df_res.loc[df_res[col_link_type] == setting_filter]
        # computes a normalized by maximum ms1 intensity; not used atm
        df_res[col_area_sum_norm_total] = df_res[col_area_sum_total] / df_res[col_area_sum_total].max()
        # computes the light/heavy log2ratio
        df_res[col_lh_log2ratio] = np.log2(df_res[col_area_sum_light] / df_res[col_area_sum_heavy])

        # df_res[col_area_sum_norm_total] = (df_res[col_area_sum_total] - df_res[col_area_sum_total].min()) / (df_res[col_area_sum_total].max() - df_res[col_area_sum_total].min())
        df_exp_list.append(df_res)
    df_final = pd.DataFrame()
    for dfl in df_exp_list:
        df_final = df_final.append(dfl)
    # computing the same violations as extract and removing them; right now only works on uid col_level
    df_final = get_violation_removed_df(df_final, bag_cont)

    # computing the log2ratio between the two experiments; hardcoded right now; at least reference should be a user setting
    df_log2 = pd.pivot_table(df_final, values=col_area_sum_total, index=[col_level], columns=col_exp).reset_index()
    df_log2 = df_log2.dropna()
    df_log2[col_log2ratio] = np.log2(df_log2.iloc[:,2]/df_log2.iloc[:,1])
    df_final = pd.merge(df_final, df_log2[[col_level,col_log2ratio]], on=[col_level], how='left')
    return df_final


# removes uids with violations; only checks for light/heavy log2ratio violations
# TODO: implement violation which requires all charge states of a peptide to be present in both experiments
# TODO: split up into violation assign (use separate columns) and filtering
# TODO: keep violation columns from xTract for comparison
def get_violation_removed_df(df, bag_cont):
    name = set(df[col_origin])
    print("Shape of {0} before filtering zero intensities: {1}.".format(name, df.shape))
    df = df[(df[col_area_sum_light] > 0) | (df[col_area_sum_heavy] > 0)]
    print("Shape of {0} before filtering lh log2ratio: {1}.".format(name, df.shape))
    df = df[(df[col_lh_log2ratio] < 1) & (df[col_lh_log2ratio] > -1)]
    print("Shape of {0} after filtering: {1}.".format(name, df.shape))
    # ms1 intensities have to be divided by tech_replicates*bio_replicates when coming from a details container
    # TODO: don't hardcode the number but get it from the input file
    if bag_cont.cont_type == row_details_string:
        df[[col_area_sum_total, col_area_sum_light, col_area_sum_heavy]] = \
            df[[col_area_sum_total, col_area_sum_light, col_area_sum_heavy]].apply(lambda x: x / 6)
    return df