import pandas as pd
import time
from functools import reduce



class xQuestDB(object):
    def __init__(self):
        self.uxid_string = 'uxID'
        self.pos1_string = 'AbsPos1'
        self.pos2_string = 'AbsPos2'
        self.prot1_string = 'Protein1'
        self.prot2_string = 'Protein2'
        self.score_string = 'ld-Score'
        self.fdr_string = 'FDR'
        self.type_string = 'Type'
        # self.xltype_string = 'XLType'
        self.type_xlink_string = 'xlink'
        self.type_mono_string = 'monolink'
        self.spectrum_string = 'Spectrum'

        self.type = 'xQuestDB'


class xTractDB(object):
    def __init__(self):
        self.uxid_string = 'uID'
        self.log2_string = 'log2ratio'
        self.fdr_string = 'FDR'
        self.pval_string = 'pvalue'
        self.bonf_string = 'Bonf'
        self.type_string = 'type'
        self.type_xlink_string = 'xlink'
        self.type_mono_string = 'monolink'
        self.signi_string = 'significance'
        self.exp_string = 'experiment'
        self.exp_ref_string = 'referenceexperiment'
        self.type = 'xTractDB'
        self.link_group_string = 'link_group'
        self.dist_string = "dist_delta"
        self.pos_string = "Positions"
        self.prot_string = "Proteins"

def get_count_df(df, vals_list, merge_vals_list, sort_key, count_string='count'):
    df_new = pd.DataFrame(df)
    df_list = []
    for vals in vals_list:
        df_count = df_new.groupby(
                    vals).size().reset_index(name=count_string)
        df_count = df_count.rename(index=str, columns={k:merge_vals_list[i] for i,k in enumerate(vals)})
        df_list.append(df_count)
    # using reduce to merge any number of dataframes. see: https://stackoverflow.com/questions/38089010/merge-a-list-of-pandas-dataframes
    df_merge = reduce(lambda x, y: pd.merge(x, y, on = merge_vals_list), df_list).set_index(merge_vals_list).sum(axis=1).reset_index(name=count_string)
    df_merge = df_merge.sort_values([sort_key, count_string],ascending=False).reset_index(drop=True)
    return df_merge


def input_log2_ref(exp_list):
    print("Please select your reference experiment for log2ratio and p-values calculation")
    print("{0}".format({no:exp for no,exp in enumerate(exp_list)}))
    exp_ref = int(input("Enter a number (Default: 0): ") or "0")
    if exp_ref < 0 or exp_ref > len(exp_list)-1:
        print("ERROR: Incorrect experiment selected: {0}".format(exp_ref))
        exit(1)
    return exp_list[exp_ref]


def get_prot_name_and_link_pos(entry_list):
    # temp df; splitting uxid into positions and protein names
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


def timeit(f):
    def wrap(*args):
        time1 = time.time()
        ret = f(*args)
        time2 = time.time()
        print('{:s} function took {:.3f} ms'.format(f.__name__, (time2-time1)*1000.0))

        return ret
    return wrap

def get_xtract_df(bag_cont, incl_tech=False):
    xt_db = xTractDB()
    if incl_tech:
        sum_list = [bag_cont.col_exp, bag_cont.col_bio_rep, bag_cont.col_tech_rep, bag_cont.col_weight_type]
    else:
        sum_list = [bag_cont.col_exp, bag_cont.col_bio_rep, bag_cont.col_weight_type]
    sum_list_log2 = [bag_cont.col_exp, bag_cont.col_bio_rep, bag_cont.col_tech_rep, bag_cont.col_weight_type]
    mean_list = [bag_cont.col_exp]
    exp_ref = input_log2_ref(bag_cont.exp_list)
    df_pval = bag_cont.get_two_sided_ttest(sum_list, sum_list, ref=exp_ref)
    df_log2 = bag_cont.getlog2ratio(sum_list_log2, mean_list, ref=exp_ref)
    # n-way merge
    df = reduce(lambda left, right: pd.merge(left, right, on=[bag_cont.col_level, bag_cont.col_exp]),
                [df_log2, df_pval])
    df = df.sort_values([bag_cont.col_exp, bag_cont.col_level])
    df = df.rename(index=str, columns={bag_cont.col_level: xt_db.uxid_string, bag_cont.col_exp: xt_db.exp_string,
                                       bag_cont.col_log2ratio_ref: xt_db.exp_ref_string,
                                       bag_cont.col_log2ratio: xt_db.log2_string,
                                       bag_cont.col_pval: xt_db.pval_string,
                                       bag_cont.col_fdr: xt_db.fdr_string})
    return df