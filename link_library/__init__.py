import pandas as pd
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
