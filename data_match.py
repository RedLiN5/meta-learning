from feature_reader import FeatureReader
import pandas as pd
import numpy as np
import pickle
from scipy import stats

def _select_candicate(series_row, candidates_list):
    i = 1
    for _ in range(len(series_row)):
        value = series_row.nlargest(i)[-1]
        name = series_row.index[series_row==value][0]
        if name in candidates_list:
            i += 1
        else:
            break
    return name, value

class DataMatch(object):

    def __init__(self, ):
        fr = FeatureReader(X=None, y=None,
                           id='test0',
                           numeric_names=None,
                           category_names=None,
                           industry=None,
                           business_func=None)
        self.features_info = fr.run()

    def _load_metabase(self):
        with open('metabase', 'rb') as fp:
            all_features = pickle.load(fp)

        return all_features

    def _scoring_numeric(self, features_dict):
        features_input = self.features_info['Info_Numeric']
        features_base = features_dict['Info_Numeric']
        input_colnames = features_input.columns
        base_colnames = features_base.columns
        score_table = pd.DataFrame(columns=base_colnames,
                                   index=input_colnames)
        for i in input_colnames:
            scores = features_base.apply(lambda x: stats.ttest_ind(features_input[i],
                                                                   x)[1])
            score_table.ix[i, :] = scores.values

        score_table = score_table.apply(pd.to_numeric,
                                        errors = 'ignore')
        candidates = []
        candidate_scores =[]
        if len(input_colnames) > len(base_colnames):
            pass
        else:
            for i in range(len(input_colnames)):
                if len(candidates) < len(base_colnames):
                    row = score_table.loc[i]
                    name, value = _select_candicate(series_row=row,
                                                    candidates_list=candidates)
                    candidates.append(name)
                    candidate_scores.append(value)

        return candidates, candidate_scores



