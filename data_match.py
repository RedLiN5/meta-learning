from feature_reader import FeatureReader
import pandas as pd
import numpy as np
import pickle
import collections
from scipy import stats

def _select_numeric_candicate_short(series_row, candidates_list):
    i = 1
    for _ in range(len(series_row)):
        value = series_row.nlargest(i)[-1]
        name = series_row.index[series_row==value][0]
        if name in candidates_list:
            i += 1
        else:
            break
    return name, value

def _select_numeric_candicate_long(score_df):
    max_arg = score_df.apply(lambda x: x.argmax())
    max_arg.dropna(inplace=True)
    count = collections.Counter(max_arg).items()
    dup = [_item for _item, _count in count if _count > 1]
    uni = [_item for _item, _count in count if _count == 1]
    if len(np.unique(max_arg)) == len(max_arg):
        col = max_arg.index.tolist()
    else:
        col = []
        for value in np.unique(max_arg):
            pos = max_arg == value
            col.append(max_arg.index[pos][0])
    max_value = score_df.max().loc[col].tolist()
    for _row in dup:
        try:
            col.append(score_df.loc[_row].argmax())
            max_value.append(score_df.loc[_row].max())
        except Exception as e:
            print(e)
        uni.append(_row)
    return uni, col, max_value


class DataMatch(object):

    def __init__(self, X=None, y=None,id='test0',numeric_names=None,
                 category_names=None, industry=None, business_func=None):
        fr = FeatureReader(X=X, y=y, id=id,
                           numeric_names=numeric_names,
                           category_names=category_names,
                           industry=industry,
                           business_func=business_func)
        self.features_info = fr.run()

    def _load_metabase(self):
        with open('metabase', 'rb') as fp:
            all_features = pickle.load(fp)

        return all_features

    def _scoring_numeric(self, features_base):
        """
        Return names and scores of input numerical variables who matches local variables best.
        Returns:
            candidates: list
                Names of best matchers
            candidate_scores: list
                Scores of best matchers
        """
        features_input = self.features_info['Info_Numeric']
        features_base = features_base['Info_Numeric']
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

        if len(input_colnames) > len(base_colnames):
            _score_table = score_table.copy()
            for _ in range(len(input_colnames)):
                candidates, colnames, candidate_scores =\
                    _select_numeric_candicate_long(score_df=_score_table)
                _score_table= score_table.drop(colnames, axis=1).drop(candidates, axis=0)
                if _score_table.shape[0] == 0:
                    break
        else:
            candidates = []
            candidate_scores = []
            for i in range(len(input_colnames)):
                if len(candidates) < len(base_colnames):
                    row = score_table.ix[i,:]
                    name, value = _select_numeric_candicate_short(series_row=row,
                                                                candidates_list=candidates)
                    candidates.append(name)
                    candidate_scores.append(value)

        return candidates, candidate_scores

    def _scoring_category(self, features_base):
        """
        Return names and scores of input categorical variables who matches local variables best.
        Returns:
            candidates: list
                Names of best matchers
            candidate_scores: list
                Scores of best matchers
        """
        features_input = self.features_info['Info_Category']
        features_base = features_base['Info_Category']
        input_colnames = features_input.columns
        base_colnames = features_base.columns
        score_table = pd.DataFrame(columns=base_colnames,
                                   index=input_colnames)
        # TODO chi_squared test on percentage data

    def _calculate_scores(self):
        all_features = self._load_metabase()
        ids = []
        scores = []
        for key in all_features.keys():
            feature_dict = all_features[key]
            _, candidate_scores = self._scoring_numeric(features_base=feature_dict)
            ids.append(key)
            scores.append(np.mean(candidate_scores))
        return ids, scores

    def run(self, best_match=5):
        ids, scores = self._calculate_scores()
        sorted_scores = sorted(((v, k) for (k, v) in enumerate(scores)),
                               reverse=True)
        index = list(map(lambda x: x[1], sorted_scores[:best_match]))
        ids_best = np.array(ids)[index]
        print('End:', ids_best)

