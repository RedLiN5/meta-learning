
import pandas as pd
import numpy as np
import pickle
import collections
from scipy import stats


class FeatureReader(object):

    def __init__(self, X, y, id, numeric_names,
                 category_names):
        """
        Args:
            X: pandas.DataFrame
                Predictors
            y: pandas.Series
                Response
            id: string
                Unique ID
            numeric_names: pandas.indexes.base.Index
                Column names of numerical columns
            category_names: pandas.indexes.base.Index
                Column names of categorical columns
        """
        self.X = X
        self.y = y
        self.id = id
        self.numeric_names = numeric_names
        self.category_names = category_names
        self.features_info = {}

    def _extract_info(self):
        """
        Extract basic feature information from numeric and categorical features
        Returns:

        """
        X_numeric = self.X[self.numeric_names]
        X_category = self.X[self.category_names]
        ncol_numeric = X_numeric.shape[1]
        ncol_category = self.X.shape[1] - ncol_numeric
        if ncol_category != len(self.category_names):
            raise ValueError("Length of 'X_category' and 'ncol_category' are not consistent.")

        # --*--*-- Numeric process --*--*--
        index_numeric = ['kurtosis', 'skewness', 'mean', 'std', 'min', 'max',
                         'median', 'first_quartile', 'third_quartile']
        colnames_numeric = ['var'+str(n) for n in np.arange(1, ncol_numeric+1)]

        kurtosis = X_numeric.apply(lambda x: stats.kurtosis(x,
                                                            nan_policy='omit')).values
        skewness = X_numeric.apply(lambda x: stats.skew(x,
                                                        nan_policy='omit')).values
        mean = X_numeric.apply(np.mean).values
        std = X_numeric.apply(np.std).values
        min = X_numeric.apply(np.min).values
        max = X_numeric.apply(np.max).values
        median = X_numeric.apply(np.median).values
        first_quartile = X_numeric.apply(lambda x: np.percentile(x, 25)).values
        third_quartile = X_numeric.apply(lambda x: np.percentile(x, 75)).values
        data_numeric = np.array([kurtosis, skewness, mean, std, min, max, median, first_quartile, third_quartile])

        info_numeric = pd.DataFrame(data=data_numeric,
                                    index=index_numeric,
                                    columns=colnames_numeric)

        # --*--*-- Category process --*--*--
        index_category = ['num_category', 'counter']
        colnames_category = ['var' + str(n) for n in np.arange(1, ncol_category + 1)]
        num_category = X_category.apply(lambda x: len(np.unique(x))).values
        counter = X_category.apply(collections.Counter).values
        data_category = np.array([num_category, counter])

        info_category = pd.DataFrame(data=data_category,
                                     index=index_category,
                                     columns=colnames_category)

        return info_numeric, info_category

    def _encapsule(self):
        info_numeric, info_category = self._extract_info()
        self.features_info['Info_Numeric'] = info_numeric
        self.features_info['Info_Category'] = info_category
        return self.features_info

    def _append(self, features_info):
        try:
            with open('config/metabase', 'rb') as fp:
                itemdict = pickle.load(fp)
                if self.id in itemdict.keys():
                    print(BaseException("WARNING: ID \'{0}\' has existed in database.".format(self.id)))
                else:
                    itemdict[self.id] = features_info
        except Exception as e:
            print(e, 'in reading metabase')
            itemdict = dict()
            itemdict[self.id] = features_info

        with open('config/metabase', 'wb') as fp:
            pickle.dump(itemdict, fp)

    def run(self):
        features_info = self._encapsule()
        self._append(features_info=features_info)
        return features_info



