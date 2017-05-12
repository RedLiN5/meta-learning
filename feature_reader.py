
import pandas as pd
import numpy as np
from scipy import stats


class FeatureReader(object):

    def __init__(self, X, y, numeric_names, category_names, industry, business_func):
        """

        Args:
            X: pandas.DataFrame
                Predictors
            y: pandas.Series
                Response
            numeric_names: pandas.indexes.base.Index
                Column names of numerical columns
            category_names: pandas.indexes.base.Index
                Column names of categorical columns
            industry: string
                Industry of this data set
            business_func: string
                Business function of this project
        """
        self.X = X
        self.y = y
        self.numeric_names = numeric_names
        self.category_names = category_names
        industry = get_industry(industry=industry)
        business_func = get_business_func(business_func)
        self.features_info = dict(Industry=industry,
                                  Business_function=business_func)

    def _extract_info(self):
        """
        Extract basic feature information from numeric and categorical features
        Returns:

        """
        X_numeric = self.X[self.numeric_names]
        X_category = self.X[self.category_names]
        ncol_numeric = X_numeric.shape[1]
        index_numeric = ['kurtosis', 'skewness', 'mean', 'std', 'min', 'max',
                         'median', 'first_quartile', 'third_quartile']
        colnames = ['var'+str(n) for n in np.arange(1, ncol_numeric+1)]
        kurtosis = X_numeric.apply(stats.kurtosis).values
        skewness = X_numeric.apply(stats.skew).values
        mean = X_numeric.apply(np.mean).values
        std = X_numeric.apply(np.std).values
        min = X_numeric.apply(np.min).values
        max = X_numeric.apply(np.max).values
        median = X_numeric.apply(np.median).values
        first_quartile = X_numeric.apply(lambda x: np.percentile(x, 25)).values
        third_quartile = X_numeric.apply(lambda x: np.percentile(x, 75)).values
        data = np.array([kurtosis, skewness, mean, std, min, max, median, first_quartile, third_quartile])
        info_numeric = pd.DataFrame(data=data,
                                    index=index_numeric,
                                    columns=colnames)








industries_dict = dict(banks='Banks', ecommerce='E-commerce', education='Education',
                       entertainment='Entertainment', financial_service='Financial Services',
                       retailer='General Retailers', health_care='Health Care', internet='Internet',
                       life_insurance='Life Insurance', public_services='Public Services',
                       software_computer='Software and Computer Services', telecom='Telecommunication')

business_func_dict = dict(academics='Academics', customer_support='Customer Support',
                          disease_diagnosis='Disease Diagnosis', disease_treatment='Disease Treatment',
                          finance='Finance', human_resource='Human Resource', it='IT', investment='Investment',
                          marketing='Marketing', quality_assurance='Quality Assurance', sales='Sales',
                          risk_management='Risk Management')

def get_industry(industry):
    if isinstance(industry, str):
        try:
            industry_valid = industries_dict[industry.lower()]
            return industry_valid
        except KeyError:
            industries = industries_dict.keys()
            raise ValueError('{0} is not a valid industry value.'
                             'Valid options are {1}'.format(industry, sorted(industries)))
    else:
        raise ValueError('{0} is not a string'.format(industry))

def get_business_func(buz_func):
    if isinstance(buz_func, str):
        try:
            business_func = business_func_dict[buz_func.lower()]
            return business_func
        except KeyError:
            buz_funcs = business_func_dict.keys()
            raise ValueError('{0} is not a valid industry value.'
                             'Valid options are {1}'.format(buz_func, sorted(buz_funcs)))
    else:
        raise ValueError('{0} is not a string'.format(buz_func))