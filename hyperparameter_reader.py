
import pandas as pd
import numpy as np
import pickle
from regressors import *
from classifiers import *

class HyperparameterReader(object):

    def __init__(self):
        pass

    def _load(self):
        try:
            with open('config/hyperparameter', 'rb') as fp:
                itemdict = pickle.load(fp)
        except Exception as e:
            print(e, 'in reading hyperparameter')
            itemdict = dict()
        return itemdict

    def _save(self):
        itemdict = self._load()
        itemdict[self.id] = self.algorithm_params
        try:
            with open('config/hyperparameter', 'wb') as fp:
                pickle.dump(itemdict, fp)
        except Exception as e:
            print(e, 'in saving hyperparameter')


    def save_new_hyperparameter(self, id=None,
                            algorithm=None):
        self.id = id
        try:
            algorithm_name = algorithm.__class__.__name__
            hyperparameters = algorithm.get_params()
            self.algorithm_params = {algorithm_name: hyperparameters}
        except Exception as e:
            print(e, 'in reading hyperparameter')
        self._save()

    def read_return(self, id=None):
        itemdict = self._load()
        algorithm_params = itemdict[id]
        algorithm_name = list(algorithm_params.keys())[0]
        if algorithm_name in classifier_dict.keys():
            algorithm = classifier_dict[algorithm_name]
        if algorithm_name in regressor_dict.keys():
            algorithm = regressor_dict[algorithm_name]
        hyperparameters = algorithm_params[algorithm_name]
        algorithm_util = algorithm().set_params(**hyperparameters)
        return algorithm_util






classifier_dict = {'logistic':logistic,
                   'SVC': SVC,
                   'KNeighborsClassifier':KNeighborsClassifier,
                   'GaussianNB':GaussianNB,
                   'BernoulliNB':BernoulliNB,
                   'DecisionTreeClassifier':DecisionTreeClassifier,
                   'RandomForestClassifier':RandomForestClassifier,
                   'AdaBoostClassifier':AdaBoostClassifier,
                   'GradientBoostingClassifier':GradientBoostingClassifier,
                   'LinearDiscriminantAnalysis':LinearDiscriminantAnalysis,
                   'QuadraticDiscriminantAnalysis':QuadraticDiscriminantAnalysis
                   }


regressor_dict = {'Ridge':Ridge,
                  'Lasso':Lasso,
                  'KNeighborsRegressor':KNeighborsRegressor,
                  'GradientBoostingRegressor':GradientBoostingRegressor,
                  'AdaBoostRegressor':AdaBoostRegressor,
                  'RandomForestRegressor':RandomForestRegressor,
                  'DecisionTreeRegressor':DecisionTreeRegressor,
                  }
