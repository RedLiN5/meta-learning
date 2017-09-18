
import numpy as np
import pandas as pd

from classifiers import *
from regressors import *
from optimize_algorithms.base import *
from optimize_algorithms.param_interval import *



class Optimizer():

    def __init__(self, algorithm, X, y, max_iter, metric='accuracy'):
        '''

        Args:
            algorithm: str, optional
                the name of an algorithm
                * 'Logistic':LogisticRegression,
                * 'SVC': SVC,
                * 'KNeighborsClassifier':KNeighborsClassifier,
                * 'GaussianNB':GaussianNB,
                * 'BernoulliNB':BernoulliNB,
                * 'DecisionTreeClassifier':DecisionTreeClassifier,
                * 'RandomForestClassifier':RandomForestClassifier,
                * 'AdaBoostClassifier':AdaBoostClassifier,
                * 'GradientBoostingClassifier':GradientBoostingClassifier,
                * 'LinearDiscriminantAnalysis':LinearDiscriminantAnalysis,
                * 'QuadraticDiscriminantAnalysis':QuadraticDiscriminantAnalysis,
                * 'XGBClassifier':XGBClassifier
            X: numpy.array or pandas.DataFrame
                predictors
            y: numpy.array or pandas.Series
                target
            max_iter: int
                maximum iteration times
            metric: str, optional (default='accuracy')
                metric names
                * 'accuracy': Accuracy
                * 'auc': Area under the ROC-curve
        '''
        if algorithm not in classifier_dict.keys() and algorithm not in regressor_dict.keys():
            raise("Please input valid algorithm names in {0}".format(list(classifier_dict.keys())+
                                                                     list(regressor_dict.keys())))
        else:
            self.X = X
            self.y = y
            self.algorithm = algorithm
            self.max_iter = max_iter
            self.metric = metric

    def run(self):
        model = classifier_dict[self.algorithm]()
        param_names, param_bounds, param_types = params_dict[self.algorithm]
        abs_opt = abstract_optimizer(model=model,
                                     X=self.X, y=self.y,
                                     param_types=param_types,
                                     param_bounds=param_bounds,
                                     param_names=param_names,
                                     metric=self.metric)
        opt_params = abs_opt.opt_run(max_iter=self.max_iter)
        return opt_params

