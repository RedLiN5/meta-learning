
from .base import *
from sklearn.ensemble import RandomForestClassifier
from collections import OrderedDict

class param_interval():

    def __init__(self, algorithm, X):
        self.algorithm = algorithm
        self.nrow, self.ncol = X.shape


    def rfc(self):
        depth_max = int(np.log2(self.ncol+1))+1
        param_names = ['n_estimators', 'max_depth', 'criterion']
        param_bounds = [[20, 1000], [1, depth_max],['gini', 'entropy']]
        param_types = ['int', 'int', 'str']
        return param_names, param_bounds, param_types


    def logistic(self):
        param_names = ['C', 'solver']
        param_bounds = [[1e-10, 10],['newton-cg', 'lbfgs','liblinear', 'sag']]
        param_types = ['float', 'str']
        return param_names, param_bounds, param_types

    def svc(self):
        param_names = ['C', 'kernel']
        param_bounds = [[1e-10, 10], ['linear', 'poly', 'rbf', 'sigmoid']]
        param_types = ['float', 'str']
        return param_names, param_bounds, param_types

    def knn(self):
        param_names = ['n_neighbors', 'weights', 'algorithm']
        param_bounds = [[2, 20], ['uniform', 'distance'], ['ball_tree', 'kd_tree', 'brute'] ]
        param_types = ['int', 'str', 'str']
        return param_names, param_bounds, param_types

    def gaussiannb(self):
        param_names = []
        param_bounds = []
        param_types = []
        return param_names, param_bounds, param_types

    def bernoullinb(self):
        param_names = []
        param_bounds = []
        param_types = []
        return param_names, param_bounds, param_types

    def dt(self):
        depth_max = int(np.log2(self.ncol+1))+1
        param_names = ['max_depth', 'max_features', 'criterion']
        param_bounds = [[1, depth_max], ['sqrt', 'log2', 'auto'], ['gini', 'entropy']]
        param_types = ['int', 'str', 'str']
        return param_names, param_bounds, param_types

    def adaboost(self):
        param_names = ['n_estimators', 'learning_rate', 'algorithm']
        param_bounds = [[20, 1000], [1e-10, 10], ['SAMME', 'SAMME.R']]
        param_types = ['int', 'float', 'str']
        return param_names, param_bounds, param_types

    def gb(self):
        depth_max = int(np.log2(self.ncol+1))+1
        param_names = ['learning_rate', 'n_estimators', 'max_depth', 'max_features', 'loss']
        param_bounds = [[1e-10, 10], [20, 1000], [1, depth_max], ['sqrt', 'log2', 'auto'], ['deviance', 'exponential']]
        param_types = ['float', 'int', 'int', 'str', 'str']
        return param_names, param_bounds, param_types

    def lda(self):
        param_names = ['solver']
        param_bounds = [['svd', 'lsqr', 'eigen']]
        param_types = ['str']
        return param_names, param_bounds, param_types

    def qda(self):
        param_names = []
        param_bounds = []
        param_types = []
        return param_names, param_bounds, param_types

    def xgb(self):
        depth_max = int(np.log2(self.ncol+1))+1
        param_names = ['max_depth', 'n_estimators', 'learning_rate', 'subsample']
        param_bounds = [[1, depth_max], [20, 1000], [1e-10, 10], [0.5, 1]]
        param_types = ['int', 'int', 'float', 'float']
        return param_names, param_bounds, param_types

    def get(self):
        params_dict = {'Logistic': self.logistic(),
                       'SVC': self.svc(),
                       'KNeighborsClassifier': self.knn(),
                       'GaussianNB': self.gaussiannb(),
                       'BernoulliNB': self.bernoullinb(),
                       'DecisionTreeClassifier': self.dt(),
                       'RandomForestClassifier': self.rfc(),
                       'AdaBoostClassifier': self.adaboost(),
                       'GradientBoostingClassifier': self.gb(),
                       'LinearDiscriminantAnalysis': self.lda(),
                       'QuadraticDiscriminantAnalysis': self.qda(),
                       'XGBClassifier': self.xgb()
                       }
        return params_dict[self.algorithm]





