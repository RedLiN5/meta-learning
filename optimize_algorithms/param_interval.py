
from .base import *
from sklearn.ensemble import RandomForestClassifier
from collections import OrderedDict

def rfc():
    param_names = ['n_estimators', 'max_depth', 'criterion']
    param_bounds = [[20, 1000], [1,10],['gini', 'entropy']]
    param_types = ['int', 'int', 'str']
    return param_names, param_bounds, param_types


def logistic():
    param_names = ['C', 'solver']
    param_bounds = [[1e-10, 10],['newton-cg', 'lbfgs','liblinear', 'sag']]
    param_types = ['float', 'str']
    return param_names, param_bounds, param_types

def svc():
    param_names = ['C', 'kernel']
    param_bounds = [[1e-10, 10], ['linear', 'poly', 'rbf', 'sigmoid']]
    param_types = ['float', 'str']
    return param_names, param_bounds, param_types

def knn():
    param_names = ['n_neighbors', 'weights', 'algorithm']
    param_bounds = [[2, 20], ['uniform', 'distance'], ['ball_tree', 'kd_tree', 'brute'] ]
    param_types = ['int', 'str', 'str']
    return param_names, param_bounds, param_types

def gaussiannb():
    param_names = []
    param_bounds = []
    param_types = []
    return param_names, param_bounds, param_types

def bernoullinb():
    param_names = []
    param_bounds = []
    param_types = []
    return param_names, param_bounds, param_types

def dt():
    param_names = ['max_depth', 'max_features', 'criterion']
    param_bounds = [[1,10], ['sqrt', 'log2', 'auto'], ['gini', 'entropy']]
    param_types = ['int', 'str', 'str']
    return param_names, param_bounds, param_types

def adaboost():
    param_names = ['n_estimators', 'learning_rate', 'algorithm']
    param_bounds = [[20, 1000], [1e-10, 10], ['SAMME', 'SAMME.R']]
    param_types = ['int', 'float', 'str']
    return param_names, param_bounds, param_types

def gb():
    param_names = ['learning_rate', 'n_estimators', 'max_depth', 'max_features', 'loss']
    param_bounds = [[1e-10, 10], [20, 1000], [1,10], ['sqrt', 'log2', 'auto'], ['deviance', 'exponential']]
    param_types = ['float', 'int', 'int', 'str', 'str']
    return param_names, param_bounds, param_types

def lda():
    param_names = ['solver']
    param_bounds = [['svd', 'lsqr', 'eigen']]
    param_types = ['str']
    return param_names, param_bounds, param_types

def qda():
    param_names = []
    param_bounds = []
    param_types = []
    return param_names, param_bounds, param_types

def xgb():
    param_names = ['max_depth', 'n_estimators', 'learning_rate', 'subsample']
    param_bounds = [[1,10], [20, 1000], [1e-10, 10], [0.5, 1]]
    param_types = ['int', 'int', 'float', 'float']
    return param_names, param_bounds, param_types


params_dict = {'Logistic': logistic(),
               'SVC': svc(),
               'KNeighborsClassifier': knn(),
               'GaussianNB': gaussiannb(),
               'BernoulliNB': bernoullinb(),
               'DecisionTreeClassifier': dt(),
               'RandomForestClassifier': rfc(),
               'AdaBoostClassifier': adaboost(),
               'GradientBoostingClassifier': gb(),
               'LinearDiscriminantAnalysis': lda(),
               'QuadraticDiscriminantAnalysis': qda(),
               'XGBClassifier': xgb()
               }





