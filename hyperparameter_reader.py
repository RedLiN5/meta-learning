
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
        """
        Save an algorithm and its hyperparameters
        Args:
            id: project ID
            algorithm: object

        Returns:

        """
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
        try:
            algorithm_params = itemdict[id]
            algorithm_name = list(algorithm_params.keys())[0]
            if algorithm_name in classifier_dict.keys():
                algorithm = classifier_dict[algorithm_name]
            elif algorithm_name in regressor_dict.keys():
                algorithm = regressor_dict[algorithm_name]
            else:
                raise ValueError("Algorithm does not exist in metabase")
            hyperparameters = algorithm_params[algorithm_name]
            algorithm_util = algorithm().set_params(**hyperparameters)
            return algorithm_util
        except Exception as e:
            print(ValueError('ID {0} does not have a corresponding algorithm'))









