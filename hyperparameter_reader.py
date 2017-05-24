
import pandas as pd
import numpy as np
import pickle

class HyperparameterReader(object):

    def __init__(self):
        pass

    def _load(self):
        try:
            with open('config/hyperparameter', 'rb') as fp:
                itemdict = pickle.load(fp)
        except Exception as e:
            print(e, 'in reading')
            itemdict = dict()
        return itemdict

    def save(self):
        itemdict = self._load()
        itemdict[self.id] = self.algorithm_params
        try:
            with open('config/hyperparameter', 'wb') as fp:
                pickle.dump(itemdict, fp)
        except Exception as e:
            print(e, 'in saving')


    def read_hyperparameter(self, id=None,
                            algorithm=None):
        self.id = id
        try:
            class_name = algorithm.__class__.__name__
            hyperparameters = algorithm.get_params()
            self.algorithm_params = {class_name: hyperparameters}
        except Exception as e:
            print(e)



