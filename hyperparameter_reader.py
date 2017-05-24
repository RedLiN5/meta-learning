
import pandas as pd
import numpy as np
import pickle

class HyperparameterReader(object):

    def __init__(self, id):
        self.id = id

    def _load(self):
        try:
            with open('config/hyperparameter', 'rb') as fp:
                itemdict = pickle.load(fp)
        except Exception as e:
            print(e, 'in reading')
            itemdict = dict()
        return itemdict

    def _save(self, itemdict):
        try:
            with open('config/hyperparameter', 'wb') as fp:
                pickle.dump(itemdict, fp)
        except Exception as e:
            print(e, 'in saving')


    def _read_hyperparameter(self, algorithm=None):
        try:
            class_name = algorithm.__class__.__name__
            hyperparameters = algorithm.get_params()
            algorithm_params = {class_name: hyperparameters}

        except Exception as e:
            print(e)

