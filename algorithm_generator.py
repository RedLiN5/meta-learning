from classifiers import *
from regressors import *
import pickle


def append(id, algorithm, hyperparameters):
    with open('config/hyperparameter', 'rb') as fp:
        itemdict = pickle.load(fp)

    algorithm_name = algorithm.__class__.__name__
    if isinstance(hyperparameters, dict):
        algorithm_params = {algorithm_name: hyperparameters}
        itemdict[id] = algorithm_params
    else:
        raise ValueError("'hyperparameters' must be a dictionary")

    with open('config/hyperparameter', 'wb') as fp:
        pickle.dump(itemdict, fp)


