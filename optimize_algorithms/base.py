
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn import metrics
import concurrent.futures
from functools import partial

#TODO 写一个 recorder 类, 将每个 model里前20的param_name, param, perfrom_score都记录下来. 最终可以用来排序, 做ensemble
class abstract_optimizer():

    def __init__(self, model, X, y, param_names, param_bounds, param_types, metric, random_state):
        if not len(param_types) == len(param_bounds) == len(param_names):
            raise("'param_types', 'param_bounds' and 'param_names' must have same length")
        else:
            self.param_types = param_types
            self.param_bounds = param_bounds
            self.param_names = param_names

        self.model = model
        self.seed = random_state
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y,
                                                                                test_size=0.2,
                                                                                random_state=self.seed)
        self.X, self.y = X, y
        self.metric = metric


    def get_init_param(self, param_type, upper_bound, lower_bound, n):
        if upper_bound > self.param_bound[1]:
            upper_bound = self.param_bound[1]
        if lower_bound < self.param_bound[0]:
            lower_bound = self.param_bound[0]
        np.random.seed(self.seed)
        if param_type == 'float':
            init_param = np.random.uniform(lower_bound, upper_bound, n)
        elif param_type == 'int':
            init_param = np.random.randint(lower_bound, upper_bound, n)
        else:
            init_param = None
        init_param = np.unique(init_param)
        return init_param

    def _performance_score(self, fix_params, param_name, p):
        fix_params[param_name] = p
        self.model.set_params(**fix_params)
        self.model.set_params(**{'random_state': self.seed})
        self.model.fit(self.X_train, self.y_train)
        if self.metric == 'accuracy':
            score = self.model.score(self.X_test, self.y_test)
        elif self.metric == 'auc':
            pred_prob = self.model.predict_proba(self.X_test)[:, 1]
            fpr, tpr, thresholds = metrics.roc_curve(self.y_test, pred_prob)
            score = metrics.auc(fpr, tpr)
        else:
            raise ValueError("Value 'metric' is not specified")
        return score

    def get_performance(self, param_name, param_list, fix_params=None):
        if fix_params:
            pass
        else:
            fix_params = {}
        perform_scores = []
        f = partial(self._performance_score, fix_params, param_name)
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for param_value, score in zip(param_list, executor.map(f, param_list)):
                perform_scores.append(round(score, 4))
        return param_list, perform_scores

    def acquisition(self, mean, std, best):
        z = (mean - best) / std
        f = (mean - best) * norm.cdf(z) + std * norm.pdf(z)
        f[std == 0.0] = 0.0
        return f

    def get_param(self, x, y):
        g = GaussianProcessRegressor()
        g.set_params(**{'random_state': self.seed})
        x = np.array(x).reshape(-1,1)
        g.fit(x, y)
        mean, std = g.predict(x, return_std=True)
        ei = self.acquisition(mean=mean, std=std, best=0)
        candidate = x[np.argmax(ei)][0]
        best = np.max(ei)
        return candidate, best

    def get_str_param(self, param_name, param_list, fix_params=None):
        perform_scores = []
        if fix_params:
            pass
        else:
            fix_params = {}
        for param in param_list:
            fix_params[param_name] = param
            self.model.set_params(**fix_params)
            self.model.set_params(**{'random_state': self.seed})
            self.model.fit(self.X_train, self.y_train)
            if self.metric == 'accuracy':
                score = self.model.score(self.X_test, self.y_test)
            elif self.metric == 'auc':
                pred_prob = self.model.predict_proba(self.X_test)[:,1]
                fpr, tpr, thresholds = metrics.roc_curve(self.y_test, pred_prob)
                score = metrics.auc(fpr, tpr)
            else:
                raise ValueError("Value 'metric' is not specified")
            perform_scores.append(round(score, 4))
        opt_param = param_list[np.argmax(perform_scores)]
        return opt_param


    def opt_run(self, max_iter):
        params = {}
        if len(self.param_types) > 0:
            for i in range(len(self.param_types)):
                param_type = self.param_types[i]
                if param_type != 'str':
                    param_name = self.param_names[i]
                    param_bound = self.param_bounds[i]
                    self.param_bound = self.param_bounds[i]
                    best_param, best_ei = self.get_opt_param(param_type=param_type,
                                                             param_name=param_name,
                                                             param_bound=param_bound,
                                                             max_iter=max_iter)
                    params[param_name] = best_param
                elif param_type == 'str':
                    param_name = self.param_names[i]
                    param_list = self.param_bounds[i]
                    opt_param = self.get_str_param(param_name=param_name,
                                                   param_list=param_list,
                                                   fix_params=params)
                    params[param_name] = opt_param
            return params
        elif len(self.param_types) == 0:
            return params
        else:
            raise ValueError("'param_types', 'param_names' or 'param_bounds' is invalid")



    def get_opt_param(self, param_type, param_name, param_bound, max_iter):
        if param_type == 'str':
            pass
        else:
            upper_bound = param_bound[1]
            lower_bound = param_bound[0]
            print('Initial parameters for {0}.'.format(param_name))
            init_param = self.get_init_param(param_type=param_type,
                                             upper_bound=upper_bound,
                                             lower_bound=lower_bound,
                                             n=100)
            x, y = self.get_performance(param_name=param_name,
                                        param_list=init_param)
            candidate, best = self.get_param(x, y)
            if len(x) < 30:
                best_param = candidate
                best_ei = best
            else:
                count = 0
                converge = Converge()
                while count <= max_iter:
                    new_upper = candidate * 1.2
                    new_lower = candidate * 0.8
                    try:
                        new_param = self.get_init_param(param_type=param_type,
                                                        lower_bound=new_lower,
                                                        upper_bound=new_upper,
                                                        n=30)
                        x, y = self.get_performance(param_name=param_name,
                                                    param_list=new_param)
                        candidate, best = self.get_param(x, y)
                        converge.add(param=candidate,
                                     ei=best)
                        if converge.check():
                            print('Converge!')
                            break
                    except Exception as e:
                        print('Error:', e)
                        break
                best_param = candidate
                best_ei = best
            return best_param, best_ei


class Converge():

    def __init__(self):
        self.param = []
        self.ei = []

    def add(self, param, ei):
        self.param.append(param)
        self.ei.append(ei)

    def check(self):
        if len(self.ei) >= 5:
            times = 0
            for i in [4,3,2,1]:
                minuend = self.ei[-(i+1)]
                subtrahend = self.ei[-i]
                rate = abs(subtrahend-minuend)/minuend
                if rate <= 0.02:
                    times += 1
            if times >= 3:
                return True
            else:
                return False
        else:
            return False




















