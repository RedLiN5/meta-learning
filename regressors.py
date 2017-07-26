from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor


regressor_dict = {'Ridge':Ridge,
                  'Lasso':Lasso,
                  'KNeighborsRegressor':KNeighborsRegressor,
                  'GradientBoostingRegressor':GradientBoostingRegressor,
                  'AdaBoostRegressor':AdaBoostRegressor,
                  'RandomForestRegressor':RandomForestRegressor,
                  'DecisionTreeRegressor':DecisionTreeRegressor,
                  'XGBRegressor':XGBRegressor
                  }