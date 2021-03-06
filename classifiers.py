from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier

classifier_dict = {'Logistic':LogisticRegression,
                   'SVC': SVC,
                   'KNeighborsClassifier':KNeighborsClassifier,
                   'GaussianNB':GaussianNB,
                   'BernoulliNB':BernoulliNB,
                   'DecisionTreeClassifier':DecisionTreeClassifier,
                   'RandomForestClassifier':RandomForestClassifier,
                   'AdaBoostClassifier':AdaBoostClassifier,
                   'GradientBoostingClassifier':GradientBoostingClassifier,
                   'LinearDiscriminantAnalysis':LinearDiscriminantAnalysis,
                   'QuadraticDiscriminantAnalysis':QuadraticDiscriminantAnalysis,
                   'XGBClassifier':XGBClassifier
                   }