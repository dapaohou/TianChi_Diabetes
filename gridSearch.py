import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import Lasso, ElasticNet  # 批量导入要实现的回归算法
from sklearn.svm import SVR  # SVM中的回归算法
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor  # 集成算法
from sklearn.model_selection import cross_val_score  # 交叉检验
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score  # 批量导入指标算法
from sklearn.model_selection import GridSearchCV
from catboost import CatBoostClassifier
from tqdm import *
import xgboost as xgb
import lightgbm as lgb

from cleanData import *

import warnings
warnings.filterwarnings("ignore")

from matplotlib import rcParams
rcParams['font.sans-serif'] = ['SimHei']
rcParams['font.family'] = 'sans-serif'


df = pd.read_csv(".\\data\\level2\\train_feat_dummies.csv", encoding='gb18030')
y = df['label']
X = np.array(df.drop(['label'], 1))
X = scale_load(X, 'x')

out = pd.DataFrame()

for i in tqdm(range(5)):
    estimator = CatBoostClassifier(
        l2_leaf_reg=3,
        loss_function='Logloss',
        random_seed=i)

    param_grid = {'learning_rate': [0.001, 0.05, 0.1, 0.5, 0.8, 1], 'depth': range(3, 7, 1),
                  'iterations': [1000, 3000]}

    gbm = GridSearchCV(estimator, param_grid, scoring='f1', cv=5)

    gbm.fit(X, y)

    print('Best parameters found by grid search are:', gbm.best_params_)
    print('Best score found by grid search are:', gbm.best_score_)
    print('Grid scores found by grid search are:', gbm.cv_results_)
