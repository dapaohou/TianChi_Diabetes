import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.externals import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from matplotlib import rcParams
rcParams['font.sans-serif'] = ['SimHei']
rcParams['font.family'] = 'sans-serif'

delcols = ['SNP54', 'SNP55', 'ACEID']


def checknan(df):
    colnan = df.count()
    rownan = df.count(axis=1) / df.shape[1]
    print("cols nan:", colnan, "rows nan", rownan)

# def check_nan(df):
#     print(df.index[np.where(np.isnan(df))[0]])
#     print(df.columns[np.where(np.isnan(df))[1]])


def encode(df, encode_cols):
    ohe = preprocessing.OneHotEncoder(categorical_features=encode_cols)
    ohe.fit(df.values)
    df = ohe.transform(df.values)
    # df = pd.get_dummies(df)
    return df


def listEncodeCols(df):
    cols = df.columns
    encode_cols = []
    for col in cols:
        if 'SNP' in col and col not in delcols or col in ['BMI分类', 'DM家族史']:
            encode_cols.append(col)
    return encode_cols


def drop_fill(df):
    df.drop(delcols, 1, inplace=True)
    # df.replace(0, np.nan, inplace=True)
    # df.fillna(-999, inplace=True)
    df.fillna(df.median(), inplace=True)
    # 根据指标的正常值设置偏差量作为特征
    return df


def calculate_mse(_x, _y):
    return np.linalg.norm((_x - _y)) / len(_y)


def scale_features(x, scaler_name):
    scaler = preprocessing.StandardScaler().fit(x)
    scaled_x = scaler.transform(x)
    joblib.dump(scaler, ".\\model\\{}_scaler.save".format(scaler_name))
    return scaled_x


def scale_load(x, scaler_name):
    scaler = joblib.load(".\\model\\{}_scaler.save".format(scaler_name))
    scaled_x = scaler.transform(x)
    return scaled_x


def select_features(X_train, y_train, df, size=1):
    del_cols = []
    threadshold = int(size*X_train.shape[1])
    feat_labels = df.columns[1:]  # 特征列名
    forest = RandomForestRegressor()
    forest.fit(X_train, y_train)
    importances = forest.feature_importances_  # feature_importances_特征列重要性占比
    indices = np.argsort(importances)[::-1]  # 对参数从小到大排序的索引序号取逆,即最重要特征索引——>最不重要特征索引
    for f in range(X_train.shape[1]):
        print("%2d) %-*s %f" % (f + 1, 30, feat_labels[indices[f]], importances[indices[f]]))
        # -*s表示左对齐字段feat_labels[indices[f]]宽为30
        if f > threadshold:
            del_cols.append(feat_labels[indices[f]])
    plt.title('Feature Importances')
    plt.bar(range(X_train.shape[1]), importances[indices], color='lightblue', align='center')
    plt.xticks(range(X_train.shape[1]), feat_labels[indices], rotation=90)
    plt.xlim([-1, X_train.shape[1]])
    plt.tight_layout()
    # plt.savefig('./random_forest.png', dpi=300)
    plt.show()
    return del_cols

