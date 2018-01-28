import numpy as np
from sklearn import preprocessing
from sklearn.externals import joblib
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from matplotlib import rcParams
rcParams['font.sans-serif'] = ['SimHei']
rcParams['font.family'] = 'sans-serif'

sexdic = {'男': 0, '女': 1, '??': 0}
delcols = ['id', '体检日期', '乙肝表面抗原', '乙肝表面抗体', '乙肝e抗原',
           '乙肝e抗体', '乙肝核心抗体']
gaomidu_dic = {'男': 1, '女': 1.3, '??': 1.3}


def gaomidu(x):
    return gaomidu_dic[x]

def check_nan(df):
    print(df.index[np.where(np.isnan(df))[0]])
    print(df.columns[np.where(np.isnan(df))[1]])


def sexencode(df):
    df['性别'] = df['性别'].map(sexdic)
    return df


def show(df):
    print(df.head())


def drop_fill(df):
    df.drop(delcols, 1, inplace=True)
    # df.replace(0, np.nan, inplace=True)
    df.fillna(df.mean().round(decimals=3), inplace=True)
    df['甘油三酯超标值!!!'] = df['甘油三酯'] / 1.7
    df['高密度脂蛋白胆固醇超标值!!!'] = df['高密度脂蛋白胆固醇'] / df['性别'].apply(gaomidu)
    df['低高胆固醇ratio!!!'] = df['低密度脂蛋白胆固醇'] / df['高密度脂蛋白胆固醇']
    df['低密度脂蛋白胆固醇超标值!!!'] = df['低密度脂蛋白胆固醇'] / 4.14
    # df.fillna(0, inplace=True)  # 先用均值填充NaN,如果还有NaN说明该列全为零，用0填充，让后面删掉
    return df


def calculate_mse(_x, _y):
    return np.linalg.norm((_x - _y)) / len(_y)


def scale_features(x, scaler_name):
    scaler = preprocessing.StandardScaler().fit(x)
    scaled_x = scaler.transform(x)
    joblib.dump(scaler, ".\\model\\{}_scaler.save".format(scaler_name))
    return scaled_x

def scale_y(y):
    y_scaler = preprocessing.StandardScaler().fit()
    scaled_y = y_scaler.transform(y)
    joblib.dump(y_scaler, 'model/y_scaler.save')
    return np.array(scaled_y)


def scale_load(x, scaler_name):
    scaler = joblib.load(".\\model\\{}_scaler.save".format(scaler_name))
    scaled_x = scaler.transform(x)
    return scaled_x


def inverse_scale(x, scaler_name):
    scaler = joblib.load(".\\model\\{}_scaler.save".format(scaler_name))
    original_x = scaler.inverse_transform(x)
    return original_x

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

