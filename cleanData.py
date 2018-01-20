import numpy as np
from sklearn import preprocessing
from sklearn.externals import joblib
import matplotlib.pyplot as plt

# from sklearn.feature_selection import SelectKBest, chi2
# from sklearn.svm import LinearSVC
# from sklearn.feature_selection import VarianceThreshold
# from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestRegressor

sexdic = {'男': 0, '女': 1, '??': 2}
delcols = ['id', '体检日期', '乙肝表面抗原', '乙肝表面抗体', '乙肝e抗原', '乙肝e抗体', '乙肝核心抗体']


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
    # df.fillna(0, inplace=True)  # 先用均值填充NaN,如果还有NaN说明该列全为零，用0填充，让后面删掉
    return df


def calculate_mse(_x, _y):
    return np.linalg.norm((_x - _y)) / len(_y)


def scale_features(x):

    ##单变量特征选择-卡方检验，选择相关性最高的前100个特征  
    # X_chi2 = SelectKBest(chi2, k=2000).fit_transform(label_X_scaler, label_y)
    # print("训练集有 %d 行 %d 列" % (X_chi2.shape[0],X_chi2.shape[1]))
    # df_X_chi2=pd.DataFrame(X_chi2)
    # feature_names = df_X_chi2.columns.tolist()#显示列名
    # print('单变量选择的特征：\n',feature_names)

    ##基于L1的特征选择  
    ##lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(label_X_scaler, label_y)  
    ##model = SelectFromModel(lsvc, prefit=True)  
    ##X_lsvc = model.transform(label_X_scaler)  
    ##df_X_lsvc=pd.DataFrame(X_chi2)  
    ##feature_names = df_X_lsvc.columns.tolist()#显示列名  
    ##print('L1选择的特征：\n',feature_names)  

    ##基于树的特征选择，并按重要性阈值选择特征  
    # clf = ExtraTreesClassifier()#基于树模型进行模型选择
    # clf = clf.fit(label_X_scaler, label_y)
    # model = SelectFromModel(clf, threshold='1.00*mean',prefit=True)#选择特征重要性为1倍均值的特征，数值越高特征越重要
    # X_trees = model.transform(label_X_scaler)#返回所选的特征
    # df_X_trees=pd.DataFrame(X_chi2)
    # feature_names = df_X_trees.columns.tolist()#显示列名
    # print('树选择的特征：\n',feature_names)
    scaler = preprocessing.StandardScaler().fit(x)
    scaled_x = scaler.transform(x)
    joblib.dump(scaler, ".\\model\\scaler.save")
    return scaled_x


def scale_load(x):
    scaler = joblib.load(".\\model\\scaler.save")
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
    # plt.title('Feature Importances')
    # plt.bar(range(X_train.shape[1]), importances[indices], color='lightblue', align='center')
    # plt.xticks(range(X_train.shape[1]), feat_labels[indices], rotation=90)
    # plt.xlim([-1, X_train.shape[1]])
    # plt.tight_layout()
    # # plt.savefig('./random_forest.png', dpi=300)
    # plt.show()
    return del_cols