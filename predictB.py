from cleanData import *
import pandas as pd
import datetime
from sklearn.metrics import mean_squared_error
model_names = ['Lasso', 'ElasticNet', 'catboost',
               'GBR', 'LightGBM']  # 不同模型的名称列表


df = pd.read_csv(".\\data\\test_B.csv", encoding="gbk")
df = drop_fill(df)
df = encode(df)
show(df)

X = np.array(df.values)
print(X.shape)
X = scale_load(X, 'x')
out = pd.DataFrame()
y = 0.0


# 模型融合
modelindex = 0
for model in model_names:
    model = joblib.load(".\\model_add_feature\\" + model_names[modelindex] + ".m")
    y += model.predict(X)
    modelindex += 1
y = y/len(model_names)


out = pd.DataFrame({'y': y})
out.to_csv(r'./result/{}_B_average.csv'.format(datetime.datetime.now().strftime('%m%d_%H%M')), index=None, header=None, float_format='%.4f')




