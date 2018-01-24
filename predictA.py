from cleanData import *
import pandas as pd
from sklearn.metrics import mean_squared_error
model_names = ['Lasso', 'ElasticNet', 'catboost',
               'GBR', 'KRR', 'LightGBM']  # 不同模型的名称列表

df = pd.read_csv(".\\data\\test.csv", encoding="gbk")
df = drop_fill(df)
df = sexencode(df)
show(df)

X = np.array(df.values)
print(X.shape)
X = scale_load(X)

out = pd.DataFrame()
y = 0.0
modelindex = 0
for model in model_names:
    model = joblib.load(".\\model\\" + model_names[modelindex] + ".m")
    y += model.predict(X)
    modelindex += 1
y = y/len(model_names)
out = pd.DataFrame({'y': y})
out.to_csv('.\\result\\124A_%s.csv' % 'average', index=None, header=None, float_format='%.4f')




