from cleanData import *
import pandas as pd

model_names = ['LassoCV', 'DecisionTree', 'ElasticNet', 'SVR', 'GBR']

df = pd.read_csv(".\\data\\test.csv", encoding="gbk")
out = df[['id']]
df = drop_fill(df)
df = sexencode(df)
show(df)

X = np.array(df.values)
print(X.shape)
X = scale_load(X)

modelindex = 0
for model in model_names:
    model = joblib.load(".\\model\\" + model_names[modelindex] + ".m")
    y = model.predict(X)
    print(y[0:20])
    out['y'] = y
    print('model %s out' % model_names[modelindex], out)
    out.to_csv('.\\result\\120A_%s.csv' % model_names[modelindex])
    modelindex += 1


