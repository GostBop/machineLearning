import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import matplotlib.pyplot as plt

x = np.array(pd.read_csv("./train/train.csv"))
y = pd.read_csv("./label/label.csv")

#划分训练集
x_train_all, x_predict, y_train_all, y_predict = train_test_split(x, y, test_size=0.10, random_state=100)

x_train, x_test, y_train, y_test = train_test_split(x_train_all, y_train_all, test_size=0.2, random_state=100)

train_data = lgb.Dataset(data=x_train,label=y_train)
test_data = lgb.Dataset(data=x_test,label=y_test)

#开始训练
param = {
    'objective':'regression', 
    
    'num_leaves':119, 
    'max_depth':7, 
    
    'learning_rate': 0.01, 
    'metric': 'rmse',
    'min_data_in_leaf': 2000,
    'feature_fraction': 0.54,
    'bagging_fraction': 1.0,
    'bagging_freq': 5,

    'colsample_bytree': 0.7,
}

num_round = 1000
evals_result = {}
bst = lgb.train(param, train_data, num_round, valid_sets=[test_data], early_stopping_rounds=2,evals_result=evals_result)

bst.save_model('model.txt')

bst = lgb.Booster(model_file='model.txt') 

#测试集预测
ypred = bst.predict(x_predict, num_iteration=bst.best_iteration)

RMSE = np.sqrt(mean_squared_error(y_predict, ypred))

print("RMSE of predict :",RMSE)

from sklearn.metrics import r2_score
r2_score_ = r2_score(y_predict, ypred)
print('r2 score of predict :', r2_score_)

print('plt result...')
ax = lgb.plot_metric(evals_result, metric='rmse')
plt.show()

print('rank the feature...')
ax = lgb.plot_importance(bst, max_num_features=20)
plt.show()

#预测结果写入
test = pd.read_csv("./test/test.csv", header=None)
x_pred = np.array(test)
y_pred = bst.predict(x_pred, num_iteration=bst.best_iteration)
id_ = []
for i in range(len(y_pred)):
    id_.append(i + 1)
    
dataframe = pd.DataFrame({'id':id_,'Predicted':y_pred})
dataframe.to_csv("result.csv",index=False,sep=',')