import xgboost as xgb
from sklearn.datasets import make_classification
 
X, Y = make_classification(1000, 20)
dtrain = xgb.DMatrix(X, Y)
dtest = xgb.DMatrix(X)
 
param = {'max_depth':10, 'min_child_weight':1, 'learning_rate':0.1}
num_round = 200
bst = xgb.train(param, dtrain, num_round)
res = bst.predict(dtest, pred_leaf=True)
print(res)