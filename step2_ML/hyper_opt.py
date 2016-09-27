from __future__ import division, print_function

import numpy as np
import xgboost as xgb
from bayes_opt import BayesianOptimization
from scipy.integrate import cumtrapz
from sklearn.metrics.regression import mean_absolute_error
from sklearn import cross_validation as cv

data = np.loadtxt('db-c0.35-clusters-md-crack-300K-open-12-withf-gamma-phd.csv', delimiter=',')
_, y_TS = np.loadtxt('db-cryst.csv', delimiter=',', skiprows=2).T

n_data = min([len(data), len(y_TS)])

data = data[:n_data]
y_TS = y_TS[:n_data]

X = data[:, :-1].copy()
colvar = data[:,0].copy()
X = np.hstack([X, y_TS[:,None]])
y = data[:,-1].copy()

data2 = np.loadtxt('db-ashift_70.0-clusters-md-crack-300K-open-12-chunk1-withf-gamma-phd.csv', delimiter=',')
_, y_TS2 = np.loadtxt('db-amorph.csv', delimiter=',', skiprows=2).T

n_data2 = min([len(data2), len(y_TS2)])

data2 = data[:n_data2]
y_TS2 = y_TS2[:n_data2]

X2 = data2[:, :-1].copy()
colvar2 = data2[:,0].copy()
X2 = np.hstack([X2, y_TS2[:,None]])
y2 = data2[:,-1].copy()

X = np.vstack([X, X2])
colvar = np.hstack([colvar, colvar2])
y = np.hstack([y, y2])

params = {
    'eta' : 0.05,
    'max_depth' : 5,
    'min_child_weight' : 1,
    'subsample' : 0.8,
    'gamma' : 1,
    'colsample_bytree' : 0.8,
    'early_stopping_rounds': 100,
    'objective': 'reg:linear',
    'lambda': 1.,
    'n_jobs': 4,
    'eval_metric': 'mae',
    'seed': 42,
    'silent' : 1}

ttsplits = []
for i in range(5):
    X_train, X_test, y_train, y_test = cv.train_test_split(X, y, test_size=0.9)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    ttsplits.append([dtrain, dtest])
    
def score(
        # n_estimators,
        max_depth,
        min_child_weight,
        subsample,
        gamma,
        colsample_bytree):
    params['max_depth'] = int(max_depth)
    params['min_child_weight'] = min_child_weight
    params['subsample'] = subsample
    params['gamma'] = gamma
    params['colsample_bytree'] = colsample_bytree
    scores = []
    for (dtrain, dtest) in ttsplits:
        model = xgb.train(params,
                          dtrain,
                          500)
                          # int(n_estimators))
        y_pred = model.predict(dtest)
        score = mean_absolute_error(dtest.get_label(), y_pred)
        scores.append(score)
    return - np.mean(score)



BO = BayesianOptimization(score,
                          {
                              # 'n_estimators': [100, 2000],
                              'max_depth' : [4, 8],
                              'min_child_weight': [1, 8],
                              'subsample' : [0.6, 0.8],
                              'gamma' : [0.01, 1],
                              'colsample_bytree': [0.1, 1]
                          })

BO.maximize(init_points=2**6, n_iter=2000, xi=0.01, acq="ei")
