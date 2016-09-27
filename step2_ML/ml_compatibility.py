from __future__ import division, print_function

import xgboost as xgb

# from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
import sklearn.preprocessing as prep
from sklearn import cluster

# from sklearn import cross_validation as cv
# from sklearn.feature_selection import RFE, RFECV
# from sklearn.metrics import regression as metrics
# import pandas as pd
import numpy as np
# import itertools
from scipy.integrate import cumtrapz
from matplotlib import pyplot as plt
import seaborn as sns
from matplotlib import rcParams

sns.set_context('paper', font_scale=2, rc={"lines.linewidth": 2})
sns.set_style('ticks')
rcParams['figure.figsize'] = 8, 6

def plot_the_integral(y, x, label='', color=None):
    sort_idx = np.argsort(x)
    xx = np.array(x)[sort_idx]
    yy = np.array(y)[sort_idx]
    iy = cumtrapz(yy, xx)
    iy -= iy[xx[1:] < 4].min() 
    if color:
        pl = plt.plot(xx[1:], iy, color=color, label=label, alpha=0.6, lw=3)
    else:
        pl = plt.plot(xx[1:], iy, label=label, lw=3)
    return iy, pl


def get_train_test_idx(X, n_interp, do_clustering=True, do_birch=True):
    if do_clustering:
        Xscaled = prep.StandardScaler().fit_transform(X)
        if do_birch:
            threshold_birch = 1.
            n_clusters = int(len(Xscaled) / n_interp)
            while True:
                try:
                    clus = cluster.Birch(threshold=threshold_birch, n_clusters=n_clusters)
                    clus.fit(Xscaled)
                    if len(np.unique(clus.labels_)) == n_clusters:
                        break
                    else:
                        threshold_birch *=0.5
                except:
                    threshold_birch *=0.5
        else:
            n_clusters = int(len(Xscaled) / n_interp)
            clus = cluster.KMeans(n_clusters=n_clusters)
            clus.fit(Xscaled)
        clus_indices = [np.where(clus.labels_ == i)[0] for i in np.unique(clus.labels_)]
        idx_train = []
        for label in np.unique(clus.labels_):
            cluster_center = Xscaled[clus_indices[label]].mean(axis=0)
            dists = ((Xscaled - cluster_center)**2).sum(axis=1)**0.5
            idx_train.append(np.argmin(dists))

    else:
        idx_train = list(np.arange(0, len(X), n_interp))

    idx_test = set(np.arange(0, len(X))).difference(idx_train)
    idx_test = list(idx_test)
    return idx_train, idx_test


def load_data(rtype):
    
    if rtype == 'crystal':
        data = np.loadtxt('db-c0.35-clusters-md-crack-300K-open-12-withf-gamma-phd.csv', delimiter=',')
        _, y_TS = np.loadtxt('db-cryst.csv', delimiter=',', skiprows=2).T
    elif rtype == 'amorph':
        data =  np.loadtxt('db-ashift_70.0-clusters-md-crack-300K-open-12-chunk1-withf-gamma-phd.csv', delimiter=',')
        _, y_TS = np.loadtxt('db-amorph.csv', delimiter=',', skiprows=2).T

    n_data = min([len(data), len(y_TS)])
    data = data[:n_data]
    y_TS = y_TS[:n_data]
    
    colvar = data[:, 0].copy()
    X = data[:, 1:-1].copy()
    y = data[:,-1].copy()
    X = np.hstack([X, y_TS[:,None]])
    data = None
    return X, y, y_TS, colvar


X1, y1, y_TS1, colvar1 = load_data('crystal')
X2, y2, y_TS2, colvar2 = load_data('amorph')

X = np.vstack([X1, X2])
y = np.hstack([y1, y2])
y_TS = np.hstack([y_TS1, y_TS2])

n_data1 = len(y1)
n_data2 = len(y2)
n_interp = 50

# X = prep.StandardScaler().fit_transform(X)
# X = prep.RobustScaler().fit_transform(X)
# poly = prep.PolynomialFeatures(degree=2, interaction_only=False).fit_transform(X)
# X = np.hstack([X, poly])
# X = prep.StandardScaler().fit_transform(X)

# from sklearn import decomposition
# X = decomposition.MiniBatchSparsePCA().fit_transform(X)

# X = prep.RobustScaler().fit_transform(X)
params = {
    'eta': 0.05,
    'colsample_bytree': 1,
    'gamma' : 0.01,
    'max_depth': 4,
    'min_child_weight': 1,
    'n_estimators': 500,
    'subsample': 0.6,
    'silent': True,
    'early_stopping_rounds': 100,
    'objective': 'reg:linear',
    'lambda': 1.,
    'n_jobs': -1,
    'eval_metric': "mae",
    'seed': 42}

idx_train, idx_test = get_train_test_idx(X, n_interp, do_clustering=True, do_birch=False)

X_train = X[idx_train]
X_test  = X[idx_test]
y_train = y[idx_train]
y_test = y[idx_test]

if True:
    dtrain = xgb.DMatrix(np.array(X_train), label=y_train)
    model = xgb.train(params,
                      dtrain,
                      params['n_estimators'])
    y_ML = model.predict(xgb.DMatrix(X))
else:
    from sklearn.gaussian_process import GaussianProcess
    model = GaussianProcess(nugget=0.03)
    model.fit(X_train, y_train)
    y_ML = model.predict(X)


y_ML1 = y_ML[:n_data1]
y_ML2 = y_ML[n_data1:]

colors = sns.color_palette(palette='Set2', n_colors=4)


plt.clf()
iy1, _ = plot_the_integral(y1, colvar1, label='DFT c', color=colors[0])
iy_ML1, _ = plot_the_integral(y_ML1, colvar1, label='ML  c', color=colors[1])

iy2, _ = plot_the_integral(y2, colvar2, label='DFT a', color=colors[2])
iy_ML2, _ = plot_the_integral(y_ML2, colvar2, label='ML  a', color=colors[3])

plt.xlim(3,7.2)
plt.ylim(-3, 2)
# plt.scatter(colvar[idx_train], [plt.ylim()[0]]*len(idx_train), marker='|', s=500, c=colors[0], alpha=1, linewidths=2)
plt.xlabel("Reaction coordinate [A]")
plt.ylabel("Free energy [eV]")
sns.despine()
plt.tight_layout()
plt.legend()
plt.savefig("f_ml_compat_%d.pdf" % n_interp)


from sklearn.metrics.regression import mean_absolute_error as MAE
print("MAE crystal: %.3f" % MAE(iy1, iy_ML1))
print("MAE amorph: %.3f" % MAE(iy2, iy_ML2))

import sys
sys.exit(0)


plt.plot(colvar[1:], iy, '-', label='1', color='black')
for i, n_interp in enumerate([5,10,15,20,25]):
    y_DFT_interp = np.interp(colvar, colvar[::n_interp], y[::n_interp])
    iy_DFTinterp, _ = plot_the_integral(y_DFT_interp, colvar, label='%d' % n_interp, color=colors[i])


def get_train_test_idx2(X, n_train):
    Xscaled = prep.StandardScaler().fit_transform(X)
    n_clusters = n_train # np.max([n_train, len(X)])
    clus = cluster.KMeans(n_clusters=n_clusters)
    clus.fit(Xscaled)
    clus_indices = [np.where(clus.labels_ == i)[0] for i in np.unique(clus.labels_)]
    idx_train = []
    for label in np.unique(clus.labels_):
        cluster_center = Xscaled[clus_indices[label]].mean(axis=0)
        dists = ((Xscaled - cluster_center)**2).sum(axis=1)**0.5
        idx_train.append(np.argmin(dists))

    idx_test = set(np.arange(0, len(X))).difference(idx_train)
    idx_test = list(idx_test)
    return idx_train, idx_test


import pandas as pd
from sklearn.metrics.regression import mean_absolute_error as MAE
boo = xgb.XGBRegressor(n_estimators = 500,
                       gamma = 0.01,
                       max_depth = 4,
                       learning_rate = 0.05,
                       subsample = 0.6)


results = []
res_errors = []

for n_train in np.logspace(1, np.log10(0.8 * len(X)), 20).astype('int'):
    errors = []
    y_preds = []
    for i in range(10):
        idx_train, idx_test = get_train_test_idx2(X, n_train)
        X_train = X[idx_train]
        X_test = X[idx_test]
        y_train = y[idx_train]
        y_test = y[idx_test]

        boo.fit(X_train, y_train)
        y_pred = boo.predict(X_test)
        error = MAE(y_pred, y_test)
        print("MAE = %.5f" % error)
        y_preds.append(y_pred)
        errors.append(error)
        r = [n_interp, error]
        results.append(r)
        for a, b in zip(y_pred, y_test):
            e = np.abs(a - b)
            res_errors.append([n_train, e])

df = pd.DataFrame(data=np.array(res_errors), columns=['n_train', 'error'])
df['n_train'] = df['n_train'].astype('int')
grp = df.groupby('n_train')
r = grp.mean()

# plt.scatter(r.index, r.error)
res = grp['error'].agg([np.mean, np.std])
sns.set_style('ticks')
rcParams['figure.figsize'] = 8, 6

plt.plot(r.index, r.error, 'o-', lw=3, label='ML', color='black')
plt.plot(r.index, [MAE(y,y_TS)] * len(r.index), '--', lw=3, label='TS', color='black')
plt.xscale('log')
plt.ylim(0,0.40)
plt.xlim(10,1000)
plt.legend(loc=3)
plt.xlabel("Training set size")
plt.ylabel(r"MAE [$eV/\AA{}$]")
sns.despine()
plt.tight_layout()

# r = [n_interp, np.mean(errors), np.std(errors)]
    # results.append(r)
