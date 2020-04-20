import os, sys, glob
import numpy as np
import pandas as pd

import time
import datetime

from joblib import Parallel, delayed
from sklearn.metrics import f1_score, log_loss, classification_report
from sklearn.model_selection import StratifiedKFold

import lightgbm as lgb


def read_feat(path, test_mode=False):
    df = pd.read_csv(path)
    df = df.iloc[::-1]

    if test_mode:
        df_feat = [df['渔船ID'].iloc[0], df['type'].iloc[0]]
        df = df.drop(['type'], axis=1)
    else:
        df_feat = [df['渔船ID'].iloc[0]]

    df['time'] = df['time'].apply(lambda x: datetime.datetime.strptime(x, "%m%d %H:%M:%S"))
    df_diff = df.diff(1).iloc[1:]
    df_diff['time_seconds'] = df_diff['time'].dt.total_seconds()
    df_diff['dis'] = np.sqrt(df_diff['x'] ** 2 + df_diff['y'] ** 2)

    df_feat.append(df['time'].dt.day.nunique())
    df_feat.append(df['time'].dt.hour.min())
    df_feat.append(df['time'].dt.hour.max())
    df_feat.append(df['time'].dt.hour.value_counts().index[0])

    df_feat.append(df['速度'].min())
    df_feat.append(df['速度'].max())
    df_feat.append(df['速度'].mean())

    # df_feat.append(df_diff['time'].min())
    # df_feat.append(df_diff['time'].max())
    # df_feat.append(df_diff['time'].mean())

    df_feat.append(df_diff['速度'].min())
    df_feat.append(df_diff['速度'].max())
    df_feat.append(df_diff['速度'].mean())
    df_feat.append((df_diff['速度'] > 0).mean())
    df_feat.append((df_diff['速度'] == 0).mean())

    df_feat.append(df_diff['方向'].min())
    df_feat.append(df_diff['方向'].max())
    df_feat.append(df_diff['方向'].mean())
    df_feat.append((df_diff['方向'] > 0).mean())
    df_feat.append((df_diff['方向'] == 0).mean())

    df_feat.append((df_diff['x'].abs() / df_diff['time_seconds']).min())
    df_feat.append((df_diff['x'].abs() / df_diff['time_seconds']).max())
    df_feat.append((df_diff['x'].abs() / df_diff['time_seconds']).mean())
    df_feat.append((df_diff['x'] > 0).mean())
    df_feat.append((df_diff['x'] == 0).mean())

    df_feat.append((df_diff['y'].abs() / df_diff['time_seconds']).min())
    df_feat.append((df_diff['y'].abs() / df_diff['time_seconds']).max())
    df_feat.append((df_diff['y'].abs() / df_diff['time_seconds']).mean())
    df_feat.append((df_diff['y'] > 0).mean())
    df_feat.append((df_diff['y'] == 0).mean())

    df_feat.append(df_diff['dis'].min())
    df_feat.append(df_diff['dis'].max())
    df_feat.append(df_diff['dis'].mean())

    df_feat.append((df_diff['dis'] / df_diff['time_seconds']).min())
    df_feat.append((df_diff['dis'] / df_diff['time_seconds']).max())
    df_feat.append((df_diff['dis'] / df_diff['time_seconds']).mean())

    return df_feat


read_feat('../input/hy_round1_train_20200102/4285.csv', True)

train_feat = Parallel(n_jobs=10)(delayed(read_feat)(path, True)
                                 for path in glob.glob('../input/hy_round1_train_20200102/*')[:])
train_feat = pd.DataFrame(train_feat)

test_feat = Parallel(n_jobs=10)(delayed(read_feat)(path, False)
                                 for path in glob.glob('../input/hy_round1_testA_20200102/*')[:])
test_feat = pd.DataFrame(test_feat)
test_feat = test_feat.sort_values(by=0)

train_feat[1] = train_feat[1].map({'围网':0,'刺网':1,'拖网':2})

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

n_fold = 10
skf = StratifiedKFold(n_splits=n_fold, shuffle=True)
eval_fun = f1_score


def run_oof(clf, X_train, y_train, X_test, kf):
    print(clf)
    preds_train = np.zeros((len(X_train), 3), dtype=np.float)
    preds_test = np.zeros((len(X_test), 3), dtype=np.float)
    train_loss = [];
    test_loss = []

    i = 1
    for train_index, test_index in kf.split(X_train, y_train):
        x_tr = X_train[train_index];
        x_te = X_train[test_index]
        y_tr = y_train[train_index];
        y_te = y_train[test_index]
        clf.fit(x_tr, y_tr, eval_set=[(x_te, y_te)], early_stopping_rounds=500, verbose=False)

        train_loss.append(eval_fun(y_tr, np.argmax(clf.predict_proba(x_tr)[:], 1), average='macro'))
        test_loss.append(eval_fun(y_te, np.argmax(clf.predict_proba(x_te)[:], 1), average='macro'))

        preds_train[test_index] = clf.predict_proba(x_te)[:]
        preds_test += clf.predict_proba(X_test)[:]

        print('{0}: Train {1:0.7f} Val {2:0.7f}/{3:0.7f}'.format(i, train_loss[-1], test_loss[-1], np.mean(test_loss)))
        print('-' * 50)
        i += 1
    print('Train: ', train_loss)
    print('Val: ', test_loss)
    print('-' * 50)
    print('Train{0:0.5f}_Test{1:0.5f}\n\n'.format(np.mean(train_loss), np.mean(test_loss)))
    preds_test /= n_fold
    return preds_train, preds_test


params = {
    'learning_rate': 0.01,
    'min_child_samples': 5,
    'max_depth': 7,
    'lambda_l1': 2,
    'boosting': 'gbdt',
    'objective': 'multiclass',
    'n_estimators': 2000,
    'metric': 'multi_error',
    'num_class': 3,
    'feature_fraction': .75,
    'bagging_fraction': .85,
    'seed': 99,
    'num_threads': 20,
    'verbose': -1
}

train_pred, test_pred = run_oof(lgb.LGBMClassifier(**params),
                                train_feat.iloc[:, 2:].values,
                                train_feat.iloc[:, 1].values,
                                test_feat.iloc[:, 1:].values,
                                skf)
test_feat['label'] = np.argmax(test_pred, 1)
test_feat['label'] = test_feat['label'].map({0:'围网',1:'刺网',2:'拖网'})
test_feat[[0, 'label']].to_csv('baseline.csv',index=None, header=None)