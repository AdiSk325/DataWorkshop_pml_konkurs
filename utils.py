from itertools import product
import sys

import numpy as np
from numpy.random import uniform, normal, lognormal, exponential

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from sklearn.model_selection import learning_curve
from sklearn.metrics import mean_absolute_error as mae


#MODELE
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
import xgboost as xgb
import lightgbm as lgb
import catboost as ctb

BLACK_LIST = ['CO', 'log_CO', 'CO_cat', 'id', 'sample']
TARGET = 'CO'

FEATS = ['TIT', 'TEY', 'TAT', 'AFDP', 'AT + NOX', 'TIT - TEY', 'NOX','GTEP',
         'AH', 'TAT - TIT', 'AT', 'AP', 'CO', 'log_TAT', 'log_TIT', 'AFDP_NOX']

def prepare_data():

    df_train = pd.read_hdf("../module3/input/train_power_plant.h5")
    df_test = pd.read_hdf("../module3/input/test_power_plant.h5")
    
    df_train['sample'] = 'train'    
    df_test['sample'] = 'test'
    df_train['log_CO'] = np.log1p(df_train['CO'])
    
    df_all = pd.concat([df_train, df_test])
    return df_all


def split_data(df_all):
    
    df_train, df_test = df_all[df_all['CO'].notna()],  df_all[df_all['CO'].isna()]
    return df_train, df_test
    
    
def feature_engeenering(df_all):
    
    feats = get_feats(df_all)    
    df_all['CO_cat'] = (df_all['CO'] < df_all['CO'].median()).astype(int)
    
    for col in ['TAT', 'TIT', 'TEY']:
        df_all[f'log_{col}'] = np.log1p(df_all[col])
    
    df_all['TAT - TIT'] =  df_all['TAT'] - df_all['TIT']
    df_all['TAT - TEY'] =  df_all['TAT'] - df_all['TEY']
    df_all['TIT - TEY'] =  df_all['TIT'] - df_all['TEY']
    
    df_all['TIT / TEY'] = df_all['TIT'] / df_all['TEY']
    df_all['TAT / TEY'] = df_all['TAT'] / df_all['TEY']
    df_all['TIT / TAT'] = df_all['TIT'] / df_all['TAT']
        
    df_all['AT + NOX']  = df_all['AT'] + df_all['NOX']
        
    for feat, bins in product(['TIT', 'TAT', 'TEY'], [2,3,5,10]):
        
        df_all[f'{feat}_cat_{bins}'] = pd.qcut(df_all[feat], bins).apply(lambda x: x.left)
    
    df_all['GTEP_cat'] = pd.qcut(df_all['GTEP'], 3).apply(lambda x: x.left)
    df_all['AFDP_NOX'] = df_all['AFDP'] + df_all['NOX']
            
    return df_all


def add_random_feats(df, no_feats=5, dis_type=['uniform', 'normal', 'lognormal', 'exponential'] ):
    
    for idx, dist in product(range(no_feats), dis_type):
        
        if f'random_{dist}_{idx}' not in df:
            df[f'random_{dist}_{idx}'] = eval(dist+'(size=df.shape[0])')
    
    return df

def get_feats(df):
    
    return [col for col in df.columns if col not in BLACK_LIST]

def get_X(df):
    return df[ get_feats(df) ].values

def get_y(df, target_var=TARGET):
    return df[target_var].values

def get_models():
    
    ctb_params = {'n_estimators': 1100.0, 'l2_leaf_reg': 2.0391558389346702, 'learning_rate': 0.050701938569694155, 'max_depth': 15.0}
    
    return [
        #('dtr-10md', DecisionTreeRegressor(max_depth=10, random_state=0)),
        #('rf-10md-200tr', RandomForestRegressor(max_depth=10, n_estimators=200, random_state=0)),
        #('xgb-10md-200tr', xgb.XGBRegressor(max_depth=10, n_estimators=200, random_state=0)),
        #('lgb-10md-200tr', lgb.LGBMRegressor(max_depth=10, n_estimators=200, random_state=0)),
        #('lgb-10md-200tr', ctb.CatBoostRegressor(depth=10, n_estimators=1000, random_state=0, verbose=0)),
        ('ctb-15md-1100tr', ctb.CatBoostRegressor(**ctb_params, random_state=0)),
    ]

def run_cv(model, X, y, folds=4, target_log=False,cv_type=KFold, success_metric=mae):
    cv = cv_type(n_splits=folds)
    
    scores = []
    for train_idx, test_idx in cv.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if target_log:
            y_train = np.log1p(y_train)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        if target_log:
            y_pred = np.expm1(y_pred)
            y_pred[y_pred < 0] = 0 #czasem może być wartość ujemna

        score = success_metric(y_test, y_pred)
        scores.append( score )
        
    return np.mean(scores), np.std(scores)


def plot_learning_curve(model, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5), target_log=False):
    
    plt.figure(figsize=(12,8))
    plt.title(title)
    if ylim is not None:plt.ylim(*ylim)

    plt.xlabel("Training examples")
    plt.ylabel("Score")
    
    if target_log:
        y = np.log1p(y)
    
    def my_scorer(model, X, y):
        y_pred = model.predict(X)
        
        if target_log:
            y = np.expm1(y)
            y_pred = np.expm1(y_pred)
            y_pred[ y_pred<0 ] = 0
        
        return mae(y, y_pred)

        
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring=my_scorer)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


def run(train, plot_lc=False, folds=3, ylim=(0, 2), target_log=False):
    X, y  = get_X(train), get_y(train)

    for model_name, model in get_models():
        score_mean, score_std = run_cv(model, X, y, folds=folds, target_log=target_log)
        print("[{0}]: {1} +/-{2}".format(model_name, score_mean, score_std))
        sys.stdout.flush() #wypisujemy wynik natychmiast, bez buforowania

        if False == plot_lc: continue
        plt = plot_learning_curve(model, model_name, X, y, ylim=ylim, cv=folds, target_log=target_log)
        plt.show()

def fit_predict_model(model, train, test, target_log=False):
    
    X_train, y_train  = get_X(train), get_y(train)
    X_test = get_X(test)
    
    if target_log:
            y_train = np.log1p(y_train)
            
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    if target_log:
        y_pred = np.expm1(y_pred)
    
    return y_pred



def xgb_objective(space):
    
    xgb_params = {
        'max_depth': int(space['max_depth']),
        'colsample_bytree': space['colsample_bytree'],
        'learning_rate': space['learning_rate'],
        'subsample': space['subsample'],
        'random_state': int(space['random_state']),
        'min_child_weight': int(space['min_child_weight']),
        'reg_alpha': space['reg_alpha'],
        'reg_lambda': space['reg_lambda'],
        'n_estimators': 100,
        'objective': 'reg:squarederror'
    }
    
    model = xgb.XGBRegressor(**xgb_params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    score = mean_squared_error(y_test, y_pred)
    
    return{'loss':score, 'status': STATUS_OK }
    
    
    
    
    