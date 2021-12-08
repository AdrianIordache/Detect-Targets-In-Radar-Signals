from utils import *

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier, plot_importance

import os
import random
import numpy as np
import pandas as pd
from IPython.display import display

from sklearn.svm import SVC
from sklearn.experimental import enable_hist_gradient_boosting  
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.experimental import enable_hist_gradient_boosting  
from sklearn.ensemble import BaggingClassifier, AdaBoostRegressor, RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import  accuracy_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, train_test_split

FOLD_MODELS = [
    [(1, 1, 42, 0, 0.77)], #(1, 1, 43, 0, 0.77)],
    [(1, 1, 42, 1, 0.76)], #(1, 1, 43, 1, 0.76)],
    [(1, 1, 42, 2, 0.76)], #(1, 1, 43, 2, 0.76)],
    [(1, 1, 42, 3, 0.76)], #(1, 1, 43, 3, 0.76)],
    [(1, 1, 42, 4, 0.76)], #(1, 1, 43, 4, 0.76)],
]
    
for MODELS in FOLD_MODELS:
    dataset = pd.read_csv(PATH_TO_TRAIN_META)

    dataset = generate_folds(
        data         = dataset, 
        skf_column   = 'label', 
        n_folds      = 5, 
        random_state = SEED
    )

    for i, (stage, gpu, version, fold, baseline) in enumerate(MODELS):
        model_name      = f'train_stage_{stage}_gpu_{gpu}_version_{version}_fold_{fold}_baseline_{baseline}.csv'
        embeddings_path = os.path.join(PATH_TO_EMBEDDINGS, model_name)
        embeddings      = pd.read_csv(embeddings_path)
        dataset         = pd.merge(dataset, embeddings, on = 'id')


    fold = MODELS[0][3]
    train_df = dataset[dataset['fold'] != fold]
    valid_df = dataset[dataset['fold'] == fold]

    X_train = train_df.drop(['id', 'label', 'fold'], inplace = False, axis = 1).values
    X_valid = valid_df.drop(['id', 'label', 'fold'], inplace = False, axis = 1).values 

    y_train = train_df['label'].values
    y_valid = valid_df['label'].values

    svm_model = SVC(C = 0.3)
    svm_model.fit(X_train, y_train)

    svm_predictions = svm_model.predict(X_valid)
    print("SVM Accuracy: {}".format(accuracy_score(y_valid, svm_predictions)))

    # xgb_model = LGBMClassifier()
    # xgb_model.fit(X_train, y_train)

    # xgb_predictions = xgb_model.predict(X_valid)
    # print("XGB Accuracy: {}".format(accuracy_score(y_valid, xgb_predictions)))
