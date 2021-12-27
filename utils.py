import numpy as np
import pandas as pd

from pprint import pprint
from copy import deepcopy
from sklearn.metrics import f1_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


# iterator for GridSearch
def folds_iterator(n_folds, samples_per_fold, size):
    for i in range(n_folds):
        yield np.arange(0, size - samples_per_fold * (i + 1)), \
              np.arange(size - samples_per_fold * (i + 1), size - samples_per_fold * i)


def validate_model(model, x, y, params, n_folds, samples_per_fold):
    grid_search = GridSearchCV(estimator=model,
                               param_grid=params,
                               scoring={"F1_Score": make_scorer(f1_score, average='micro')},
                               refit="F1_Score",
                               n_jobs=-1,
                               error_score='raise')
    grid_search.fit(x, y)
    best_index = grid_search.best_index_
    return grid_search.best_estimator_, grid_search.cv_results_["mean_test_F1_Score"][best_index], grid_search.best_params_


def validate_ML_models(X, y, config_models, n_folds, samples_per_fold, seed=25, show_plots=False, debug_mode=False):
    results_df = pd.DataFrame(columns=('Model_Name', 'F1_Score',
                                       'Model_Best_Params'))

    best_f1_score = -np.Inf
    best_model = None
    best_model_name = 'No model'
    best_params = None
    idx = 0
    for model_config in config_models:
        cur_model, cur_f1_score, cur_params = validate_model(deepcopy(model_config['model']),
                                                             X, y, model_config['params'],
                                                             n_folds, samples_per_fold)
        print('Model name: ', model_config['model_name'])
        print('Best model params: ')
        pprint(cur_params)
        results_df.loc[idx] = [model_config['model_name'],
                               cur_f1_score,
                               cur_params]
        idx += 1

        if cur_f1_score > best_f1_score:
            best_f1_score = cur_f1_score
            best_model_name = model_config['model_name']
            best_model = cur_model
            best_params = cur_params

    # add visualizations here
    return results_df, best_model, best_model_name


def scale_normalize(df, features):
    # separate the features
    feature_data = df.loc[:, features].values

    # scale and center data (mean = 0, variance = 1)
    scaled_data = StandardScaler().fit_transform(feature_data)

    # show the result
    scaled_df = pd.DataFrame(data=scaled_data, columns=features)
    return scaled_df
