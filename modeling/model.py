import joblib
import os
import pandas as pd

from tune_sklearn import TuneSearchCV

from helpers.helpers import make_directories_if_not_exists


def train_model(x_train, y_train, get_pipeline_function, model_name, model, param_space, n_trials, cv_times, scoring):
    """
    Trains a machine learning model, optimizes the hyperparameters, saves the serialized model into the
    MODELS_DIRECTORY, and saves the cross validation results as a csv into the DIAGNOSTICS_DIRECTORY.

    :param x_train: x_train dataframe
    :param y_train: y_train series
    :param get_pipeline_function: callable that takes model to produce a scikit-learn pipeline
    :param model_name: name of the model
    :param model: instantiated model
    :param param_space: the distribution of hyperparameters to search over
    :param n_trials: number of trial to search for optimal hyperparameters
    :param cv_times: number of times to cross validation
    :param scoring: scoring method used for cross validation
    :returns: scikit-learn pipeline
    """
    print(f'training {model_name}...')
    pipeline = get_pipeline_function(model)
    search = TuneSearchCV(pipeline, param_distributions=param_space, n_trials=n_trials, scoring=scoring, cv=cv_times,
                          verbose=1, n_jobs=-1, search_optimization='hyperopt')
    search.fit(x_train, y_train)
    best_pipeline = search.best_estimator_
    cv_results = pd.DataFrame(search.cv_results_).sort_values(by=['rank_test_score'], ascending=True)
    make_directories_if_not_exists([os.path.join(model_name, 'models')])
    joblib.dump(best_pipeline, os.path.join(model_name, 'models', 'model.pkl'), compress=3)
    make_directories_if_not_exists([os.path.join(model_name, 'diagnostics', 'cv_results')])
    cv_results.to_csv(os.path.join(model_name, 'diagnostics', 'cv_results', 'cv_results.csv'), index=False)
    return best_pipeline