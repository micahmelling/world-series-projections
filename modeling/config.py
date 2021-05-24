from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from hyperopt import hp
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score, f1_score, balanced_accuracy_score


TARGET = 'target'
TEST_SET_START_YEAR = 2017
CV_SCORING = 'neg_log_loss'
CLASS_CUTOFF = 0.5
CV_SPLITS = 3
DROP_COLUMNS = ['team_yearID', 'teamIDwinner', 'team_teamID']


FOREST_PARAM_GRID = {
    'model__max_depth': hp.uniformint('model__max_depth', 3, 16),
    'model__min_samples_leaf': hp.uniform('model__min_samples_leaf', 0.001, 0.01),
    'model__max_features': hp.choice('model__max_features', ['log2', 'sqrt'])
}


GRADIENT_BOOSTING_PARAM_GRID = {
    'model__learning_rate': hp.uniform('model__learning_rate', 0.01, 0.5),
    'model__n_estimators': hp.uniformint('model__n_estimators', 75, 150),
    'model__max_depth': hp.uniformint('model__max_depth', 2, 16)
}


XGBOOST_PARAM_GRID = {
    'model__learning_rate': hp.uniform('model__learning_rate', 0.01, 0.5),
    'model__n_estimators': hp.randint('model__n_estimators', 75, 150),
    'model__max_depth': hp.randint('model__max_depth', 2, 16),
}


LIGHTGBM_PARAM_GRID = {
    'model__learning_rate': hp.uniform('model__learning_rate', 0.01, 0.5),
    'model__n_estimators': hp.randint('model__n_estimators', 75, 150),
    'model__max_depth': hp.randint('model__max_depth', 2, 16),
    'model__num_leaves': hp.randint('model__num_leaves', 10, 100)
}


CAT_BOOST_PARAM_GRID = {
    'model__depth': hp.randint('model__depth', 2, 16),
    'model__l2_leaf_reg': hp.randint('model__l2_leaf_reg', 1, 10),
    'model__learning_rate': hp.uniform('model__learning_rate', 0.01, 0.5),
    'model__min_data_in_leaf': hp.uniformint('model__min_data_in_leaf', 1, 10)
}


MODEL_TRAINING_DICT = {
    'random_forest': [RandomForestClassifier(), FOREST_PARAM_GRID, 50],
    'extra_trees': [ExtraTreesClassifier(), FOREST_PARAM_GRID, 50],
    'gradient_boosting': [GradientBoostingClassifier(), GRADIENT_BOOSTING_PARAM_GRID, 50],
    'xgboost': [XGBClassifier(), XGBOOST_PARAM_GRID, 50],
    'light_gbm': [LGBMClassifier(), LIGHTGBM_PARAM_GRID, 50],
}


MODEL_EVALUATION_LIST = [
    ('1_prob', log_loss, 'log_loss'),
    ('1_prob', brier_score_loss, 'brier_score'),
    ('1_prob', roc_auc_score, 'roc_auc'),
    ('predicted_class', f1_score, 'f1'),
    ('predicted_class', balanced_accuracy_score, 'balanced_accuracy'),
]
