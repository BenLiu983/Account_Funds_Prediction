import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
import lightgbm as lgb
from hyperopt import hp, tpe, fmin, Trials, STATUS_OK, space_eval
from hyperopt.pyll.base import scope


## 1. lightgbm

class lightgbm_dev:
    def __init__(self, X_train, y_train, X_test, y_test,
                 baseline_params=None, baseline_ind=1,
                 search_space=None, max_evals=20):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.max_evals = max_evals
        self.baseline_ind = baseline_ind
        self.baseline_params = baseline_params if baseline_params is not None else {}
        self.search_space = search_space if search_space is not None else {
            'max_depth': int(hp.quniform('max_depth', 3,10,1))
        }

    def baseline_model(self):
        model = lgb.LGBMRegressor(**self.baseline_params)
        model.fit(self.X_train, self.y_train)
        return model

    def hyperopt_model(self, max_evals=50):
        def objective(params):
            model = lgb.LGBMRegressor(**params)
            model.fit(self.X_train, self.y_train)
            y_pred = model.predict(self.X_test)
            loss = mean_squared_error(self.y_test, y_pred)
            return loss

        # optimize
        trials = Trials()
        best_params = fmin(
            fn=objective,
            space=self.search_space,
            algo=tpe.suggest,
            max_evals = max_evals,
            trials = trials
        )

        best_params_int = {
            # int
            'num_leaves': int(best_params['num_leaves']),
            'max_depth': int(best_params['max_depth']),
            'n_estimators': int(best_params['n_estimators']),
            'min_data_in_leaf': int(best_params['min_data_in_leaf']),

            # decimal
            'learning_rate': best_params['learning_rate'],
            'feature_fraction': best_params['feature_fraction'],
            'bagging_fraction': best_params['bagging_fraction'],
            'lambda_l1': best_params['lambda_l1']
        }

        model = lgb.LGBMRegressor(**best_params_int)
        model.fit = (self.X_train, self.y_train)

        print(f"best param: {best_params_int}")

        return model

    def run_model(self, baseline_ind=1):
        if baseline_ind == 1:
            return self.baseline_model()
        else:
            return self.hyperopt_model(max_evals=self.max_evals)