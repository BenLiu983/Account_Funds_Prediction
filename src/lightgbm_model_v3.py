import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
import lightgbm as lgb
from hyperopt import hp, tpe, fmin, Trials, STATUS_OK, space_eval
from hyperopt.pyll.base import scope
import mlflow
import mlflow.lightgbm
from utils.metrics import PowerRatio

## 1. lightgbm

class lightgbm_dev_v1:
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
            'max_depth': scope.int(hp.quniform('max_depth', 3,8,1)),
            'n_estimators': scope.int(hp.quniform('n_estimators', 50, 200, 50)),
            'learning_rate': hp.uniform('learning_rate', 0.01, 0.1)
        }


    def baseline_model(self):
        with mlflow.start_run() as run:
            mlflow.lightgbm.autolog()
            model = lgb.LGBMRegressor(**self.baseline_params)
            model.fit(self.X_train, self.y_train)
            
            # prediction and metrics
            y_train_pred = model.predict(self.X_train)
            y_test_pred = model.predict(self.X_test)
            
            rmse_train = np.sqrt(mean_squared_error(self.y_train, y_train_pred))
            rmse_test = np.sqrt(mean_squared_error(self.y_test, y_test_pred))
            r2_train = r2_score(self.y_train, y_train_pred)
            r2_test = r2_score(self.y_test, y_test_pred)
            pr_train = PowerRatio(y_train_pred, self.y_train)
            pr_test = PowerRatio(y_test_pred, self.y_test)
            
            # log metrics
            mlflow.log_metric("rmse_train", rmse_train)
            mlflow.log_metric("rmse_test", rmse_test)
            mlflow.log_metric("r2_train", r2_train)
            mlflow.log_metric("r2_test", r2_test)
            mlflow.log_metric("powerratio_train", pr_train)
            mlflow.log_metric("powerratio_test", pr_test)
            
            # log id
            run_id = run.info.run_id
            mlflow.log_param("run_id", run_id)
            print(f"MLflow Run ID: {run_id}")
            
        return model, run_id

    def hyperopt_model(self, max_evals=50):
        def objective(params):
            
            params['num_leaves'] = int(params['num_leaves'])
            params['max_depth'] = int(params['max_depth'])
            params['n_estimators'] = int(params['n_estimators'])
            params['min_data_in_leaf'] = int(params['min_data_in_leaf'])
            
            model = lgb.LGBMRegressor(**params)
            model.fit(self.X_train, self.y_train)
            
            # predictions and metrics
            y_train_pred = model.predict(self.X_train)
            y_test_pred = model.predict(self.X_test)
            
            rmse_train = np.sqrt(mean_squared_error(self.y_train, y_train_pred))
            rmse_test = np.sqrt(mean_squared_error(self.y_test, y_test_pred))
            r2_train = r2_score(self.y_train, y_train_pred)
            r2_test = r2_score(self.y_test, y_test_pred)
            pr_train = PowerRatio(y_train_pred, self.y_train)
            pr_test = PowerRatio(y_test_pred, self.y_test)
            
            loss = mean_squared_error(self.y_test, y_test_pred)       
            
            # logging metrics
            mlflow.log_metric("rmse_train", rmse_train)
            mlflow.log_metric("rmse_test", rmse_test)
            mlflow.log_metric("r2_train", r2_train)
            mlflow.log_metric("r2_test", r2_test)
            mlflow.log_metric("powerratio_train", pr_train)
            mlflow.log_metric("powerratio_test", pr_test)
            
            mlflow.log_metric("loss", loss)
                   
            return {'loss': loss, 'status': STATUS_OK}


        with mlflow.start_run() as run:
            
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
            
            mlflow.log_params(best_params_int)

            model = lgb.LGBMRegressor(**best_params_int)
            model.fit(self.X_train, self.y_train)
            
            # Predictions and metrics
            y_train_pred = model.predict(self.X_train)
            y_test_pred = model.predict(self.X_test)
            
            rmse_train = np.sqrt(mean_squared_error(self.y_train, y_train_pred))
            rmse_test = np.sqrt(mean_squared_error(self.y_test, y_test_pred))
            r2_train = r2_score(self.y_train, y_train_pred)
            r2_test = r2_score(self.y_test, y_test_pred)
            pr_train = PowerRatio(y_train_pred, self.y_train)
            pr_test = PowerRatio(y_test_pred, self.y_test)

            # Logging metrics
            mlflow.log_metric("rmse_train", rmse_train)
            mlflow.log_metric("rmse_test", rmse_test)
            mlflow.log_metric("r2_train", r2_train)
            mlflow.log_metric("r2_test", r2_test)
            mlflow.log_metric("powerratio_train", pr_train)
            mlflow.log_metric("powerratio_test", pr_test)
            
            # Log the run ID
            run_id = run.info.run_id
            mlflow.log_param("run_id", run_id)
            
            print(f"MLflow Run ID: {run_id}")
            print(f"best param: {best_params_int}")

        return model, run_id

    def run_model(self):
        if self.baseline_ind == 1:
            return self.baseline_model()
        else:
            return self.hyperopt_model(max_evals=self.max_evals)

