import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor

import mlflow
import mlflow.sklearn
#import xgboost as xgb
#import mlflow.xgboost

from hyperopt import hp, tpe, fmin, Trials, STATUS_OK, space_eval
from hyperopt.pyll.base import scope

from utils.metrics import PowerRatio

# v1
class random_forest_dev_v1:
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
            'min_samples_split': scope.int(hp.quniform('min_samples_split', 2, 10, 1)),
        }

    def baseline_model(self):
        with mlflow.start_run(run_name = "Baseline Model") as run:

            model = RandomForestRegressor(**self.baseline_params)
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
            mlflow.log_params(self.baseline_params)
            mlflow.log_param("run_id", run_id)
            mlflow.sklearn.log_model(model, "baseline model")
            print(f"MLflow Run ID: {run_id}")
            
        return model, run_id

    def hyperopt_model(self, max_evals=50):

        def objective(params):
            
            params['max_depth'] = int(params['max_depth'])
            params['n_estimators'] = int(params['n_estimators'])
            params['min_samples_split'] = int(params['min_samples_split'])  

            model = RandomForestRegressor(**params)
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
                   
            return {'loss': loss, 'status': STATUS_OK}


        with mlflow.start_run(run_name = "Hyperopt Model") as run:
            
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
                'max_depth': int(best_params['max_depth']),
                'n_estimators': int(best_params['n_estimators']),
                'min_samples_split': int(best_params['min_samples_split']),

                # categorical
                'max_features': best_params['max_features'],
                'oob_score': best_params['oob_score']
                
            }

            model = RandomForestRegressor(**best_params_int)
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
            mlflow.log_params(best_params_int)
            run_id = run.info.run_id
            mlflow.log_param("run_id", run_id)
            mlflow.sklearn.log_model(model, "hyperopt model")
            
            print(f"MLflow Run ID: {run_id}")
            print(f"best param: {best_params_int}")

        return model, run_id

    def run_model(self):
        if self.baseline_ind == 1:
            return self.baseline_model()
        else:
            return self.hyperopt_model(max_evals=self.max_evals)


# v2
class random_forest_dev_v2:
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
            'min_samples_split': scope.int(hp.quniform('min_samples_split', 2, 10, 1)),
        }

    def baseline_model(self):
        with mlflow.start_run(run_name = "Baseline Model") as run:

            model = RandomForestRegressor(**self.baseline_params)
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
            mlflow.log_params(self.baseline_params)
            mlflow.log_param("run_id", run_id)
            mlflow.sklearn.log_model(model, "baseline model")
            print(f"MLflow Run ID: {run_id}")
            
        return model, run_id

    def hyperopt_model(self, max_evals=50):

        def objective(params):
            
            params['max_depth'] = int(params['max_depth'])
            params['n_estimators'] = int(params['n_estimators'])
            params['min_samples_split'] = int(params['min_samples_split'])  

            model = RandomForestRegressor(**params)
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

            loss = -pr_test
            #oss = mean_squared_error(self.y_test, y_test_pred)       
                   
            return {'loss': loss, 'status': STATUS_OK}


        with mlflow.start_run(run_name = "Hyperopt Model") as run:
            
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
                'max_depth': int(best_params['max_depth']),
                'n_estimators': int(best_params['n_estimators']),
                'min_samples_split': int(best_params['min_samples_split']),

                # categorical
                'max_features': best_params['max_features'],
                'oob_score': best_params['oob_score']
                
            }

            model = RandomForestRegressor(**best_params_int)
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
            mlflow.log_params(best_params_int)
            run_id = run.info.run_id
            mlflow.log_param("run_id", run_id)
            mlflow.sklearn.log_model(model, "hyperopt model")
            
            print(f"MLflow Run ID: {run_id}")
            print(f"best param: {best_params_int}")

        return model, run_id

    def run_model(self):
        if self.baseline_ind == 1:
            return self.baseline_model()
        else:
            return self.hyperopt_model(max_evals=self.max_evals)