import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV

import statsmodels.api as sm

from hyperopt import hp, tpe, fmin, Trials, STATUS_OK, space_eval
from hyperopt.pyll.base import scope
import mlflow

from utils.metrics import PowerRatio



class linear_regression_v1:
    def __init__(self, X_train, y_train, X_test, y_test, model_type = 'baseline', significance_level=0.05):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.model_type = model_type
        self.significance_level = significance_level

    # m1: baseline with all fea
    def baseline_model(self):

        X_train_with_intercept = sm.add_constant(self.X_train)
        model = sm.OLS(self.y_train, X_train_with_intercept).fit()
            
        return model

    # m2: backward fea selection
    def backward_model(self):
        
        def backward_selection(X, y, significance_level=0.05):
            while True:

                model = sm.OLS(y, X).fit()
                
                # p-values of the features
                p_values = model.pvalues[1:]  # Exclude the intercept
                max_p_value = p_values.max()
                
                # if the highest p-value > significance_level, remove the feature
                if max_p_value > significance_level:
                    
                    # get feature with highest p-value
                    feature_to_remove = p_values.idxmax()  
                    X = X.drop(columns=[feature_to_remove])
                    
                else:
                    break
            
            return X, model


        # adding a constant to the model (intercept)
        X_train_with_intercept = sm.add_constant(self.X_train)

        # apply backward selection
        X_selected_backward, backward_model = backward_selection(X_train_with_intercept, self.y_train)

        model = sm.OLS(self.y_train, X_selected_backward).fit()

        return model, X_selected_backward


    # m3: forward fea selection
    def forward_model(self):
        
        def forward_selection(X, y, significance_level=0.05):
            selected_features = []  # list to store selected features
            remaining_features = list(X.columns)  # list of all feature names
            while remaining_features:
                
                p_values = []
                for feature in remaining_features:
            
                    # model with selected features + the current feature
                    model = sm.OLS(y, sm.add_constant(X[selected_features + [feature]])).fit()
            
                    # store p-value for the current feature
                    p_values.append((feature, model.pvalues[feature]))  

                    # sort features by p-value
                p_values.sort(key=lambda x: x[1])  # Sort by p-value

                # if the best p-value < significance level, add that feature
                best_feature, best_p_value = p_values[0]
                
                if best_p_value < significance_level:
                    selected_features.append(best_feature)
                    remaining_features.remove(best_feature)  # remove the fea 
                else:
                    break  
        
            X_selected = X[selected_features]
            model = sm.OLS(y, sm.add_constant(X_selected)).fit()
        
            return X_selected, model

        
        # adding a constant to the model (intercept)
        X_train_with_intercept = sm.add_constant(self.X_train)

        # apply forward selection
        X_selected_forward, forward_model = forward_selection(X_train_with_intercept, self.y_train)

        model = sm.OLS(self.y_train, X_selected_forward).fit()

        return model, X_selected_forward
        
        
    def run_model(self):
        if self.model_type == 'baseline':
            return self.baseline_model()
        elif self.model_type == 'backward':
            return self.backward_model()
        elif self.model_type == 'forward':
            return self.forward_model()