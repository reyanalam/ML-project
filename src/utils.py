import os
import sys
import numpy as np
import pandas as pd
import dill
from sklearn.metrics import r2_score
from src.exception import CustomeExceptionClass
from sklearn.model_selection import GridSearchCV

def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,'wb') as file_obj:
            dill.dump(obj,file_obj)

    except Exception as e:
        raise CustomeExceptionClass(e,sys)
def evaluation_model(x_train, y_train, x_test, y_test, models, params):
    try:
        report = {}
        for model_name, model in models.items():
            param_grid = params.get(model_name, {})
            
            # Initialize best_model to None
            best_model = None

            if param_grid:  # If params are present, use GridSearchCV
                grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
                grid_search.fit(x_train, y_train)
                best_model = grid_search.best_estimator_
            else:  # Directly fit models without parameters
                model.fit(x_train, y_train)
                best_model = model
            
            # Use best_model for predictions
            y_train_pred = best_model.predict(x_train)  # Predict on train set
            y_test_pred = best_model.predict(x_test)    # Predict on test set

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            report[model_name] = test_model_score  # Store test score in report
        return report
    except Exception as e:
        raise CustomeExceptionClass(e, sys)
