import os
import sys
import numpy as np
import pandas as pd
import dill
from sklearn.metrics import r2_score
from src.exception import CustomeExceptionClass

def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,'wb') as file_obj:
            dill.dump(obj,file_obj)

    except Exception as e:
        raise CustomeExceptionClass(e,sys)
def evaluation_model(x_train,y_train,x_test,y_test,models):
    try:
        report = {}
        for model_name, model in models.items():
            model.fit(x_train, y_train)  # Fit the model
            y_train_pred = model.predict(x_train)  # Predict on train set
            y_test_pred = model.predict(x_test)  # Predict on test set

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            report[model_name] = test_model_score  # Store test score in report
        return report
    except Exception as e:
        raise CustomeExceptionClass(e,sys)