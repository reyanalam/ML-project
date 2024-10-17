import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomeExceptionClass
from src.logger import logging
from src.utils import save_object,evaluation_model

@dataclass
class modeltrainerconfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = modeltrainerconfig()
    
    def initiate_model_trainer(self,train_arr,test_arr):
        try:
            logging.info('split training and test input data')
            x_train,y_train,x_test,y_test=(
                train_arr[:,:-1],train_arr[:,-1],test_arr[:,:-1],test_arr[:,-1]
            )
            models = {
                'RandomForest': RandomForestRegressor(),
                'Decision Tree': DecisionTreeRegressor(),
                'Gradient Boosting':GradientBoostingRegressor(),
                'Linear Regression':LinearRegression(),
                'K-Neighbour' : KNeighborsRegressor(),
                'XGB':XGBRegressor(),
                'CatBoosting':CatBoostRegressor(verbose=False),
                'AdaBoost':AdaBoostRegressor()
            }
            model_report:dict = evaluation_model(x_train=x_train,y_train=y_train,x_test = x_test,y_test=y_test,models = models)
            best_model_score = max(model_report.values())
            best_model_name = max(model_report, key=model_report.get)
            best_model = models[best_model_name]
            if best_model_score < 0.6:
                raise CustomeExceptionClass('No best model found')
            logging.info('Found the best model')

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(x_test)
            r2_square = r2_score(y_test,predicted)
            return r2_square
        except Exception as e:
            raise CustomeExceptionClass(e,sys)

