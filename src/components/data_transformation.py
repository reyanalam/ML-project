#we do feature negineering,data cleaning here

import sys
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomeExceptionClass
from src.logger import logging

from src.utils import save_object

@dataclass
class datatransformationconfig:
    preprocessor_obj_path_path = os.path.join('artifacts','preprocessor.pkl')

class datatransformation:
    def __init__(self):
        self.datatransformationconfig = datatransformationconfig()
    
    def get_data_transformer_object(self):
        try:
            cat_feature = ['gender','race_ethnicity','parental_level_of_education','lunch','test_preparation_course']  
            num_feature = ['reading_score','writing_score']

            num_pipeline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler())
                ]
            )
            cat_pipeline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('encoder',OneHotEncoder()),
                    ('scaling',StandardScaler(with_mean=False))
                ]
            )
            logging.info('Numerical column standard scaling completed')
            logging.info('Categorical column encoding completed')

            preprocessor = ColumnTransformer(
                [
                    ('num_pipeline',num_pipeline,num_feature),
                    ('cat_pipeline',cat_pipeline,cat_feature)
                ]
            )
            return preprocessor
        except Exception as e:
            raise CustomeExceptionClass(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Reading train and test data')
            logging.info('Obtaining preprocessing object')

            preprocessing_obj = self.get_data_transformer_object()
            target_feature = 'math_score'
            num_feature = ['reading_score','writing_score']

            input_feature_train_df = train_df.drop(columns=[target_feature],axis=1)
            target_feature_train_df = train_df[target_feature]

            input_feature_test_df = test_df.drop(columns=[target_feature],axis=1)
            target_feature_test_df = test_df[target_feature]

            logging.info('Applying preprocessing obj on training and testing dataframes')

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr,np.array(target_feature_train_df)
            ]
            test_arr = np.c_[
                input_feature_test_arr,np.array(target_feature_test_df)
            ]

            logging.info('saved preprocessing object')
            
            save_object(
                file_path = self.datatransformationconfig.preprocessor_obj_path_path,
                obj = preprocessing_obj
            )

            return(
                train_arr,
                test_arr,
                self.datatransformationconfig.preprocessor_obj_path_path,
            )
            
        except Exception as e:
            raise CustomeExceptionClass(e,sys)

