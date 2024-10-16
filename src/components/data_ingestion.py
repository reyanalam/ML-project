import os
import sys
from src.exception import CustomeExceptionClass
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import datatransformation , datatransformationconfig
@dataclass
class dataingestionconfig:
    train_data_path : str=os.path.join('artifacts','train.csv')
    test_data_path : str=os.path.join('artifacts','test.csv')
    raw_data_path : str=os.path.join('artifacts','raw.csv')

class dataingestion:
    def __init__(self):
        self.ingestion_config = dataingestionconfig()
    
    def initiatedataingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df=pd.read_csv(r'notebook\data\stud.csv')
            logging.info('Read the dataset as dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.train_data_path)

            logging.info('Train Test Split')
            train_set,test_set = train_test_split(df,test_size=0.2,random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,index = False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index = False,header=True)
            
            logging.info('Ingestion of data is completed')

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomeExceptionClass(e,sys)

if __name__ == '__main__':
    obj = dataingestion()
    train_data , test_data = obj.initiatedataingestion()

    data_transformation = datatransformation()
    data_transformation.initiate_data_transformation(train_data,test_data)
