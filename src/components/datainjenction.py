import pandas as pd
import numpy as np
import os
import sys
from sklearn.model_selection import train_test_split
from src.logger import logging
from src.exception import CustmizedException
from dataclasses import dataclass
from src.components.data_transformation import DataTransfomation
from src.components.model_trainer import ModelTrainer

@dataclass
#class DataIngenctionConfig:
#    train_data_path=os.path.join('artifacts', 'train.csv')
#    test_data_path=os.path.join('artifacts','test.csv')
#    raw_data_path=os.path.join('artifacts','raw.csv')'''

class DataIngenctionConfig:
    train_data_folder = os.path.join('artifacts', 'train_data')
    test_data_folder = os.path.join('artifacts', 'test_data')
    raw_data_folder = os.path.join('artifacts', 'raw_data')

    train_data_path = os.path.join(train_data_folder, 'train.csv')
    test_data_path = os.path.join(test_data_folder, 'test.csv')
    raw_data_path = os.path.join(raw_data_folder, 'raw.csv')

    def create_data_folders(self):
        os.makedirs(self.train_data_folder, exist_ok=True)
        os.makedirs(self.test_data_folder, exist_ok=True)
        os.makedirs(self.raw_data_folder, exist_ok=True)

class DataIngenction:
    def __init__(self):
        self.ingenction_config=DataIngenctionConfig()
    
    def initialise_data_injenction(self):
        logging.info('Data Ingenction Started')
        try:
            logging.info('Data reading from pandas')
            data=pd.read_csv(os.path.join('notebook/data','income_cleandata.csv'))
            logging.info('Data reading completed')

            os.makedirs(os.path.dirname(self.ingenction_config.raw_data_path),exist_ok=True)
            data.to_csv(self.ingenction_config.raw_data_path,index=False, header=True)
            logging.info('Data split in train and test')

            train_set, test_set= train_test_split(data, test_size=0.2, random_state=43)
            os.makedirs(os.path.dirname(self.ingenction_config.train_data_path),exist_ok=True)
            train_set.to_csv(self.ingenction_config.train_data_path, index=False, header=True)
            os.makedirs(os.path.dirname(self.ingenction_config.test_data_path),exist_ok=True)
            test_set.to_csv(self.ingenction_config.test_data_path, index=False, header= True)
            logging.info('Data Ingestion completed')
            return ( self.ingenction_config.train_data_path,
                    self.ingenction_config.test_data_path)
        
        except Exception as e:
            logging.info('Error occured in Data Injenction')
            raise CustmizedException(e,sys)

if __name__=='__main__':
    obj=DataIngenction()
    train_path,test_path=obj.initialise_data_injenction()

    data_transformation =DataTransfomation()
    train_array, test_array, _ = data_transformation.initiate_data_transfromation(train_path=train_path, test_path=test_path)

    modeltrainer=ModelTrainer()
    print(modeltrainer.initialise_model_trainer(train_array,test_array))
