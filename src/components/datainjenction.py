import pandas as pd
import numpy as np
import os
import sys
from sklearn.model_selection import train_test_split
from src.logger import logging
from src.exception import CustmizedException
from dataclasses import dataclass

@dataclass
class DataIngenctionConfig:
    train_data_path=os.path.join('artifacts', 'train.csv')
    test_data_path=os.path.join('artifacts','test.csv')
    raw_data_path=os.path.join('artifacts','raw.csv')

class DataIngenction:
    def __init__(self):
        self.ingenction_config=DataIngenctionConfig()
    
    def initialise_data_injenction(self):
        logging.info('Data Ingenction Started')
        try:
            logging.info('Data reading from pandas')
            data=pd.read_csv(os.path.join('notebook/data','cleaned_hotel_booking.csv'))
            logging.info('Data reading completed')

            os.makedirs(os.path.dirname(self.ingenction_config.raw_data_path),exist_ok=True)
            data.to_csv(self.ingenction_config.raw_data_path,index=False, header=True)
            logging.info('Data split in train and test')

            train_set, test_set= train_test_split(data, test_size=0.2, random_state=43)
            
            train_set.to_csv(self.ingenction_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingenction_config.test_data_path, index=False, header= True)
            logging.info('Data Ingestion completed')
            return ( self.ingenction_config.train_data_path,
                    self.ingenction_config.test_data_path)
        
        except Exception as e:
            logging.info('Error occured in Data Injenction')
            raise CustmizedException(e,sys)

if __name__=='__main__':
    obj=DataIngenction()
    obj.initialise_data_injenction()