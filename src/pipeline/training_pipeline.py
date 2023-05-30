import os, sys
from src.components.data_transformation import DataTransfomation
from src.components.datainjenction import DataIngenction
from src.components.model_trainer import ModelTrainer
from src.exception import CustmizedException
from src.logger import logging
from dataclasses import dataclass

if __name__ =='__main__':
    obj=DataIngenction()
    train_data_path, test_data_path= obj.initialise_data_injenction()

    data_transformation=DataTransfomation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transfromation(train_data_path,test_data_path)

    model_train=ModelTrainer()
    model_train.initialise_model_trainer(train_arr,test_arr)