import os
import sys
from src.logger import logging
from flask import Flask
from src.exception import CustmizedException
#from src.components.dataingenction import DataIngenction

app=Flask(__name__)

@app.route('/', methods=['GET','POST'])
def index():
    try:
        #obj=DataIngenction()
        #obj.initialise_data_injenction()
        logging.info('Artifacts folder created with raw, train and test data')
        return 'Welcome to Versatile Commerce'
        
    except Exception as e:
        abc=CustmizedException(e,sys)
        logging.info(abc.error_message)
        return 'My name is Vishal Bharate'
    

if __name__=='__main__':
    app.run(debug=True)
