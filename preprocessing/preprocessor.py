from path.path import RAW_DATA, PROCESSED_DATA
import pandas as pd
from log.logging import LOGGER
import logging

class Preprocessor:
    """
    1. Read data 
    2. Drop Unnamed: 0 column
    3. Set datetime index
    4. Drop unnecessary columns 'LPS', 'Pressure_switch', 'Oil_level', 'Caudal_impulses'
    5. Add column status to label the data as healthy(0) and faulty(1)
    """ 
    def preprocess(self):
        try:
            data = pd.read_csv(RAW_DATA)
            data = data.drop(['Unnamed: 0'], axis=1)
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            data = data.set_index(['timestamp'])
            data = data.drop(['LPS', 'Pressure_switch', 'Oil_level', 'Caudal_impulses'], axis=1)
        
            data['status'] = 0
            data.loc[ '4/18/2020 0:00' : '4/18/2020 23:59' ,'status'] = 1 
            data.loc[ '5/29/2020 23:30' : '5/30/2020 6:00' ,'status'] = 1
            data.loc[ '6/5/2020 10:00' : '6/7/2020 14:30' ,'status'] = 1 
            data.loc[ '7/15/2020 14:30' : '7/15/2020 19:00' ,'status'] = 1 
        
            data.to_csv(PROCESSED_DATA)

            LOGGER.log_preprocessing("Raw data preprocessed successfullt", logging.INFO)

        
        except Exception as e:
           LOGGER.log_preprocessing(f"Error in preprocessing - {e}", logging.ERROR)