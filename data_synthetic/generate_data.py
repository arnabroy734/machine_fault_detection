import numpy as np
import pandas as pd

class HealthyData:
    """
    This class will generate healthy data for trending
    """
    def __init__(self):
        """
        1. Read the healthy.csv file
        
        """
        try:
            self.data = pd.read_csv("./data_synthetic/healthy.csv")
        except Exception as e:
            raise Exception(f"Healthy data cannot be read - {e}")
        
    def get_data_by_id(self, idx):
        """
        Description: get_data_by_id (id): returns (single_row, id) 
        Return a single row by dataframe.iloc[id] and the id itself
        """       
        try:
            return (self.data.iloc[idx], idx)
        
        except: # Reset the id to 0 if it goes outside range
            return (self.data.iloc[0], 0)
    
    def get_column_names(self):
        """
        Return a list of column names of the existing dataframe
        """
        return list(self.data.columns)
    
    def get_bulk_data(self, start_idx, size):
        """
        Get data.iloc[start_idx:start_idx+size]
        """
        return (self.data.iloc[start_idx : start_idx+size], start_idx+size-1)
    

class FaultyData:
    """
    This class will generate faulty data for trending
    """
    def __init__(self):
        """
        1. Read the faulty.csv file
        
        """
        try:
            self.data = pd.read_csv("./data_synthetic/faulty.csv")
        except Exception as e:
            raise Exception(f"faulty data cannot be read - {e}")
        
    def get_data_by_id(self, idx):
        """
        Description: get_data_by_id (id): returns (single_row, id) 
        Return a single row by dataframe.iloc[id] and the id itself
        """       
        try:
            return (self.data.iloc[idx], idx)
        
        except: # Reset the id to 0 if it goes outside range
            return (self.data.iloc[0], 0)
    
    
    
    
